"""
Combined Chat — RAG Orchestrator (v4 — Multi-hop)

Hybrid RAG with 4 parallel retrieval branches:
  1. Semantic Text Search    — Ollama embeddings → Neo4j vector index
  2. Graph Structure Search  — FastRP topology embeddings → cosine similarity
  3. Multi-hop Graph Traversal — direct variable-depth Cypher (up to 10 hops)
  4. LLM-generated Cypher    — intent-driven query for complex questions

Key improvements in v4:
  - Smart multi-hop traversal that can reach any connected node
  - Schema is enriched with the actual folder label prefix
  - Wider context (30 items vs 15) from semantic and FastRP
  - Answer prompt enforces tables for multi-item results
"""
import logging
import asyncio
import re
import difflib
from typing import List, Dict, Any, Optional
from app.combined_chat.llm_service import get_llm_service
from app.combined_chat.embedding_service import EmbeddingService
from app.combined_chat.fastrp_service import FastRPService
from app.combined_chat.gds_service import GDSCombinedService
from app.combined_chat.web_search_service import get_web_search_service
from app.db.connections import get_neo4j_driver, get_redis_client
from app.core.config import settings
import json
import time
import uuid

logger = logging.getLogger(__name__)

# Cypher execution hard timeout (seconds)
_CYPHER_TIMEOUT = 6
# FastRP structural search timeout (seconds)
_FASTRP_TIMEOUT = 4
# Orchestration (LLM intent) timeout (seconds)
_ORCH_TIMEOUT = 6
# Overall retrieval phase timeout
_RETRIEVAL_TIMEOUT = 7
# Multi-hop traversal timeout
_MULTIHOP_TIMEOUT = 5
# Query expansion timeout
_EXPANSION_TIMEOUT = 3
# Semantic node resolution timeout
_SEMANTIC_RESOLVE_TIMEOUT = 5
# Fast-path retrieval timeout
_FAST_RETRIEVAL_TIMEOUT = 4
# Fast-path context cap
_FAST_CONTEXT_CHARS = 8000
# Maximum context chars sent to synthesis LLM
_MAX_CONTEXT_CHARS = 20000


class CombinedRAGService:
    def __init__(self):
        self.llm = get_llm_service()
        self.vector_engine = EmbeddingService()
        self.fastrp_engine = FastRPService()
        self.gds_suite = GDSCombinedService()
        self.neo4j = get_neo4j_driver()
        self.redis = get_redis_client()
        self._schema_cache: Optional[str] = None
        self._schema_expiry: float = 0
        self._schema_version: int = 1
        self._chain_depth_cache: Dict[str, int] = {}  # folder_id -> max chain depth
        self._greetings = {"hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening"}

    def invalidate_schema_cache(self):
        """
        Clear the in-memory schema cache so the next query fetches fresh schema.
        Should be called after any CRUD operation that changes the graph structure
        (add/update/delete nodes or relationships).
        """
        self._schema_cache = None
        self._schema_expiry = 0
        self._schema_version += 1
        self._chain_depth_cache.clear()
        logger.info(f"🗑️ RAG schema cache invalidated (CRUD change detected, v{self._schema_version})")

    def _fast_answer_cache_key(
        self,
        question: str,
        folder_id: Optional[str],
        user_id: str,
        catalog_mode: bool,
        combined_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        normalized = self._normalize_question(question)
        scope = folder_id or "global"
        mode = "catalog" if catalog_mode else "fast"
        history_window = max(1, int(settings.RAG_FAST_HISTORY_WINDOW_MESSAGES))
        history_slice = (combined_history or [])[-history_window:]
        history_fingerprint = "|".join(
            f"{m.get('role', '')}:{self._normalize_question(str(m.get('content', '')))}"
            for m in history_slice
            if str(m.get("content", "")).strip()
        )
        digest = uuid.uuid5(
            uuid.NAMESPACE_OID,
            f"{user_id}|{scope}|{mode}|{normalized}|{history_fingerprint}|{self._schema_version}",
        )
        return f"chat:fast_answer:{digest}"

    async def _get_cached_fast_answer(
        self,
        question: str,
        folder_id: Optional[str],
        user_id: str,
        catalog_mode: bool,
        combined_history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        try:
            key = self._fast_answer_cache_key(
                question,
                folder_id,
                user_id,
                catalog_mode,
                combined_history=combined_history,
            )
            value = await self.redis.get(key)
            if value:
                logger.info("⚡ Fast-answer cache HIT")
                return str(value)
        except Exception as e:
            logger.warning(f"Fast-answer cache read failed: {e}")
        return None

    async def _set_cached_fast_answer(
        self,
        question: str,
        folder_id: Optional[str],
        user_id: str,
        catalog_mode: bool,
        answer: str,
        combined_history: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        if not answer or len(answer.strip()) < 16:
            return
        try:
            ttl = max(30, int(settings.RAG_FAST_ANSWER_CACHE_TTL_SECONDS))
            key = self._fast_answer_cache_key(
                question,
                folder_id,
                user_id,
                catalog_mode,
                combined_history=combined_history,
            )
            await self.redis.set(key, answer, ex=ttl)
        except Exception as e:
            logger.warning(f"Fast-answer cache write failed: {e}")

    async def _persist_fast_history(
        self,
        question: str,
        answer: str,
        user_id: str,
        folder_id: Optional[str],
        session_id: Optional[str],
    ) -> None:
        try:
            history_key = f"chat:history:{user_id}:{folder_id or 'global'}"
            await self.redis.rpush(history_key, json.dumps({"role": "user", "content": question}))
            await self.redis.rpush(history_key, json.dumps({"role": "assistant", "content": answer}))
            await self.redis.ltrim(history_key, -20, -1)

            if session_id:
                await self.redis.hset(
                    f"chat:meta:{session_id}",
                    mapping={
                        "folder_id": folder_id or "",
                        "updated_at": str(int(time.time())),
                    },
                )
        except Exception as e:
            logger.warning(f"Failed to persist fast chat history: {e}")

        try:
            from sqlalchemy import text as sa_text
            from app.db.connections import get_postgres_session

            db_session_id = str(session_id or uuid.uuid4())
            try:
                db_session_id = str(uuid.UUID(db_session_id))
            except Exception:
                db_session_id = str(uuid.uuid5(uuid.NAMESPACE_OID, db_session_id))

            async with get_postgres_session() as session:
                await session.execute(
                    sa_text("""
                        INSERT INTO neural_nexus.chat_history (user_id, session_id, role, message)
                        VALUES (:user_id, :session_id, 'user', :message)
                    """),
                    {"user_id": user_id, "session_id": db_session_id, "message": question}
                )
                await session.execute(
                    sa_text("""
                        INSERT INTO neural_nexus.chat_history (user_id, session_id, role, message)
                        VALUES (:user_id, :session_id, 'assistant', :message)
                    """),
                    {"user_id": user_id, "session_id": db_session_id, "message": answer}
                )
                await session.commit()
        except Exception as e:
            logger.warning(f"Failed to persist fast chat session: {e}")

    def _normalize_question(self, question: str) -> str:
        return re.sub(r"\s+", " ", question.lower().strip())

    async def _expand_query_terms(self, question: str) -> List[str]:
        """
        Intent-aware query expansion using the LLM.
        
        Goes beyond simple synonyms — understands user INTENT and generates
        database-appropriate search terms. Examples:
          - "what helps with stress?" → ["anxiety", "adaptogen", "nervine", "calming", "anxiolytic"]
          - "benefits of tulasi" → ["tulsi", "holy basil", "ocimum tenuiflorum", "sacred basil", "therapeutic uses"]
          - "what is good for skin?" → ["dermatological", "skin care", "anti-inflammatory", "wound healing", "cosmetic"]
        
        Fast: ~5s timeout, non-streaming, fails gracefully.
        """
        try:
            prompt = (
                "You are a search query optimizer for a knowledge graph database. "
                "The user asked a question but the database may use different terminology.\n\n"
                "Your job:\n"
                "1. Understand the user's INTENT (what they really want to know)\n"
                "2. Generate search terms the DATABASE might use instead\n"
                "3. Include: synonyms, scientific names, common names, related medical/technical terms, "
                "alternate spellings, abbreviations, and conceptually related terms\n\n"
                "RULES:\n"
                "- Return ONLY a comma-separated list of search terms\n"
                "- Include both the obvious synonyms AND the conceptual bridges\n"
                "- Max 10 terms, no explanations, no numbering\n"
                "- If the user uses a common name, include scientific name and vice versa\n"
                "- If the user asks about effects/benefits/uses, include the medical/pharmacological terms\n\n"
                f"User question: \"{question}\"\n\n"
                "Search terms:"
            )
            response = await asyncio.wait_for(
                self.llm.generate_response(prompt),
                timeout=6.0,
            )
            raw = response.strip().lower()
            if not raw or raw == "none" or "error" in raw:
                return []
            # Parse comma-separated terms
            terms = [t.strip().strip('"\'.-') for t in raw.split(",") if t.strip() and len(t.strip()) > 1]
            # Filter out noise and overly long terms
            terms = [t for t in terms if 1 < len(t) < 60 and t != "none" and not t.startswith("search")][:10]
            return terms
        except Exception as e:
            logger.debug(f"Intent-aware expansion failed (non-critical): {e}")
            return []

    async def _resolve_followup(self, question: str, history: List[Dict[str, str]]) -> str:
        """
        Resolve follow-up questions using conversation history.

        Detects ambiguous questions like:
          - "what are they?" → "what are the major uses of tulasi?"
          - "tell me more" → "tell me more about the therapeutic uses of tulasi"
          - "and the side effects?" → "what are the side effects of tulasi?"

        Returns the original question if it's already self-contained.
        """
        normalized = question.lower().strip()
        words_list = normalized.split()
        word_count = len(words_list)

        # Quick check: does this question likely need resolution?
        needs_resolution = False

        # If the question contains a specific entity name (capitalized word ≥ 4 chars),
        # it's self-contained — no resolution needed.
        original_words = question.strip().split()
        has_named_entity = any(
            w[0].isupper() and len(w) >= 4 and w.lower() not in {
                "what", "which", "where", "when", "who", "that", "this", "name",
                "list", "show", "find", "tell", "give", "does", "have", "they",
            }
            for w in original_words
        )
        if has_named_entity:
            return question

        # Pronoun-only / very short questions (≤ 4 words)
        short_vague_markers = [
            "they", "them", "those", "these", "that", "this", "it",
            "its", "their", "the same",
        ]
        if word_count <= 4:
            for marker in short_vague_markers:
                if marker in normalized:
                    needs_resolution = True
                    break

        # Explicit back-references in any length question
        back_references = [
            "the above", "mentioned above", "above uses", "above benefits",
            "previous", "mentioned earlier", "tell me more", "elaborate on",
            "go on", "what about",
        ]
        for ref in back_references:
            if ref in normalized:
                needs_resolution = True
                break

        # Questions starting with conjunctions are follow-ups
        if normalized.startswith(("and ", "but ", "also ", "or ")):
            needs_resolution = True

        if not needs_resolution:
            return question

        # Build a compact history summary (last 4 messages)
        recent = history[-4:]
        history_text = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:200]}"
            for m in recent
        )

        try:
            prompt = (
                "You are a question resolver. Given the conversation history and a follow-up question, "
                "rewrite the follow-up into a COMPLETE, SELF-CONTAINED question.\n\n"
                "RULES:\n"
                "- Replace all pronouns (they, it, those, etc.) with the actual entities from the conversation\n"
                "- Make the question understandable WITHOUT any conversation history\n"
                "- Keep it concise — just the rewritten question, nothing else\n"
                "- If the question is ALREADY self-contained, return it EXACTLY as-is\n"
                "- Do NOT add extra context or explanation\n\n"
                f"CONVERSATION:\n{history_text}\n\n"
                f"FOLLOW-UP QUESTION: {question}\n\n"
                "REWRITTEN QUESTION:"
            )
            response = await asyncio.wait_for(
                self.llm.generate_response(prompt),
                timeout=5.0,
            )
            rewritten = response.strip().strip('"\'')
            # Sanity check: don't accept empty or absurdly long rewrites
            if rewritten and 3 < len(rewritten) < 300:
                return rewritten
            return question
        except Exception as e:
            logger.debug(f"Follow-up resolution failed (non-critical): {e}")
            return question

    def _detect_topic_change(self, question: str, history: Optional[List[Dict[str, str]]]) -> bool:
        """
        Detect if the user's question is about a DIFFERENT topic than the conversation history.
        
        Returns True if the question is about a new topic (history should NOT be passed to LLM).
        Returns False if the question is likely a follow-up (history SHOULD be passed).
        """
        if not history or len(history) < 2:
            return False
        
        normalized = question.lower().strip()
        words = normalized.split()
        
        # If the question uses pronouns or follow-up markers, it's a continuation
        follow_up_markers = {
            "they", "them", "those", "these", "that", "this", "it",
            "its", "their", "the same", "above", "previous",
            "tell me more", "elaborate", "go on", "what about",
            "and ", "but ", "also ", "or ",
        }
        for marker in follow_up_markers:
            if marker in normalized:
                return False
        
        # Extract significant words from the question (potential entity names)
        stopwords = {
            "what", "which", "where", "when", "who", "how", "why", "does",
            "is", "are", "was", "were", "the", "and", "for", "with",
            "from", "have", "has", "been", "being", "into", "about",
            "commonly", "known", "called", "used", "belong", "belongs",
            "plant", "family", "name", "common", "scientific",
            "list", "show", "find", "tell", "give", "many", "much",
        }
        q_keywords = {w.strip("?,.'\"!") for w in words
                      if len(w.strip("?,.'\"!")) > 3 and w.strip("?,.'\"!").lower() not in stopwords}
        
        if not q_keywords:
            return False
        
        # Check if any of these keywords appeared in the last assistant answer
        last_assistant = ""
        for m in reversed(history):
            if m.get("role") == "assistant":
                last_assistant = m.get("content", "").lower()
                break
        
        if not last_assistant:
            return False
        
        # If NONE of the question's key entities appear in the last answer,
        # the user has switched topics
        overlap = sum(1 for kw in q_keywords if kw in last_assistant)
        if overlap == 0:
            logger.info(f"🔄 Topic change detected: question keywords {q_keywords} not in last answer")
            return True
        
        return False

    def _select_retrieval_mode(self, question: str, folder_id: Optional[str]) -> str:
        normalized = self._normalize_question(question)
        if normalized in self._greetings:
            return "greeting"

        deep_phrases = (
            "connected to", "connection", "connections", "relationship", "relationships",
            "path ", "paths", "chain", "travers", "bridge", "bottleneck", "cluster",
            "community", "communities", "central", "influence", "important", "rank",
            "ranking", "similar", "similarity", "predict", "missing link", "algorithm",
            "pagerank", "articlerank", "betweenness", "closeness", "degree centrality",
            "hits", "louvain", "leiden", "wcc", "kcore", "triangle", "topological",
            "multi-hop", "multihop", "why is", "why are", "how is", "how are",
        )
        if any(phrase in normalized for phrase in deep_phrases):
            return "deep"

        catalog_phrases = (
            "how many", "count ", "counts ", "list all", "what are all", "what are the",
            "which are all", "show all", "give all", "all of the", "total number", "number of",
        )
        if folder_id and any(phrase in normalized for phrase in catalog_phrases):
            return "catalog"

        return "fast"

    async def _stream_fast_answer(
        self,
        question: str,
        folder_id: Optional[str],
        file_id: Optional[str],
        combined_history: Optional[List[Dict[str, str]]],
        user_id: str,
        session_id: Optional[str],
        web_search: bool,
        catalog_mode: bool = False,
    ):
        yield json.dumps({"type": "step", "id": 2, "status": "Searching relevant sources..."}) + "\n"

        normalized_question = self._normalize_question(question)

        # ── Query Expansion: generate synonyms/alternate terms ──
        # Handles: tulasi→tulsi, stress→anxiety, benefits→uses, etc.
        expanded_terms = await self._expand_query_terms(question)
        search_question = normalized_question
        if expanded_terms:
            search_question = f"{normalized_question} {' '.join(expanded_terms)}"
            logger.info(f"🔍 Query expanded: {expanded_terms}")

        vector_task = asyncio.create_task(
            asyncio.wait_for(
                self.vector_engine.vector_search(search_question, folder_id),
                timeout=_FAST_RETRIEVAL_TIMEOUT,
            )
        )

        named_tasks: List[tuple] = [("Semantic Search", vector_task)]
        if folder_id:
            named_tasks.append((
                "Focused Entity Scan",
                asyncio.create_task(
                    asyncio.wait_for(self._focused_entity_scan(search_question, folder_id), timeout=_FAST_RETRIEVAL_TIMEOUT)
                ),
            ))
        if catalog_mode and folder_id:
            named_tasks.append((
                "Database Facts",
                asyncio.create_task(
                    asyncio.wait_for(self._type_enumeration(folder_id), timeout=_FAST_RETRIEVAL_TIMEOUT)
                ),
            ))

        web_search_task = None
        if web_search:
            logger.info("🌐 Integrated Web Search enabled")
            web_search_task = asyncio.create_task(get_web_search_service().search(question))

        names = [t[0] for t in named_tasks]
        tasks = [t[1] for t in named_tasks]
        try:
            raw_outputs = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=_FAST_RETRIEVAL_TIMEOUT + 1,
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Fast retrieval timeout ({_FAST_RETRIEVAL_TIMEOUT}s) — using partial results")
            raw_outputs = [asyncio.TimeoutError()] * len(tasks)

        fused = []
        for name, result in zip(names, raw_outputs):
            if isinstance(result, (Exception, type(None))) or not result:
                if isinstance(result, Exception):
                    logger.warning(f"Fast retrieval source '{name}' raised: {result}")
                continue
            if isinstance(result, list):
                res_str = json.dumps(result[:20], default=str)
            else:
                res_str = str(result)
            fused.append(f"=== {name} ===\n{res_str}")

        context = "\n\n".join(fused)

        # ── Intent-aware Re-retrieval: if context is thin, try again with LLM-rewritten query ──
        if len(context.strip()) < 50 and folder_id:
            logger.info("🔄 Thin context — attempting intent-aware re-retrieval...")
            yield json.dumps({"type": "step", "id": 2, "status": "Refining search with intent analysis..."}) + "\n"
            try:
                rewrite_prompt = (
                    f"The user asked: \"{question}\"\n"
                    "No direct matches were found in the knowledge graph database.\n"
                    "Think about what entities, concepts, or relationships the database MIGHT contain "
                    "that are RELATED to this question.\n"
                    "Generate 5-8 short search keywords that a knowledge graph database about "
                    "herbs, plants, biomarkers, compounds, diseases, or similar topics would likely contain.\n"
                    "Return ONLY a comma-separated list. No explanations."
                )
                rewrite_resp = await asyncio.wait_for(
                    self.llm.generate_response(rewrite_prompt), timeout=6.0
                )
                rewrite_terms = [t.strip().strip('"\'') for t in rewrite_resp.split(",")
                                if t.strip() and 1 < len(t.strip()) < 60][:8]
                if rewrite_terms:
                    rewrite_query = " ".join(rewrite_terms)
                    logger.info(f"🔄 Re-retrieval query: {rewrite_query}")
                    retry_result = await asyncio.wait_for(
                        self._focused_entity_scan(rewrite_query, folder_id),
                        timeout=_FAST_RETRIEVAL_TIMEOUT,
                    )
                    if retry_result and len(str(retry_result).strip()) > 50:
                        fused.append(f"=== Intent-Matched Re-retrieval ===\n{retry_result}")
                        context = "\n\n".join(fused)
                        logger.info(f"✅ Re-retrieval succeeded: {len(context)} chars of context")
            except Exception as e:
                logger.debug(f"Re-retrieval failed (non-critical): {e}")

        if len(context) > _FAST_CONTEXT_CHARS:
            logger.info(f"✂️ Trimming fast context from {len(context)} to {_FAST_CONTEXT_CHARS} chars")
            context = context[:_FAST_CONTEXT_CHARS]

        suggest_web_search = True
        if web_search and web_search_task:
            try:
                web_result = await web_search_task
                if web_result and not web_result.get("error"):
                    yield json.dumps({
                        "type": "web_search_result",
                        "data": {
                            "answer": web_result.get("answer", ""),
                            "sources": (
                                web_result.get("grounding_metadata", {}).get("grounding_chunks", [])
                                if web_result.get("grounding_metadata")
                                else []
                            )
                        }
                    }) + "\n"
                    suggest_web_search = False
            except Exception as e:
                logger.warning(f"🌐 Integrated Web Search failed: {e}")

        thin_context = len(context.strip()) < 50
        is_grounded = len(context.strip()) >= 50
        yield json.dumps({"type": "web_search_suggestion", "data": suggest_web_search, "emphasized": thin_context}) + "\n"
        yield json.dumps({"type": "data_grounding", "data": {"grounded": is_grounded, "source_count": len(fused), "context_chars": len(context)}}) + "\n"

        yield json.dumps({"type": "step", "id": 3, "status": "Synthesizing answer..."}) + "\n"

        full_answer = ""
        chunk_count = 0
        # SMART HISTORY: Only pass history if the question is a follow-up.
        # If it's a new topic, clear history to prevent LLM confusion.
        is_new_topic = self._detect_topic_change(question, combined_history)
        if is_new_topic:
            answer_history = []  # Don't poison the LLM with irrelevant context
            logger.info("🧹 New topic detected — history cleared for synthesis")
        else:
            answer_history = (combined_history or [])[-max(1, int(settings.RAG_FAST_HISTORY_WINDOW_MESSAGES)):]
        async for chunk in self.llm.astream_response(
            self._build_answer_prompt(question, context, folder_id, fast_mode=True),
            answer_history
        ):
            if chunk:
                full_answer += chunk
                chunk_count += 1
                yield json.dumps({"type": "content", "data": chunk}) + "\n"

        logger.info(f"📝 Streamed {chunk_count} content chunks")
        yield json.dumps({"type": "step", "id": 4, "status": "Research completed."}) + "\n"

        if not web_search:
            await self._set_cached_fast_answer(
                question=question,
                folder_id=folder_id,
                user_id=user_id,
                catalog_mode=catalog_mode,
                answer=full_answer,
                combined_history=combined_history,
            )

        await self._persist_fast_history(
            question=question,
            answer=full_answer,
            user_id=user_id,
            folder_id=folder_id,
            session_id=session_id,
        )

    # ────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────

    async def stream_answer(
        self,
        question: str,
        folder_id: Optional[str] = None,
        file_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        web_search: bool = False,
    ):
        """Streaming Orchestrator with strict folder isolation and massive parallelization."""
        t_start = time.time()

        logger.info(f"🚀 Research starting for folder {folder_id or 'global'} | Question: {question}")
        yield json.dumps({"type": "step", "id": 1, "status": "Analyzing research intent..."}) + "\n"

        # ── History ────────────────────────────────────────────
        history_key = f"chat:history:{user_id}:{folder_id or 'global'}"
        stored_history = []
        try:
            history_window = max(1, int(settings.RAG_HISTORY_WINDOW_MESSAGES))
            raw_history = await self.redis.lrange(history_key, -history_window, -1)
            stored_history = [json.loads(m) for m in raw_history]
        except Exception as e:
            logger.warning(f"Failed to load history from Redis: {e}")

        combined_history = []
        seen_msgs = set()
        for msg in (stored_history + (history or [])):
            msg_str = f"{msg.get('role', '')}:{msg.get('content', '')}"
            if msg_str not in seen_msgs and msg.get('content', '').strip():
                seen_msgs.add(msg_str)
                combined_history.append(msg)

        # ── Fast-path: Greeting ────────────────────────────────
        clean_q = question.lower().strip().strip("?!.")
        if clean_q in self._greetings:
            ans = (
                "Hello! 👋 I'm your Neural Nexus research assistant. "
                "I'm ready to explore the data in your active folder. "
                "What would you like to know?"
            )
            yield json.dumps({"type": "content", "data": ans}) + "\n"
            yield json.dumps({"type": "step", "id": 4, "status": "Done"}) + "\n"
            return

        # ── Follow-up Resolution: rewrite ambiguous questions using conversation context ──
        # Handles: "what are they?", "tell me more", "explain that", "and the side effects?"
        resolved_question = question
        if combined_history and len(combined_history) >= 2:
            resolved_question = await self._resolve_followup(question, combined_history)
            if resolved_question != question:
                logger.info(f"🔗 Follow-up resolved: '{question}' → '{resolved_question}'")

        retrieval_mode = self._select_retrieval_mode(resolved_question, folder_id)
        logger.info(f"⚡ Retrieval mode selected: {retrieval_mode}")
        if retrieval_mode in {"fast", "catalog"}:
            if not web_search:
                cached_answer = await self._get_cached_fast_answer(
                    question=question,
                    folder_id=folder_id,
                    user_id=user_id,
                    catalog_mode=(retrieval_mode == "catalog"),
                    combined_history=combined_history,
                )
                if cached_answer:
                    yield json.dumps({"type": "step", "id": 2, "status": "Using cached response..."}) + "\n"
                    yield json.dumps({"type": "web_search_suggestion", "data": False, "emphasized": False}) + "\n"
                    yield json.dumps({"type": "step", "id": 3, "status": "Finalizing answer..."}) + "\n"
                    yield json.dumps({"type": "content", "data": cached_answer}) + "\n"
                    yield json.dumps({"type": "step", "id": 4, "status": "Research completed."}) + "\n"
                    await self._persist_fast_history(
                        question=question,
                        answer=cached_answer,
                        user_id=user_id,
                        folder_id=folder_id,
                        session_id=session_id,
                    )
                    return
            async for chunk in self._stream_fast_answer(
                question=resolved_question,
                folder_id=folder_id,
                file_id=file_id,
                combined_history=combined_history,
                user_id=user_id,
                session_id=session_id,
                web_search=web_search,
                catalog_mode=(retrieval_mode == "catalog"),
            ):
                yield chunk
            return

        # Determine fallback dynamic depth
        fallback_depth = await self._get_dynamic_depth(folder_id, resolved_question)
        # Use resolved question for all retrieval in the deep path
        question = resolved_question

        # ══════════════════════════════════════════════════════════
        #  MASSIVE PARALLEL LAUNCH: Fire EVERYTHING at once
        #  - Query expansion (LLM)      → runs in parallel
        #  - Schema fetch (cached)       → runs in parallel
        #  - All 7 retrieval branches    → start immediately with raw question
        #    (don't wait for expansion — we'll re-search with expanded terms if ready)
        # ══════════════════════════════════════════════════════════

        # LLM tasks (fire and forget — will be awaited later)
        expansion_task = asyncio.create_task(
            asyncio.wait_for(
                self._expand_query(question, folder_id),
                timeout=_EXPANSION_TIMEOUT
            )
        )
        schema_task = asyncio.create_task(self._get_schema())

        # Retrieval branches that DON'T need expansion or orchestration
        # (start immediately with raw question for speed)
        vector_task = asyncio.create_task(
            self.vector_engine.vector_search(question, folder_id)
        )
        fastrp_task = asyncio.create_task(
            asyncio.wait_for(
                self.fastrp_engine.structural_search(question, folder_id),
                timeout=_FASTRP_TIMEOUT
            )
        )
        multihop_task = asyncio.create_task(
            asyncio.wait_for(
                self._multihop_traversal(question, folder_id),
                timeout=_MULTIHOP_TIMEOUT
            )
        )
        type_enum_task = asyncio.create_task(
            asyncio.wait_for(
                self._type_enumeration(folder_id),
                timeout=5
            )
        )
        entity_scan_task = asyncio.create_task(
            asyncio.wait_for(
                self._focused_entity_scan(question, folder_id),
                timeout=6
            )
        )

        web_search_task = None
        if web_search:
            logger.info("🌐 Integrated Web Search enabled")
            web_search_task = asyncio.create_task(
                get_web_search_service().search(question)
            )

        # ── Wait for expansion + schema (needed for orchestration + expansion-dependent branches) ──
        expanded_terms = []
        try:
            expanded_terms = await expansion_task
        except asyncio.TimeoutError:
            logger.warning("⏱️ Query expansion timed out")
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")

        expanded_question = question
        if expanded_terms:
            logger.info(f"🔎 Query Expansion: {expanded_terms}")
            expanded_question = question + " " + " ".join(expanded_terms)

        schema = await schema_task
        t_phase1 = time.time()
        logger.info(f"⏱️ Phase 1 (expansion+schema): {t_phase1 - t_start:.2f}s")

        # Launch expansion-dependent branches NOW (they run in parallel with already-running branches)
        property_task = asyncio.create_task(
            asyncio.wait_for(
                self._property_search(question, folder_id, expanded_terms, fallback_depth),
                timeout=6
            )
        )
        semantic_resolve_task = asyncio.create_task(
            asyncio.wait_for(
                self._semantic_node_resolution(question, folder_id, expanded_terms, fallback_depth),
                timeout=_SEMANTIC_RESOLVE_TIMEOUT
            )
        )

        # ── Orchestration (in parallel with retrieval) ──────────
        orchestration_task = asyncio.create_task(
            asyncio.wait_for(
                self._orchestrate_retrieval(question, schema, folder_id or "global", combined_history),
                timeout=_ORCH_TIMEOUT
            )
        )

        yield json.dumps({"type": "step", "id": 2, "status": "Searching the knowledge graph..."}) + "\n"

        # ── Gather ONLY the 7 retrieval branches (no orchestration!) ────
        # Orchestration runs in background — we check it AFTER retrieval finishes
        named_tasks: List[tuple] = [
            ("Semantic Search", vector_task),
            ("Structural Search (FastRP)", fastrp_task),
            ("Multi-hop Graph Traversal", multihop_task),
            ("Property Search", property_task),
            ("Semantic Node Resolution", semantic_resolve_task),
            ("Database Facts", type_enum_task),
            ("Focused Entity Scan", entity_scan_task),
        ]

        # Gather retrieval only — orchestration does NOT block this!
        names = [t[0] for t in named_tasks]
        tasks = [t[1] for t in named_tasks]
        try:
            raw_outputs = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=_RETRIEVAL_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Global retrieval timeout ({_RETRIEVAL_TIMEOUT}s) — using partial results")
            raw_outputs = [asyncio.TimeoutError()] * len(tasks)

        # ── Check if orchestration finished while retrieval was running ──
        # Give it 1 more second grace if it's still running
        if not orchestration_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(orchestration_task), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass

        if orchestration_task.done() and not orchestration_task.cancelled():
            try:
                orch_result = orchestration_task.result()
                intent = orch_result if isinstance(orch_result, dict) else {"use_cypher": False, "cypher_query": None, "use_gds": False, "gds_algo": None, "research_strategy": "Fallback"}
            except Exception as e:
                logger.warning(f"⏱️ Orchestration failed: {e}")
                intent = {"use_cypher": False, "cypher_query": None, "use_gds": False, "gds_algo": None, "research_strategy": "Timeout Fallback"}
        else:
            logger.info("⏱️ Orchestration still running — skipping Cypher/GDS to save time")
            orchestration_task.cancel()
            intent = {"use_cypher": False, "cypher_query": None, "use_gds": False, "gds_algo": None, "research_strategy": "Fast Fallback"}

        logger.info(f"🧠 Research Intent: {intent.get('research_strategy', 'Standard')}")
        if intent.get("cypher_query"):
            logger.info(f"🔗 Generated Cypher: {intent['cypher_query']}")

        # Read the AI-determined search depth
        req_depth = intent.get("search_depth")
        if isinstance(req_depth, int) and req_depth > 0:
            max_depth = min(req_depth, 20)
        else:
            max_depth = fallback_depth

        logger.info(f"📏 AI-determined chain depth for folder {folder_id}: {max_depth} hops")
        yield json.dumps({"type": "intent", "data": intent}) + "\n"

        # ── Quick second pass: Cypher/GDS (only if orchestration succeeded) ──
        algo = None
        extra_tasks: List[tuple] = []
        if intent.get("use_cypher") and intent.get("cypher_query"):
            extra_tasks.append(("Graph Traversal", asyncio.create_task(
                self._execute_cypher_safe(intent["cypher_query"], folder_id, file_id)
            )))
        if intent.get("use_gds"):
            algo = intent.get("gds_algo", "centrality")
            gds_coro = self._dispatch_gds(algo, folder_id)
            if gds_coro:
                extra_tasks.append((f"Graph Algorithm ({algo})", asyncio.create_task(gds_coro)))

        extra_names = []
        extra_outputs = []
        if extra_tasks:
            extra_names = [t[0] for t in extra_tasks]
            extra_task_list = [t[1] for t in extra_tasks]
            try:
                extra_outputs = await asyncio.wait_for(
                    asyncio.gather(*extra_task_list, return_exceptions=True),
                    timeout=4  # tight timeout for cypher/gds
                )
            except asyncio.TimeoutError:
                logger.warning("⏱️ Cypher/GDS second pass timed out")
                extra_outputs = [asyncio.TimeoutError()] * len(extra_task_list)

        t_retrieval = time.time()
        logger.info(f"⏱️ Phase 2 (retrieval): {t_retrieval - t_phase1:.2f}s")

        # ── Fuse with labels (compact serialization) ──────────
        fused = []
        # Process main retrieval results
        for name, result in zip(names, raw_outputs):
            if isinstance(result, (Exception, type(None))):
                if isinstance(result, Exception):
                    logger.warning(f"Retrieval source '{name}' raised: {result}")
                continue
            if not result:
                continue
            if isinstance(result, list):
                res_str = json.dumps(result[:30], default=str)
                if name.startswith("Graph Algorithm"):
                    current_algo = algo if algo else "analytics"
                    yield json.dumps({"type": "gds_results", "data": {"algorithm": current_algo, "results": result}}) + "\n"
            else:
                res_str = str(result)
            fused.append(f"=== {name} ===\n{res_str}")

        # Process extra (Cypher/GDS) results
        for name, result in zip(extra_names, extra_outputs):
            if isinstance(result, (Exception, type(None))):
                if isinstance(result, Exception):
                    logger.warning(f"Extra source '{name}' raised: {result}")
                continue
            if not result:
                continue
            if isinstance(result, list):
                res_str = json.dumps(result[:30], default=str)
                if name.startswith("Graph Algorithm"):
                    current_algo = algo if algo else "analytics"
                    yield json.dumps({"type": "gds_results", "data": {"algorithm": current_algo, "results": result}}) + "\n"
            else:
                res_str = str(result)
            fused.append(f"=== {name} ===\n{res_str}")

        context = "\n\n".join(fused)

        # ── Safety net: if context is empty, try a direct neighbor query ─
        if not context.strip() and folder_id:
            logger.warning("All retrieval layers returned empty – running folder neighbor safety net")
            safety_ctx = await self._folder_neighbor_context(folder_id)
            if safety_ctx:
                fused.append(f"=== Folder Overview ===\n{safety_ctx}")
                context = "\n\n".join(fused)

        # ── Cap context to prevent slow synthesis (source-aware) ──
        if len(context) > _MAX_CONTEXT_CHARS:
            logger.info(f"✂️ Trimming context from {len(context)} to {_MAX_CONTEXT_CHARS} chars")
            # Proportional trimming: give each source a fair share
            budget_per_source = _MAX_CONTEXT_CHARS // max(len(fused), 1)
            trimmed_fused = []
            for f in fused:
                if len(f) > budget_per_source:
                    trimmed_fused.append(f[:budget_per_source] + "\n[...truncated]")
                else:
                    trimmed_fused.append(f)
            context = "\n\n".join(trimmed_fused)
            # Final safety cap
            if len(context) > _MAX_CONTEXT_CHARS:
                context = context[:_MAX_CONTEXT_CHARS]

        logger.info(f"🧪 Context Fusion | Sources: {len(fused)} | Size: {len(context)} chars")

        # ── Web Search Suggestion ──────────────────────────────
        suggest_web_search = True
        if web_search and web_search_task:
            try:
                web_result = await web_search_task
                if web_result and not web_result.get("error"):
                    yield json.dumps({
                        "type": "web_search_result",
                        "data": {
                            "answer": web_result.get("answer", ""),
                            "sources": (
                                web_result.get("grounding_metadata", {}).get("grounding_chunks", [])
                                if web_result.get("grounding_metadata")
                                else []
                            )
                        }
                    }) + "\n"
                    suggest_web_search = False # Already performed
            except Exception as e:
                logger.warning(f"🌐 Integrated Web Search failed: {e}")

        thin_context = len(context.strip()) < 50
        is_grounded = len(context.strip()) >= 50
        yield json.dumps({"type": "web_search_suggestion", "data": suggest_web_search, "emphasized": thin_context}) + "\n"
        yield json.dumps({"type": "data_grounding", "data": {"grounded": is_grounded, "source_count": len(fused), "context_chars": len(context), "algorithm": algo}}) + "\n"

        # ── Step 3: Synthesize (STREAMING token-by-token) ──────
        yield json.dumps({"type": "step", "id": 3, "status": "Synthesizing research results..."}) + "\n"

        # Flush padding: small JSON chunks (<100 bytes) get buffered by TCP/proxy.
        # Adding space padding forces the buffer to flush each chunk immediately.


        full_answer = ""
        chunk_count = 0
        # SMART HISTORY: detect topic changes in deep path too
        is_new_topic = self._detect_topic_change(question, combined_history)
        deep_history = [] if is_new_topic else combined_history
        if is_new_topic:
            logger.info("🧹 New topic detected in deep path — history cleared for synthesis")
        async for chunk in self.llm.astream_response(
            self._build_answer_prompt(question, context, folder_id),
            deep_history
        ):
            if chunk:
                full_answer += chunk
                chunk_count += 1
                yield json.dumps({"type": "content", "data": chunk}) + "\n"
        
        logger.info(f"📝 Streamed {chunk_count} content chunks")

        # ── Save to PostgreSQL for persistent cross-session history ──
        try:
            from sqlalchemy import text as sa_text
            from app.db.connections import get_postgres_session

            # Convert frontend session_id to a valid UUID
            db_session_id = session_id or str(uuid.uuid4())
            try:
                db_session_id = str(uuid.UUID(db_session_id))
            except (ValueError, TypeError):
                db_session_id = str(uuid.uuid5(uuid.NAMESPACE_OID, db_session_id))

            async with get_postgres_session() as session:
                await session.execute(
                    sa_text("""
                        INSERT INTO neural_nexus.chat_history (user_id, session_id, role, message)
                        VALUES (:user_id, :session_id, 'user', :message)
                    """),
                    {"user_id": user_id, "session_id": db_session_id, "message": question}
                )
                await session.execute(
                    sa_text("""
                        INSERT INTO neural_nexus.chat_history (user_id, session_id, role, message)
                        VALUES (:user_id, :session_id, 'assistant', :message)
                    """),
                    {"user_id": user_id, "session_id": db_session_id, "message": full_answer}
                )
                await session.commit()
            logger.info(f"💾 Saved chat to PostgreSQL (session: {db_session_id})")
        except Exception as e:
            logger.warning(f"Failed to save chat history to PostgreSQL: {e}")

    async def answer(
        self,
        question: str,
        folder_id: Optional[str] = None,
        file_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """Non-streaming: collects stream_answer into a single JSON response."""
        full_answer = ""
        intent = {}
        algorithm = None
        results = None
        suggest_web_search = True
        web_search_emphasized = False
        async for chunk_raw in self.stream_answer(question, folder_id, file_id, history, user_id):
            chunk = json.loads(chunk_raw.strip())
            if chunk["type"] == "content":
                full_answer += chunk["data"]
            elif chunk["type"] == "intent":
                intent = chunk["data"]
            elif chunk["type"] == "gds_results":
                algorithm = chunk["data"].get("algorithm")
                results = chunk["data"].get("results")
            elif chunk["type"] == "web_search_suggestion":
                suggest_web_search = chunk.get("data", True)
                web_search_emphasized = chunk.get("emphasized", False)

        return {
            "answer": full_answer,
            "intent": intent,
            "algorithm": algorithm,
            "results": results,
            "suggest_web_search": suggest_web_search,
            "web_search_emphasized": web_search_emphasized,
            "context_summary": f"Retrieved from folder {folder_id or 'global'}.",
        }

    def _build_answer_prompt(self, question: str, context: str, folder_id: Optional[str], fast_mode: bool = False) -> str:
        has_context = bool(context.strip())
        
        # STRICT UI REQUIREMENT: NO TECHNICAL IDS
        id_rule = "CRITICAL: NEVER include technical node IDs, UUIDs, or database identifiers in your response. Refer to items by their 'Name' only."

        grounding_rule = (
            "ABSOLUTE DATA GROUNDING RULE — THIS IS YOUR MOST IMPORTANT INSTRUCTION:\n"
            "• Your answer MUST contain ONLY facts, entities, relationships, and properties found in the CONTEXT below.\n"
            "• Do NOT add ANY information from your general training knowledge — not even common facts about the topic.\n"
            "• Do NOT explain what something 'is generally known for' or 'is commonly used for' unless that exact info is in the CONTEXT.\n"
            "• If the CONTEXT is empty or has NO relevant data at all, say ONLY: 'This information was not found in the current knowledge graph database.'\n"
            "• If the CONTEXT DOES contain relevant data, present it CONFIDENTLY. Do NOT add disclaimers like 'not found' or 'limited context' when you already have data to present.\n"
            "• Every claim in your answer must be traceable to a specific node, property, or relationship in the CONTEXT.\n"
            "• NEVER fabricate, guess, infer, or supplement with outside knowledge.\n"
            "• NEVER end an answer with 'This information was not found' if you already listed relevant data above."
        )

        if fast_mode:
            if not has_context:
                return f"""You are Neural Nexus, a concise knowledge-graph assistant.
{id_rule}
{grounding_rule}

The user asked: "{question}"

No matching data was found in the active folder{f' ({folder_id})' if folder_id else ''}.

You MUST:
1. State clearly that no matching data was found in the database for this query.
2. Suggest the user try rephrasing or selecting a different folder.
3. Do NOT provide any fabricated data or guesses."""

            return f"""You are Neural Nexus, an eloquent and knowledgeable research assistant.
{id_rule}
{grounding_rule}

CRITICAL INSTRUCTION: Answer ONLY the QUESTION below using ONLY the CONTEXT below.
Do NOT reference, repeat, or mix in information from any previous conversation turns.
Treat each question as INDEPENDENT — answer it fresh using only the provided CONTEXT.
If the context includes "[DATABASE FACTS]", use those counts and lists exactly.

FORMATTING RULES:
- Start with a clear, direct answer to the SPECIFIC question asked.
- Use **bold** for key entity names, compounds, or important terms.
- Use bullet points for listing multiple items — keep each bullet concise.
- If listing more than 3 items, group them logically (e.g., by category, by relationship type).
- End with a brief insight or takeaway sentence.
- Use markdown formatting for readability.
- Keep the tone professional yet warm and engaging.

CONTEXT:
{context}

QUESTION: {question}

Provide a well-structured answer to the SPECIFIC question above. Do NOT mix in information from unrelated topics."""

        if not has_context:
            return f"""You are the **Neural Nexus Research Assistant** — a friendly, knowledgeable expert.
{id_rule}
{grounding_rule}

The user asked: "{question}"

No matching data was found in the active folder{f' ({folder_id})' if folder_id else ''}.

You MUST respond:
1. Clearly state that you searched the knowledge graph database but found NO matching results for this specific query.
2. Suggest possible reasons (e.g., the data might be in a different folder, or the question might need rephrasing).
3. Offer helpful follow-up suggestions based on the question topic.
4. Do NOT fabricate or guess any data. Keep your tone warm, professional, and encouraging."""

        return f"""You are Neural Nexus — a sharp, knowledgeable research assistant that delivers beautifully formatted answers.
{id_rule}
{grounding_rule}

CONTEXT:
{context}

QUESTION: {question}

FORMATTING & STYLE RULES:
1. **Opening**: Start with 1-2 sentences that directly answer the question. Use **bold** for key findings.
2. **Body**: Present the data in ONE of these formats (pick the best fit):
   - Bullet list for enumerating items, properties, or connections
   - Markdown table for comparing scores, rankings, or multi-attribute data
   - Grouped sections with sub-headers for complex multi-part answers
3. **Key entities** should be in **bold**, relationships in *italics*
4. **Tables** (when used): include a score column, short headers, wrap with context sentences
5. **Closing**: End with a brief insight, pattern observation, or actionable takeaway
6. **Tone**: Professional, warm, and engaging — like a knowledgeable colleague explaining findings
7. **Length**: Be thorough but not verbose. Quality over quantity.

HONESTY: If context is truly incomplete for a specific sub-question, say so clearly in one sentence."""



    # ────────────────────────────────────────────────────────────
    #  Schema Discovery (cached 5 min)
    # ────────────────────────────────────────────────────────────

    async def _get_schema(self) -> str:
        if self._schema_cache and time.time() < self._schema_expiry:
            return str(self._schema_cache)

        async with self.neo4j.session() as session:
            try:
                # Single optimized batch query for schema
                schema_res = await session.run("""
                    CALL {
                        CALL db.labels() YIELD label RETURN collect(label) AS labels
                    }
                    CALL {
                        CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS relTypes
                    }
                    CALL {
                        CALL db.schema.visualization() YIELD relationships
                        UNWIND relationships AS rel
                        WITH startNode(rel) AS s, type(rel) AS t, endNode(rel) AS e
                        RETURN collect(DISTINCT [labels(s)[0], t, labels(e)[0]]) AS patterns
                    }
                    RETURN labels, relTypes, patterns
                """)
                schema_data = await schema_res.single()
                
                if not schema_data:
                    return "Schema: [unavailable]"

                labels = schema_data["labels"]
                rel_types = schema_data["relTypes"]
                patterns = schema_data["patterns"]

                res_str = "KNOWLEDGE GRAPH SCHEMA:\n"
                res_str += "Node Labels: " + ", ".join(labels) + "\n"
                res_str += "Relationship Types: " + ", ".join(rel_types) + "\n"
                res_str += "Patterns:\n" + "\n".join(
                    f"  ({p[0]})-[:{p[1]}]->({p[2]})" for p in patterns if len(p) == 3
                )

                logger.info(f"📊 Schema: {len(labels)} labels, {len(rel_types)} rel types (batched)")

                self._schema_cache = res_str
                self._schema_expiry = time.time() + 600  # 10 min cache
                return res_str
            except Exception as e:
                logger.error(f"Schema discovery failed: {e}")
                return "Labels: [Unknown], Patterns: [Unknown]"

    # ────────────────────────────────────────────────────────────
    #  Dynamic Traversal Depth Calculation
    # ────────────────────────────────────────────────────────────

    async def _get_dynamic_depth(self, folder_id: str, question: str) -> int:
        """
        Determines the maximum relationship traversal depth dynamically.
        1. If the user explicitly asks for N hops/degrees/steps, use that (up to 100).
        2. Otherwise, default to 6 hops to handle deep chains (like Herb→Part→Compound→Effect)
           without causing memory explosion in dense graphs.
        """
        import re
        # Look for explicit depth requests in the user's question
        match = re.search(r'(\d+)\s*(?:hops?|degrees?|steps?|levels?|jumps?)', question, re.IGNORECASE)
        if match:
            requested_depth = int(match.group(1))
            # Cap at 100 strictly to prevent neo4j from running out of memory
            # and limit to at least 1 hop
            return min(max(requested_depth, 1), 100)
            
        # Default to 6: deep enough for complex chains, shallow enough to run < 1 second
        return 6

    # ────────────────────────────────────────────────────────────
    #  Intent Orchestration
    # ────────────────────────────────────────────────────────────

    async def _orchestrate_retrieval(self, question: str, schema: str, folder_id: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        folder_label = f"F_{folder_id.replace('-', '_')}"
        
        # Format recent history for the prompt (last 4 turns)
        history_context = ""
        if history and len(history) > 0:
            recent_msgs = history[-4:]
            history_lines = []
            for m in recent_msgs:
                role = m.get('role', 'user').upper()
                content = m.get('content', '').replace('\n', ' ')
                history_lines.append(f"{role}: {content}")
            history_context = "\nRECENT CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n"

        prompt = f"""
TASK: Orchestrate data retrieval for a Knowledge Graph Research Agent.
FOLDER LABEL: {folder_label}
SCHEMA (actual graph structure for this folder):
{schema}
{history_context}

CRITICAL CONTEXT:
- The graph uses folder-scoped labels. Every node in this folder carries the label `{folder_label}`.
- Use ONLY the relationship types and node labels listed in the SCHEMA above — do NOT invent new ones.
- The data domain is UNKNOWN — it could be pharma, finance, legal, social, or anything. Adapt your query to the actual schema.
- CONVERSATION CONTINUITY: The user's question might refer to answers or entities from the RECENT CONVERSATION HISTORY (e.g., using "them", "it", "those"). Use the history to figure out what entity type or specific name they are asking about before deciding on the query.

MULTI-HOP REASONING RULES:
- Answers often require traversing 2-6 hops through the graph.
- For connections between two entity types that are not directly linked, always try variable-length paths.
- ALWAYS use UNDIRECTED relationship patterns (no arrow) to catch relationships in BOTH directions:
  Use: MATCH (a:{folder_label})-[*1..5]-(b:{folder_label}) WHERE a.name IS NOT NULL RETURN ...
  NEVER use -[*1..5]-> (directed). ALWAYS use -[*1..5]- (undirected).
  Reason: Relationship directions in the graph are inconsistent — some go A→B, others B→A.
  Undirected patterns catch BOTH, ensuring no data is missed.
- (EXAMPLE ONLY — for illustration — actual domain may differ):
  e.g. Person -[:WorksAt]- Company -[:Located]- City
  e.g. Product -[:HasComponent]- Material -[:SourcedFrom]- Country

SOP:
1. PATH DISCOVERY: ALWAYS use undirected -[*1..5]-(m) for multi-hop questions. NEVER use directed ->. Never assume direct 1-hop connection.
2. RELATIONSHIPS: Use ONLY rel types from the SCHEMA. Never hallucinate.
3. NEO4J 5 SYNTAX: Use COUNT {{{{ (n)--() }}}} not size((n)--()).
4. NO HARDCODING: Use label comparisons and patterns. Never hardcode node names.
5. TYPE MATCHING: When filtering by entity type, ALWAYS use BOTH methods to catch all nodes:
   - Label-based: MATCH (n:TypeLabel_{folder_label})
   - Property-based (fallback for manually-created nodes): OR toLower(n.type) = 'typename'
   Example: MATCH (n:{folder_label}) WHERE (n:Student_{folder_label} OR toLower(n.type) = 'student') RETURN count(n)
   This is CRITICAL because some nodes use labels while others use the `type` property.

GDS ALGORITHM SELECTION — set `use_gds: true` and pick the right `gds_algo` ONLY when the question genuinely needs graph-algorithmic analysis:

INTELLIGENT GDS DECISION GUIDE:
- DO NOT use GDS for simple lookups, listings, counting, or direct relationship queries. These are handled perfectly by Cypher alone.
  Examples that do NOT need GDS: "How many students?", "List all herbs", "What is X connected to?", "What are the properties of Y?"
- DO use GDS when the question requires mathematical graph analysis that Cypher alone cannot provide accurately:
  • Ranking by structural importance (not just counting connections — PageRank considers the quality of connections, not just quantity)
  • Finding hidden communities or clusters that are not obvious from labels
  • Measuring similarity between nodes based on neighborhood overlap
  • Finding bridges/bottlenecks in the network
  • Predicting missing connections
  Examples that NEED GDS: "Which is the most important hub?", "Find natural clusters", "What nodes are structurally similar?", "Which node is the biggest bottleneck?", "Rank all herbs by influence"
- GDS and Cypher are NOT mutually exclusive. You can set BOTH `use_cypher: true` AND `use_gds: true` when the question benefits from both direct data AND algorithmic analysis.
- Be ACCURATE: only trigger GDS when the algorithm genuinely adds insight the Cypher query cannot.

— CENTRALITY (Who/What is most important?) —
• "pagerank": Influence, importance, popularity, "best", "most central", ranking, "most important".
• "articlerank": Like PageRank but better for diverse graphs. "Most authoritative".
• "betweenness": Bridges, bottlenecks, connecting groups, flow control.
• "closeness": Reachability, "closest to all others", central access point.
• "degree": Most connections, "most active", "most linked", "most connected", "hub".
• "hits": Hubs vs authorities. "Which are hubs" vs "which are authorities".

— COMMUNITY DETECTION (How is the data grouped?) —
• "louvain": Communities, clusters, groups, "who belongs together", "natural cluster".
• "leiden": Same as louvain but higher quality. "Better clustering".
• "wcc": Islands, disconnected parts, "isolated groups".
• "kcore": Core structure, "tight-knit core", inner circle vs periphery.
• "triangle_count": Local density, "tight clusters", "tightly-knit groups".

— SIMILARITY —
• "similarity": "What is similar to X", "which nodes share neighbors", "most similar", Jaccard.

— LINK PREDICTION (What connections are missing?) —
• "link_prediction_common": Predict missing links by shared neighbors.
• "link_prediction_adamic": Advanced link prediction weighted by rare connections.
• "link_prediction_resource": Flow-based link prediction.

— TOPOLOGY —
• "topological_sort": Logical sequence, dependency order, timeline for DAGs.

Question: {question}

JSON OUTPUT:
{{
  "use_cypher": bool,
  "cypher_query": "CYPHER or null",
  "use_gds": bool,
  "gds_algo": "pagerank|articlerank|betweenness|closeness|degree|hits|louvain|leiden|wcc|kcore|triangle_count|similarity|link_prediction_common|link_prediction_adamic|link_prediction_resource|topological_sort|null",
  "research_strategy": "brief description",
  "search_depth": int // Estimated number of hops needed to traverse the graph to answer this question. Example: 1 for direct mapping, 3 for components, 5 for deep supply chain. Defaults to 5 if unsure. Cap at 10.
}}
        """
        try:
            return await self.llm.generate_json(prompt)
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {
                "use_cypher": False,
                "cypher_query": None,
                "use_gds": False,
                "gds_algo": None,
                "research_strategy": "Fallback",
                "search_depth": 6
            }

    # ────────────────────────────────────────────────────────────
    #  Cypher Execution (with timeout + recovery)
    # ────────────────────────────────────────────────────────────

    async def _execute_cypher_safe(self, cypher: str, folder_id: str, file_id: Optional[str] = None) -> str:
        """Execute Cypher with a hard timeout. On failure, run a safe fallback query."""
        try:
            return await asyncio.wait_for(
                self._execute_cypher(cypher, folder_id, file_id),
                timeout=_CYPHER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Cypher timed out after {_CYPHER_TIMEOUT}s — running fallback")
            return await self._cypher_fallback(folder_id)
        except Exception as e:
            logger.error(f"Cypher wrapper error: {e}")
            return await self._cypher_fallback(folder_id)

    async def _execute_cypher(self, cypher: str, folder_id: str, file_id: Optional[str] = None) -> str:
        if not cypher:
            return ""
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()

        # Security: Block write operations
        if any(keyword in cypher.upper() for keyword in ["CREATE", "DELETE", "SET ", "MERGE", "REMOVE"]):
            return "[Security Error]: Write operations blocked."

        try:
            async with self.neo4j.session() as session:
                res = await session.run(cypher)
                data = await res.data()
                logger.info(f"📊 Cypher Success | {len(data)} results found.")
                if not data:
                    return ""
                return f"[Graph Results (Scoped to Folder {folder_id})]:\n{json.dumps(data, indent=2, default=str)[:4000]}"
        except Exception as e:
            logger.error(f"❌ Cypher Execution Failed: {str(e)}")
            # Instead of returning an error to the LLM context, try a safe fallback
            return await self._cypher_fallback(folder_id)

    async def _cypher_fallback(self, folder_id: str) -> str:
        """Safe fallback: get top connected nodes in this folder."""
        if not folder_id:
            return ""
        try:
            async with self.neo4j.session() as session:
                res = await session.run("""
                    MATCH (n:Entity {folder_id: $folder_id})-[r]-(m:Entity {folder_id: $folder_id})
                    WITH n.name AS entity, n.type AS type, 
                         collect(DISTINCT type(r) + ' → ' + m.name)[..5] AS connections,
                         count(r) AS degree
                    ORDER BY degree DESC
                    LIMIT 10
                    RETURN entity, type, connections, degree
                """, folder_id=folder_id)
                data = await res.data()
                if data:
                    return f"[Graph Overview (Folder {folder_id})]:\n{json.dumps(data, indent=2, default=str)}"
                return ""
        except Exception as e:
            logger.warning(f"Cypher fallback also failed: {e}")
            return ""

    # ────────────────────────────────────────────────────────────
    #  Multi-hop Graph Traversal (Smart path engine)
    # ────────────────────────────────────────────────────────────

    async def _multihop_traversal(self, question: str, folder_id: Optional[str]) -> str:
        """
        Generic multi-hop graph traversal — works for ANY domain.
        Does NOT hardcode any entity types, relationship names, or domain knowledge.
        Auto-discovers the active graph structure from the folder label and
        runs two lean parallel queries:
          1. 2-hop neighbor scan  — fast, broad coverage
          2. 3-hop chain sample   — discovers compound connections
        Both queries target only folder-scoped nodes via the folder label.
        """
        if not folder_id:
            return ""

        folder_label = f"F_{folder_id.replace('-', '_')}"

        try:
            async with self.neo4j.session() as session:

                # ── Query 1: 2-hop neighbors (fast, broad) ──
                # Uses UNDIRECTED patterns to catch relationships in BOTH directions.
                # This is critical: Herb→HAS_PART→PlantPart and PlantPart→CONTAINS→Phytoconstituent
                # may have different directions, so directed queries miss entire branches.
                q1 = f"""
                    MATCH (a:{folder_label})-[r1]-(b:{folder_label})
                    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
                    OPTIONAL MATCH (b)-[r2]-(c:{folder_label})
                    WHERE c.name IS NOT NULL AND c <> a
                    RETURN
                        a.name AS source,
                        type(r1) AS rel1,
                        b.name AS target1,
                        type(r2) AS rel2,
                        c.name AS target2
                    LIMIT 80
                """

                # ── Query 2: 3-hop path sample (richer chains) ──
                # Discovers longer chains like Herb→PlantPart→Phytoconstituent→Biomarker
                # Undirected to follow relationship chains regardless of direction.
                q2 = f"""
                    MATCH (a:{folder_label})-[r1]-(b:{folder_label})-[r2]-(c:{folder_label})-[r3]-(d:{folder_label})
                    WHERE a.name IS NOT NULL AND d.name IS NOT NULL
                      AND a <> d AND a <> c AND b <> d
                    RETURN
                        a.name AS node_a,
                        type(r1) AS rel_ab,
                        b.name AS node_b,
                        type(r2) AS rel_bc,
                        c.name AS node_c,
                        type(r3) AS rel_cd,
                        d.name AS node_d
                    LIMIT 100
                """

                # Run both queries sequentially in the same session
                # (Neo4j async sessions are single-use per statement)
                async def _fetch(q: str):
                    try:
                        r = await session.run(q)
                        return await r.data()
                    except Exception as ex:
                        return ex

                res1 = await _fetch(q1)
                res2 = await _fetch(q2)

                results = []
                if isinstance(res1, list) and res1:
                    results.append(f"[2-hop Graph Connections]:\n{json.dumps(res1, default=str)}")
                elif isinstance(res1, Exception):
                    logger.warning(f"Multi-hop Q1 failed: {res1}")

                if isinstance(res2, list) and res2:
                    results.append(f"[3-hop Entity Chains]:\n{json.dumps(res2, default=str)}")
                elif isinstance(res2, Exception):
                    logger.warning(f"Multi-hop Q2 failed: {res2}")

                if results:
                    logger.info(f"🔗 Multi-hop: {len(res1) if isinstance(res1, list) else 0} 2-hop rows, {len(res2) if isinstance(res2, list) else 0} 3-hop chains")
                    return "\n\n".join(results)

                return ""
        except Exception as e:
            logger.warning(f"Multi-hop traversal failed: {e}")
            return ""


    # ────────────────────────────────────────────────────────────
    #  LLM Query Expansion (turns "stress" → ["stress physiological", ...])
    # ────────────────────────────────────────────────────────────

    async def _expand_query(self, question: str, folder_id: Optional[str]) -> List[str]:
        """
        Uses the LLM to expand the user's question into possible node names
        that could exist in the graph. This bridges the gap between how users
        phrase questions and how entities are actually named in the graph.

        Example:
          Input:  "What helps for stress?"
          Output: ["stress physiological", "psychological stress",
                   "stress response", "anxiety", "cortisol"]
        """
        if not folder_id:
            return []

        folder_label = f"F_{folder_id.replace('-', '_')}"

        # First, get a sample of actual node names from the folder so the LLM
        # can understand the naming conventions used in this specific dataset
        try:
            async with self.neo4j.session() as session:
                res = await session.run(f"""
                    MATCH (n:{folder_label})
                    WHERE n.name IS NOT NULL
                    RETURN n.name AS name
                    LIMIT 60
                """
                )
                sample_rows = await res.data()
                sample_names = [r["name"] for r in sample_rows]
        except Exception as e:
            logger.warning(f"Failed to fetch sample names for query expansion: {e}")
            sample_names = []

        if not sample_names:
            return []

        prompt = f"""You are a search query expander for a knowledge graph.

The user asked: "{question}"

Here are REAL node names from the graph (these are examples of the naming convention):
{json.dumps(sample_names[:50], indent=0)}

Your task: Based on the naming patterns above, generate a list of possible node names
that the user might be referring to. Think about:
- Partial matches (user says "stress" but node might be "stress physiological")
- Scientific/formal variants (user says "turmeric" but node is "curcuma longa")
- Related concepts (user says "anxiety" but there's a "stress" node)
- Alternate phrasing (user says "blood pressure" but node is "hypertension")

IMPORTANT:
- Look at the actual names above to understand the naming pattern
- Include ONLY terms that might realistically match nodes in THIS graph
- Return 3-8 expanded terms
- Do NOT include the original question words themselves

Return ONLY a JSON array of strings, nothing else.
Example: ["stress physiological", "anxiety disorder", "cortisol"]"""

        try:
            result = await self.llm.generate_json(prompt)
            if isinstance(result, list):
                expanded = [str(t).lower().strip() for t in result if isinstance(t, str) and len(str(t).strip()) > 2]
                logger.info(f"🔎 Query expanded: {expanded}")
                return expanded[:8]
            elif isinstance(result, dict) and "error" not in result:
                # Sometimes it returns {"terms": [...]} or similar
                for v in result.values():
                    if isinstance(v, list):
                        return [str(t).lower().strip() for t in v if isinstance(t, str)][:8]
            return []
        except Exception as e:
            logger.warning(f"Query expansion LLM call failed: {e}")
            return []

    # ────────────────────────────────────────────────────────────
    #  Semantic Node Resolution (vector → neighborhood traversal)
    # ────────────────────────────────────────────────────────────

    async def _semantic_node_resolution(self, question: str, folder_id: Optional[str], expanded_terms: List[str] = None, max_depth: int = 6) -> str:
        """
        Finds the closest semantically matching nodes using vector similarity,
        then fetches their full neighborhood (relationships + connected nodes + properties).

        Domain-agnostic: works with any graph data — traverses up to max_depth hops
        to discover the full relationship chain from matched nodes.
        1. Vector similarity finds closest nodes even with partial/informal names
        2. Then traverses ALL relationships dynamically to find connected entities
        3. Returns both the matched nodes AND their full relationship context
        """
        if not folder_id:
            return ""

        folder_label = f"F_{folder_id.replace('-', '_')}"

        # Step 1: Find semantically similar nodes via vector search
        try:
            # Embed the question
            from app.services.ai_service import get_ai_service
            ai = get_ai_service()
            emb = await ai.embed(question)
        except Exception as e:
            logger.warning(f"Semantic resolution embedding failed: {e}")
            return ""

        try:
            async with self.neo4j.session() as session:
                # Vector similarity search to find closest nodes
                vector_query = f"""
                    CALL db.index.vector.queryNodes('{settings.VECTOR_INDEX_NAME}', $top_k, $emb)
                    YIELD node, score
                    WHERE node.name IS NOT NULL
                      AND ($folder_id IS NULL OR node.folder_id = $folder_id)
                      AND score > 0.5
                    RETURN
                        node.name AS name,
                        labels(node) AS labels,
                        properties(node) AS props,
                        score
                    ORDER BY score DESC
                    LIMIT 8
                """
                res = await session.run(vector_query, emb=emb, folder_id=folder_id, top_k=30)
                matched_nodes = await res.data()

                if not matched_nodes:
                    logger.info("[SemanticResolve] No vector matches above threshold")
                    return ""

                logger.info(f"[SemanticResolve] Found {len(matched_nodes)} semantic matches: {[r['name'] for r in matched_nodes]}")

                # Step 2: For each matched node, fetch its MULTI-HOP neighborhood
                # This is crucial: data like Herb→PlantPart→Compound→TherapeuticUse
                # requires 3+ hops to connect the full chain
                resolved_names = [r["name"] for r in matched_nodes[:8]]

                # Query A: Direct 1-hop neighbors (fast, essential)
                q_1hop = f"""
                    MATCH (n:{folder_label})
                    WHERE n.name IN $names
                    OPTIONAL MATCH (n)-[r]-(neighbor:{folder_label})
                    WHERE neighbor.name IS NOT NULL
                    RETURN
                        n.name AS source_node,
                        properties(n) AS source_props,
                        type(r) AS relationship,
                        neighbor.name AS connected_to,
                        labels(neighbor) AS connected_labels,
                        properties(neighbor) AS connected_props
                    LIMIT 100
                """
                res2 = await session.run(q_1hop, names=resolved_names)
                neighborhood = await res2.data()

                # Query B: Dynamic multi-hop extended chain
                # Traverses up to the auto-detected max chain depth for this graph
                q_multihop = f"""
                    MATCH (n:{folder_label})
                    WHERE n.name IN $names
                    MATCH path = (n)-[*1..{max_depth}]-(far:{folder_label})
                    WHERE far.name IS NOT NULL AND far <> n
                    WITH n, far, [r IN relationships(path) | type(r)] AS rel_chain,
                         [nd IN nodes(path) | nd.name] AS node_chain
                    RETURN DISTINCT
                        n.name AS source_node,
                        far.name AS connected_to,
                        labels(far) AS connected_labels,
                        properties(far) AS connected_props,
                        rel_chain,
                        node_chain
                    LIMIT 120
                """
                try:
                    res3 = await session.run(q_multihop, names=resolved_names)
                    multihop_data = await res3.data()
                except Exception as mh_err:
                    logger.warning(f"Multi-hop neighborhood query failed: {mh_err}")
                    multihop_data = []

                if not neighborhood:
                    # Return just the matched nodes with their properties
                    lines = []
                    for r in matched_nodes:
                        props = r.get("props") or {}
                        clean = {k: v for k, v in props.items()
                                 if k not in ("id", "embedding", "folder_id", "file_id",
                                              "fastrp_embedding", "created_at", "updated_at")
                                 and v not in (None, "", [], {})}
                        lines.append(f"  Node: {r['name']} (score: {r['score']:.2f}) | {json.dumps(clean, default=str)}")
                    return "[Semantically Resolved Nodes]:\n" + "\n".join(lines)

                # Step 3: Format the rich neighborhood context
                lines = []
                current_node = None

                # Format 1-hop results
                for r in neighborhood:
                    src = r.get("source_node")
                    if src != current_node:
                        current_node = src
                        src_props = r.get("source_props") or {}
                        clean_src = {k: v for k, v in src_props.items()
                                     if k not in ("id", "embedding", "folder_id", "file_id",
                                                  "fastrp_embedding", "created_at", "updated_at")
                                     and v not in (None, "", [], {})}
                        lines.append(f"\n  📌 {src} | Properties: {json.dumps(clean_src, default=str)}")

                    rel = r.get("relationship", "?")
                    target = r.get("connected_to", "")
                    target_props = r.get("connected_props") or {}
                    clean_target = {k: v for k, v in target_props.items()
                                    if k not in ("id", "embedding", "folder_id", "file_id",
                                                 "fastrp_embedding", "created_at", "updated_at")
                                    and v not in (None, "", [], {})}
                    if target:
                        lines.append(f"    —[{rel}]→ {target} | {json.dumps(clean_target, default=str)}")

                # Format multi-hop chain results (the crucial addition)
                if multihop_data:
                    lines.append("\n  🔗 Extended chains (multi-hop):")
                    seen_chains = set()
                    for r in multihop_data:
                        src = r.get("source_node", "?")
                        target = r.get("connected_to", "?")
                        rel_chain = r.get("rel_chain", [])
                        node_chain = r.get("node_chain", [])
                        chain_key = f"{src}->{target}"
                        if chain_key in seen_chains:
                            continue
                        seen_chains.add(chain_key)

                        # Format as: Stress physiological →[TREATED_BY]→ Geraniol →[FOUND_IN]→ ... →[PART_OF]→ Centella asiatica
                        chain_str = node_chain[0] if node_chain else src
                        for i, rel_name in enumerate(rel_chain):
                            next_node = node_chain[i + 1] if i + 1 < len(node_chain) else target
                            chain_str += f" →[{rel_name}]→ {next_node}"

                        target_props = r.get("connected_props") or {}
                        clean_t = {k: v for k, v in target_props.items()
                                   if k not in ("id", "embedding", "folder_id", "file_id",
                                                "fastrp_embedding", "created_at", "updated_at")
                                   and v not in (None, "", [], {})}
                        lines.append(f"    {chain_str}" + (f" | {json.dumps(clean_t, default=str)}" if clean_t else ""))

                total_connections = len(neighborhood) + len(multihop_data)
                result = f"[Semantic Node Resolution — matched nodes + full neighborhood (up to {max_depth} hops)]:\n" + "\n".join(lines)
                logger.info(f"🧩 Semantic Resolution: {len(matched_nodes)} nodes, {total_connections} connections (1-hop: {len(neighborhood)}, multi-hop: {len(multihop_data)})")
                return result

        except Exception as e:
            logger.warning(f"Semantic node resolution failed: {e}")
            return ""

    # ────────────────────────────────────────────────────────────
    #  Focused Entity Scan (exhaustive neighborhood for mentioned entities)
    # ────────────────────────────────────────────────────────────

    async def _focused_entity_scan(self, question: str, folder_id: Optional[str]) -> str:
        """
        When the question mentions a specific entity (e.g. 'Tamarind', 'Aspirin'),
        finds that entity and exhaustively traverses ALL its connections up to 4 hops.

        Unlike semantic search (top-K limited) or multi-hop (random scan),
        this is TARGETED: it starts from the mentioned entity and follows
        EVERY relationship chain in BOTH directions with NO result limit.

        This ensures questions like "What plant parts of Tamarind..." get
        ALL plant parts AND all their connections (phytoconstituents, biomarkers, etc.)
        """
        if not folder_id:
            return ""

        folder_label = f"F_{folder_id.replace('-', '_')}"

        # Extract potential entity names from question (words > 3 chars, not stopwords)
        stopwords = {"what", "which", "where", "when", "that", "this", "those",
                      "there", "their", "them", "they", "with", "from", "have",
                      "does", "about", "along", "mention", "used", "medicinally",
                      "contains", "contain", "plant", "parts", "list", "give",
                      "show", "find", "tell", "many", "much", "most", "more",
                      "some", "other", "also", "been", "being", "into", "each",
                      "only", "your", "very", "just"}
        words = [w.strip("?,.'\"!").lower() for w in question.split()
                 if len(w.strip("?,.'\"!")) > 3 and w.strip("?,.'\"!").lower() not in stopwords]

        if not words:
            return ""

        try:
            _SKIP_PROPS = ['id', 'embedding', 'folder_id', 'file_id', 'fastrp_embedding',
                           'created_at', 'updated_at', 'source_count']

            async with self.neo4j.session() as session:
                # Step 1: Find keyword matches across ALL string properties (not just name).
                # This catches commonName, scientificName, synonyms, origin, family, etc.
                keywords = words[:6]
                find_query = f"""
                    MATCH (n:{folder_label})
                    WHERE n.name IS NOT NULL
                      AND any(kw IN $keywords WHERE
                          toLower(n.name) CONTAINS kw
                          OR toLower(coalesce(n.description, '')) CONTAINS kw
                          OR toLower(coalesce(n.commonName, '')) CONTAINS kw
                          OR toLower(coalesce(n.scientificName, '')) CONTAINS kw
                          OR toLower(coalesce(n.family, '')) CONTAINS kw
                          OR toLower(coalesce(n.origin, '')) CONTAINS kw
                          OR toLower(coalesce(n.text, '')) CONTAINS kw
                          OR toLower(coalesce(toString(n.type), '')) CONTAINS kw
                      )
                    RETURN DISTINCT n.name AS name
                    LIMIT 8
                """
                res = await session.run(find_query, keywords=keywords, skip_props=_SKIP_PROPS)
                found = await res.data()

                # Step 1b: If nothing matched, do typo-tolerant fuzzy candidate lookup
                # against ALL string property values (not just name).
                if not found:
                    names_query = f"""
                        MATCH (n:{folder_label})
                        WHERE n.name IS NOT NULL
                        WITH n, n.name AS name,
                             [val IN [n.description, n.commonName, n.scientificName, n.family, n.origin, n.text, toString(n.type)]
                              WHERE val IS NOT NULL | toLower(toString(val))] AS prop_values
                        RETURN DISTINCT name, prop_values
                        LIMIT 4000
                    """
                    names_res = await session.run(names_query, skip_props=_SKIP_PROPS)
                    rows = await names_res.data()
                    # Build a flat list of all searchable text per node
                    candidates = []
                    candidate_names = {}  # searchable_text → node name
                    for row in rows:
                        node_name = str(row.get("name", "")).strip()
                        if not node_name:
                            continue
                        # Add the name itself
                        candidates.append(node_name.lower())
                        candidate_names[node_name.lower()] = node_name
                        # Add all property values
                        for pval in (row.get("prop_values") or []):
                            pval_str = str(pval).strip().lower()
                            if pval_str and len(pval_str) > 2:
                                candidates.append(pval_str)
                                candidate_names[pval_str] = node_name

                    fuzzy_hits = set()
                    for kw in keywords:
                        for hit in difflib.get_close_matches(kw, candidates, n=5, cutoff=0.65):
                            matched_name = candidate_names.get(hit)
                            if matched_name:
                                fuzzy_hits.add(matched_name)
                    if fuzzy_hits:
                        found = [{"name": n} for n in fuzzy_hits][:8]

                if not found:
                    return ""

                entity_names = [r["name"] for r in found]
                logger.info(f"🎯 Focused scan: found entities {entity_names} from question keywords")

                # Step 2: Exhaustive 4-hop neighborhood scan (undirected, no limit)
                # This follows ALL relationship chains from the found entities
                scan_query = f"""
                    MATCH (root:{folder_label})
                    WHERE root.name IN $names
                    MATCH path = (root)-[*1..4]-(connected:{folder_label})
                    WHERE connected.name IS NOT NULL AND connected <> root
                    WITH root, connected,
                         [r IN relationships(path) | type(r)] AS rel_chain,
                         [nd IN nodes(path) | nd.name] AS node_chain,
                         length(path) AS hops
                    RETURN DISTINCT
                        root.name AS from_entity,
                        connected.name AS to_entity,
                        connected.type AS to_type,
                        rel_chain,
                        node_chain,
                        hops
                    ORDER BY hops, from_entity, to_type
                    LIMIT 300
                """
                res2 = await session.run(scan_query, names=entity_names)
                paths = await res2.data()

                if not paths:
                    return ""

                # Format as structured context
                lines = [f"[FOCUSED ENTITY SCAN — Complete neighborhood for: {', '.join(entity_names)}]:"]
                by_entity: dict = {}
                for p in paths:
                    root = p["from_entity"]
                    chain_str = " → ".join(
                        f"({p['node_chain'][i]})-[{p['rel_chain'][i]}]"
                        for i in range(min(len(p['node_chain'])-1, len(p['rel_chain'])))
                    ) + f" → ({p['to_entity']})"

                    if root not in by_entity:
                        by_entity[root] = []
                    by_entity[root].append(f"    {chain_str}")

                for entity, chains in by_entity.items():
                    lines.append(f"  From '{entity}' ({len(chains)} paths):")
                    # Show all unique paths
                    unique_chains = list(dict.fromkeys(chains))  # deduplicate preserving order
                    for c in unique_chains[:60]:
                        lines.append(c)

                logger.info(f"🎯 Focused scan: {len(paths)} paths from {len(entity_names)} entities")
                return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Focused entity scan failed: {e}")
            return ""

    # ────────────────────────────────────────────────────────────
    #  Direct Type Enumeration (authoritative database counts)
    # ────────────────────────────────────────────────────────────

    async def _type_enumeration(self, folder_id: Optional[str]) -> str:
        """
        Provides authoritative, complete entity counts and names grouped by type.

        This is the GROUND TRUTH source for questions like:
        - "How many students?"    → exact count from database
        - "List all courses"      → complete list, no approximation

        Unlike semantic search (top-K limited) or multi-hop (direction-dependent),
        this queries ALL nodes in the folder using the `type` property.
        The result is always complete and accurate.
        """
        if not folder_id:
            return ""

        folder_label = f"F_{folder_id.replace('-', '_')}"

        try:
            async with self.neo4j.session() as session:
                # Query ALL nodes grouped by type with counts and full name lists
                query = f"""
                    MATCH (n:{folder_label})
                    WHERE n.name IS NOT NULL AND n.type IS NOT NULL
                    WITH n.type AS entity_type, n.name AS name
                    ORDER BY entity_type, name
                    WITH entity_type, collect(DISTINCT name) AS names, count(DISTINCT name) AS total
                    RETURN entity_type, total, names
                    ORDER BY total DESC
                """
                result = await session.run(query)
                data = await result.data()

                if not data:
                    return ""

                # Format as clear, authoritative facts for the LLM
                lines = ["[DATABASE FACTS — Authoritative entity counts from the graph database]:"]
                total_entities = 0
                for row in data:
                    etype = row["entity_type"]
                    count = row["total"]
                    names = row["names"]
                    total_entities += count
                    # Show all names for types with ≤ 30 entities, truncate for larger types
                    if count <= 30:
                        names_str = ", ".join(names)
                        lines.append(f"  {etype}: {count} total — [{names_str}]")
                    else:
                        sample = ", ".join(names[:20])
                        lines.append(f"  {etype}: {count} total — [{sample}, ... and {count - 20} more]")

                lines.append(f"  TOTAL ENTITIES IN FOLDER: {total_entities}")

                logger.info(f"📋 Type Enumeration: {len(data)} types, {total_entities} entities")
                return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Type enumeration failed: {e}")
            return ""

    # ────────────────────────────────────────────────────────────
    #  Property-Aware Node Search (parallel branch)
    # ────────────────────────────────────────────────────────────

    async def _property_search(self, question: str, folder_id: Optional[str], expanded_terms: List[str] = None, max_depth: int = 6) -> str:
        """
        Fetches ALL properties of nodes that match any keyword from the question
        OR any expanded term from LLM query expansion.

        Enhanced with:
        - LLM-expanded search terms (e.g. "stress" → also searches "stress physiological")
        - Fuzzy partial matching across ALL property values
        - Multi-word phrase matching for compound node names

        Domain-agnostic: works for herbs, drugs, people, companies — any dataset.
        """
        if not folder_id:
            return ""

        folder_label = f"F_{folder_id.replace('-', '_')}"

        # Extract meaningful keywords from the question (skip stopwords)
        stopwords = {"what", "is", "are", "the", "a", "an", "of", "for", "in",
                     "to", "and", "or", "with", "how", "which", "where", "does",
                     "do", "its", "their", "can", "has", "have", "give", "me",
                     "tell", "show", "find", "list", "get", "name", "names",
                     "help", "helps", "that", "this", "about", "from", "all",
                     "any", "been", "being", "but", "by", "could", "each",
                     "had", "into", "may", "might", "more", "most", "much",
                     "not", "only", "other", "our", "out", "own", "should",
                     "some", "such", "than", "them", "then", "there", "these",
                     "those", "through", "under", "very", "was", "were",
                     "will", "would", "your"}
        keywords = [
            w.strip("?,.").lower()
            for w in question.split()
            if len(w.strip("?,.")) > 2 and w.strip("?,.").lower() not in stopwords
        ]

        # Merge with LLM-expanded terms (the crucial enhancement)
        if expanded_terms:
            keywords.extend(expanded_terms)

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        keywords = unique_keywords

        if not keywords:
            return ""

        # Build conditions: match on node name OR any property value
        # Support multi-word expanded terms (e.g. "stress physiological")
        # Escape single quotes to prevent Cypher injection/syntax errors
        kw_conditions = " OR ".join(
            f"toLower(n.name) CONTAINS '{kw.replace(chr(39), chr(92)+chr(39))}'"
            for kw in keywords[:10]  # increased cap for expanded terms
        )

        cypher = f"""
            MATCH (n:{folder_label})
            WHERE {kw_conditions}
            RETURN
                n.name AS name,
                labels(n) AS labels,
                properties(n) AS props
            LIMIT 20
        """

        try:
            async with self.neo4j.session() as session:
                res = await session.run(cypher)
                rows = await res.data()

            if not rows:
                # Second pass: search INSIDE property VALUES using a broader scan
                # This catches cases like: name is "Asparagus racemosus" but
                # a property says common_name: "Shatavari"
                broad_cypher = f"""
                    MATCH (n:{folder_label})
                    WHERE n.name IS NOT NULL
                    RETURN
                        n.name AS name,
                        labels(n) AS labels,
                        properties(n) AS props
                    LIMIT 300
                """
                async with self.neo4j.session() as session2:
                    res2 = await session2.run(broad_cypher)
                    all_rows = await res2.data()

                # Filter client-side: any keyword appears in any property value
                # OR any keyword appears as a SUBSTRING of the node name
                rows = [
                    r for r in all_rows
                    if any(
                        kw in str(v).lower()
                        for kw in keywords[:10]
                        for v in list((r.get("props") or {}).values()) + [r.get("name", "")]
                    )
                ][:20]

            # Step 3: For matched nodes, fetch DYNAMIC MULTI-HOP RELATIONSHIPS
            # Traverses up to max_depth hops to discover the full chain
            # e.g. Entity_A ←[REL_1]← Entity_B ←[REL_2]← ... ←[REL_N]← Entity_Z
            if rows and folder_id:
                matched_names = [r["name"] for r in rows if r.get("name")]
                if matched_names:
                    try:
                        async with self.neo4j.session() as session3:
                            # 1-hop direct relationships
                            rel_query = f"""
                                MATCH (n:{folder_label})-[r]-(m:{folder_label})
                                WHERE n.name IN $names AND m.name IS NOT NULL
                                RETURN
                                    n.name AS source,
                                    type(r) AS relationship,
                                    m.name AS target,
                                    properties(m) AS target_props
                                LIMIT 40
                            """
                            res3 = await session3.run(rel_query, names=matched_names)
                            rel_rows = await res3.data()

                        # Dynamic multi-hop chain traversal (separate session)
                        async with self.neo4j.session() as session4:
                            chain_query = f"""
                                MATCH (n:{folder_label})
                                WHERE n.name IN $names
                                MATCH path = (n)-[*1..{max_depth}]-(far:{folder_label})
                                WHERE far.name IS NOT NULL AND far <> n
                                WITH n, far, [r IN relationships(path) | type(r)] AS rel_chain,
                                     [nd IN nodes(path) | nd.name] AS node_chain
                                RETURN DISTINCT
                                    n.name AS source,
                                    far.name AS target,
                                    labels(far) AS target_labels,
                                    properties(far) AS target_props,
                                    rel_chain,
                                    node_chain
                                LIMIT 60
                            """
                            res4 = await session4.run(chain_query, names=matched_names)
                            chain_rows = await res4.data()

                        rel_lines = []
                        # Format 1-hop
                        if rel_rows:
                            for rr in rel_rows:
                                t_props = rr.get("target_props") or {}
                                clean_t = {k: v for k, v in t_props.items()
                                           if k not in ("id", "embedding", "folder_id", "file_id",
                                                        "fastrp_embedding", "created_at", "updated_at")
                                           and v not in (None, "", [], {})}
                                rel_lines.append(
                                    f"  {rr['source']} —[{rr['relationship']}]→ {rr['target']}"
                                    + (f" | {json.dumps(clean_t, default=str)}" if clean_t else "")
                                )

                        # Format multi-hop chains
                        if chain_rows:
                            rel_lines.append("  --- Extended chains (multi-hop) ---")
                            seen_chains = set()
                            for cr in chain_rows:
                                src = cr.get("source", "?")
                                target = cr.get("target", "?")
                                chain_key = f"{src}->{target}"
                                if chain_key in seen_chains:
                                    continue
                                seen_chains.add(chain_key)
                                rel_chain = cr.get("rel_chain", [])
                                node_chain = cr.get("node_chain", [])
                                chain_str = node_chain[0] if node_chain else src
                                for i, rel_name in enumerate(rel_chain):
                                    next_node = node_chain[i + 1] if i + 1 < len(node_chain) else target
                                    chain_str += f" →[{rel_name}]→ {next_node}"
                                t_props = cr.get("target_props") or {}
                                clean_t = {k: v for k, v in t_props.items()
                                           if k not in ("id", "embedding", "folder_id", "file_id",
                                                        "fastrp_embedding", "created_at", "updated_at")
                                           and v not in (None, "", [], {})}
                                rel_lines.append(
                                    f"  {chain_str}"
                                    + (f" | {json.dumps(clean_t, default=str)}" if clean_t else "")
                                )

                        if rel_lines:
                            rows_extra_context = "\n[Relationships of matched nodes (up to 3 hops)]:\n" + "\n".join(rel_lines)
                        else:
                            rows_extra_context = ""
                    except Exception as e:
                        logger.warning(f"Property search relationship fetch failed: {e}")
                        rows_extra_context = ""
                else:
                    rows_extra_context = ""
            else:
                rows_extra_context = ""

            if not rows:
                return ""

            # Format: human-readable node property dump
            lines = []
            for r in rows:
                props = r.get("props") or {}
                # Remove internal/technical keys that add noise
                clean_props = {
                    k: v for k, v in props.items()
                    if k not in ("id", "embedding", "folder_id", "file_id",
                                 "fastrp_embedding", "created_at", "updated_at")
                    and v not in (None, "", [], {})
                }
                lines.append(f"Node: {r.get('name')} | Properties: {json.dumps(clean_props, default=str)}")

            result = "[Node Properties (matched to question keywords)]:\n" + "\n".join(lines)
            if rows_extra_context:
                result += "\n" + rows_extra_context

            logger.info(f"🏷️  Property Search: {len(rows)} nodes found (expanded terms: {len(expanded_terms or [])}")
            return result

        except Exception as e:
            logger.warning(f"Property search failed: {e}")
            return ""


    async def _folder_neighbor_context(self, folder_id: str) -> str:
        """Last-resort: grab entity names + relationships from the folder."""
        try:
            async with self.neo4j.session() as session:
                res = await session.run("""
                    MATCH (n:Entity {folder_id: $folder_id})
                    OPTIONAL MATCH (n)-[r]-(m:Entity {folder_id: $folder_id})
                    WITH n.name AS entity, n.type AS type,
                         coalesce(n.description, '') AS description,
                         collect(DISTINCT type(r) + ' → ' + m.name)[..3] AS sample_connections
                    ORDER BY size(sample_connections) DESC
                    LIMIT 20
                    RETURN entity, type, description, sample_connections
                """, folder_id=folder_id)
                data = await res.data()
                if data:
                    return json.dumps(data, indent=2, default=str)
                return ""
        except Exception as e:
            logger.warning(f"Folder neighbor context failed: {e}")
            return ""

    # ────────────────────────────────────────────────────────────
    #  GDS Dispatch (full algorithm coverage)
    # ────────────────────────────────────────────────────────────

    def _dispatch_gds(self, algo: str, folder_id: Optional[str]):
        """Create the correct GDS task based on algorithm name."""
        fid = folder_id or "global"
        try:
            # ── Centrality ──
            if algo == "pagerank":
                return self.gds_suite.get_pagerank_context(fid)
            elif algo == "articlerank":
                return self.gds_suite.get_articlerank_context(fid)
            elif algo == "betweenness":
                return self.gds_suite.get_betweenness_context(fid)
            elif algo == "closeness":
                return self.gds_suite.get_closeness_context(fid)
            elif algo == "degree":
                return self.gds_suite.get_degree_context(fid)
            elif algo == "hits":
                return self.gds_suite.get_hits_context(fid)
            # ── Community Detection ──
            elif algo == "louvain":
                return self.gds_suite.get_louvain_context(fid)
            elif algo == "leiden":
                return self.gds_suite.get_leiden_context(fid)
            elif algo == "wcc":
                return self.gds_suite.get_wcc_context(fid)
            elif algo == "kcore":
                return self.gds_suite.get_kcore_context(fid)
            elif algo == "triangle_count":
                return self.gds_suite.get_triangle_count_context(fid)
            # ── Similarity ──
            elif algo == "similarity":
                return self.gds_suite.get_similarity_context(fid)
            # ── Link Prediction ──
            elif algo == "link_prediction_common":
                return self.gds_suite.get_link_prediction_common_context(fid)
            elif algo == "link_prediction_adamic":
                return self.gds_suite.get_link_prediction_adamic_context(fid)
            elif algo == "link_prediction_resource":
                return self.gds_suite.get_link_prediction_resource_context(fid)
            # ── Topology ──
            elif algo == "topological_sort":
                return self.gds_suite.get_topological_sort_context(fid)
            # ── Legacy fallback aliases ──
            elif algo == "centrality":
                return self.gds_suite.get_centrality_context(fid)
            elif algo == "community":
                return self.gds_suite.get_community_context(fid)
            elif algo == "paths":
                return self.gds_suite.get_path_context("", "", fid)
            else:
                # Default fallback to ArticleRank centrality
                logger.warning(f"Unknown GDS algo '{algo}' — falling back to centrality")
                return self.gds_suite.get_centrality_context(fid)
        except Exception as e:
            logger.error(f"GDS dispatch failed: {e}")
            return None
