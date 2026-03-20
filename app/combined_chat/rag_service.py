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
from typing import List, Dict, Any, Optional
from app.combined_chat.gemini_service import GeminiService
from app.combined_chat.embedding_service import EmbeddingService
from app.combined_chat.fastrp_service import FastRPService
from app.combined_chat.gds_service import GDSCombinedService
from app.db.connections import get_neo4j_driver, get_redis_client
import json
import time

logger = logging.getLogger(__name__)

# Cypher execution hard timeout (seconds)
_CYPHER_TIMEOUT = 12
# FastRP structural search timeout (seconds)
_FASTRP_TIMEOUT = 6
# Orchestration (LLM intent) timeout (seconds)
_ORCH_TIMEOUT = 10
# Overall retrieval phase timeout
_RETRIEVAL_TIMEOUT = 20
# Multi-hop traversal timeout
_MULTIHOP_TIMEOUT = 12


class CombinedRAGService:
    def __init__(self):
        self.gemini = GeminiService()
        self.vector_engine = EmbeddingService()
        self.fastrp_engine = FastRPService()
        self.gds_suite = GDSCombinedService()
        self.neo4j = get_neo4j_driver()
        self.redis = get_redis_client()
        self._schema_cache: Optional[str] = None
        self._schema_expiry: float = 0
        self._greetings = {"hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening"}

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
    ):
        """Streaming Orchestrator with strict folder isolation."""

        logger.info(f"🚀 Research starting for folder {folder_id or 'global'} | Question: {question}")
        yield json.dumps({"type": "step", "id": 1, "status": "Analyzing research intent..."}) + "\n"

        # ── History ────────────────────────────────────────────
        history_key = f"chat:history:{user_id}:{folder_id or 'global'}"
        stored_history = []
        try:
            raw_history = await self.redis.lrange(history_key, -10, -1)
            stored_history = [json.loads(m) for m in raw_history]
        except Exception as e:
            logger.warning(f"Failed to load history from Redis: {e}")

        combined_history = (stored_history + history) if history else stored_history

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

        # ── Step 1: Schema + Intent + Vector + FastRP (all fully parallel) ─
        schema_task  = asyncio.create_task(self._get_schema())
        vector_task  = asyncio.create_task(self.vector_engine.vector_search(question, folder_id))
        fastrp_task  = asyncio.create_task(
            asyncio.wait_for(self.fastrp_engine.structural_search(question, folder_id), timeout=_FASTRP_TIMEOUT)
        )

        schema = await schema_task

        orchestration_task = asyncio.create_task(
            asyncio.wait_for(
                self._orchestrate_retrieval(question, schema, folder_id or "global"),
                timeout=_ORCH_TIMEOUT
            )
        )

        try:
            intent = await orchestration_task
        except asyncio.TimeoutError:
            logger.warning("⏱️ Orchestration timed out — using vector-only fallback")
            intent = {"use_cypher": False, "cypher_query": None, "use_gds": False, "gds_algo": None, "research_strategy": "Timeout Fallback"}
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            intent = {"use_cypher": False, "cypher_query": None, "use_gds": False, "gds_algo": None, "research_strategy": "Fallback"}
        logger.info(f"🧠 Research Intent: {intent.get('research_strategy', 'Standard')}")
        if intent.get("cypher_query"):
            logger.info(f"🔗 Generated Cypher: {intent['cypher_query']}")

        yield json.dumps({"type": "intent", "data": intent}) + "\n"

        # ── Step 2: Retrieval (all parallel branches) ──────────
        logger.info(f"🔍 Retrieval Phase | Vector=True, FastRP=True, MultiHop=True, Cypher={intent.get('use_cypher')}, GDS={intent.get('use_gds')}")
        yield json.dumps({"type": "step", "id": 2, "status": "Searching the knowledge graph..."}) + "\n"

        named_tasks: List[tuple] = [
            ("Semantic Search", vector_task),
            ("Structural Search (FastRP)", fastrp_task),
            # Always run multi-hop graph traversal for each question
            ("Multi-hop Graph Traversal", asyncio.create_task(
                asyncio.wait_for(self._multihop_traversal(question, folder_id), timeout=_MULTIHOP_TIMEOUT)
            )),
        ]

        # Cypher
        if intent.get("use_cypher") and intent.get("cypher_query"):
            named_tasks.append(("Graph Traversal", self._execute_cypher_safe(intent["cypher_query"], folder_id, file_id)))

        # GDS
        if intent.get("use_gds"):
            algo = intent.get("gds_algo", "centrality")
            gds_task = self._dispatch_gds(algo, folder_id)
            if gds_task:
                named_tasks.append((f"Graph Algorithm ({algo})", gds_task))

        # Gather all retrieval outputs with global timeout
        names  = [t[0] for t in named_tasks]
        tasks  = [t[1] for t in named_tasks]
        try:
            raw_outputs = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=_RETRIEVAL_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Global retrieval timeout ({_RETRIEVAL_TIMEOUT}s) — using partial results")
            raw_outputs = [asyncio.TimeoutError()] * len(tasks)

        # ── Fuse with labels (compact serialization) ──────────
        fused = []
        for name, result in zip(names, raw_outputs):
            if isinstance(result, (Exception, type(None))):
                if isinstance(result, Exception):
                    logger.warning(f"Retrieval source '{name}' raised: {result}")
                continue
            if not result:
                continue
            if isinstance(result, list):
                # Compact serialization — no indent, limit items
                res_str = json.dumps(result[:30], default=str)
                if name.startswith("Graph Algorithm"):
                    current_algo = algo if "algo" in dir() else "analytics"
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

        logger.info(f"🧪 Context Fusion | Sources: {len(fused)} | Size: {len(context)} chars")

        # ── Step 3: Synthesize ─────────────────────────────────
        yield json.dumps({"type": "step", "id": 3, "status": "Synthesizing research results..."}) + "\n"

        full_answer = ""
        prompt = self._build_answer_prompt(question, context, folder_id)

        async for chunk in self.gemini.astream_response(prompt, combined_history):
            full_answer += chunk
            yield json.dumps({"type": "content", "data": chunk}) + "\n"

        # ── Step 4: Finalize ───────────────────────────────────
        logger.info(f"✅ Research completed for user {user_id}")
        yield json.dumps({"type": "step", "id": 4, "status": "Research completed."}) + "\n"

        try:
            await self.redis.rpush(history_key, json.dumps({"role": "user", "content": question}))
            await self.redis.rpush(history_key, json.dumps({"role": "assistant", "content": full_answer}))
            await self.redis.ltrim(history_key, -20, -1)
            await self.redis.expire(history_key, 86400)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    async def answer(
        self,
        question: str,
        folder_id: Optional[str] = None,
        file_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """Non-streaming wrapper for backward compatibility."""
        full_answer = ""
        intent = {}
        async for chunk_raw in self.stream_answer(question, folder_id, file_id, history, user_id):
            chunk = json.loads(chunk_raw.strip())
            if chunk["type"] == "content":
                full_answer += chunk["data"]
            elif chunk["type"] == "intent":
                intent = chunk["data"]

        return {
            "answer": full_answer,
            "intent": intent,
            "context_summary": f"Retrieved from folder {folder_id or 'global'}.",
        }

    # ────────────────────────────────────────────────────────────
    #  Answer Prompt (polished, user-friendly)
    # ────────────────────────────────────────────────────────────

    def _build_answer_prompt(self, question: str, context: str, folder_id: Optional[str]) -> str:
        has_context = bool(context.strip())
        
        if not has_context:
            return f"""You are the **Neural Nexus Research Assistant** — a friendly, knowledgeable expert.

The user asked: "{question}"

Unfortunately, no matching data was found in the active folder{f' ({folder_id})' if folder_id else ''}.

Please respond politely:
1. Acknowledge that you searched but found no matching results for this specific query.
2. Suggest possible reasons (e.g., the data might be in a different folder, or the question might need rephrasing).
3. Offer helpful follow-up suggestions based on the question topic.

Keep your tone warm, professional, and encouraging. Do NOT fabricate data."""

        return f"""You are the **Neural Nexus Research Assistant** — a precise, knowledgeable expert who synthesizes graph database results into clear, direct answers.

─── GROUND RULES ───
• Base your answer **strictly** on the CONTEXT below. This data comes directly from the user's knowledge graph.
• The context contains ACTUAL data extracted from the graph via semantic search, graph traversal, and structural analysis.
• Do NOT say "the data does not contain" if the answer IS in the context — look carefully!
• If the question is multi-hop (e.g., "which herbs contain X that treats Y"), trace the path through the data provided.
• Do NOT use any prior knowledge beyond what is in the CONTEXT.

─── CONTEXT FROM ACTIVE FOLDER ───
{context}

─── USER QUESTION ───
{question}

─── RESPONSE GUIDELINES ───
1. **ANSWER FIRST**: Start with a direct, specific answer. If the question asks "which herbs", NAME THE HERBS in the first sentence.
2. **Tone**: Professional yet friendly. Address the user directly ("Based on your data…").
3. **Structure**:
   • Start with a concise **Summary** (1-2 sentences with the direct answer).
   • Follow with a **markdown table** if comparing multiple items (ALWAYS use tables for lists of 3+ items).
   • Match table columns to the actual data returned (could be products, diseases, people, molecules, or any domain).
   • If Graph Algorithm data is present, include an **⚡ Graph Analysis** section.
   • End with **Key Takeaways** bullet list for complex answers.
4. **Tables**: ALWAYS use markdown tables when listing multiple entities, relationships, or any structured data.
5. **Honesty**: If context is truly insufficient for a specific part of the question, say so for THAT part only.
6. **Trace Multi-hop Paths**: If the context shows A→B and B→C, explicitly connect them in natural language.
7. **Conciseness**: Every sentence must add value. No filler text."""

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
    #  Intent Orchestration
    # ────────────────────────────────────────────────────────────

    async def _orchestrate_retrieval(self, question: str, schema: str, folder_id: str) -> Dict[str, Any]:
        folder_label = f"F_{folder_id.replace('-', '_')}"
        prompt = f"""
TASK: Orchestrate data retrieval for a Knowledge Graph Research Agent.
FOLDER LABEL: {folder_label}
SCHEMA (actual graph structure for this folder):
{schema}

CRITICAL CONTEXT:
- The graph uses folder-scoped labels. Every node in this folder carries the label `{folder_label}`.
- Use ONLY the relationship types and node labels listed in the SCHEMA above — do NOT invent new ones.
- The data domain is UNKNOWN — it could be pharma, finance, legal, social, or anything. Adapt your query to the actual schema.

MULTI-HOP REASONING RULES:
- Answers often require traversing 2-6 hops through the graph.
- For connections between two entity types that are not directly linked, always try variable-length paths.
- Use: MATCH (a:{folder_label})-[*1..5]->(b:{folder_label}) WHERE a.name IS NOT NULL RETURN ...
- (EXAMPLE ONLY — for illustration — actual domain may differ):
  e.g. Person → WorksAt → Company → Located → City
  e.g. Product → HasComponent → Material → SourcedFrom → Country

SOP:
1. PATH DISCOVERY: Use (n)-[*1..5]->(m) for multi-hop questions. Never assume direct 1-hop connection.
2. IMPORTANCE/RANKING: For "most important", "widest range", "top N" → set `use_gds: true`, `gds_algo: centrality`.
3. GROUPS/CLUSTERS: For "groups of", "clusters" → set `use_gds: true`, `gds_algo: community`.
4. RELATIONSHIPS: Use ONLY rel types from the SCHEMA. Never hallucinate.
5. NEO4J 5 SYNTAX: Use COUNT {{ (n)--() }} not size((n)--()).
6. NO HARDCODING: Use label comparisons and patterns. Never hardcode node names.

Question: {question}

JSON OUTPUT:
{{
  "use_cypher": bool,
  "cypher_query": "CYPHER or null",
  "use_gds": bool,
  "gds_algo": "centrality|community|similarity|null",
  "research_strategy": "brief description"
}}
        """
        try:
            return await self.gemini.generate_json(prompt)
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {
                "use_cypher": False,
                "cypher_query": None,
                "use_gds": False,
                "gds_algo": None,
                "research_strategy": "Fallback",
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
                # Finds what each node connects to within 2 hops
                # Returns: source, rel_type, target for both hops
                q1 = f"""
                    MATCH (a:{folder_label})-[r1]->(b:{folder_label})
                    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
                    OPTIONAL MATCH (b)-[r2]->(c:{folder_label})
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
                # Discovers longer chains without variable-length overhead
                q2 = f"""
                    MATCH (a:{folder_label})-[r1]->(b:{folder_label})-[r2]->(c:{folder_label})-[r3]->(d:{folder_label})
                    WHERE a.name IS NOT NULL AND d.name IS NOT NULL
                      AND a <> d
                    RETURN
                        a.name AS node_a,
                        type(r1) AS rel_ab,
                        b.name AS node_b,
                        type(r2) AS rel_bc,
                        c.name AS node_c,
                        type(r3) AS rel_cd,
                        d.name AS node_d
                    LIMIT 60
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
    #  GDS Dispatch
    # ────────────────────────────────────────────────────────────

    def _dispatch_gds(self, algo: str, folder_id: Optional[str]):
        """Create the correct GDS task based on algorithm name."""
        fid = folder_id or "global"
        try:
            if algo == "similarity":
                return self.gds_suite.get_similarity_context(fid)
            elif algo == "community":
                return self.gds_suite.get_community_context(fid)
            elif algo == "paths":
                return self.gds_suite.get_path_context("", "", fid)
            else:
                return self.gds_suite.get_centrality_context(fid)
        except Exception as e:
            logger.error(f"GDS dispatch failed: {e}")
            return None
