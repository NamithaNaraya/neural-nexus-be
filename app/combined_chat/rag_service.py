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
from app.core.config import settings
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
_RETRIEVAL_TIMEOUT = 25
# Multi-hop traversal timeout
_MULTIHOP_TIMEOUT = 12
# Query expansion timeout
_EXPANSION_TIMEOUT = 6
# Semantic node resolution timeout
_SEMANTIC_RESOLVE_TIMEOUT = 10


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
        self._chain_depth_cache.clear()
        logger.info("🗑️ RAG schema cache invalidated (CRUD change detected)")

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

        # ── Step 0.5: Query Expansion (LLM expands user terms) ─
        # e.g. user says "stress" → LLM expands to ["stress physiological", ...]
        expanded_terms = []
        try:
            expanded_terms = await asyncio.wait_for(
                self._expand_query(question, folder_id),
                timeout=_EXPANSION_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning("⏱️ Query expansion timed out")
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")

        # Determine fallback dynamic depth if LLM fails
        fallback_depth = await self._get_dynamic_depth(folder_id, question)

        # Build expanded question for searches that benefit from it
        expanded_question = question
        if expanded_terms:
            logger.info(f"🔎 Query Expansion: {expanded_terms}")
            expanded_question = question + " " + " ".join(expanded_terms)

        # ── Step 1: Schema + Intent + Vector + FastRP (all fully parallel) ─
        schema_task  = asyncio.create_task(self._get_schema())
        vector_task  = asyncio.create_task(self.vector_engine.vector_search(expanded_question, folder_id))
        fastrp_task  = asyncio.create_task(
            asyncio.wait_for(self.fastrp_engine.structural_search(expanded_question, folder_id), timeout=_FASTRP_TIMEOUT)
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
            intent = {"use_cypher": False, "cypher_query": None, "use_gds": False, "gds_algo": None, "research_strategy": "Timeout Fallback"}
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            intent = {"use_cypher": False, "cypher_query": None, "use_gds": False, "gds_algo": None, "research_strategy": "Fallback"}
        logger.info(f"🧠 Research Intent: {intent.get('research_strategy', 'Standard')}")
        if intent.get("cypher_query"):
            logger.info(f"🔗 Generated Cypher: {intent['cypher_query']}")

        # Read the AI-determined search depth, fallback to regex/default 6
        req_depth = intent.get("search_depth")
        if isinstance(req_depth, int) and req_depth > 0:
            max_depth = min(req_depth, 20)  # cap at 20 hops for safety
        else:
            max_depth = fallback_depth
            
        logger.info(f"📏 AI-determined chain depth for folder {folder_id}: {max_depth} hops")

        yield json.dumps({"type": "intent", "data": intent}) + "\n"

        # ── Step 2: Retrieval (all parallel branches) ──────────
        logger.info(f"🔍 Retrieval Phase | Vector=True, FastRP=True, MultiHop=True, Cypher={intent.get('use_cypher')}, GDS={intent.get('use_gds')}")
        yield json.dumps({"type": "step", "id": 2, "status": "Searching the knowledge graph..."}) + "\n"

        named_tasks: List[tuple] = [
            ("Semantic Search", vector_task),
            ("Structural Search (FastRP)", fastrp_task),
            # Always run multi-hop graph traversal
            ("Multi-hop Graph Traversal", asyncio.create_task(
                asyncio.wait_for(self._multihop_traversal(question, folder_id), timeout=_MULTIHOP_TIMEOUT)
            )),
            # Property-aware search — now enhanced with LLM-expanded terms
            ("Property Search", asyncio.create_task(
                asyncio.wait_for(self._property_search(question, folder_id, expanded_terms, max_depth), timeout=8)
            )),
            # Semantic Node Resolution — uses vector similarity to find closest
            # matching nodes, then traverses their full neighborhood dynamically
            ("Semantic Node Resolution", asyncio.create_task(
                asyncio.wait_for(
                    self._semantic_node_resolution(question, folder_id, expanded_terms, max_depth),
                    timeout=_SEMANTIC_RESOLVE_TIMEOUT
                )
            )),
            # Direct type enumeration — authoritative counts & complete lists
            # This is the GROUND TRUTH source for "how many X" and "list all X" questions.
            # Unlike semantic search (top-K limited), this queries the ENTIRE database.
            ("Database Facts", asyncio.create_task(
                asyncio.wait_for(
                    self._type_enumeration(folder_id),
                    timeout=5
                )
            )),
            # Focused Entity Scan — exhaustive neighborhood for entities mentioned in the question
            # Ensures ALL connections (e.g., Tamarind → ALL plant parts → ALL phytoconstituents)
            ("Focused Entity Scan", asyncio.create_task(
                asyncio.wait_for(
                    self._focused_entity_scan(question, folder_id),
                    timeout=8
                )
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

        return f"""You are Neural Nexus — a sharp, knowledgeable assistant that answers questions from a knowledge graph.

RULES:
• Answer ONLY from the CONTEXT below. It is real data from the user's active knowledge graph.
• Never say "the data doesn't contain" if it actually does — read carefully.
• Never pad with boilerplate like "Certainly!", "Great question", "Executive Summary", "Key Findings", or "Key Takeaways".
• NEVER use section headers (##, ###) unless the answer is genuinely multi-part.
• Be direct. Answer first, then support with data.
• PRIORITY: If the context contains "DATABASE FACTS", those are AUTHORITATIVE counts and complete lists queried directly from the graph database. For counting questions ("how many X?") or listing questions ("list all X"), ALWAYS use the DATABASE FACTS numbers — they are exact and complete. Other sources (semantic search, vector search) are approximate and may be incomplete.

CONTEXT:
{context}

QUESTION: {question}

RESPONSE STRUCTURE — always follow this pattern based on question type:

─── PATTERN A: "Which X / List / What are all" questions ───
1. **Opening sentence**: Give a direct answer (e.g. "Your data contains 3 herbs with anti-inflammatory phytochemicals.").
2. **Table**: Show ALL relevant columns with every data point from the context.
3. **Summary** (2-3 sentences): After the table, explain what this means in plain English — highlight any notable patterns, the most common items, or key takeaway from the data.

─── PATTERN B: Simple fact / yes-no ───
Answer in 1-2 sentences only. No table, no headers.

─── PATTERN C: How / Why / Explain ───
Short paragraphs, **bold** key terms. Max 4-5 sentences total.

─── PATTERN D: Comparison / ranking ───
Opening sentence → table with comparison columns → 1-2 sentence takeaway.

─── PATTERN E: Algorithm / graph analysis ───
Opening sentence → ranked table or list → 2-sentence interpretation of what the algorithm found.

TABLE RULES:
- Include ALL data columns you have evidence for — never omit relevant columns.
- Column headers should be short and clear.
- Always wrap the table with an opening sentence AND a closing summary.

AFTER-TABLE SUMMARY RULES:
- 2-3 sentences max.
- Mention the count (e.g. "4 phytochemicals were identified"), the most notable finding, and any pattern.
- Write in plain English — no jargon, no bullet points in the summary.

HONESTY: If context is truly incomplete for a specific sub-question, say it in one sentence only."""



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

GDS ALGORITHM SELECTION — set `use_gds: true` and pick the right `gds_algo` when the question fits:

— CENTRALITY (Who/What is most important?) —
• "pagerank": Influence, importance, popularity, "best", "most central", ranking.
• "articlerank": Like PageRank but better for diverse graphs. "Most authoritative".
• "betweenness": Bridges, bottlenecks, connecting groups, flow control.
• "closeness": Reachability, "closest to all others", central access point.
• "degree": Most connections, "most active", "most linked".
• "hits": Hubs vs authorities. "Which are hubs" vs "which are authorities".

— COMMUNITY DETECTION (How is the data grouped?) —
• "louvain": Communities, clusters, groups, "who belongs together".
• "leiden": Same as louvain but higher quality. "Better clustering".
• "wcc": Islands, disconnected parts, "isolated groups".
• "kcore": Core structure, "tight-knit core", inner circle vs periphery.
• "triangle_count": Local density, "tight clusters", "tightly-knit groups".

— SIMILARITY —
• "similarity": "What is similar to X", "which nodes share neighbors", Jaccard.

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
            return await self.gemini.generate_json(prompt)
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
                    LIMIT 150
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
            result = await self.gemini.generate_json(prompt)
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
            async with self.neo4j.session() as session:
                # Step 1: Find matching entities by fuzzy name matching
                kw_conditions = " OR ".join(
                    f"toLower(n.name) CONTAINS '{kw}'" for kw in words[:6]
                )
                find_query = f"""
                    MATCH (n:{folder_label})
                    WHERE ({kw_conditions}) AND n.name IS NOT NULL
                    RETURN DISTINCT n.name AS name
                    LIMIT 5
                """
                res = await session.run(find_query)
                found = await res.data()

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
        kw_conditions = " OR ".join(
            f"toLower(n.name) CONTAINS '{kw}'"
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
