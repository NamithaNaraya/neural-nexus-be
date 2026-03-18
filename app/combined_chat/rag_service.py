"""
Combined Chat — RAG Orchestrator (v3)

Hybrid RAG with 3 parallel retrieval branches (matching architecture blueprint):
  1. Semantic Text Search    — Ollama embeddings → Neo4j vector index
  2. Graph Structure Search  — FastRP topology embeddings → cosine similarity
  3. Cypher Query Generation — LLM-generated Cypher from schema

Key improvements:
  - Cypher execution with timeout + auto-recovery query
  - Empty-context safety net (folder neighbor scan)
  - Polished, user-friendly answer prompt
  - Full parallelism across all 3 retrieval branches
"""
import logging
import asyncio
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

        # ── Step 1: Schema + Intent + Vector + FastRP (all in parallel) ─
        schema_task  = asyncio.create_task(self._get_schema())
        vector_task  = asyncio.create_task(self.vector_engine.vector_search(question, folder_id))
        fastrp_task  = asyncio.create_task(self.fastrp_engine.structural_search(question, folder_id))

        schema = await schema_task

        orchestration_task = asyncio.create_task(
            self._orchestrate_retrieval(question, schema, folder_id or "global")
        )

        intent = await orchestration_task
        logger.info(f"🧠 Research Intent: {intent.get('research_strategy', 'Standard')}")
        if intent.get("cypher_query"):
            logger.info(f"🔗 Generated Cypher: {intent['cypher_query']}")

        yield json.dumps({"type": "intent", "data": intent}) + "\n"

        # ── Step 2: Retrieval (3 parallel branches) ────────────
        logger.info(f"🔍 Retrieval Phase | Vector=True, FastRP=True, Cypher={intent.get('use_cypher')}, GDS={intent.get('use_gds')}")
        yield json.dumps({"type": "step", "id": 2, "status": "Searching the knowledge graph..."}) + "\n"

        named_tasks: List[tuple] = [
            ("Semantic Search", vector_task),
            ("Structural Search (FastRP)", fastrp_task),
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

        # Gather all retrieval outputs
        names  = [t[0] for t in named_tasks]
        tasks  = [t[1] for t in named_tasks]
        raw_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        # ── Fuse with labels ───────────────────────────────────
        fused = []
        for name, result in zip(names, raw_outputs):
            if isinstance(result, Exception):
                logger.warning(f"Retrieval source '{name}' raised: {result}")
                continue
            if not result:
                continue
            if isinstance(result, list):
                res_str = json.dumps(result[:20], indent=2, default=str)
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

        return f"""You are the **Neural Nexus Research Assistant** — a friendly, knowledgeable expert who provides clear, well-organized answers.

─── GROUND RULES ───
• Base your answer **strictly** on the CONTEXT below. This data comes from the user's active folder.
• Do NOT use any prior knowledge or data from other folders/projects.
• If the context only partially answers the question, say so honestly.

─── CONTEXT FROM ACTIVE FOLDER ───
{context}

─── USER QUESTION ───
{question}

─── RESPONSE GUIDELINES ───
1. **Tone**: Professional yet friendly. Address the user directly ("Based on your data…").
2. **Structure**:
   • Start with a concise **Executive Summary** (2-3 sentences answering the core question).
   • Follow with detailed findings, organized with headers/bullets.
   • If Graph Algorithm data is present, include an **⚡ Algorithmic Insights** section highlighting hidden patterns, rankings or communities.
   • End with a **Key Takeaways** bullet list if the answer is complex.
3. **Tables**: Use markdown tables when comparing multiple items.
4. **Honesty**: If the context is insufficient, say "Based on the available data…" rather than guessing.
5. **Clean output**: Do NOT mention data source methods (e.g., "From Semantic Search"). Present facts naturally.
6. **Conciseness**: Prefer quality over quantity. Every sentence should add value."""

    # ────────────────────────────────────────────────────────────
    #  Schema Discovery (cached 5 min)
    # ────────────────────────────────────────────────────────────

    async def _get_schema(self) -> str:
        if self._schema_cache and time.time() < self._schema_expiry:
            return str(self._schema_cache)

        async with self.neo4j.session() as session:
            try:
                # 1. Node Labels & Properties
                node_props_res = await session.run("""
                    CALL db.labels() YIELD label 
                    MATCH (n) WHERE label IN labels(n) 
                    WITH label, keys(n) AS keys LIMIT 100
                    RETURN label, collect(DISTINCT keys) AS property_samples
                """)
                node_data = await node_props_res.data()
                label_info = [f"Node Label: {r['label']} (Props: {r['property_samples']})" for r in node_data]

                # 2. Relationship Types
                rel_res = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                rel_data = await rel_res.data()
                rel_types = [r["relationshipType"] for r in rel_data]
                rel_prop_info = [f"Available Rel Type: {t}" for t in rel_types]

                # 3. Relationship Property Samples
                rel_props_res = await session.run("""
                    MATCH ()-[r]->() 
                    WITH type(r) AS type, keys(r) AS keys LIMIT 100
                    RETURN type, collect(DISTINCT keys) AS property_samples
                """)
                rel_prop_data = await rel_props_res.data()
                rel_prop_info += [f"Rel Type {r['type']} Props: {r['property_samples']}" for r in rel_prop_data]

                # 4. Active Patterns
                patterns_res = await session.run("""
                    CALL db.schema.visualization() YIELD relationships
                    UNWIND relationships AS rel
                    WITH startNode(rel) AS s, type(rel) AS t, endNode(rel) AS e
                    RETURN DISTINCT labels(s)[0] AS source, t AS type, labels(e)[0] AS target
                """)
                pattern_data = await patterns_res.data()
                patterns = [f"({r['source']})-[:{r['type']}]->({r['target']})" for r in pattern_data]

                res_str = "KNOWLEDGE GRAPH SCHEMA (GROUND TRUTH):\n"
                res_str += "\n".join(label_info) + "\n"
                res_str += "\n".join(rel_prop_info) + "\n"
                res_str += "\nValidated connection Patterns:\n" + "\n".join(patterns)

                logger.info(f"📊 Schema Discovery: {len(label_info)} labels, {len(rel_types)} relationship types.")

                self._schema_cache = res_str
                self._schema_expiry = time.time() + 300
                return res_str
            except Exception as e:
                logger.error(f"Schema discovery failed: {e}")
                return "Labels: [Unknown], Patterns: [Unknown]"

    # ────────────────────────────────────────────────────────────
    #  Intent Orchestration
    # ────────────────────────────────────────────────────────────

    async def _orchestrate_retrieval(self, question: str, schema: str, folder_id: str) -> Dict[str, Any]:
        prompt = f"""
        TASK: Orchestrate data retrieval for a Knowledge Graph Research Agent.
        FOLDER_ID: {folder_id}
        SCHEMA:
        {schema}
        
        SOP FOR COMPLEX GRAPH RESEARCH:
        1. PATH DISCOVERY: For "X treats Y" or "X connected to Y", use 2-3 hop Cypher: (n)-[*1..3]->(m).
        2. IMPORTANCE/RANKING (*GDS MANDATORY*): For "most important", "widest range", or "top entities", you MUST set `use_gds` to true and select `centrality`.
        3. GROUPS/CLUSTERS (*GDS MANDATORY*): For "groups of diseases" or "clusters", you MUST set `use_gds` to true and select `community`.
        4. DEGREE FALLBACK: Use Neo4j 5 syntax if sorting in Cypher: MATCH (n) WHERE n.folder_id = '{folder_id}' RETURN n.name, COUNT {{ (n)--() }} as deg ORDER BY deg DESC.
        5. DIVERSITY: count(DISTINCT neighbor) of a target label.
        6. SIMILARITY: Find nodes sharing neighbors (hubs). (a)-[:REL]->(hub)<-[:REL]-(b).
        7. STRUCTURE SIMILARITY: Use [SIMILAR_TO] relationships and the 'basis' property if present.
        8. PLANT PARTS: Use the 'part_of_plant' property on Relationships if filtering by plant organ.

        STRICT CYPHER RULES:
        - FOLDER ISOLATION: EVERY node variable in your MATCH must be filtered by folder_id. 
          Example: MATCH (h:Herb), (p:Phytochemical) WHERE h.folder_id = '{folder_id}' AND p.folder_id = '{folder_id}' ...
        - NO HALLUCINATIONS: You MUST ONLY use Relationship Types listed in the SCHEMA above. 
          (e.g., if SCHEMA has HAS_USE, do NOT use HAS_THERAPEUTIC_USE).
        - MULTI-HOP PATHS: If the question requires connecting A to C, and A is connected to B and B to C, use (a)-[*1..3]->(c) or explicit hops.
        - NEO4J 5 SYNTAX: Do NOT use `size((n)--())`. You MUST use the `COUNT {{ (n)--() }}` subquery pattern instead.
        - No Hardcoding: Use labels and property comparisons.

        Question: {question}
        
        JSON OUTPUT:
        {{
          "use_cypher": bool,
          "cypher_query": "CYPHER",
          "use_gds": bool,
          "gds_algo": "centrality|community|similarity|null",
          "research_strategy": "..."
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
    #  Folder Neighbor Context (safety net)
    # ────────────────────────────────────────────────────────────

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
