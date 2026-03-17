import logging
import asyncio
from typing import List, Dict, Any, Optional
from app.combined_chat.gemini_service import GeminiService
from app.combined_chat.embedding_service import EmbeddingService
from app.combined_chat.gds_service import GDSCombinedService
from app.db.connections import get_neo4j_driver, get_redis_client
import json
import time

logger = logging.getLogger(__name__)

class CombinedRAGService:
    def __init__(self):
        self.gemini = GeminiService()
        self.vector_engine = EmbeddingService()
        self.gds_suite = GDSCombinedService()
        self.neo4j = get_neo4j_driver()
        self.redis = get_redis_client()
        self._schema_cache: Optional[str] = None
        self._schema_expiry: float = 0
        self._greetings = {"hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening"}

    async def answer(self, question: str, folder_id: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, user_id: str = "anonymous") -> Dict[str, Any]:
        """🚀 The 5-Stage Pipeline Orchestrator."""
        
        # Redis-based isolation: Load history for this specific User-Folder pair
        history_key = f"chat:history:{user_id}:{folder_id or 'global'}"
        stored_history = []
        try:
            raw_history = await self.redis.lrange(history_key, -10, -1)
            stored_history = [json.loads(m) for m in raw_history]
        except Exception as e:
            logger.warning(f"Failed to load history from Redis: {e}")

        # Combine provided history with stored history
        combined_history = (stored_history + history) if history else stored_history
        
        # Fast Path: Greeting Detector
        clean_q = question.lower().strip().strip('?!.')
        if clean_q in self._greetings:
            return {
                "answer": "Hello! I am your Neural Nexus research assistant. How can I help you explore your knowledge graph today?",
                "intent": {"use_cypher": False, "use_gds": False, "use_vector": False},
                "context_summary": "Greeting detected. Short-circuiting pipeline for speed."
            }
        
        # Stage 1: Dynamic Schema Introspection
        schema = await self._get_schema()
        
        # Stage 2: Concurrent Master Orchestration & Vector Search
        orchestration_task = asyncio.create_task(self._orchestrate_retrieval(question, schema, folder_id or "global"))
        vector_task = asyncio.create_task(self.vector_engine.vector_search(question, folder_id))
        
        # Wait for the "Brain" to decide the plan
        intent = await orchestration_task
        logger.info(f"Execution Plan: {intent}")

        retrieval_tasks = [vector_task]
        
        # Stage 3: Dynamic Data Retrieval
        if intent.get("use_cypher") and intent.get("cypher_query"):
            retrieval_tasks.append(self._execute_cypher(intent["cypher_query"]))
        
        if intent.get("use_gds"):
            algo = intent.get("gds_algo", "centrality")
            if algo == "similarity":
                retrieval_tasks.append(self._format_gds_res("Node Similarity/PageRank", self.gds_suite.get_similarity_context(folder_id)))
            elif algo == "community":
                retrieval_tasks.append(self._format_gds_res("Louvain Community Detection", self.gds_suite.get_community_context(folder_id)))
            elif algo == "paths":
                retrieval_tasks.append(self.gds_suite.get_path_context("", "", folder_id))
            else:
                retrieval_tasks.append(self._format_gds_res("ArticleRank Centrality", self.gds_suite.get_centrality_context(folder_id)))

        raw_results = await asyncio.gather(*retrieval_tasks)
        
        # Stage 4: Context Fusion
        context = self._fuse_context(raw_results)
        
        # Stage 5: Research Synthesis
        answer = await self._synthesize_answer(question, context, combined_history, folder_id or "global")
        
        # Save to Redis for isolation
        try:
            await self.redis.rpush(history_key, json.dumps({"role": "user", "content": question}))
            await self.redis.rpush(history_key, json.dumps({"role": "assistant", "content": answer}))
            await self.redis.ltrim(history_key, -20, -1) 
            await self.redis.expire(history_key, 86400) 
        except Exception as e:
            logger.warning(f"Failed to save history to Redis: {e}")

        return {
            "answer": answer,
            "intent": intent,
            "context_summary": f"Retrieved context from {len(retrieval_tasks)} parallel streams within folder {folder_id or 'global'}."
        }

    async def _get_schema(self) -> str:
        if self._schema_cache and time.time() < self._schema_expiry:
            return str(self._schema_cache)
            
        async with self.neo4j.session() as session:
            try:
                # 1. Get ALL Labels
                labels_res = await session.run("CALL db.labels()")
                labels = [r[0] for r in await labels_res.records()]
                
                # 2. Get connecting patterns
                patterns_res = await session.run("""
                    CALL db.schema.visualization() YIELD relationships
                    UNWIND relationships AS rel
                    WITH startNode(rel) AS s, type(rel) AS t, endNode(rel) AS e
                    RETURN labels(s)[0] AS source, t AS type, labels(e)[0] AS target
                """)
                data = await patterns_res.data()
                patterns = [f"({r['source']})-[:{r['type']}]->({r['target']})" for r in data]
                
                res_str = f"Available Labels (CRITICAL): {labels}\nKey Patterns: " + ", ".join(patterns[:12])
                if not patterns:
                    res_str = f"Available Labels: {labels}\nSchema: Generic Graph Nodes"
                    
                self._schema_cache = res_str
                self._schema_expiry = time.time() + 600
                return res_str
            except Exception as e:
                logger.error(f"Schema introspection failed: {e}")
                return "Common Labels: [Item, Concept, Category], Relationships: [RELATED_TO, PART_OF, CONTAINS]"

    async def _orchestrate_retrieval(self, question: str, schema: str, folder_id: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze user question against this Knowledge Graph Schema:
        {schema}
        
        STRICT RULES:
        1. Only use Node Labels exactly as they appear in the 'Available Labels' list above.
        2. Filter every node by 'folder_id' or 'folderId' using the value: "{folder_id}".
        3. SYNONYM MAPPING: Map user terms like "benefits", "properties", "effects", or "uses" to the most relevant labels in the schema (e.g. TherapeuticUse, Phytoconstituent).
        4. Use CONTAINS and toLower() for robust property matching.

        Question: {question}
        
        RETURN ONLY JSON:
        {{
          "is_social": bool,
          "is_out_of_scope": bool,
          "use_cypher": bool,
          "cypher_query": "string | null",
          "use_gds": bool,
          "gds_algo": "centrality|community|similarity|null"
        }}
        """
        try:
            return await self.gemini.generate_json(prompt)
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {"is_social": False, "is_out_of_scope": False, "use_cypher": False, "cypher_query": None, "use_gds": False, "gds_algo": None}

    async def _format_gds_res(self, algo_name: str, task: Any) -> str:
        try:
            results = await task
            if isinstance(results, list):
                formatted_data = json.dumps(results, indent=2, default=str)
                return f"[Graph Algorithm Results: {algo_name}]:\n{formatted_data}"
            return str(results)
        except Exception as e:
            return f"[Algorithm Error ({algo_name})]: {str(e)}"

    async def _execute_cypher(self, cypher: str) -> str:
        if not cypher: return ""
        cypher = cypher.replace('```cypher', '').replace('```', '').strip()
        if any(keyword in cypher.upper() for keyword in ["CREATE", "DELETE", "SET", "MERGE", "REMOVE"]):
             return "[Security Error]: Write operations blocked."

        try:
            async with self.neo4j.session() as session:
                res = await session.run(cypher)
                data = await res.data()
                return f"[Graph Results]:\n{json.dumps(data, indent=2, default=str)[:3000]}"
        except Exception as e:
            return f"[Graph Error]: {str(e)}"

    def _fuse_context(self, results: List[Any]) -> str:
        stream_names = ["Semantic Search", "Graph Traversal", "Graph Algorithms"]
        fused = []
        for i, r in enumerate(results):
            name = stream_names[i] if i < len(stream_names) else "Extra Intel"
            if not r: continue
            fused.append(f"=== {name} ===\n{str(r)}")
        return "\n\n".join(fused)

    async def _synthesize_answer(self, question: str, context: str, history: List[Dict[str, str]], folder_id: str) -> str:
        prompt = f"""
        You are the **Neural Nexus Research Assistant**, a professional expert in Knowledge Graph synthesis.
        
        CONTEXT FROM DATA RETRIEVAL:
        {context}
        
        USER QUESTION: 
        {question}
        
        INSTRUCTIONS:
        1. CHARACTER: Be informative, professional, and clear. Use **simple, natural English**.
        2. DOMAIN NEUTRAL: Do not assume the data is about any specific topic. Adapt your terminology to context.
        3. NO TECHNICAL CODES: You must NOT mention technical IDs, Folder IDs (like '{folder_id}'), or internal labels.
        4. NATURAL LANGUAGE: Translate technical relationship names into simple verbs like "contained in", "associated with", "linked to", etc.
        5. RESPONSE STRUCTURE:
           - **Executive Summary**: A clear 4-5 sentence story explaining what was found in the current folder.
           - **Evidence Table**: Use a **Markdown Table** to organize the key data found.
        6. RELIABILITY: Only answer based on context. If data is missing for this folder, explain that clearly.
        """
        return await self.gemini.generate_response(prompt, history)
