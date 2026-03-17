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

    async def stream_answer(self, question: str, folder_id: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, user_id: str = "anonymous"):
        """🚀 Streaming Orchestrator with Status Updates."""
        
        # 1. Load History
        yield json.dumps({"type": "step", "id": 1, "status": "Reading history..."}) + "\n"
        history_key = f"chat:history:{user_id}:{folder_id or 'global'}"
        stored_history = []
        try:
            raw_history = await self.redis.lrange(history_key, -10, -1)
            stored_history = [json.loads(m) for m in raw_history]
        except Exception as e:
            logger.warning(f"Failed to load history from Redis: {e}")

        combined_history = (stored_history + history) if history else stored_history
        
        # Fast Path: Greeting Detector
        clean_q = question.lower().strip().strip('?!.')
        if clean_q in self._greetings:
            full_ans = "Hello! I am your Neural Nexus research assistant. How can I help you explore your knowledge graph today?"
            yield json.dumps({"type": "content", "data": full_ans}) + "\n"
            yield json.dumps({"type": "step", "id": 13, "status": "Done"}) + "\n"
            return

        # 2. Schema Introspection
        yield json.dumps({"type": "step", "id": 4, "status": "Introspecting graph schema..."}) + "\n"
        schema = await self._get_schema()
        
        # 3. Dynamic Orchestration
        yield json.dumps({"type": "step", "id": 5, "status": "Analyzing request intent..."}) + "\n"
        orchestration_task = asyncio.create_task(self._orchestrate_retrieval(question, schema, folder_id or "global"))
        vector_task = asyncio.create_task(self.vector_engine.vector_search(question, folder_id))
        
        intent = await orchestration_task
        yield json.dumps({"type": "intent", "data": intent}) + "\n"

        # 4. Data Retrieval
        yield json.dumps({"type": "step", "id": 9, "status": "Extracting graph & semantic data..."}) + "\n"
        retrieval_tasks = [vector_task]
        
        # 1. Cypher Execution
        if intent.get("use_cypher") and intent.get("cypher_query"):
            retrieval_tasks.append(self._execute_cypher(intent["cypher_query"]))
        
        # 2. GDS Algorithm Execution
        if intent.get("use_gds"):
            algo = intent.get("gds_algo", "centrality")
            gds_res = []
            try:
                if algo == "similarity":
                    gds_res = await self.gds_suite.get_similarity_context(folder_id)
                elif algo == "community":
                    gds_res = await self.gds_suite.get_community_context(folder_id)
                elif algo == "paths":
                    path_str = await self.gds_suite.get_path_context("", "", folder_id)
                    retrieval_tasks.append(asyncio.create_task(asyncio.to_thread(lambda: path_str)))
                else:
                    gds_res = await self.gds_suite.get_centrality_context(folder_id)
                
                if gds_res:
                    yield json.dumps({"type": "gds_results", "data": {"algorithm": algo, "results": gds_res}}) + "\n"
                    # Add to Gemini context
                    gds_context = f"[Graph Algorithm Results: {algo}]:\n{json.dumps(gds_res[:10], indent=2, default=str)}"
                    retrieval_tasks.append(asyncio.create_task(asyncio.to_thread(lambda: gds_context)))
            except Exception as e:
                logger.error(f"GDS Execution Failed: {e}")

        # Wait for all Retrieval outputs
        raw_retrieval_outputs = await asyncio.gather(*retrieval_tasks)
        context = self._fuse_context(list(raw_retrieval_outputs))
        
        # 5. Synthesis (Streaming)
        yield json.dumps({"type": "step", "id": 11, "status": "Synthesizing research..."}) + "\n"
        
        full_answer = ""
        prompt = f"""
        You are the **Neural Nexus Research Assistant**, a professional expert in Knowledge Graph synthesis.
        
        CONTEXT FROM DATA RETRIEVAL:
        {context}
        
        USER QUESTION: 
        {question}
        
        INSTRUCTIONS:
        1. CHARACTER: Be informative and professional. Use **simple, natural English**.
        2. DOMAIN NEUTRAL: Do not assume the data topic. Adapt terminology.
        3. NO TECHNICAL CODES: NO IDs, Folder IDs, or internal labels.
        4. NATURAL LANGUAGE: Translate technical relationship names into simple verbs.
        5. RESPONSE STRUCTURE:
           - **Executive Summary**: 4-5 sentences.
           - **Evidence Table**: Markdown Table.
        6. PROACTIVE SYNTHESIS: Focus on what **is** in the data. Avoid negative statements like "not possible to identify" or "cannot be generated". If direct evidence is thin, synthesize based on related entities or general graph patterns found in the context.
        """
        
        async for chunk in self.gemini.astream_response(prompt, combined_history):
            full_answer += chunk
            yield json.dumps({"type": "content", "data": chunk}) + "\n"
        
        # 6. Post-processing & Redis Save
        yield json.dumps({"type": "step", "id": 13, "status": "Finalizing..."}) + "\n"
        try:
            await self.redis.rpush(history_key, json.dumps({"role": "user", "content": question}))
            await self.redis.rpush(history_key, json.dumps({"role": "assistant", "content": full_answer}))
            await self.redis.ltrim(history_key, -20, -1) 
            await self.redis.expire(history_key, 86400) 
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    async def answer(self, question: str, folder_id: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, user_id: str = "anonymous") -> Dict[str, Any]:
        """Non-streaming wrapper for backward compatibility."""
        full_answer = ""
        intent = {}
        async for chunk_raw in self.stream_answer(question, folder_id, history, user_id):
            chunk = json.loads(chunk_raw.strip())
            if chunk["type"] == "content":
                full_answer += chunk["data"]
            elif chunk["type"] == "intent":
                intent = chunk["data"]
        
        return {
            "answer": full_answer,
            "intent": intent,
            "context_summary": f"Retrieved from folder {folder_id or 'global'}."
        }

    async def _get_schema(self) -> str:
        if self._schema_cache and time.time() < self._schema_expiry:
            return str(self._schema_cache)
            
        async with self.neo4j.session() as session:
            try:
                # 1. Get ALL Labels
                labels_res = await session.run("CALL db.labels()")
                labels_data = await labels_res.data()
                # CALL db.labels() returns rows like {'label': '...'}
                labels = [list(r.values())[0] for r in labels_data]
                
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
        3. ALGORITHM BIAS: If user asks about "similarity", "centrality", "connections", "groups", or "important items", set "use_gds" to true.
        4. SYNONYM MAPPING: Map user terms like "benefits", "properties", "effects", or "uses" to the most relevant labels in the schema.
        5. Use CONTAINS and toLower() for robust property matching.

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
