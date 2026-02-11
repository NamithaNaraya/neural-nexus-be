"""
Hybrid RAG Service - LangGraph refactor

Vector + Graph reasoning chain for natural language queries.
Uses LangGraph for orchestration with sliding window context.
"""
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Annotated, TypedDict, Union

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# Sliding window size for conversation context
SLIDING_WINDOW_SIZE = 5

class ChatMessage(TypedDict):
    """Single chat message for LangGraph state."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    citations: Optional[List[Dict[str, Any]]]

class RAGState(TypedDict):
    """
    State for the Hybrid RAG LangGraph.
    """
    # Inputs
    question: str
    session_id: str
    scope: Optional[Dict[str, Any]]
    
    # Context
    history: List[Dict[str, str]]
    enhanced_question: str
    vector_results: List[Dict[str, Any]]
    strategic_results: List[Dict[str, Any]]
    graph_context: Dict[str, Any]
    
    # Outputs
    answer: str
    citations: List[Dict[str, Any]]
    related_nodes: List[str]
    
    # Metadata
    error: Optional[str]
    schema: Optional[Dict[str, Any]]

# Node Implementations
async def context_loading_node(state: RAGState) -> Dict[str, Any]:
    """Step 1: Get conversation context and build enhanced question."""
    logger.info(f"RAG Graph: Loading context for session {state['session_id']}")
    
    history = state.get("history", [])
    last_questions = [msg["content"] for msg in history if msg["role"] == "user"]
    
    enhanced_question = state["question"]
    if last_questions:
        context_summary = " | ".join(last_questions[-3:])
        enhanced_question = f"Previous context: {context_summary}\n\nCurrent question: {state['question']}"
        
    return {"enhanced_question": enhanced_question}

async def vector_search_node(state: RAGState) -> Dict[str, Any]:
    """Step 2: Perform vector similarity search."""
    # This node needs access to neo4j. In LangGraph we usually pass tools or services in config
    # For now we'll assume the service instance handles the neo4j connection
    return {"vector_results": []} # To be implemented in the class wrapper

async def graph_expansion_node(state: RAGState) -> Dict[str, Any]:
    """Step 3: Expand context via graph traversal."""
    return {"graph_context": {}} # To be implemented in the class wrapper

async def answer_generation_node(state: RAGState) -> Dict[str, Any]:
    """Step 4: Generate final answer with citations."""
    return {"answer": "", "citations": []} # To be implemented in the class wrapper


class HybridRAGService:
    """
    Hybrid RAG Service orchestrated by LangGraph.
    """
    
    def __init__(self, neo4j_driver, ai_service):
        self.neo4j = neo4j_driver
        self.ai = ai_service
        self.schema_cache = None
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(RAGState)
        
        # Add nodes (using instance methods to access neo4j/ai)
        workflow.add_node("load_context", context_loading_node)
        workflow.add_node("vector_search", self._vector_search_node)
        workflow.add_node("strategic_scout", self._strategic_scout_node)
        workflow.add_node("graph_expansion", self._graph_expansion_node)
        workflow.add_node("generate_answer", self._answer_generation_node)
        
        # Set edges
        workflow.set_entry_point("load_context")
        workflow.add_edge("load_context", "vector_search")
        workflow.add_edge("vector_search", "strategic_scout")
        workflow.add_edge("strategic_scout", "graph_expansion")
        workflow.add_edge("graph_expansion", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()

    # --- Node Implementation Methods ---

    async def _vector_search_node(self, state: RAGState) -> Dict[str, Any]:
        raw_question = state["question"].lower()
        question_embedding = await self.ai.embed(state["question"])
        
        # 1. Scope handling
        params = {
            "terms": [w.strip("?,.!") .lower() for w in raw_question.split() if len(w) > 2][:10],
            "embedding": question_embedding,
            "top_k": 10
        }
        scope_filter = ""
        if state["scope"]:
            s = state["scope"]
            logger.info(f"[RAG] Applying scope filter: {s}")
            if s.get("type") == "folder":
                scope_filter = "AND node.folder_id = $scope_id"
                params["scope_id"] = s.get("id")
            elif s.get("type") == "file":
                scope_filter = "AND node.file_id = $scope_id"
                params["scope_id"] = s.get("id")
            elif s.get("type") == "selection":
                node_ids = s.get("id").split(",")
                scope_filter = "AND (node.id IN $node_ids OR elementId(node) IN $node_ids)"
                params["node_ids"] = node_ids
        else:
            logger.info("[RAG] No scope filter applied (Global Search)")

        # 2. Independent Search Steps
        all_results = []
        vector_results = []
        lexical_results = []

        # -- Step A: Vector Search (Semantic) --
        vector_query = f"""
        CALL db.index.vector.queryNodes('embedding_idx', $top_k, $embedding) YIELD node, score
        WHERE node.name IS NOT NULL {scope_filter}
        RETURN 
            COALESCE(node.id, elementId(node)) as node_id,
            node.name as name,
            COALESCE(node.description, '') as description,
            labels(node)[0] as type,
            score
        """
        try:
            async with self.neo4j.session() as session:
                result = await session.run(vector_query, params)
                vector_results = await result.data()
                logger.debug(f"[RAG] Cypher Params: { {k: v for k, v in params.items() if k != 'embedding'} }")
                logger.info(f"[RAG] Vector search found {len(vector_results)} matches.")
        except Exception as e:
            logger.warning(f"[RAG] Vector index search failed (index might not be ready): {e}")

        # -- Step B: Lexical Search (Keyword) --
        lexical_query = f"""
        MATCH (node)
        WHERE node.name IS NOT NULL
        AND (
            ANY(term IN $terms WHERE toLower(node.name) CONTAINS term OR toLower(node.description) CONTAINS term)
        )
        {scope_filter} 
        RETURN 
            COALESCE(node.id, elementId(node)) as node_id,
            node.name as name,
            COALESCE(node.description, '') as description,
            labels(node)[0] as type,
            0.85 as score
        LIMIT 20
        """
        try:
            async with self.neo4j.session() as session:
                result = await session.run(lexical_query, params)
                lexical_results = await result.data()
                logger.info(f"[RAG] Lexical search found {len(lexical_results)} matches.")
        except Exception as e:
            logger.error(f"[RAG] Lexical search failed: {e}")

        # 3. Combine and Deduplicate
        seen_ids = set()
        
        # Prioritize vector results
        for r in vector_results:
            if r["node_id"] not in seen_ids:
                all_results.append(r)
                seen_ids.add(r["node_id"])

        # Add lexical results if not already present
        for r in lexical_results:
            if r["node_id"] not in seen_ids:
                all_results.append(r)
                seen_ids.add(r["node_id"])

        # Sort by score and limit
        all_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = all_results[:20]
        
        logger.info(f"[RAG] Total unique context matches: {len(final_results)}")
        if not final_results:
            logger.warning(f"[RAG] No information found for question: '{state['question']}'")

        return {"vector_results": final_results}

    async def _strategic_scout_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Step 3: Strategic Scout - Generate and execute problem-specific Cypher.
        Uncovers multi-hop pathways and structural insights dynamically.
        """
        logger.info(f"[RAG] Strategic Scout analyzing question: {state['question']}")
        
        # 1. Get/Refresh Schema
        if not self.schema_cache:
            try:
                async with self.neo4j.session() as s:
                    # db.labels() returns a list of strings
                    res1 = await s.run("CALL db.labels()")
                    labels_data = await res1.data()
                    labels = [list(r.values())[0] for r in labels_data]
                    
                    # db.relationshipTypes() returns a list of strings
                    res2 = await s.run("CALL db.relationshipTypes()")
                    rels_data = await res2.data()
                    rels = [list(r.values())[0] for r in rels_data]
                    
                    self.schema_cache = {"labels": labels, "relationships": rels}
                    logger.info(f"[RAG] Schema cached: {len(labels)} labels, {len(rels)} relationships.")
            except Exception as e:
                logger.warning(f"Failed to fetch schema: {e}")
                self.schema_cache = {"labels": [], "relationships": []}

        # 2. Identify core entities from vector search
        entities = [r["name"] for r in state["vector_results"][:5]]
        entity_hint = f"Relevant entities in play: {', '.join(entities)}"

        # 3. Generate Cypher via LLM
        scout_prompt = f"""
        You are the Neural Nexus Strategic Scout. Your task is to generate a READ-ONLY Cypher query to answer complex multi-hop questions.
        
        SCHEMA:
        - Labels: {self.schema_cache['labels']}
        - Relationships: {self.schema_cache['relationships']}
        
        {entity_hint}
        
        HISTORICAL STRATEGIC PATTERNS (FEW-SHOT):
        - Q: "Show the chain for Shatavari from herb to outcomes."
          A: MATCH (h:Herb {{name:'Shatavari'}}) OPTIONAL MATCH (h)-[:HAS_QUALITY]->(q:Quality) OPTIONAL MATCH path=(q)-[:FACILITATES_EFFECT]->(:Effect)-[:ENABLES_KARMA]->(:Karma)-[:LEADS_TO_OUTCOME]->(:Outcome) RETURN h, q, path
          
        - Q: "Which qualities of Shatavari contribute most to outcomes?"
          A: MATCH (h:Herb {{name:'Shatavari'}})-[:HAS_QUALITY]->(q:Quality)-[:FACILITATES_EFFECT]->(:Effect)-[:ENABLES_KARMA]->(:Karma)-[:LEADS_TO_OUTCOME]->(o:Outcome) RETURN q.name AS quality, count(*) AS paths ORDER BY paths DESC
          
        QUESTION: {state['question']}
        
        RULES:
        1. Output ONLY a JSON object: {{"reasoning": "...", "cypher": "..."}}
        2. Use only labels and relationships from the SCHEMA.
        3. Keep the query efficient (LIMIT 50).
        4. If the question is simple/factual, return an empty cypher string.
        """
        
        try:
            prediction = await self.ai.chat_json([
                {"role": "system", "content": "You generate high-precision Cypher for graph analytics."},
                {"role": "user", "content": scout_prompt}
            ])
            
            cypher = prediction.get("cypher")
            if not cypher or cypher.strip() == "":
                return {"strategic_results": []}
                
            logger.info(f"[RAG] Executing Strategic Cypher: {cypher}")
            
            async with self.neo4j.session() as s:
                result = await s.run(cypher)
                records = await result.data()
                logger.info(f"[RAG] Strategic Scout found {len(records)} structural insights.")
                return {"strategic_results": records[:10]}
                
        except Exception as e:
            logger.error(f"Strategic Scout failed: {e}")
            return {"strategic_results": []}

    async def _graph_expansion_node(self, state: RAGState) -> Dict[str, Any]:
        if not state["vector_results"]:
            return {"graph_context": {"nodes": [], "relationships": [], "backbone_types": []}}
            
        # seed nodes for expansion (include 20+ to ensure deep strategic paths aren't missed)
        node_ids = [r["node_id"] for r in state["vector_results"][:20]]
        
        # Query 0: Identify dynamic 'backbone' relationships (most frequent in the graph)
        backbone_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        MATCH ()-[r]-() WHERE type(r) = relationshipType
        RETURN relationshipType, count(r) as freq
        ORDER BY freq DESC
        LIMIT 8
        """

        # Query 1: Direct Neighbors
        neighbors_query = """
        UNWIND $node_ids AS nodeId
        MATCH (n)
        WHERE n.id = nodeId OR elementId(n) = nodeId
        OPTIONAL MATCH (n)-[r]-(related)
        RETURN 
            n.name as source_name,
            type(r) as rel_type,
            related.name as related_name
        LIMIT 30
        """
        
        # Query 2: Pathfinding between top entities
        path_query = """
        MATCH (n)
        WHERE (n.id IN $node_ids OR elementId(n) IN $node_ids)
        WITH collect(n) as seedNodes
        UNWIND seedNodes as n1
        UNWIND seedNodes as n2
        WITH n1, n2 WHERE elementId(n1) < elementId(n2)
        // STRATEGIC PATHFINDING: Increased depth to 10 for very far relationships
        MATCH p = shortestPath((n1)-[*..10]-(n2))
        RETURN [node in nodes(p) | node.name] as names, [rel in relationships(p) | type(rel)] as types
        LIMIT 15
        """
        
        try:
            nodes = set()
            rels = []
            
            async with self.neo4j.session() as session:
                # Get Neighbors
                res1 = await session.run(neighbors_query, {"node_ids": node_ids})
                records1 = await res1.data()
                
                for r in records1:
                    source = r["source_name"]
                    rel = r["rel_type"]
                    target = r["related_name"]
                    nodes.add(source)
                    if target:
                        nodes.add(target)
                        rels.append(f"{source} -[{rel}]-> {target}")
                
                # Get Paths
                res2 = await session.run(path_query, {"node_ids": node_ids})
                records2 = await res2.data()
                
                for r in records2:
                    names = r["names"]
                    types = r["types"]
                    path_str = ""
                    for i in range(len(types)):
                        path_str += f"{names[i]} -[{types[i]}]-> "
                    path_str += names[-1]
                    rels.append(f"INDIRECT PATH: {path_str}")
                    for name in names: nodes.add(name)

                # Get Backbone (Dynamic Relevance)
                res0 = await session.run(backbone_query)
                backbone_records = await res0.data()
                backbone_types = [r["relationshipType"] for r in backbone_records]
                
            return {"graph_context": {"nodes": list(nodes), "relationships": rels, "backbone_types": backbone_types}}
        except Exception as e:
            logger.error(f"Graph expansion node failed: {e}")
            return {"graph_context": {"nodes": [], "relationships": []}}

    async def _answer_generation_node(self, state: RAGState) -> Dict[str, Any]:
        # Context formatting
        context = "Analyzed Entities & Attributes:\n"
        for r in state["vector_results"][:15]:
            # Provide more complete context for the AI
            type_label = r.get('type') or 'Entity'
            context += f"- {r['name']} [{type_label}]: {r['description'][:500]}\n"
            
        if state.get("strategic_results"):
            context += "\nStrategic Structural Insights (Dynamic Pathway Analysis):\n"
            context += json.dumps(state["strategic_results"], indent=2) + "\n"
            
        if state["graph_context"].get("relationships"):
            context += "\nStructural Connections (Neighborhood Expansion):\n"
            for rel in state["graph_context"]["relationships"][:15]:
                context += f"- {rel}\n"

        backbone = ", ".join(state["graph_context"].get("backbone_types", []))
        if backbone:
            context += f"\nDomain Backbone (Primary Structural Relationships): {backbone}\n"
                
        system_prompt = (
            "You are the Neural Nexus, a high-level Intelligence Expert. Your mission is to provide concise, summarized, and punchy answers that get straight to the point. "
            "1. CONCISE INTELLIGENCE: Do not be long-winded. Summarize the findings into the essential 'Aha!' moments. Focus on the core connections. "
            "2. PUNCHY STRUCTURE: Use short bullet points and brief bold headers. Eliminate all conversational 'fluff'. "
            "3. THE 'WHY' IN ONE SENTENCE: Explain the logic of the connection quickly. (e.g., 'X connects to Y because of Z'). "
            "4. GRAPH-DRIVEN: Use the 'Strategic Structural Insights' to provide the absolute direct path from start to finish. "
            "5. HUMAN CLARITY: Ensure the answer is instantly understandable. Warm, professional, but incredibly efficient."
        )
        user_prompt = f"Context (Strategic Structural Insights & Knowledge):\n{context}\n\nQuestion: {state['question']}"
        
        try:
            answer = await self.ai.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            citations = [
                {"node_id": r["node_id"], "node_name": r["name"], "score": r["score"]}
                for r in state["vector_results"][:15]
            ]
            
            return {
                "answer": answer,
                "citations": citations,
                "related_nodes": [r["node_id"] for r in state["vector_results"][:15]]
            }
        except Exception as e:
            return {"answer": f"Error generating answer: {e}", "citations": []}

    async def query(
        self,
        question: str,
        session_id: str,
        history: List[Dict[str, str]] = None,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the LangGraph RAG query."""
        initial_state = {
            "question": question,
            "session_id": session_id,
            "history": history or [],
            "scope": scope,
            "vector_results": [],
            "strategic_results": [],
            "graph_context": {},
            "answer": "",
            "citations": [],
            "related_nodes": [],
            "error": None
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            return {
                "answer": final_state["answer"],
                "citations": final_state["citations"],
                "session_id": session_id,
                "related_nodes": final_state["related_nodes"],
                "error": final_state["error"]
            }
        except Exception as e:
            logger.error(f"LangGraph RAG execution failed: {e}")
            return {"answer": f"System error: {e}", "citations": [], "session_id": session_id}


# Singleton instance
_rag_service: Optional[HybridRAGService] = None

def get_rag_service(neo4j, ai_service) -> HybridRAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = HybridRAGService(neo4j, ai_service)
    return _rag_service
