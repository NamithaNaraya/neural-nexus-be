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
    graph_context: Dict[str, Any]
    
    # Outputs
    answer: str
    citations: List[Dict[str, Any]]
    related_nodes: List[str]
    
    # Metadata
    error: Optional[str]

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
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(RAGState)
        
        # Add nodes (using instance methods to access neo4j/ai)
        workflow.add_node("load_context", context_loading_node)
        workflow.add_node("vector_search", self._vector_search_node)
        workflow.add_node("graph_expansion", self._graph_expansion_node)
        workflow.add_node("generate_answer", self._answer_generation_node)
        
        # Set edges
        workflow.set_entry_point("load_context")
        workflow.add_edge("load_context", "vector_search")
        workflow.add_edge("vector_search", "graph_expansion")
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
            
        if state["graph_context"].get("relationships"):
            context += "\nStructural Connections (Deep Path Discovery):\n"
            for rel in state["graph_context"]["relationships"][:20]:
                context += f"- {rel}\n"
        
        backbone = ", ".join(state["graph_context"].get("backbone_types", []))
        if backbone:
            context += f"\nDomain Backbone (Primary Structural Relationships): {backbone}\n"
                
        system_prompt = (
            "You are the Neural Nexus Strategic Intelligence Engine. Your role is to uncover and synthesize deep structural connections within the knowledge graph. "
            "1. STRATEGIC PATHFINDING: Trace 'INDIRECT PATHS' to connect entities that aren't directly linked. Search up to 10 levels deep if necessary. "
            "2. DYNAMIC DOMAIN RELEVANCE: Prioritize connections using 'Domain Backbone' relationship types. These are the mandatory structural ties for this specific dataset. "
            "3. BE SYNTHETIC: Explain the logical chain of connection (e.g., 'Entity A is connected to Entity B because of their shared relationship with Entity C'). "
            "4. BE ACCURATE & PROFESSIONAL: Cite specific path segments. Only claim ignorance if the search results are empty."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {state['question']}"
        
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
