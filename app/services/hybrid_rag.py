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
        params = {"query": state["enhanced_question"], "top_k": 10}
        scope_filter = ""
        
        if state["scope"]:
            s = state["scope"]
            if s.get("type") == "folder":
                scope_filter = "AND (n.folder_id = $scope_id OR n.folderId = $scope_id)"
                params["scope_id"] = s.get("id")
            elif s.get("type") == "file":
                scope_filter = "AND (n.file_id = $scope_id OR n.fileId = $scope_id)"
                params["scope_id"] = s.get("id")

        # Vector/Text search query
        query = f"""
        MATCH (n:Entity)
        WHERE (toLower(n.name) CONTAINS toLower($query) OR toLower(n.description) CONTAINS toLower($query))
        {scope_filter}
        RETURN 
            COALESCE(n.id, elementId(n)) as node_id,
            n.name as name,
            n.description as description,
            n.type as type,
            1.0 as score
        LIMIT 10
        """
        
        try:
            # Note: Using session.run or similar depending on neo4j driver version
            # Assuming self.neo4j is a driver instance
            async with self.neo4j.session() as session:
                result = await session.run(query, params)
                records = await result.data()
                
            return {"vector_results": records}
        except Exception as e:
            logger.error(f"Vector search node failed: {e}")
            return {"vector_results": [], "error": str(e)}

    async def _graph_expansion_node(self, state: RAGState) -> Dict[str, Any]:
        if not state["vector_results"]:
            return {"graph_context": {"nodes": [], "relationships": []}}
            
        node_ids = [r["node_id"] for r in state["vector_results"][:5]]
        
        query = """
        UNWIND $node_ids AS nodeId
        MATCH (n)
        WHERE n.id = nodeId OR elementId(n) = nodeId
        OPTIONAL MATCH (n)-[r]-(related)
        RETURN 
            n.name as source_name,
            type(r) as rel_type,
            related.name as related_name
        LIMIT 20
        """
        
        try:
            async with self.neo4j.session() as session:
                result = await session.run(query, {"node_ids": node_ids})
                records = await result.data()
                
            nodes = set()
            rels = []
            for r in records:
                nodes.add(r["source_name"])
                if r["related_name"]:
                    nodes.add(r["related_name"])
                    rels.append(f"{r['source_name']} -[{r['rel_type']}]-> {r['related_name']}")
                    
            return {"graph_context": {"nodes": list(nodes), "relationships": rels}}
        except Exception as e:
            logger.error(f"Graph expansion node failed: {e}")
            return {"graph_context": {"nodes": [], "relationships": []}}

    async def _answer_generation_node(self, state: RAGState) -> Dict[str, Any]:
        # Context formatting
        context = "Relevant Entities:\n"
        for r in state["vector_results"][:5]:
            context += f"- {r['name']} ({r['type']}): {r['description'][:150]}\n"
            
        if state["graph_context"].get("relationships"):
            context += "\nRelationships:\n"
            for rel in state["graph_context"]["relationships"][:10]:
                context += f"- {rel}\n"
                
        system_prompt = "You are a Knowledge Graph assistant. Answer based ONLY on the context. If unknown, say so."
        user_prompt = f"Context:\n{context}\n\nQuestion: {state['question']}"
        
        try:
            answer = await self.ai.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            citations = [
                {"node_id": r["node_id"], "node_name": r["name"], "score": r["score"]}
                for r in state["vector_results"][:5]
            ]
            
            return {
                "answer": answer,
                "citations": citations,
                "related_nodes": [r["node_id"] for r in state["vector_results"][:5]]
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
