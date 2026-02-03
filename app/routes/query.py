"""
Query Routes

Natural language queries using Hybrid RAG (Vector + Graph).
Supports scoped queries and conversational context.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import logging
import uuid

from app.core.security import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Natural language query request."""
    question: str
    scope: Optional[Dict[str, Any]] = None  # type: folder/file/selection, id/ids
    session_id: Optional[str] = None
    clear_history: bool = False


class Citation(BaseModel):
    """Source citation for an answer."""
    node_id: str
    node_name: str
    chunk_text: str
    confidence: float


class QueryResponse(BaseModel):
    """Response to a natural language query."""
    answer: str
    citations: List[Citation]
    session_id: str
    related_nodes: List[str]


@router.post("/query", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
) -> QueryResponse:
    """
    Execute a natural language query using Hybrid RAG.
    
    Steps:
    1. Get conversation context (sliding window - last 5)
    2. Vector Search - Find relevant text chunks
    3. Graph Reasoning - Verify and expand with relationships
    4. LLM Generation - Generate answer with citations
    """
    from app.db.connections import get_neo4j_driver
    from app.services.ai_service import AIService
    from app.services.hybrid_rag import get_rag_service
    
    # Create session if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Get services
        neo4j = get_neo4j_driver()
        ai_service = AIService()
        rag_service = get_rag_service(neo4j, ai_service)
        
        # Execute query
        result = await rag_service.query(
            question=request.question,
            session_id=session_id,
            scope=request.scope,
            clear_history=request.clear_history,
        )
        
        return QueryResponse(
            answer=result.get("answer", "No answer generated"),
            citations=[
                Citation(
                    node_id=c.get("node_id", ""),
                    node_name=c.get("node_name", ""),
                    chunk_text=c.get("chunk_text", ""),
                    confidence=c.get("confidence", 0.0),
                )
                for c in result.get("citations", [])
            ],
            session_id=session_id,
            related_nodes=result.get("related_nodes", []),
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return QueryResponse(
            answer=f"Error processing query: {str(e)}",
            citations=[],
            session_id=session_id,
            related_nodes=[],
        )


@router.get("/chat/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get chat history for a session (Sliding Window: last 5 for context)."""
    # TODO: Query PostgreSQL for chat history
    return {
        "session_id": session_id,
        "messages": [],
        "total_count": 0,
    }


@router.get("/chat/sessions")
async def list_chat_sessions(
    current_user: dict = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List all chat sessions for the current user."""
    # TODO: Query PostgreSQL for user's sessions
    return []


@router.delete("/chat/session/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, str]:
    """Delete a chat session and its history."""
    # TODO: Delete from PostgreSQL
    return {"message": f"Session {session_id} deleted"}
