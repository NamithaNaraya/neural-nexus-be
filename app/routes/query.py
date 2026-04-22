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

from sqlalchemy import text
from app.core.security import get_current_user
from app.db.connections import get_postgres_session

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Natural language query request."""
    question: str
    scope: Optional[Dict[str, Any]] = None  # type: folder/file/selection, id/ids
    session_id: Optional[str] = None
    clear_history: bool = False

def _to_uuid(sid: str) -> str:
    """Safely convert a string into a UUID string, deterministically hashing non-UUIDs."""
    try:
        # If it's already a valid UUID, this succeeds
        return str(uuid.UUID(sid))
    except (ValueError, TypeError):
        # Otherwise, generate a deterministic UUID
        return str(uuid.uuid5(uuid.NAMESPACE_OID, sid))


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
    # Enhanced RAG fields
    grounding_score: float = 0.0          # Feature 10: 0-1 confidence
    needs_clarification: bool = False      # Feature 2: ask-back flag
    ml_insights_count: int = 0             # Feature 3+6: similar nodes found
    predictions_count: int = 0             # Feature 4+5: ML predictions injected


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
    from app.services.rag import get_enhanced_rag_service
    
    # Create session if not provided, and ensure UUID format
    session_id_raw = request.session_id or str(uuid.uuid4())
    session_id = _to_uuid(session_id_raw)
    
    try:
        # Get services
        neo4j = get_neo4j_driver()
        ai_service = AIService()
        rag_service = get_enhanced_rag_service(neo4j, ai_service)
        
        # 1. Fetch persistent history from PostgreSQL
        history = []
        if not request.clear_history:
            async with get_postgres_session() as session:
                hist_result = await session.execute(
                    text("""
                        SELECT role, message as content 
                        FROM neural_nexus.chat_history 
                        WHERE session_id = :session_id 
                        ORDER BY timestamp DESC 
                        LIMIT 10
                    """),
                    {"session_id": session_id}
                )
                # Reverse to get chronological order for LangGraph
                history = [dict(r) for r in reversed(hist_result.mappings().all())]

        # 2. Execute query via LangGraph-powered RAG Service
        logger.info(f"Initiating RAG service query for session: {session_id}")
        result = await rag_service.query(
            question=request.question,
            session_id=session_id,
            history=history,
            scope=request.scope,
        )
        
        # 3. Store new interaction in PostgreSQL
        async with get_postgres_session() as session:
            # Store User question
            await session.execute(
                text("""
                    INSERT INTO neural_nexus.chat_history (user_id, session_id, role, message)
                    VALUES (:user_id, :session_id, 'user', :message)
                """),
                {"user_id": current_user['id'], "session_id": session_id, "message": request.question}
            )
            # Store Assistant answer
            await session.execute(
                text("""
                    INSERT INTO neural_nexus.chat_history (user_id, session_id, role, message)
                    VALUES (:user_id, :session_id, 'assistant', :message)
                """),
                {"user_id": current_user['id'], "session_id": session_id, "message": result.get("answer", "")}
            )
            await session.commit()
        
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
            session_id=session_id_raw,
            related_nodes=result.get("related_nodes", []),
            grounding_score=result.get("grounding_score", 0.0),
            needs_clarification=result.get("needs_clarification", False),
            ml_insights_count=result.get("ml_insights_count", 0),
            predictions_count=result.get("predictions_count", 0),
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return QueryResponse(
            answer=f"Error processing query: {str(e)}",
            citations=[],
            session_id=session_id,
            related_nodes=[],
        )


@router.get("/query/chat/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 10,  # 5 Q&A pairs
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    db_session_id = _to_uuid(session_id)
    
    async with get_postgres_session() as session:
        result = await session.execute(
            text("""
                SELECT role, message, citations, timestamp
                FROM neural_nexus.chat_history
                WHERE session_id = :session_id AND user_id = :user_id
                ORDER BY timestamp DESC
                LIMIT :limit
            """),
            {"session_id": db_session_id, "user_id": current_user['id'], "limit": limit}
        )
        messages = [dict(r) for r in reversed(result.mappings().all())]
        
        return {
            "session_id": session_id,
            "messages": messages,
            "total_count": len(messages),
        }


@router.get("/query/chat/v2/sessions")
async def list_chat_sessions_v2(
    current_user: dict = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """
    List all chat sessions for the current user with metadata for V2.
    Uses an optimized query to get the last message and timestamp for each session.
    """
    async with get_postgres_session() as session:
        # Optimized window function approach for clean distinct sessions with latest message
        result = await session.execute(
            text("""
                WITH session_stats AS (
                    SELECT session_id, count(*) as message_count
                    FROM neural_nexus.chat_history
                    WHERE user_id = :user_id
                    GROUP BY session_id
                ),
                latest_messages AS (
                    SELECT 
                        session_id, 
                        message as last_message, 
                        timestamp as last_activity,
                        ROW_NUMBER() OVER(PARTITION BY session_id ORDER BY timestamp DESC) as rn
                    FROM neural_nexus.chat_history
                    WHERE user_id = :user_id
                )
                SELECT lm.session_id, lm.last_message, lm.last_activity, ss.message_count
                FROM latest_messages lm
                JOIN session_stats ss ON lm.session_id = ss.session_id
                WHERE lm.rn = 1
                ORDER BY lm.last_activity DESC
            """),
            {"user_id": current_user['id']}
        )
        return [dict(r) for r in result.mappings().all()]


@router.get("/query/chat/sessions")
async def list_chat_sessions(
    current_user: dict = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List all chat sessions for the current user."""
    async with get_postgres_session() as session:
        result = await session.execute(
            text("""
                SELECT DISTINCT ON (session_id) 
                    session_id, 
                    message as last_message, 
                    timestamp as last_activity
                FROM neural_nexus.chat_history
                WHERE user_id = :user_id
                ORDER BY session_id, timestamp DESC
            """),
            {"user_id": current_user['id']}
        )
        return [dict(r) for r in result.mappings().all()]


@router.delete("/query/chat/session/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, str]:
    db_session_id = _to_uuid(session_id)
    async with get_postgres_session() as session:
        await session.execute(
            text("DELETE FROM neural_nexus.chat_history WHERE session_id = :session_id AND user_id = :user_id"),
            {"session_id": db_session_id, "user_id": current_user['id']}
        )
        await session.commit()
        
    return {"message": f"Session {session_id} deleted"}
