from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.core.security import get_current_user
from app.combined_chat.rag_service import CombinedRAGService
from app.combined_chat.web_search_service import get_web_search_service, WebSearchService

router = APIRouter()
_rag_service: Optional[CombinedRAGService] = None

def get_rag_service() -> CombinedRAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = CombinedRAGService()
    return _rag_service


def invalidate_rag_caches():
    """
    Invalidate the RAG service caches (schema, depth, etc.).
    Called by CRUD routes after graph mutations so the very next
    question sees the updated data.
    """
    global _rag_service
    if _rag_service is not None:
        _rag_service.invalidate_schema_cache()


class ChatRequest(BaseModel):
    question: str
    folder_id: Optional[str] = None
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = []
    web_search: bool = False


class WebSearchRequest(BaseModel):
    question: str
    context_hint: Optional[str] = None
    session_id: Optional[str] = None

from fastapi.responses import StreamingResponse

@router.post("/answer")
async def get_combined_answer(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    service: CombinedRAGService = Depends(get_rag_service)
):
    user_id = current_user.get("id") or current_user.get("sub")
    return await service.answer(
        question=request.question,
        folder_id=request.folder_id,
        history=request.history,
        user_id=user_id
    )

@router.post("/stream-answer")
async def stream_combined_answer(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    service: CombinedRAGService = Depends(get_rag_service)
):
    user_id = current_user.get("id") or current_user.get("sub")
    return StreamingResponse(
        service.stream_answer(
            question=request.question,
            folder_id=request.folder_id,
            history=request.history,
            user_id=user_id,
            session_id=request.session_id,
            web_search=request.web_search,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        }
    )


@router.post("/web-search")
async def web_search(
    request: WebSearchRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Perform a web-grounded search using Gemini + Google Search.
    Saves results to PostgreSQL for persistence.
    """
    service = get_web_search_service()
    result = await service.search(
        question=request.question,
        context_hint=request.context_hint,
    )

    # Save web search result to PostgreSQL for persistent history
    if request.session_id:
        try:
            import uuid
            import json
            from sqlalchemy import text as sa_text
            from app.db.connections import get_postgres_session
            import logging

            user_id = current_user.get("id") or current_user.get("sub")
            db_session_id = request.session_id
            try:
                db_session_id = str(uuid.UUID(db_session_id))
            except (ValueError, TypeError):
                db_session_id = str(uuid.uuid5(uuid.NAMESPACE_OID, db_session_id))

            # Persist sources as a real JSON array so history reload can parse them
            # consistently across frontend versions.
            sources = (
                result.get("grounding_metadata", {}).get("grounding_chunks", [])
                if result.get("grounding_metadata")
                else []
            )
            sources_json = json.dumps(sources)
            web_attachment_json = json.dumps(
                {
                    "answer": result.get("answer", ""),
                    "sources": sources,
                }
            )

            async with get_postgres_session() as session:
                try:
                    await session.execute(
                        sa_text("""
                            INSERT INTO neural_nexus.chat_history (user_id, session_id, role, message, citations)
                            VALUES (:user_id, :session_id, 'web_search', :message, CAST(:citations AS JSONB))
                        """),
                        {
                            "user_id": user_id,
                            "session_id": db_session_id,
                            "message": result.get("answer", ""),
                            "citations": sources_json,
                        }
                    )
                except Exception as insert_error:
                    logging.getLogger(__name__).warning(
                        f"Falling back to assistant-row web search persistence: {insert_error}"
                    )
                    updated = await session.execute(
                        sa_text("""
                            UPDATE neural_nexus.chat_history
                            SET citations = jsonb_set(
                                CASE
                                    WHEN citations IS NULL THEN '{}'::jsonb
                                    WHEN jsonb_typeof(citations) = 'object' THEN citations
                                    ELSE jsonb_build_object('rag_citations', citations)
                                END,
                                '{web_search_attachment}',
                                CAST(:web_attachment AS JSONB),
                                true
                            )
                            WHERE id = (
                                SELECT id
                                FROM neural_nexus.chat_history
                                WHERE user_id = :user_id
                                  AND session_id = :session_id
                                  AND role = 'assistant'
                                ORDER BY timestamp DESC
                                LIMIT 1
                            )
                        """),
                        {
                            "user_id": user_id,
                            "session_id": db_session_id,
                            "web_attachment": web_attachment_json,
                        }
                    )

                    if updated.rowcount == 0:
                        await session.execute(
                            sa_text("""
                                INSERT INTO neural_nexus.chat_history (user_id, session_id, role, message, citations)
                                VALUES (:user_id, :session_id, 'assistant', '', CAST(:citations AS JSONB))
                            """),
                            {
                                "user_id": user_id,
                                "session_id": db_session_id,
                                "citations": json.dumps({"web_search_attachment": json.loads(web_attachment_json)}),
                            }
                        )
                await session.commit()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save web search to PostgreSQL: {e}")

    return result


class GeneralAnswerRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


@router.post("/general-answer")
async def general_answer(
    request: GeneralAnswerRequest,
    current_user: dict = Depends(get_current_user),
    service: CombinedRAGService = Depends(get_rag_service),
):
    """
    Generate an answer using ONLY the LLM's general knowledge (no RAG retrieval).
    Used when the user clicks "Answer outside DB" after an initial no-data response.
    """
    import json

    prompt = (
        f"You are a knowledgeable assistant. The user asked a question that had no matching data "
        f"in their knowledge graph database. They have now requested a general knowledge answer.\n\n"
        f"Question: {request.question}\n\n"
        f"Provide a helpful, accurate answer based on your general training knowledge. "
        f"Start your response with: '⚠️ **General Knowledge Answer** (not from your database):\\n\\n' "
        f"Then answer the question thoroughly but concisely."
    )

    async def stream_general():
        async for chunk in service.llm.astream_response(prompt):
            if chunk:
                yield json.dumps({"type": "content", "data": chunk}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        stream_general(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
