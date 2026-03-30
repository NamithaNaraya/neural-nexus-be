from fastapi import APIRouter, Depends, Query
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
    history: Optional[List[Dict[str, str]]] = []


class WebSearchRequest(BaseModel):
    question: str
    context_hint: Optional[str] = None

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
            user_id=user_id
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
    Can be used standalone or to supplement a RAG answer.
    """
    service = get_web_search_service()
    return await service.search(
        question=request.question,
        context_hint=request.context_hint,
    )
