from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.core.security import get_current_user
from app.combined_chat.rag_service import CombinedRAGService

router = APIRouter()
_rag_service: Optional[CombinedRAGService] = None

def get_rag_service() -> CombinedRAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = CombinedRAGService()
    return _rag_service

class ChatRequest(BaseModel):
    question: str
    folder_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = []

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
