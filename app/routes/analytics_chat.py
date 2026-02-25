from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from app.core.security import get_current_user
from app.services.analytics_chat.analytic_chat_service import get_analytic_chat_service, AnalyticChatService

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    folder_id: Optional[str] = None
    node_ids: Optional[List[str]] = None

@router.post("/query")
async def process_analytic_query(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
    service: AnalyticChatService = Depends(get_analytic_chat_service)
):
    """
    Process a natural language query using graph algorithms.
    """
    return await service.process_query(
        query=request.query,
        folder_id=request.folder_id,
        node_ids=request.node_ids
    )
