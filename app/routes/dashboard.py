"""
Dashboard Routes

API endpoints for delivering dashboard metrics and activity.
"""
from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from app.core.security import get_current_user
from app.services.dashboard_service import get_dashboard_service

router = APIRouter()

@router.get("/stats")
async def get_dashboard_stats(
    current_user: dict = Depends(get_current_user),
    service = Depends(get_dashboard_service)
) -> List[Dict[str, Any]]:
    """Return global stats for the dashboard."""
    return await service.get_stats()

@router.get("/activity")
async def get_recent_activity(
    limit: int = 5,
    current_user: dict = Depends(get_current_user),
    service = Depends(get_dashboard_service)
) -> List[Dict[str, Any]]:
    """Return recent system activity."""
    return await service.get_recent_activity(limit)
