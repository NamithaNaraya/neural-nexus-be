import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from app.db.connections import get_postgres_session, get_neo4j_driver
from app.services.ai_service import get_ollama_service
from app.services.encounter_service import EncounterService
from app.agents.reasoning_pipeline import ReasoningPipeline
from app.core.security import get_current_user

router = APIRouter(prefix="/reasoning", tags=["reasoning"])

@router.post("/consult")
async def reasoning_consult(
    payload: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    Execute the 13-step reasoning pipeline for a domain-specific query.
    """
    user_input = payload.get("message")
    user_id = current_user['id']
    folder_id = payload.get("folder_id")
    session_id = payload.get("session_id", str(uuid.uuid4()))
    
    if not user_input or not folder_id:
        raise HTTPException(status_code=400, detail="Missing message or folder_id")
        
    try:
        async with get_postgres_session() as db:
            # Initialize services
            encounter_service = EncounterService(db)
            ai_service = get_ollama_service()
            
            # Initialize and run pipeline
            pipeline = ReasoningPipeline(encounter_service, ai_service)
            result = await pipeline.run({
                "user_input": user_input,
                "user_id": user_id,
                "session_id": session_id,
                "folder_id": folder_id
            })
            
            return {
                "final_response": result.get("final_response"),
                "session_id": session_id,
                "encounter_id": result.get("encounter_id"),
                "status": result.get("status"),
                "extracted_indicators": [str(i) for i in result.get("extracted_indicators", [])],
                "inferred_states": [str(s) for s in result.get("inferred_states", [])],
                "recommendation": str(result.get("recommendation", "")),
                "safety": result.get("safety_report")
            }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def report_outcome(
    payload: Dict[str, Any]
):
    """
    Step 13: Store patient/user feedback as outcome.
    """
    encounter_id = payload.get("encounter_id")
    feedback = payload.get("feedback")
    rating = payload.get("rating")
    
    if not encounter_id or not feedback:
        raise HTTPException(status_code=400, detail="Missing encounter_id or feedback")
        
    async with get_postgres_session() as db:
        encounter_service = EncounterService(db)
        outcome = await encounter_service.store_outcome(
            encounter_id=uuid.UUID(encounter_id),
            feedback=feedback,
            rating=rating
        )
        
        return {"status": "success", "outcome_id": outcome.id}
