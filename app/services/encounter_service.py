import logging
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Encounter, Outcome

logger = logging.getLogger(__name__)

class EncounterService:
    """
    Handles persistence of reasoning sessions (Encounters) and feedback (Outcomes).
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def create_encounter(
        self, 
        user_id: uuid.UUID, 
        session_id: uuid.UUID, 
        indicators: Dict[str, Any]
    ) -> Encounter:
        """Create a new reasoning encounter."""
        encounter = Encounter(
            user_id=user_id,
            session_id=session_id,
            indicators=indicators,
            safety_status="pending"
        )
        self.db.add(encounter)
        await self.db.commit()
        await self.db.refresh(encounter)
        return encounter

    async def update_encounter_reasoning(
        self, 
        encounter_id: uuid.UUID, 
        inferred_states: Dict[str, Any],
        recommendations: Dict[str, Any],
        safety_status: str
    ):
        """Update encounter with reasoning results."""
        stmt = select(Encounter).filter(Encounter.id == encounter_id)
        result = await self.db.execute(stmt)
        encounter = result.scalar_one_or_none()
        
        if encounter:
            encounter.inferred_states = inferred_states
            encounter.recommendations = recommendations
            encounter.safety_status = safety_status
            await self.db.commit()
            return True
        return False

    async def store_outcome(
        self, 
        encounter_id: uuid.UUID, 
        feedback: str, 
        rating: Optional[int] = None
    ) -> Outcome:
        """Store patient/user feedback as an outcome."""
        outcome = Outcome(
            encounter_id=encounter_id,
            feedback=feedback,
            rating=rating
        )
        self.db.add(outcome)
        await self.db.commit()
        await self.db.refresh(outcome)
        return outcome

    async def get_history(self, user_id: uuid.UUID, limit: int = 10) -> List[Encounter]:
        """Fetch encounter history for a user."""
        stmt = select(Encounter).filter(
            Encounter.user_id == user_id
        ).order_by(Encounter.created_at.desc()).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
