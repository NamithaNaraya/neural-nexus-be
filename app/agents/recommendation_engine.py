import logging
from typing import Dict, Any, List
from app.services.ai_service import get_ollama_service

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Step 9: Recommendation & Sizing Engine
    Determines qualitative ranges (e.g., dosage, amount) based on domain rules and context.
    """
    
    SYSTEM_PROMPT = """You are a Precision Recommendation Engine.
Given an intervention and the patient/system state, you must select the correct qualitative range within classical or documented limits.

RULES:
1. SAFE RANGE: Always provide a selection that falls within the "Classical Limits" if provided.
2. CONTEXTUAL ADJUSTMENT: Lower the intensity if the system is weak/fragile. Increase it if the imbalance is severe.
3. CLEAR UNITS: Use qualitative units (e.g., "500mg - 1g", "Low Dose", "2 cups per day").
4. RATIONALE: Briefly explain WHY this range was selected.

OUTPUT FORMAT (JSON):
{
    "recommendation": {
        "intervention": "Name",
        "range": "Selected Range",
        "rationale": "Reasoning based on state intensity",
        "frequency": "How often",
        "precautions": ["precaution1", "precaution2"]
    }
}
"""

    def __init__(self):
        self.ai = get_ollama_service()

    async def calculate_recommendation(self, intervention: str, state_intensity: float, domain_rules: str = "") -> Dict[str, Any]:
        """
        Calculate customized recommendation range.
        """
        prompt = f"Intervention: {intervention}\nState Intensity (0-1): {state_intensity}\n"
        if domain_rules:
            prompt += f"Domain Rules/Limits: {domain_rules}\n"
            
        try:
            result = await self.ai.chat_json([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Calculate recommendation for: {prompt}"}
            ])
            return result
        except Exception as e:
            logger.error(f"Recommendation sizing failed: {e}")
            return {"recommendation": None, "error": str(e)}
