import logging
from typing import Dict, Any, List
from app.services.ai_service import get_ollama_service

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Step 5: State Inference Engine
    Translates indicators (e.g., symptoms) into domain states (e.g., Dosha, deficiency).
    Built to be domain-agnostic by referencing the knowledge graph schema.
    """
    
    SYSTEM_PROMPT = """You are a Logic Inference Engine.
Given a list of indicators (symptoms/observations) and their intensity levels, you must infer the underlying "States" based on the provided domain context.

RULES:
1. IDENTIFY STATES: Name the states that these indicators most likely point to (e.g., "Pitta Imbalance", "Nitrogen Deficiency").
2. QUANTIFY CONFIDENCE: Provide a confidence score (0-1) for each inferred state.
3. EXPLAIN LOGIC: Briefly explain WHY these indicators lead to this state based on the provided logic.
4. MAPPING: If possible, map indicators to specific properties (e.g., "Burning" -> "Heat").

OUTPUT FORMAT (JSON):
{
    "inferred_states": [
        {
            "state": "State Name",
            "confidence": 0.9,
            "reasoning": "...",
            "indicators": ["indicator1", "indicator2"]
        }
    ]
}
"""

    def __init__(self):
        self.ai = get_ollama_service()

    async def infer_states(self, indicators: List[Dict[str, Any]], domain_context: str = "") -> Dict[str, Any]:
        """
        Infer system states from indicators.
        """
        indicator_data = [{"name": i.get("name"), "level": i.get("level")} for i in indicators]
        
        prompt = f"Indicators: {indicator_data}\n\n"
        if domain_context:
            prompt += f"Domain Context: {domain_context}\n"
            
        try:
            result = await self.ai.chat_json([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Infer states for: {prompt}"}
            ])
            return result
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"inferred_states": [], "error": str(e)}
