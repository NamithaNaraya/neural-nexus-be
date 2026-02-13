import logging
from typing import Dict, Any, List
from app.services.ai_service import get_ollama_service

logger = logging.getLogger(__name__)

class SafetyAgent:
    """
    Step 3: Safety Engine
    Checks extracted indicators for dangerous/emergency conditions.
    """
    
    SYSTEM_PROMPT = """You are a Medical Safety Auditor.
Your job is to analyze extracted symptoms/indicators and determine if they represent a life-threatening emergency.

RED FLAGS (Emergency):
- Severe chest pain or pressure
- Difficulty breathing or severe shortness of breath
- Sudden confusion or loss of consciousness
- Severe bleeding that won't stop
- Signs of stroke (facial drooping, arm weakness, speech difficulty)
- Severe allergic reaction (swelling, hives, wheezing)
- Suicidal ideation or self-harm risk

VULNERABLE GROUPS: Infants, elderly, and pregnant women are higher risk.

OUTPUT FORMAT (JSON):
{
    "status": "SAFE" | "DANGEROUS",
    "reason": "Brief explanation if dangerous",
    "advice": "Emergency advice if dangerous (e.g., Call 911 immediately)"
}
"""

    def __init__(self):
        self.ai = get_ollama_service()

    async def check_safety(self, indicators: List[Dict[str, Any]], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze indicators for safety red-flags.
        """
        indicator_str = ", ".join([f"{i.get('name')} ({i.get('level', 'unknown')})" for i in indicators])
        
        prompt = f"Indicators: {indicator_str}\n\n"
        if context:
            prompt += f"Patient Context: {context}\n"
            
        try:
            result = await self.ai.chat_json([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Assess safety for: {prompt}"}
            ])
            return result
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return {"status": "UNKNOWN", "error": str(e)}
