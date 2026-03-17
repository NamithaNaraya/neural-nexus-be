import logging
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )

    async def generate_response(self, prompt: str, history: List[Dict[str, str]] = None) -> str:
        """Simple wrapper for text generation."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            messages = []
            if history:
                for m in history:
                    if m["role"] == "user": messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant": messages.append(AIMessage(content=m["content"]))
                    elif m["role"] == "system": messages.append(SystemMessage(content=m["content"]))
            
            messages.append(HumanMessage(content=prompt))
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Gemini Service error: {e}")
            return f"Error: {str(e)}"

    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON response."""
        try:
            response_text = await self.generate_response(prompt)
            clean = response_text.strip()
            if clean.startswith("```json"):
                clean = clean[7:-3].strip()
            elif clean.startswith("```"):
                clean = clean[3:-3].strip()
            
            import json
            return json.loads(clean)
        except Exception as e:
            logger.warning(f"JSON Parsing failed: {e}")
            return {"error": str(e), "raw": response_text}
