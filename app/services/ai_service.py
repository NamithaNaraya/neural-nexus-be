"""
AI Service - Ollama Integration

Provides AI model inference for all agents using Ollama.
Supports both chat and embedding models.
"""
import logging
import httpx
from typing import Any, Dict, List, Optional
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """
    Ollama AI Service
    
    Provides:
    - Chat completions (for extraction, validation, etc.)
    - Embeddings (for semantic search)
    """
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.embed_model = settings.OLLAMA_EMBED_MODEL
        self.timeout = settings.AI_REQUEST_TIMEOUT
        
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        """
        Send chat completion request to Ollama.
        
        Args:
            messages: List of {role, content} messages
            model: Model override (uses default if None)
            temperature: Generation temperature (lower = more deterministic)
            json_mode: If True, request JSON output format
            
        Returns:
            Model response text
        """
        model = model or self.model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        
        if json_mode:
            payload["format"] = "json"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
                
        except httpx.TimeoutException:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            raise RuntimeError("AI request timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise RuntimeError(f"AI service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise RuntimeError(f"AI service unavailable: {str(e)}")
    
    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Chat with JSON response parsing.
        
        Returns parsed JSON dict.
        """
        response = await self.chat(messages, model, temperature, json_mode=True)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Try to extract JSON from response
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(response[start:end])
            except:
                pass
            raise RuntimeError(f"Invalid JSON response from AI: {response[:200]}")
    
    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            model: Embedding model override
            
        Returns:
            Embedding vector as list of floats
        """
        model = model or self.embed_model
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": model,
                        "prompt": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
                
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise RuntimeError(f"Embedding service unavailable: {str(e)}")
    
    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Note: Ollama doesn't support batch embeddings natively,
        so we process sequentially.
        """
        embeddings = []
        for text in texts:
            embedding = await self.embed(text, model)
            embeddings.append(embedding)
        return embeddings


# Singleton instance
_ollama_service: Optional[OllamaService] = None


def get_ollama_service() -> OllamaService:
    """Get Ollama service singleton."""
    global _ollama_service
    if _ollama_service is None:
        _ollama_service = OllamaService()
    return _ollama_service
