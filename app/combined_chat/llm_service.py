import logging
import asyncio
import json
import queue
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.core.config import settings

logger = logging.getLogger(__name__)

class BaseLLMService(ABC):
    """Base interface for all LLM providers (Gemini, Ollama, etc.)."""
    
    @abstractmethod
    async def generate_response(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Simple text generation."""
        pass

    @abstractmethod
    async def astream_response(self, prompt: str, history: Optional[List[Dict[str, str]]] = None):
        """Streaming token-by-token response."""
        pass
    
    @abstractmethod
    async def check_health(self) -> bool:
        """Verify the provider is reachable and the model is available."""
        pass

    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Base implementation for JSON generation via text parsing."""
        response_text = ""
        try:
            response_text = await self.generate_response(prompt)
            clean = response_text.strip()
            # Remove possible Markdown blocks
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            
            import json
            return json.loads(clean)
        except Exception as e:
            logger.warning(f"JSON Parsing failed: {e}")
            return {"error": str(e), "raw": response_text}


class GeminiService(BaseLLMService):
    """Google Gemini Implementation."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        self._native_client = None

    def _get_native_client(self):
        """Lazy-init native google-genai client."""
        if self._native_client is None:
            try:
                from google import genai
                self._native_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            except Exception as e:
                logger.warning(f"Could not init native genai client: {e}")
        return self._native_client

    async def generate_response(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
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

    async def astream_response(self, prompt: str, history: Optional[List[Dict[str, str]]] = None):
        """True streaming using native google-genai SDK with sub-chunking."""
        client = self._get_native_client()
        if client:
            try:
                from google.genai import types

                contents = []
                if history:
                    for m in history:
                        role = "user" if m["role"] == "user" else "model"
                        contents.append(types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=m["content"])]
                        ))
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                ))

                synthesis_model = settings.GEMINI_MODEL or "gemini-2.5-flash"
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.1,
                )

                chunk_queue = queue.Queue()
                done_event = threading.Event()
                error_holder = [None]

                def _stream_worker():
                    try:
                        for chunk in client.models.generate_content_stream(
                            model=synthesis_model,
                            contents=contents,
                            config=config,
                        ):
                            if chunk.text:
                                # Split large chunks into ~8-word sub-chunks
                                words = chunk.text.split(' ')
                                sub_chunk = []
                                for w in words:
                                    sub_chunk.append(w)
                                    if len(sub_chunk) >= 8:
                                        chunk_queue.put(' '.join(sub_chunk) + ' ')
                                        sub_chunk = []
                                if sub_chunk:
                                    chunk_queue.put(' '.join(sub_chunk))
                    except Exception as e:
                        error_holder[0] = e
                    finally:
                        done_event.set()

                thread = threading.Thread(target=_stream_worker, daemon=True)
                thread.start()

                while not done_event.is_set() or not chunk_queue.empty():
                    try:
                        text = chunk_queue.get(timeout=0.05)
                        yield text
                    except queue.Empty:
                        await asyncio.sleep(0.02)

                if error_holder[0]:
                    yield f"\n[Error]: {error_holder[0]}"
                return
            except Exception as e:
                logger.warning(f"Native streaming failed, falling back to LangChain: {e}")

        # Fallback: LangChain astream
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            messages = []
            if history:
                for m in history:
                    if m["role"] == "user": messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant": messages.append(AIMessage(content=m["content"]))
                    elif m["role"] == "system": messages.append(SystemMessage(content=m["content"]))
            
            messages.append(HumanMessage(content=prompt))
            async for chunk in self.llm.astream(messages):
                yield chunk.content
        except Exception as e:
            logger.error(f"Gemini Streaming error: {e}")
            yield f"\n[Streaming Error]: {str(e)}"


    async def check_health(self) -> bool:
        """Verify Gemini connection by sending a minimal ping prompt."""
        try:
            from langchain_core.messages import HumanMessage
            # Use a tiny prompt to minimize tokens
            await self.llm.ainvoke([HumanMessage(content="ping")])
            return True
        except Exception as e:
            logger.error(f"♊ Gemini Health Check Failed: {e}")
            return False


class OllamaService(BaseLLMService):
    """Local Ollama Implementation."""
    
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1
        )

    async def generate_response(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
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
            logger.error(f"Ollama Service error: {e}")
            return f"Error: {str(e)}"

    async def astream_response(self, prompt: str, history: Optional[List[Dict[str, str]]] = None):
        """Streaming for Ollama with the same sub-chunking logic for word-by-word feel."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            messages = []
            if history:
                for m in history:
                    if m["role"] == "user": messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant": messages.append(AIMessage(content=m["content"]))
                    elif m["role"] == "system": messages.append(SystemMessage(content=m["content"]))
            
            messages.append(HumanMessage(content=prompt))
            
            async for chunk in self.llm.astream(messages):
                if chunk and chunk.content:
                    # Apply sub-chunking logic if chunk is large
                    # (Though Ollama usually streams faster, we keep parity)
                    if len(chunk.content.split(' ')) > 15:
                        words = chunk.content.split(' ')
                        sub_chunk = []
                        for w in words:
                            sub_chunk.append(w)
                            if len(sub_chunk) >= 8:
                                yield ' '.join(sub_chunk) + ' '
                                sub_chunk = []
                        if sub_chunk:
                            yield ' '.join(sub_chunk)
                    else:
                        yield chunk.content
        except Exception as e:
            logger.error(f"Ollama Streaming error: {e}")
            yield f"\n[Ollama Error]: {str(e)}"


    async def check_health(self) -> bool:
        """Verify Ollama connection by checking the local endpoint."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    # Check if the configured model exists in the tags
                    models = resp.json().get("models", [])
                    model_names = [m["name"] for m in models]
                    # Exact match or with :latest
                    target = settings.OLLAMA_MODEL
                    if target in model_names or f"{target}:latest" in model_names:
                        return True
                    logger.warning(f"🤖 Ollama is running but model '{target}' not found in {model_names}")
                    # Even if model not found, server is up
                    return True 
                return False
        except Exception as e:
            logger.error(f"🤖 Ollama Health Check Failed: {e}")
            return False


def get_llm_service() -> BaseLLMService:
    """Factory to return the configured LLM provider."""
    provider = getattr(settings, "LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        logger.info("🤖 Initializing Ollama Service")
        return OllamaService()
    else:
        logger.info("♊ Initializing Gemini Service")
        return GeminiService()
