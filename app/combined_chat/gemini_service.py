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
        # Native google-genai client for true streaming
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

    async def astream_response(self, prompt: str, history: Optional[List[Dict[str, str]]] = None):
        """
        True streaming using native google-genai SDK.
        Disables thinking to get immediate token-by-token output.
        Falls back to LangChain astream if native fails.
        """
        client = self._get_native_client()
        if client:
            try:
                import asyncio
                from google.genai import types

                # Build conversation with history
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

                # Use gemini-2.5-flash with thinking DISABLED:
                # - Quality answers (2.0-flash dumps raw JSON)
                # - Fast synthesis (~1.4s vs 14.8s with thinking)
                # - Only 5 large chunks → we split into words below
                synthesis_model = settings.GEMINI_MODEL or "gemini-2.5-flash"

                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.1,
                )

                import queue
                import threading

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
                                # for smooth word-by-word streaming
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
                    logger.error(f"Native streaming error: {error_holder[0]}")
                    yield f"\n[Error]: {error_holder[0]}"

                return  # Success — don't fall through to LangChain

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

    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON response."""
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
