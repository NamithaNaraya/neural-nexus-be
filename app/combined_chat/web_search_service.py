"""
Web Search Service — Gemini Google Search Grounding

Uses the native google-genai SDK to perform grounded web searches.
When the RAG pipeline has insufficient data, or the user explicitly
clicks "Search the Web", this service queries Gemini with the
Google Search tool enabled, returning real-time web results.
"""
import logging
from typing import Optional
from google import genai
from google.genai import types
from app.core.config import settings

logger = logging.getLogger(__name__)


class WebSearchService:
    """Performs web-grounded search using Gemini + Google Search."""

    def __init__(self):
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for WebSearchService")
        self.client = genai.Client(api_key=api_key)
        self.model = settings.GEMINI_MODEL or "gemini-2.5-flash"
        self.google_search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        logger.info(f"🌐 WebSearchService initialized (model={self.model})")

    async def search(self, question: str, context_hint: Optional[str] = None) -> dict:
        """
        Perform a web-grounded search for the given question.

        Args:
            question: The user's question to search for.
            context_hint: Optional previous RAG answer to give context.

        Returns:
            dict with keys:
                - answer: str (the web-grounded response)
                - source: "web_search"
                - grounding_metadata: dict | None (search entry point, sources)
        """
        try:
            # Build the prompt — if we have a RAG context hint,
            # tell the model to supplement/verify it
            if context_hint:
                prompt = (
                    f"The user asked: \"{question}\"\n\n"
                    f"A knowledge base returned this answer:\n{context_hint[:1000]}\n\n"
                    f"Please search the web to provide additional, up-to-date information "
                    f"that supplements or verifies the above answer. "
                    f"Be concise, factual, and cite sources when possible."
                )
            else:
                prompt = (
                    f"Please search the web and provide a comprehensive, well-structured answer "
                    f"to the following question. Be concise, factual, and cite sources when possible.\n\n"
                    f"Question: {question}"
                )

            # Use sync API wrapped for async compatibility
            # google-genai's generate_content is synchronous
            import asyncio
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[self.google_search_tool],
                ),
            )

            answer_text = response.text or "No results found from web search."

            # Extract grounding metadata if available
            grounding_meta = None
            try:
                if (response.candidates
                        and response.candidates[0].grounding_metadata):
                    gm = response.candidates[0].grounding_metadata
                    grounding_meta = {
                        "search_entry_point": (
                            gm.search_entry_point.rendered_content
                            if gm.search_entry_point else None
                        ),
                        "grounding_chunks": [
                            {
                                "title": getattr(chunk.web, "title", None),
                                "uri": getattr(chunk.web, "uri", None),
                            }
                            for chunk in (gm.grounding_chunks or [])
                            if hasattr(chunk, "web") and chunk.web
                        ],
                    }
            except Exception as meta_err:
                logger.warning(f"Could not extract grounding metadata: {meta_err}")

            logger.info(
                f"🌐 Web search completed | "
                f"Answer length: {len(answer_text)} chars | "
                f"Sources: {len(grounding_meta.get('grounding_chunks', [])) if grounding_meta else 0}"
            )

            return {
                "answer": answer_text,
                "source": "web_search",
                "grounding_metadata": grounding_meta,
            }

        except Exception as e:
            logger.error(f"🌐 Web search failed: {e}", exc_info=True)
            return {
                "answer": (
                    "I wasn't able to complete the web search at the moment. "
                    "Please try again in a few seconds."
                ),
                "source": "web_search",
                "grounding_metadata": None,
                "error": str(e),
            }


# ── Singleton accessor ──────────────────────────────────────
_web_search_service: Optional[WebSearchService] = None


def get_web_search_service() -> WebSearchService:
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService()
    return _web_search_service
