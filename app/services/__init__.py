# Services Package
# Core business logic (AI processing, Azure, Neo4j interactions)

from .ai_service import OllamaService, get_ollama_service

__all__ = [
    "OllamaService",
    "get_ollama_service",
]
