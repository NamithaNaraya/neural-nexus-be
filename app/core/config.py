"""
Application Configuration

Loads environment variables and provides typed settings.
All configuration is centralized here for easy management.
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # === Application ===
    APP_NAME: str = "Neural Nexus"
    DEBUG: bool = False
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # === CORS ===
    CORS_ORIGINS: List[str] = ["*"]  # Allow all for development to fix SSE/WS connectivity
    
    # === Database Connections ===
    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # PostgreSQL
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/knowledge_graph"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # === AI Services ===
    LLM_PROVIDER: str = "ollama"  # "ollama" or "gemini"
    EMBEDDING_PROVIDER: str = "ollama"  # "ollama" or "gemini"
    
    # Ollama (used for embeddings)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:latest"
    OLLAMA_EMBED_MODEL: str = "mxbai-embed-large:latest"
    OLLAMA_CHAT_TIMEOUT_SECONDS: int = 90
    OLLAMA_RETRY_ATTEMPTS: int = 2
    OLLAMA_NUM_PREDICT: int = 768
    OLLAMA_NUM_CTX: int = 8192
    OLLAMA_TOP_P: float = 0.9
    OLLAMA_REPEAT_PENALTY: float = 1.05
    RAG_FAST_ANSWER_CACHE_TTL_SECONDS: int = 120
    RAG_HISTORY_WINDOW_MESSAGES: int = 20
    RAG_FAST_HISTORY_WINDOW_MESSAGES: int = 8
    
    # Gemini (used for extracting and chat)
    GEMINI_MODEL: str = "gemini-1.5-pro"
    GOOGLE_API_KEY: Optional[str] = None
    
    # RAG Context & Persona
    RAG_PERSONA: str = "knowledge assistant"
    
    # === Azure Storage ===
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None
    AZURE_CONTAINER_NAME: str = "knowledge-files"
    
    # === Performance Tuning ===
    CELERY_WORKER_CONCURRENCY: int = 4
    MAX_CHUNK_PARALLEL: int = 10
    AI_REQUEST_TIMEOUT: int = 120  # Increased for large documents
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    GRAPH_ALL_DEFAULT_LIMIT: int = 10000
    GRAPH_ALL_MAX_LIMIT: int = 100000
    GRAPH_FOLDER_DEFAULT_LIMIT: int = 1000
    GRAPH_FOLDER_MAX_LIMIT: int = 10000
    GRAPH_LINK_LIMIT_MULTIPLIER: int = 2
    
    # === Feature Flags ===
    ENABLE_WEBSOCKET: bool = True
    ENABLE_PERFORMANCE_MODE: bool = True
    DEFAULT_LOD_LEVEL: str = "balanced"
    
    # === Graph Schema Settings ===
    # Priority for label selection when a node has multiple labels
    # (Domain-specific labels like "Herb" should NOT be here — they come from n.type automatically)
    GRAPH_LABEL_PRIORITY: List[str] = ["Entity"]
    # Labels that are internal/system and should be ignored for domain discovery
    GRAPH_SYSTEM_LABELS: List[str] = ["Entity", "Chunk", "File", "Folder", "Unknown"]
    # Default node types shown in dropdowns when DB is empty or query fails
    DEFAULT_NODE_TYPES: List[str] = ["Person", "Organization", "Concept", "Event", "Location", "Document", "Topic"]
    # Default relationship types shown in dropdowns when DB is empty or query fails
    DEFAULT_RELATIONSHIP_TYPES: List[str] = ["RELATED_TO", "BELONGS_TO", "PART_OF", "CREATED_BY", "WORKS_AT", "LOCATED_IN", "KNOWS"]
    # Property keys to check (in order) when resolving a node's display name
    NODE_NAME_PRIORITY_KEYS: List[str] = ["name", "label", "title", "value", "text", "display_name"]
    
    # === Vector / Embedding Settings ===
    VECTOR_INDEX_NAME: str = "embedding_idx"
    EMBEDDING_DIMENSION: int = 1024  # Must match the embedding model output (mxbai-embed-large = 1024)
    
    # === JWT Settings ===
    JWT_SECRET_KEY: str = "jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
