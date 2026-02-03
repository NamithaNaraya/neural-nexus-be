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
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
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
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma2:latest"
    OLLAMA_EMBED_MODEL: str = "mxbai-embed-large"
    GOOGLE_API_KEY: Optional[str] = None  # Optional cloud fallback
    
    # === Azure Storage ===
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None
    AZURE_CONTAINER_NAME: str = "knowledge-files"
    
    # === Performance Tuning ===
    CELERY_WORKER_CONCURRENCY: int = 4
    MAX_CHUNK_PARALLEL: int = 10
    AI_REQUEST_TIMEOUT: int = 30
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    
    # === Feature Flags ===
    ENABLE_WEBSOCKET: bool = True
    ENABLE_PERFORMANCE_MODE: bool = True
    DEFAULT_LOD_LEVEL: str = "balanced"
    
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
