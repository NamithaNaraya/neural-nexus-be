"""
Database Package

Provides database connections and ORM models.
"""
from app.db.connections import (
    init_neo4j,
    init_postgres,
    init_redis,
    close_neo4j,
    close_postgres,
    close_redis,
    get_neo4j_driver,
    get_postgres_session,
    get_redis_client,
)
from app.db.models import Base, User, Folder, File, AuditLog, ChatHistory

__all__ = [
    # Connections
    "init_neo4j",
    "init_postgres",
    "init_redis",
    "close_neo4j",
    "close_postgres",
    "close_redis",
    "get_neo4j_driver",
    "get_postgres_session",
    "get_redis_client",
    # Models
    "Base",
    "User",
    "Folder",
    "File",
    "AuditLog",
    "ChatHistory",
]
