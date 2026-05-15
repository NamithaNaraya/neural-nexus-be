"""
Database Connections Module

Provides connection management for all external services:
- Neo4j (Graph Database)
- PostgreSQL (User/Audit Data)
- Redis (Task Queue)
"""
import logging
from typing import Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from redis import asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

# === Global Connection Instances ===
_neo4j_driver: Optional[AsyncDriver] = None
_postgres_engine = None
_postgres_session_factory = None
_redis_client: Optional[aioredis.Redis] = None


# === Neo4j ===
async def init_neo4j() -> None:
    """Initialize Neo4j connection."""
    global _neo4j_driver
    _neo4j_driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    # Verify connection
    async with _neo4j_driver.session() as session:
        result = await session.run("RETURN 1 as ping")
        await result.consume()
    logger.info("Neo4j connection verified")


async def close_neo4j() -> None:
    """Close Neo4j connection."""
    global _neo4j_driver
    if _neo4j_driver:
        await _neo4j_driver.close()
        _neo4j_driver = None


def get_neo4j_driver() -> AsyncDriver:
    """Get Neo4j driver instance."""
    if not _neo4j_driver:
        raise RuntimeError("Neo4j driver not initialized")
    return _neo4j_driver


# Alias for dependency injection
get_neo4j = get_neo4j_driver


# === PostgreSQL ===
async def init_postgres() -> None:
    """Initialize PostgreSQL connection."""
    global _postgres_engine, _postgres_session_factory
    
    # Convert postgres:// to postgresql+asyncpg://
    db_url = settings.DATABASE_URL
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    _postgres_engine = create_async_engine(
        db_url,
        echo=settings.DEBUG,
        pool_size=10,
        max_overflow=20,
    )
    _postgres_session_factory = async_sessionmaker(
        _postgres_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # Verify connection
    async with _postgres_engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("PostgreSQL connection verified")


async def close_postgres() -> None:
    """Close PostgreSQL connection."""
    global _postgres_engine
    if _postgres_engine:
        await _postgres_engine.dispose()
        _postgres_engine = None


def get_postgres_session() -> AsyncSession:
    """Get PostgreSQL session."""
    if not _postgres_session_factory:
        raise RuntimeError("PostgreSQL not initialized")
    return _postgres_session_factory()


# === Redis ===
async def init_redis() -> None:
    """Initialize Redis connection."""
    global _redis_client
    _redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )

    # Verify connection with retries to tolerate transient network issues
    max_attempts = 3
    delay = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            await _redis_client.ping()
            logger.info("Redis connection verified")
            return
        except Exception as e:
            logger.warning(f"Redis ping attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(delay)
                delay *= 2

    # If we reach here, ping failed on all attempts. Keep the client object
    # so runtime calls can still attempt to recover; log a clear warning.
    logger.warning("Redis could not be reached during init — continuing without cache. Redis operations may fail until connectivity is restored.")


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


def get_redis_client() -> aioredis.Redis:
    """Get Redis client instance."""
    global _redis_client
    if not _redis_client:
        # Lazy-create client if init_redis wasn't called or failed earlier.
        try:
            _redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Redis client lazily created (no ping performed)")
        except Exception as e:
            logger.warning(f"Failed to lazily create Redis client: {e}")
            raise RuntimeError("Redis client unavailable")
    return _redis_client
