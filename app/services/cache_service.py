"""
Cache Service

Handles Redis-backed caching for graph data to optimize read performance
and ensure efficient data delivery.
"""
import json
import logging
from typing import Optional, Any
from app.db.connections import get_redis_client

logger = logging.getLogger(__name__)

class CacheService:
    """
    Service for managing graph data caching in Redis.
    """
    
    def __init__(self):
        self.redis = get_redis_client()
        self.ttl = 3600  # Default 1 hour TTL
        self.prefix = "neural_nexus:graph:"
        self.analytics_prefix = "neural_nexus:analytics:"

    async def get_cached_value(self, full_key: str) -> Optional[Any]:
        """Retrieve any cached JSON value by its full Redis key."""
        try:
            data = await self.redis.get(full_key)
            if data:
                logger.info(f"Cache HIT for key: {full_key}")
                return json.loads(data)
            logger.info(f"Cache MISS for key: {full_key}")
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set_cached_value(self, full_key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store any JSON-serializable value with an optional TTL."""
        try:
            serialized = json.dumps(data)
            await self.redis.set(full_key, serialized, ex=ttl or self.ttl)
            logger.info(f"Cache SET for key: {full_key}")
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def get_cached_graph(self, scope_key: str) -> Optional[dict]:
        """
        Retrieve cached graph data from Redis.
        """
        try:
            full_key = f"{self.prefix}{scope_key}"
            return await self.get_cached_value(full_key)
        except Exception:
            return None

    async def set_cached_graph(self, scope_key: str, data: Any) -> bool:
        """
        Store graph data in Redis with TTL.
        """
        try:
            full_key = f"{self.prefix}{scope_key}"
            return await self.set_cached_value(full_key, data)
        except Exception:
            return False

    async def get_cached_analytics(self, scope_key: str) -> Optional[dict]:
        """Retrieve cached analytics results from Redis."""
        return await self.get_cached_value(f"{self.analytics_prefix}{scope_key}")

    async def set_cached_analytics(self, scope_key: str, data: Any) -> bool:
        """Store analytics results in Redis."""
        return await self.set_cached_value(f"{self.analytics_prefix}{scope_key}", data)

    async def invalidate_graph(self, scope_key: str) -> bool:
        """
        Remove specific graph data from cache.
        """
        try:
            full_key = f"{self.prefix}{scope_key}"
            await self.redis.delete(full_key)
            logger.info(f"Cache INVALIDATED for key: {full_key}")
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def invalidate_all(self) -> bool:
        """
        Invalidate all Neural Nexus caches that depend on graph data.
        """
        try:
            keys = await self.redis.keys("neural_nexus:*")
            if keys:
                await self.redis.delete(*keys)
            logger.info("All Neural Nexus caches INVALIDATED")
            return True
        except Exception as e:
            logger.error(f"Redis flush error: {e}")
            return False

def get_cache_service() -> CacheService:
    """Dependency provider for CacheService."""
    return CacheService()
