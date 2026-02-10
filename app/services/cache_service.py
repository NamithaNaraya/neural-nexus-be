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

    async def get_cached_graph(self, scope_key: str) -> Optional[dict]:
        """
        Retrieve cached graph data from Redis.
        """
        try:
            full_key = f"{self.prefix}{scope_key}"
            data = await self.redis.get(full_key)
            if data:
                logger.info(f"Cache HIT for key: {full_key}")
                return json.loads(data)
            logger.info(f"Cache MISS for key: {full_key}")
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set_cached_graph(self, scope_key: str, data: Any) -> bool:
        """
        Store graph data in Redis with TTL.
        """
        try:
            full_key = f"{self.prefix}{scope_key}"
            serialized = json.dumps(data)
            await self.redis.set(full_key, serialized, ex=self.ttl)
            logger.info(f"Cache SET for key: {full_key}")
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

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
        Invalidate all graph related caches.
        """
        try:
            keys = await self.redis.keys(f"{self.prefix}*")
            if keys:
                await self.redis.delete(*keys)
            logger.info("All graph caches INVALIDATED")
            return True
        except Exception as e:
            logger.error(f"Redis flush error: {e}")
            return False

def get_cache_service() -> CacheService:
    """Dependency provider for CacheService."""
    return CacheService()
