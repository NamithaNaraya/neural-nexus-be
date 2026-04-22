"""
Cache Service

Handles Redis-backed caching for graph data to optimize read performance
and ensure efficient data delivery.
"""
import json
import logging
from typing import Optional, Any, Iterable, Set
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
    
    async def _collect_keys(self, pattern: str) -> Set[str]:
        """Collect Redis keys matching a pattern using SCAN."""
        collected: Set[str] = set()
        try:
            async for key in self.redis.scan_iter(match=pattern):
                collected.add(key)
        except Exception as e:
            logger.warning(f"Redis scan_iter failed for pattern '{pattern}': {e}")
            # Fallback to KEYS for compatibility; used only if SCAN is unavailable.
            try:
                keys = await self.redis.keys(pattern)
                collected.update(keys or [])
            except Exception as ke:
                logger.error(f"Redis keys fallback failed for pattern '{pattern}': {ke}")
        return collected

    async def _delete_patterns(self, patterns: Iterable[str]) -> int:
        """Delete all keys matching any of the provided patterns."""
        keys_to_delete: Set[str] = set()
        for pattern in patterns:
            keys_to_delete.update(await self._collect_keys(pattern))

        if not keys_to_delete:
            return 0

        await self.redis.delete(*keys_to_delete)
        return len(keys_to_delete)

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

    async def invalidate_graph_global(self) -> bool:
        """Invalidate global graph and analytics caches without touching unrelated keys."""
        try:
            deleted = await self._delete_patterns(
                (
                    f"{self.prefix}all_*",
                    f"{self.prefix}node-types*",
                    f"{self.prefix}relationship-types*",
                    f"{self.analytics_prefix}*",
                )
            )
            logger.info(f"Global graph cache invalidation removed {deleted} keys")
            return True
        except Exception as e:
            logger.error(f"Global graph cache invalidation failed: {e}")
            return False

    async def invalidate_folder_graph(self, folder_id: str, include_global: bool = True) -> bool:
        """Invalidate graph cache entries tied to a specific folder."""
        try:
            patterns = [f"{self.prefix}folder_{folder_id}_*"]
            if include_global:
                patterns.append(f"{self.prefix}all_*")
            deleted = await self._delete_patterns(patterns)
            logger.info(f"Folder cache invalidation for {folder_id} removed {deleted} keys")
            return True
        except Exception as e:
            logger.error(f"Folder cache invalidation failed for {folder_id}: {e}")
            return False

    async def invalidate_file_graph(self, file_id: str, folder_id: Optional[str] = None) -> bool:
        """Invalidate graph cache entries tied to a specific file (and optional folder)."""
        try:
            patterns = [f"{self.prefix}file_{file_id}", f"{self.prefix}all_*"]
            if folder_id:
                patterns.append(f"{self.prefix}folder_{folder_id}_*")
            deleted = await self._delete_patterns(patterns)
            logger.info(f"File cache invalidation for {file_id} removed {deleted} keys")
            return True
        except Exception as e:
            logger.error(f"File cache invalidation failed for {file_id}: {e}")
            return False

def get_cache_service() -> CacheService:
    """Dependency provider for CacheService."""
    return CacheService()
