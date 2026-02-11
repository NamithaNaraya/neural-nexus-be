
import asyncio
import logging
from app.db.connections import init_redis, get_redis_client, close_redis
from app.services.cache_service import CacheService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        # Initialize Redis connection
        await init_redis()
        
        # Use CacheService to invalidate all graph-related caches
        cache_service = CacheService()
        await cache_service.invalidate_all()
        
        # Also do a flushdb just to be absolutely sure everything is gone
        redis_client = get_redis_client()
        await redis_client.flushdb()
        
        logger.info("Successfully cleared all Redis cache.")
    except Exception as e:
        logger.error(f"Failed to clear Redis cache: {e}")
    finally:
        await close_redis()

if __name__ == "__main__":
    asyncio.run(main())
