import asyncio
import os
import sys
import logging
from sqlalchemy import text

# Add the current directory to sys.path to find 'app'
sys.path.append(os.getcwd())

from app.db.connections import init_postgres, get_postgres_session
from app.db.models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_db():
    """
    Ensure all tables defined in models.py exist in the database.
    This is useful for adding new tables without running manual SQL.
    """
    logger.info("Initializing PostgreSQL schema synchronization...")
    await init_postgres()
    
    from app.db.connections import _postgres_engine
    
    if _postgres_engine is None:
        logger.error("Database engine not initialized.")
        return

    try:
        async with _postgres_engine.begin() as conn:
            # We use create_all to create tables that don't exist
            # Note: This won't modify existing tables (columns), but will create new ones.
            logger.info("Running Base.metadata.create_all...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ PostgreSQL tables synchronized successfully.")
            
    except Exception as e:
        logger.error(f"❌ Failed to synchronize database: {e}")
    finally:
        from app.db.connections import close_postgres
        await close_postgres()

if __name__ == "__main__":
    asyncio.run(init_db())
