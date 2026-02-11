import asyncio
import logging
from sqlalchemy import text
from app.db.connections import get_postgres_session, init_postgres

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate():
    try:
        # Initialize PostgreSQL connection
        await init_postgres()
        
        async with get_postgres_session() as session:
            logger.info("Adding chunk_data column to entity_staging...")
            try:
                await session.execute(
                    text("ALTER TABLE neural_nexus.entity_staging ADD COLUMN IF NOT EXISTS chunk_data JSONB;")
                )
                await session.commit()
                logger.info("Successfully added chunk_data column.")
            except Exception as e:
                logger.error(f"Migration failed: {e}")
    except Exception as e:
        logger.error(f"PostgreSQL initialization failed: {e}")

if __name__ == "__main__":
    asyncio.run(migrate())
