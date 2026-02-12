
import asyncio
import logging
import os
import sys

# Add the current directory to sys.path to find 'app'
sys.path.append(os.getcwd())

from app.db.connections import init_postgres, get_postgres_session
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def wipe_sql_data():
    await init_postgres()
    async with get_postgres_session() as session:
        # Tables to wipe
        tables = [
            "neural_nexus.chat_history",
            "neural_nexus.audit_logs",
            "neural_nexus.entity_staging",
            "neural_nexus.files",
            "neural_nexus.folders"
        ]
        
        logger.info("Wiping all PostgreSQL data in neural_nexus schema...")
        for table in tables:
            try:
                # TRUNCATE with CASCADE to handle foreign keys
                await session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                logger.info(f"Wiped table: {table}")
            except Exception as e:
                logger.warning(f"Could not wipe {table}: {e}")
        
        await session.commit()
    logger.info("SQL Wipe Complete.")

if __name__ == "__main__":
    asyncio.run(wipe_sql_data())
