
import asyncio
import logging
import os
import sys

# Add the current directory to sys.path to find 'app'
sys.path.append(os.getcwd())

from app.db.connections import init_neo4j, get_neo4j_driver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_neo4j_schema():
    await init_neo4j()
    driver = get_neo4j_driver()
    
    constraints = [
        # Entity Constraints
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        
        # Performance Indexes — Entity
        "CREATE INDEX entity_folder_idx IF NOT EXISTS FOR (e:Entity) ON (e.folder_id)",
        "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "CREATE INDEX entity_user_idx IF NOT EXISTS FOR (e:Entity) ON (e.user_id)",
        
        # Chunk & Map Nodes
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE INDEX chunk_file_idx IF NOT EXISTS FOR (c:Chunk) ON (c.file_id)",
        "CREATE INDEX chunk_folder_idx IF NOT EXISTS FOR (c:Chunk) ON (c.folder_id)",
        
        # NOTE: :File and :Folder nodes are NOT created in Neo4j.
        # Entities store folder_id/file_ids as properties — no :File/:Folder label indexes needed.
    ]

    async with driver.session() as session:
        logger.info("Setting up Neo4j Constraints and Indexes...")
        for query in constraints:
            try:
                await session.run(query)
                logger.info(f"Applied: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Query failed: {e}")
        
    logger.info("Neo4j Schema Setup Complete.")

if __name__ == "__main__":
    asyncio.run(setup_neo4j_schema())
