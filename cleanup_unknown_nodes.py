import asyncio
import logging
from app.db.connections import init_neo4j, get_neo4j_driver, close_neo4j

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def cleanup():
    await init_neo4j()
    driver = get_neo4j_driver()
    
    async with driver.session() as session:
        logger.info("Starting cleanup of 'Unknown' nodes...")
        
        # 0. Orphan Adoption: Link nodes with labels but no metadata to the last known folder/file
        # (This is a safety net for failed ingestions)
        # Note: In a real cleanup, we might prompt for a file_id, but here we repair nodes that have labels.
        # For this script, we'll focus on repairing nodes that already have some metadata but are broken,
        # OR nodes that have :Herb/etc labels but are missing :Entity.
        
        # 1. Identify nodes with name 'Unknown' or missing labels
        # We look for nodes that have no properties other than maybe folder_id/file_id
        # or nodes explicitly named 'Unknown'
        
        # Merge 'Unknown' nodes with their correctly named counterparts if they share an ID
        # (This happens when variable sharing fails and creates a duplicate partial node)
        
        # For this specific case, we'll delete nodes with name 'Unknown' that are orphans
        # or nodes that don't have the :Entity label but are linked to files.
        
        result = await session.run("""
            MATCH (n)
            WHERE (n.name = 'Unknown' OR n.name IS NULL)
            AND (n.file_id IS NOT NULL OR n.folder_id IS NOT NULL)
            WITH n, [l IN labels(n) WHERE l <> 'Entity'] as other_labels
            WHERE size(other_labels) = 0
            DETACH DELETE n
            RETURN count(*) as count
        """)
        record = await result.single()
        deleted_count = record["count"] if record else 0
        logger.info(f"Deleted {deleted_count} malformed 'Unknown' nodes.")

        # 2. Repair nodes that have labels but are missing the :Entity label
        result = await session.run("""
            MATCH (n)
            WHERE NOT n:Entity 
            AND (n:Herb OR n:Quality OR n:Effect OR n:Karma OR n:Outcome)
            SET n:Entity
            RETURN count(*) as count
        """)
        record = await result.single()
        repaired_count = record["count"] if record else 0
        logger.info(f"Added :Entity label to {repaired_count} nodes.")

        # 3. Ensure 'type' property matches labels
        result = await session.run("""
            MATCH (n:Entity)
            WHERE n.type IS NULL
            WITH n, [l IN labels(n) WHERE l <> 'Entity'][0] as label_type
            WHERE label_type IS NOT NULL
            SET n.type = label_type
            RETURN count(*) as count
        """)
        record = await result.single()
        type_fix_count = record["count"] if record else 0
        logger.info(f"Fixed 'type' property for {type_fix_count} nodes.")

        # 4. Generate missing IDs (CRITICAL for frontend)
        result = await session.run("""
            MATCH (n:Entity)
            WHERE n.id IS NULL
            SET n.id = apoc.create.uuid()
            RETURN count(n) as count
        """)
        record = await result.single()
        id_fix_count = record["count"] if record else 0
        logger.info(f"Generated missing IDs for {id_fix_count} nodes.")

    await close_neo4j()

if __name__ == "__main__":
    asyncio.run(cleanup())
