
import asyncio
from app.db.connections import get_neo4j_driver, init_neo4j, close_neo4j
from app.core.config import settings

async def sync_degrees():
    await init_neo4j()
    try:
        driver = get_neo4j_driver()
        async with driver.session() as session:
            print("Starting degree synchronization...")
            query = """
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(DISTINCT r) as current_degree
            SET n.degree = current_degree
            RETURN count(n) as updated_count
            """
            result = await session.run(query)
            record = await result.single()
            count = record["updated_count"] if record else 0
            print(f"Successfully synchronized {count} nodes.")
    finally:
        await close_neo4j()

if __name__ == "__main__":
    asyncio.run(sync_degrees())
