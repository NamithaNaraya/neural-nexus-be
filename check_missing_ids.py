import asyncio
from app.db.connections import init_neo4j, get_neo4j_driver, close_neo4j

async def check_ids():
    await init_neo4j()
    driver = get_neo4j_driver()
    async with driver.session() as session:
        result = await session.run("MATCH (n:Entity) WHERE n.id IS NULL RETURN count(n) as count")
        record = await result.single()
        print(f"Nodes missing 'id': {record['count']}")
        
        if record['count'] > 0:
            result = await session.run("MATCH (n:Entity) WHERE n.id IS NULL RETURN labels(n) as labels, n.name as name LIMIT 5")
            async for rec in result:
                print(f"  Missing ID: {rec['labels']} - {rec['name']}")
    await close_neo4j()

if __name__ == "__main__":
    asyncio.run(check_ids())
