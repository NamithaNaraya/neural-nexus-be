
import asyncio
from app.db.connections import get_neo4j_driver

async def check_question_props():
    driver = get_neo4j_driver()
    async with driver.session() as session:
        result = await session.run("MATCH (n:Entity) WHERE labels(n)[0] STARTS WITH 'Question' RETURN n LIMIT 5")
        records = await result.data()
        for rec in records:
            print(f"Node: {rec['n']}")

if __name__ == "__main__":
    asyncio.run(check_question_props())
