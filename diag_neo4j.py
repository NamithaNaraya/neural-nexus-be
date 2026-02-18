import asyncio
import os
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

# Load .env from the backend root
load_dotenv()

async def diagnostic():
    uri = os.getenv("NEO4J_URI", "bolt://10.10.20.86:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    print(f"Connecting to {uri} as {user}...")
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        async with driver.session() as session:
            # Search for nodes that have a UUID as their name or specific problematic ID
            print("\n--- Searching for nodes with UUID names ---")
            try:
                query = """
                MATCH (n:Entity)
                WHERE n.name =~ '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
                OR n.id = 'd665bf79-a1f3-402f-a655-3b73cae2366c'
                RETURN n, labels(n) as labels LIMIT 10
                """
                result = await session.run(query)
                async for record in result:
                    node = record["n"]
                    labels = record["labels"]
                    print(f"\nNode ID: {node.get('id')} | Labels: {labels}")
                    print(f"Properties: {dict(node)}")
            except Exception as e:
                print(f"Search failed: {e}")
            
            print("\n--- Listing Active Constraints ---")
            try:
                constraints = await session.run("SHOW CONSTRAINTS")
                async for rec in constraints:
                    print(f"Constraint: {rec['name']} | Type: {rec['type']} | Labels: {rec['labelsOrTypes']} | Properties: {rec['properties']}")
            except Exception as const_err:
                print(f"Error fetching constraints: {const_err}")
        await driver.close()
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(diagnostic())
