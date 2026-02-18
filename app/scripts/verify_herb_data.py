
import asyncio
import httpx
import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

BASE_URL = "http://localhost:8000/api/graph"  # Assuming standard port

async def test_api():
    async with httpx.AsyncClient(timeout=10.0) as client:
        print("Testing Herb API...")
        
        # 1. Properties
        print("\n1. GET /herb/Shatavari/properties")
        try:
             # Note: Authentication might be needed if endpoints are protected. 
             # The code shows `current_user: dict = Depends(get_current_user)`.
             # Usually in dev mode or with specific mock auth it might work, 
             # but strictly hitting it from outside might fail without token.
             # However, let's assume we can hit it or we need to bypass auth for test.
             # If auth is required, this test script might fail 401. 
             # I'll check `get_current_user` dependency.
             pass
        except Exception as e:
            print(f"Skipping direct HTTP test due to Auth requirement complexity: {e}")

# Instead of HTTP request which requires Auth token validation, 
# let's invoke the route functions directly or use the neo4j driver to verify data matches query expectations.
# Actually, verifying the data in Neo4j via the exact queries used in the routes is a better proxy 
# if we can't easily generate a valid JWT token here.

from neo4j import AsyncGraphDatabase
from app.core.config import settings

async def verify_queries():
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    
    herb_name = "Shatavari"
    property_name = "Rasa"
    
    print(f"Verifying queries for Herb: {herb_name}")
    
    async with driver.session() as session:
        # Query 1: Properties
        q1 = """
        MATCH (h:Herb {name: $herb_name})-[:HAS_PROPERTY]->(p:Property)
        RETURN p.name AS property
        """
        res1 = await session.run(q1, herb_name=herb_name)
        records1 = [r["property"] for r in await res1.data()]
        print(f"Properties found: {records1}")
        
        # Query 2: Qualities
        q2 = """
        MATCH (h:Herb {name: $herb_name})-[:HAS_PROPERTY]->(p:Property {name: $property_name})
              -[r:HAS_QUALITY {herb: $herb_name}]->(q:Quality)
        RETURN q.name AS quality
        """
        res2 = await session.run(q2, herb_name=herb_name, property_name=property_name)
        records2 = [r["quality"] for r in await res2.data()]
        print(f"Qualities for {property_name}: {records2}")
        
        # Query 3: Full Profile
        q3 = """
        MATCH (h:Herb {name: $herb_name})-[:HAS_PROPERTY]->(p:Property)
        OPTIONAL MATCH (h)-[:HAS_PROPERTY]->(p)-[r:HAS_QUALITY {herb: $herb_name}]->(q:Quality)
        RETURN p.name AS property, collect(q.name) AS qualities
        ORDER BY property
        """
        res3 = await session.run(q3, herb_name=herb_name)
        records3 = await res3.data()
        print("Full Profile:")
        for r in records3:
            print(f"  {r['property']}: {r['qualities']}")
            
    await driver.close()

if __name__ == "__main__":
    asyncio.run(verify_queries())
