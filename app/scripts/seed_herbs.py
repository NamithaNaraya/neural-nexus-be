
import asyncio
import os
import sys

# Ensure backend path is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from neo4j import AsyncGraphDatabase
from app.core.config import settings

async def seed_herbs():
    print("Connecting to Neo4j...")
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )

    try:
        async with driver.session() as session:
            # 1. Constraints
            print("Creating constraints...")
            constraints = [
                "CREATE CONSTRAINT herb_unique IF NOT EXISTS FOR (h:Herb) REQUIRE h.name IS UNIQUE",
                "CREATE CONSTRAINT quality_unique IF NOT EXISTS FOR (q:Quality) REQUIRE q.name IS UNIQUE",
                "CREATE CONSTRAINT property_unique IF NOT EXISTS FOR (p:Property) REQUIRE p.name IS UNIQUE"
            ]
            for c in constraints:
                await session.run(c)
            
            # 2. Merge Data
            print("Merging herb data...")
            # Note: Splitting big block into smaller chunks if needed, but here we can try one transaction
            cypher = """
            MERGE (rasa:Property    {name:"Rasa"})
            MERGE (guna:Property    {name:"Guna"})
            MERGE (virya:Property   {name:"Virya"})
            MERGE (vipaka:Property  {name:"Vipaka"})

            MERGE (madhura:Quality  {name:"Madhura"})
            MERGE (tikta:Quality    {name:"Tikta"})
            MERGE (katu:Quality     {name:"Katu"})
            MERGE (kashaya:Quality  {name:"Kashaya"})
            MERGE (laghu:Quality    {name:"Laghu"})
            MERGE (ruksha:Quality   {name:"Ruksha"})
            MERGE (tiksna:Quality   {name:"Tiksna"})
            MERGE (guru:Quality     {name:"Guru"})
            MERGE (snigdha:Quality  {name:"Snigdha"})
            MERGE (sita:Quality     {name:"Sita"})
            MERGE (ushna:Quality    {name:"Ushna"})

            MERGE (h1:Herb {name:"Shatavari"}) SET h1.scientific_name="Asparagus racemosus"
            MERGE (h1)-[:HAS_PROPERTY]->(rasa)
            MERGE (h1)-[:HAS_PROPERTY]->(guna)
            MERGE (h1)-[:HAS_PROPERTY]->(virya)
            MERGE (h1)-[:HAS_PROPERTY]->(vipaka)
            MERGE (rasa)-[:HAS_QUALITY {herb:"Shatavari"}]->(madhura)
            MERGE (guna)-[:HAS_QUALITY {herb:"Shatavari"}]->(guru)
            MERGE (guna)-[:HAS_QUALITY {herb:"Shatavari"}]->(snigdha)
            MERGE (virya)-[:HAS_QUALITY {herb:"Shatavari"}]->(sita)
            MERGE (vipaka)-[:HAS_QUALITY {herb:"Shatavari"}]->(madhura)

            MERGE (h2:Herb {name:"Tulsi"}) SET h2.scientific_name="Ocimum sanctum"
            MERGE (h2)-[:HAS_PROPERTY]->(rasa)
            MERGE (h2)-[:HAS_PROPERTY]->(guna)
            MERGE (h2)-[:HAS_PROPERTY]->(virya)
            MERGE (rasa)-[:HAS_QUALITY {herb:"Tulsi"}]->(tikta)
            MERGE (rasa)-[:HAS_QUALITY {herb:"Tulsi"}]->(katu)
            MERGE (rasa)-[:HAS_QUALITY {herb:"Tulsi"}]->(kashaya)
            MERGE (guna)-[:HAS_QUALITY {herb:"Tulsi"}]->(laghu)
            MERGE (guna)-[:HAS_QUALITY {herb:"Tulsi"}]->(ruksha)
            MERGE (guna)-[:HAS_QUALITY {herb:"Tulsi"}]->(tiksna)
            MERGE (virya)-[:HAS_QUALITY {herb:"Tulsi"}]->(ushna)
            """
            await session.run(cypher)
            print("Herb data seeded successfully.")

    except Exception as e:
        print(f"Error seeding data: {e}")
    finally:
        await driver.close()

if __name__ == "__main__":
    asyncio.run(seed_herbs())
