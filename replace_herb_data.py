
import asyncio
import os
import sys
from sqlalchemy import text

sys.path.append(os.getcwd())

from app.db.connections import init_neo4j, get_neo4j_driver, close_neo4j, init_postgres, get_postgres_session, close_postgres

CYPHER_SCRIPT = """
CREATE CONSTRAINT herb_name_unique IF NOT EXISTS
FOR (h:Herb) REQUIRE h.name IS UNIQUE;

CREATE CONSTRAINT quality_name_unique IF NOT EXISTS
FOR (q:Quality) REQUIRE (q.name, q.dimension) IS UNIQUE;

CREATE CONSTRAINT effect_name_unique IF NOT EXISTS
FOR (e:Effect) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT karma_name_unique IF NOT EXISTS
FOR (k:Karma) REQUIRE k.name IS UNIQUE;

CREATE CONSTRAINT outcome_name_unique IF NOT EXISTS
FOR (o:Outcome) REQUIRE o.name IS UNIQUE;

MERGE (h:Herb {
  name: 'Shatavari',
  scientificName: 'Asparagus racemosus',
  layer: 'Ayurvedic',
  tradition: 'Ayurveda',
  notes: 'Ayurvedic-only graph (no modern biomarkers or phytochemistry).'
})
SET h.source = 'Classical Ayurvedic layer compiled for knowledge graph'

MERGE (rasa:Quality {name:'Rasa', dimension:'Axis'})
SET rasa.source = 'Classical dimension'
MERGE (qR1:Quality {name:'Madhura', dimension:'Rasa'})
MERGE (qR2:Quality {name:'Tikta', dimension:'Rasa'})

MERGE (guna:Quality {name:'Guna', dimension:'Axis'})
SET guna.source = 'Classical dimension'
MERGE (qG1:Quality {name:'Guru', dimension:'Guna'})
MERGE (qG2:Quality {name:'Snigdha', dimension:'Guna'})

MERGE (virya:Quality {name:'Virya', dimension:'Axis'})
SET virya.source = 'Classical dimension'
MERGE (qV1:Quality {name:'Śīta', dimension:'Virya', english:'Cooling'})

MERGE (vipaka:Quality {name:'Vipaka', dimension:'Axis'})
SET vipaka.source = 'Classical dimension'
MERGE (qVp1:Quality {name:'Madhura', dimension:'Vipaka'})

MERGE (h)-[:HAS_QUALITY]->(qR1)
MERGE (h)-[:HAS_QUALITY]->(qR2)
MERGE (h)-[:HAS_QUALITY]->(qG1)
MERGE (h)-[:HAS_QUALITY]->(qG2)
MERGE (h)-[:HAS_QUALITY]->(qV1)
MERGE (h)-[:HAS_QUALITY]->(qVp1)

MERGE (e1:Effect {name:'Balances Vata'})
MERGE (e2:Effect {name:'Balances Pitta'})
MERGE (e3:Effect {name:'Supports reproductive system'})
MERGE (e4:Effect {name:'Nourishes rasa & śukra dhātu'})
MERGE (e5:Effect {name:'Supports hormonal stability'})
MERGE (e6:Effect {name:'Reduces inflammation (Śothahara)'})
MERGE (e7:Effect {name:'Calms the nervous system'})

MERGE (qV1)-[:FACILITATES_EFFECT {rationale:'Cooling virya pacifies Pitta/heat'}]->(e2)
MERGE (qR2)-[:FACILITATES_EFFECT {rationale:'Tikta reduces heat & ama burden'}]->(e6)
MERGE (qR1)-[:FACILITATES_EFFECT {rationale:'Madhura rasa nourishes tissues'}]->(e4)
MERGE (qG2)-[:FACILITATES_EFFECT {rationale:'Snigdha unctuousness nourishes, soothes'}]->(e4)
MERGE (qG1)-[:FACILITATES_EFFECT {rationale:'Guru heaviness stabilizes & strengthens'}]->(e5)
MERGE (qV1)-[:FACILITATES_EFFECT {rationale:'Cooling calms neural agitation'}]->(e7)
MERGE (qR1)-[:FACILITATES_EFFECT {rationale:'Madhura supports vitality & ojas'}]->(e1)
MERGE (qG2)-[:FACILITATES_EFFECT {rationale:'Snigdha reduces dryness & irritability'}]->(e1)

MERGE (h)-[:EXPRESSES_EFFECT]->(e1)
MERGE (h)-[:EXPRESSES_EFFECT]->(e2)
MERGE (h)-[:EXPRESSES_EFFECT]->(e3)
MERGE (h)-[:EXPRESSES_EFFECT]->(e4)
MERGE (h)-[:EXPRESSES_EFFECT]->(e5)
MERGE (h)-[:EXPRESSES_EFFECT]->(e6)
MERGE (h)-[:EXPRESSES_EFFECT]->(e7)

MERGE (k1:Karma {name:'Śukrala', english:'Enhances reproductive fluids'})
MERGE (k2:Karma {name:'Śukraja', english:'Promotes śukra dhātu formation'})
MERGE (k3:Karma {name:'Vṛṣya', english:'Aphrodisiac / fertility enhancing'})
MERGE (k4:Karma {name:'Medhya', english:'Supports intellect & mind'})
MERGE (k5:Karma {name:'Pittahara', english:'Pacifies Pitta'})
MERGE (k6:Karma {name:'Rasāyana', english:'Rejuvenative'})
MERGE (k7:Karma {name:'Stanya-janana', english:'Galactagogue'})
MERGE (k8:Karma {name:'Balya', english:'Strength-promoting'})

MERGE (e4)-[:ENABLES_KARMA]->(k1)
MERGE (e4)-[:ENABLES_KARMA]->(k2)
MERGE (e3)-[:ENABLES_KARMA]->(k3)
MERGE (e7)-[:ENABLES_KARMA]->(k4)
MERGE (e2)-[:ENABLES_KARMA]->(k5)
MERGE (e5)-[:ENABLES_KARMA]->(k6)
MERGE (e4)-[:ENABLES_KARMA]->(k7)
MERGE (e5)-[:ENABLES_KARMA]->(k8)
MERGE (e6)-[:ENABLES_KARMA]->(k5)
MERGE (e6)-[:ENABLES_KARMA]->(k6)

MERGE (h)-[:HAS_KARMA]->(k1)
MERGE (h)-[:HAS_KARMA]->(k2)
MERGE (h)-[:HAS_KARMA]->(k3)
MERGE (h)-[:HAS_KARMA]->(k4)
MERGE (h)-[:HAS_KARMA]->(k5)
MERGE (h)-[:HAS_KARMA]->(k6)
MERGE (h)-[:HAS_KARMA]->(k7)
MERGE (h)-[:HAS_KARMA]->(k8)

MERGE (o1:Outcome {name:'Improved fertility & reproductive vitality'})
MERGE (o2:Outcome {name:'Enhanced lactation'})
MERGE (o3:Outcome {name:'Menstrual & hormonal balance'})
MERGE (o4:Outcome {name:'Digestive comfort & heat reduction'})
MERGE (o5:Outcome {name:'Calmness & emotional stability'})
MERGE (o6:Outcome {name:'Anti-inflammatory relief'})
MERGE (o7:Outcome {name:'Strength & tissue nourishment'})

MERGE (k1)-[:LEADS_TO_OUTCOME]->(o1)
MERGE (k2)-[:LEADS_TO_OUTCOME]->(o1)
MERGE (k3)-[:LEADS_TO_OUTCOME]->(o1)
MERGE (k7)-[:LEADS_TO_OUTCOME]->(o2)
MERGE (k5)-[:LEADS_TO_OUTCOME]->(o3)
MERGE (k6)-[:LEADS_TO_OUTCOME]->(o3)
MERGE (k5)-[:LEADS_TO_OUTCOME]->(o4)
MERGE (k4)-[:LEADS_TO_OUTCOME]->(o5)
MERGE (k5)-[:LEADS_TO_OUTCOME]->(o6)
MERGE (k6)-[:LEADS_TO_OUTCOME]->(o6)
MERGE (k8)-[:LEADS_TO_OUTCOME]->(o7);
"""

async def replace_data():
    await init_neo4j()
    await init_postgres()
    driver = get_neo4j_driver()
    
    folder_id = "ad572339-0ca8-42b4-a329-49752836026d" 
    
    # 0. Find the actual file_id for "Herb" in this folder
    async with get_postgres_session() as session:
        result = await session.execute(
            text("SELECT id FROM neural_nexus.files WHERE folder_id = :f_id AND filename = 'Herb' LIMIT 1"),
            {"f_id": folder_id}
        )
        row = result.fetchone()
        if not row:
            print(f"Error: Could not find file 'Herb' in folder {folder_id}")
            return
        file_id = row[0]
        print(f"Using File ID: {file_id} (Found in DB)")
    
    async with driver.session() as session:
        print(f"--- 1. Clearing old data for File: {file_id} ---")
        await session.run("MATCH (n) WHERE n.file_id = $file_id DETACH DELETE n", {"file_id": file_id})
        
        print("--- 2. Executing User Cypher Script ---")
        statements = [s.strip() for s in CYPHER_SCRIPT.split(';') if s.strip()]
        for i, stmt in enumerate(statements):
            try:
                # We do NOT pass parameters to the user script as it is raw cypher
                await session.run(stmt)
            except Exception as e:
                print(f"Error in statement {i}: {e}")

        print("--- 3. Linking Orphaned Nodes & Adding :Entity Label ---")
        # 3. Auto-link orphaned nodes or nodes from this tradition
        # We adopt any node that has labels used in this script but lacks this file_id
        result = await session.run("""
            MATCH (n)
            WHERE (n:Herb OR n:Quality OR n:Effect OR n:Karma OR n:Outcome)
            AND (n.file_id IS NULL OR NOT $file_id IN n.file_ids)
            SET n.file_ids = CASE 
                WHEN n.file_ids IS NULL THEN [$file_id]
                WHEN NOT $file_id IN n.file_ids THEN n.file_ids + $file_id
                ELSE n.file_ids
            END,
            n.file_id = $file_id, 
            n.folder_id = $folder_id,
            n.id = coalesce(n.id, apoc.create.uuid()),
            n:Entity
            RETURN count(n) as count
        """, {"file_id": file_id, "folder_id": folder_id})
        
        record = await result.single()
        node_count = record["count"]
        print(f"Linked and Labeled {node_count} nodes.")
        
        # Double check total count for file
        count_res = await session.run("MATCH (n) WHERE n.file_id = $file_id RETURN count(n) as c", {"file_id": file_id})
        total_count = (await count_res.single())["c"]
        print(f"Total Nodes for File: {total_count}")

    # 4. Update Postgres
    if total_count > 0:
        async with get_postgres_session() as session:
            await session.execute(
                text("UPDATE neural_nexus.files SET node_count = :count, status = 'completed' WHERE id = :id"),
                {"count": total_count, "id": file_id}
            )
            await session.commit()
            print(f"Updated Postgres node_count to {total_count}")

    await close_neo4j()
    await close_postgres()

if __name__ == "__main__":
    asyncio.run(replace_data())
