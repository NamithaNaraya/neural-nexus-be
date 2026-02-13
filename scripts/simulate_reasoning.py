import asyncio
import uuid
import json
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.reasoning_pipeline import ReasoningPipeline
from app.services.ai_service import get_ollama_service
from app.services.encounter_service import EncounterService
from app.db.connections import init_neo4j, init_postgres, get_postgres_session

async def simulate_reasoning():
    load_dotenv()
    
    # 1. Initialize Infrastructure
    print("\n🚀 [INIT] Initializing Backend Services...")
    await init_neo4j()
    await init_postgres()
    
    try:
        async with get_postgres_session() as db:
            # 1.5 Ensure a valid user exists for foreign key constraints
            from sqlalchemy import text
            res = await db.execute(text("SELECT id FROM neural_nexus.users LIMIT 1"))
            user_row = res.fetchone()
            
            if user_row:
                user_id = user_row[0]
                print(f"👤 [USER] Using existing user: {user_id}")
            else:
                print("👤 [USER] No users found, creating temporary test user...")
                user_id = uuid.uuid4()
                await db.execute(text("""
                    INSERT INTO neural_nexus.users (id, email, password_hash, role)
                    VALUES (:id, :email, :hash, :role)
                """), {
                    "id": user_id,
                    "email": "test@simulator.ai",
                    "hash": "mock_hash",
                    "role": "user"
                })
                await db.commit()
                print(f"👤 [USER] Created test user: {user_id}")

            # 2. Setup Pipeline
            print("⚙️  [PIPELINE] Setting up 13-Step Orchestrator...")
            encounter_service = EncounterService(db)
            ai_service = get_ollama_service()
            pipeline = ReasoningPipeline(encounter_service, ai_service)
            
            # 3. Sample Case: Mild symptoms (Safe for Graph Reasoning)
            user_input = "I have mild dry skin and feel slightly tired recently."
            session_id = uuid.uuid4()
            folder_id = "ayurveda_folder" # Matches bootstrap data in clinical_schema.cypher
            
            print(f"\n👤 [INPUT] User: \"{user_input}\"")
            print("-" * 50)
            
            # 4. Run Pipeline and Intercept Steps
            print("🧠 [STEP 1-13] Running Automated Reasoning...")
            
            # We invoke the pipeline and inspect results
            result = await pipeline.run({
                "user_input": user_input,
                "user_id": str(user_id),
                "session_id": str(session_id),
                "folder_id": folder_id
            })
            
            # 5. Visual Proof of Steps
            print("\n✅ Simulation Complete. Breakdown of Internal Logic:")
            
            print(f"\n[STEP 2] NER Extraction:")
            print(json.dumps(result.get("extracted_indicators"), indent=2))
            
            print(f"\n[STEP 3] Safety Engine Report:")
            print(json.dumps(result.get("safety_report"), indent=2))
            
            print(f"\n[STEP 5] State Inference (InferenceEngine):")
            print(json.dumps(result.get("inferred_states"), indent=2))
            
            print(f"\n[STEP 6] Graph Intelligence Layer (Neo4j Traversal):")
            if result.get("graph_interventions"):
                for inv in result.get("graph_interventions"):
                    print(f" -> Found Intervention: {inv['intervention']} ({inv['intervention_type']})")
                    print(f"    Reasoning Trace: {inv['source_state']} -[MODULATED_BY]-> {inv['target_property']}")
            else:
                print(" ! No graph paths found. (Ensure folder_id has clinical schema nodes)")

            print(f"\n[STEP 9] Recommendation Range (Qualitative Sizing):")
            print(json.dumps(result.get("recommendation"), indent=2))
            
            print(f"\n[STEP 10] Final Natural Language Output:")
            print("-" * 50)
            print(result.get("final_response"))
            print("-" * 50)
            
            print("\n[STEP 11] Logging Persistence:")
            print(f" -> Encounter stored in PostgreSQL (Session ID: {session_id})")
        
    finally:
        # DB is closed by context manager
        pass

if __name__ == "__main__":
    asyncio.run(simulate_reasoning())

if __name__ == "__main__":
    asyncio.run(simulate_reasoning())
