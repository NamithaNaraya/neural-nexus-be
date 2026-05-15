"""
Neural Nexus RAG Diagnostic Script
Identifies why chat answers are irrelevant or one-word responses
"""
import asyncio
import json
import logging
from app.combined_chat.rag_service import CombinedRAGService
from app.db.connections import (
    init_neo4j, close_neo4j, get_neo4j_driver,
    init_postgres, close_postgres, get_postgres_session,
    init_redis, close_redis, get_redis_client
)
from app.core.config import settings
from app.combined_chat.embedding_service import EmbeddingService
from app.combined_chat.llm_service import get_llm_service

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

async def diagnose():
    print("\n" + "="*80)
    print("🔍 NEURAL NEXUS RAG DIAGNOSTIC")
    print("="*80 + "\n")
    
    # Initialize connections first
    print("⚙️  INITIALIZING CONNECTIONS...")
    print("-" * 40)
    try:
        await init_neo4j()
        print("✅ Neo4j Initialized")
    except Exception as e:
        print(f"❌ Neo4j Initialization Failed: {e}")
        print("Cannot proceed without Neo4j")
        return
    
    try:
        await init_redis()
        print("✅ Redis Initialized")
    except Exception as e:
        print(f"⚠️  Redis not available (non-critical): {e}")
        print("   Chat will work but history won't be cached")
    
    try:
        await init_postgres()
        print("✅ PostgreSQL Initialized")
    except Exception as e:
        print(f"⚠️  PostgreSQL not available (non-critical): {e}")
    print()
    
    # 1. CHECK SETTINGS
    print("1️⃣  CONFIGURATION CHECK")
    print("-" * 40)
    print(f"LLM Provider: {settings.LLM_PROVIDER}")
    print(f"Embedding Provider: {settings.EMBEDDING_PROVIDER}")
    print(f"Ollama Base URL: {settings.OLLAMA_BASE_URL}")
    print(f"Vector Index Name: {settings.VECTOR_INDEX_NAME}")
    print(f"Ollama Model: {settings.OLLAMA_MODEL}")
    print(f"Ollama Embed Model: {settings.OLLAMA_EMBED_MODEL}")
    print()
    
    # 2. CHECK OLLAMA HEALTH
    print("2️⃣  OLLAMA HEALTH CHECK")
    print("-" * 40)
    llm = get_llm_service()
    try:
        health = await llm.check_health()
        print(f"✅ LLM Service Health: {'OK' if health else 'FAILED'}")
    except Exception as e:
        print(f"❌ LLM Health Error: {e}")
    
    # Check embedding
    try:
        emb_service = EmbeddingService()
        test_emb = await emb_service.ai.embed("test")
        print(f"✅ Embedding Service: OK ({len(test_emb)} dimensions)")
    except Exception as e:
        print(f"❌ Embedding Service Error: {e}")
    print()
    
    # 3. CHECK NEO4J CONNECTION & DATA
    print("3️⃣  NEO4J DATA CHECK")
    print("-" * 40)
    neo4j = get_neo4j_driver()
    try:
        async with neo4j.session() as session:
            # Check connection
            await session.run("RETURN 1")
            print(f"✅ Neo4j Connection: OK")
            
            # Count nodes
            result = await session.run("MATCH (n) RETURN COUNT(n) as count")
            row = await result.single()
            total_nodes = row["count"]
            print(f"📊 Total Nodes in DB: {total_nodes}")
            
            # Count by label
            result = await session.run("MATCH (n) RETURN DISTINCT labels(n) as labels, COUNT(n) as count")
            rows = await result.data()
            print("📋 Nodes by Label:")
            for row in rows:
                labels = row.get("labels", ["Unknown"])
                count = row.get("count", 0)
                print(f"   {labels}: {count}")
            
            # Check vector index
            try:
                result = await session.run(f"CALL db.indexes() YIELD name WHERE name = '{settings.VECTOR_INDEX_NAME}' RETURN name, state")
                rows = await result.data()
                if rows:
                    for row in rows:
                        print(f"✅ Vector Index Found: {row.get('name')} ({row.get('state')})")
                else:
                    print(f"❌ Vector Index NOT FOUND: {settings.VECTOR_INDEX_NAME}")
            except Exception as e:
                print(f"⚠️  Vector Index Check Failed: {e}")
            
            # Sample node with properties
            result = await session.run("MATCH (n) RETURN n LIMIT 1")
            row = await result.single()
            if row:
                node = row["n"]
                print(f"📝 Sample Node Properties: {dict(node)}")
                
    except Exception as e:
        print(f"❌ Neo4j Error: {e}")
    print()
    
    # 4. TEST RETRIEVAL CHAIN
    print("4️⃣  RETRIEVAL CHAIN TEST")
    print("-" * 40)
    test_question = "What is the main topic in this knowledge base?"
    print(f"Test Question: '{test_question}'")
    print()
    
    rag = CombinedRAGService()
    
    # Test embedding
    try:
        emb = await emb_service.ai.embed(test_question)
        print(f"✅ Query Embedding: {len(emb)} dimensions")
    except Exception as e:
        print(f"❌ Query Embedding Failed: {e}")
        emb = None
    
    # Test semantic search
    try:
        results = await emb_service.vector_search(test_question, folder_id=None)
        print(f"✅ Semantic Search: {len(results)} results")
        if results:
            for i, item in enumerate(results[:3], 1):
                print(f"   {i}. {item.get('name', 'Unknown')} (score: {item.get('score', 'N/A')})")
                print(f"      Text: {str(item.get('text', ''))[:80]}...")
    except Exception as e:
        print(f"❌ Semantic Search Failed: {e}")
    print()
    
    # 5. TEST CONTEXT PACKING
    print("5️⃣  CONTEXT PACKING TEST")
    print("-" * 40)
    test_context = "\n\n".join([
        "=== Sample Result 1 ===\nThis is sample retrieved context about the topic.",
        "=== Sample Result 2 ===\nMore information relevant to the question.",
    ])
    packed = rag._pack_context([test_context], 8000)
    print(f"✅ Context Packed: {len(packed)} characters")
    print(f"Preview: {packed[:200]}...\n")
    
    # 6. TEST ANSWER SYNTHESIS
    print("6️⃣  ANSWER SYNTHESIS TEST")
    print("-" * 40)
    test_context = "The main ingredient is ginger. It is used for digestion and inflammation."
    test_question = "What is the main ingredient?"
    prompt = rag._build_answer_prompt(test_question, test_context, fast_mode=True)
    print(f"✅ Answer Prompt Generated: {len(prompt)} characters")
    print(f"\nPrompt Preview:\n{prompt[:500]}...\n")
    
    # Test LLM response
    try:
        response = ""
        async for chunk in llm.astream_response(prompt):
            response += chunk
        
        print(f"✅ LLM Response Generated: {len(response)} characters")
        print(f"Response: {response}\n")
        
        # Check if response uses context
        if "ginger" in response.lower():
            print("✅ LLM used provided context")
        else:
            print("❌ LLM ignored context (hallucinating)")
            
    except Exception as e:
        print(f"❌ LLM Response Error: {e}")
    print()
    
    # 7. TEST END-TO-END
    print("7️⃣  END-TO-END RAG TEST")
    print("-" * 40)
    try:
        response = await rag.answer(
            question="Tell me about the data in this folder",
            folder_id=None,
            user_id="diagnostic"
        )
        print(f"✅ Answer Generated:")
        answer_preview = (response.get('answer', 'No answer') or '')[:300]
        print(f"Answer: {answer_preview}...")
        print(f"Data Grounding Score: {response.get('context_summary', 'N/A')}")
    except Exception as e:
        logger.exception(f"End-to-End Error Details:")
        print(f"❌ End-to-End Error: {e}")
        print(f"   This is likely due to missing data or service issue")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80 + "\n")
    
    # Cleanup
    await close_neo4j()
    await close_redis()
    await close_postgres()

async def main():
    try:
        await diagnose()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        try:
            await close_neo4j()
        except:
            pass
        try:
            await close_redis()
        except:
            pass
        try:
            await close_postgres()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
