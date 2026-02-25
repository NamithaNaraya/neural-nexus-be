import asyncio
from app.db.connections import get_neo4j_driver
from app.services.ai_service import AIService
from app.services.rag import get_enhanced_rag_service

async def main():
    print("Init Neo4j")
    neo4j = get_neo4j_driver()
    print("Init AI Service")
    ai_service = AIService()
    print("Init RAG Service")
    rag_service = get_enhanced_rag_service(neo4j, ai_service)
    print("Initialized RAG service.")
    
    print("Running query...")
    result = await rag_service.query(
        question="What are the properties of Shatavari?",
        session_id="test-1234",
        history=[],
        scope=None
    )
    print("Result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
