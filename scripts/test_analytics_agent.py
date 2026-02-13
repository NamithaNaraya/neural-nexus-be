import asyncio
import logging
from app.agents.graph_analytics_agent import GraphAnalyticsAgent
from app.services.ai_service import AIService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_analytics_agent():
    ai = AIService()
    agent = GraphAnalyticsAgent(ai)
    
    test_queries = [
        "Who are the most influential entities in this knowledge base?",
        "Are there any distinct communities or thematic clusters in my data?",
        "How is 'Shatavari' centrally positioned in the graph?",
        "What are the main bottlenecks or bridge nodes in this folder?"
    ]
    
    folder_id = "test_folder_id" # This would need a real folder in a live test
    
    print("\n=== STEP 5.5: DYNAMIC ANALYTICS TEST ===\n")
    
    for query in test_queries:
        print(f"QUERY: {query}")
        selection = await agent.analyze_intent(query)
        if selection:
            print(f"SELECTION: {selection['algorithm']} (Reason: {selection['reason']})")
            # In a real environment with Neo4j, we would call execute_algorithm
            # print(f"Executing with folder_id: {folder_id}...")
        else:
            print("SELECTION: No structural algorithm needed.")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(test_analytics_agent())
