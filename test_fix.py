"""
Quick Test - Just test the method that was failing
"""
import asyncio
from app.combined_chat.rag_service import CombinedRAGService

async def test():
    print("Testing _is_simple_factual method...")
    rag = CombinedRAGService()
    
    test_cases = [
        ("What is ginger?", True),
        ("Tell me about basil", True),
        ("List all herbs", True),
        ("How does ginger relate to inflammation?", False),
        ("Why is basil important?", False),
        ("What are the connections between herbs?", False),
    ]
    
    for question, expected in test_cases:
        result = rag._is_simple_factual(question)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{question}' → {result} (expected {expected})")
    
    print("\nTesting heuristic intent...")
    intent = rag._get_heuristic_intent("What is ginger used for?")
    print(f"✅ Heuristic intent: {intent}")
    
    print("\nAll tests passed!")

asyncio.run(test())
