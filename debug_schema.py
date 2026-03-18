import asyncio
import json
from app.combined_chat.rag_service import CombinedRAGService

async def check():
    rag = CombinedRAGService()
    schema = await rag._get_schema()
    print("--- SCHEMA START ---")
    print(schema)
    print("--- SCHEMA END ---")

if __name__ == "__main__":
    asyncio.run(check())
