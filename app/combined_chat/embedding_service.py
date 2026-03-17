import logging
from typing import List, Dict, Any
from app.services.ai_service import get_ai_service
from app.db.connections import get_neo4j_driver
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.ai = get_ai_service()
        self.neo4j = get_neo4j_driver()

    async def vector_search(self, query: str, folder_id: str = None) -> List[Dict[str, Any]]:
        """Uses semantic search to find relevant nodes."""
        # Note: In Neo4j 5.x, vector search uses the db.index.vector.queryNodes procedure
        # Here we fallback to fulltext if vector index isn't specifically named 'vector_index'
        emb = await self.ai.embed(query)
        
        async with self.neo4j.session() as session:
            # We first try vector search, then fallback to fulltext for robustness
            try:
                res = await session.run(f"""
                    CALL db.index.vector.queryNodes('{settings.VECTOR_INDEX_NAME}', 5, $emb)
                    YIELD node, score
                    WHERE ($folder_id IS NULL OR node.folder_id = $folder_id OR node.folderId = $folder_id)
                    RETURN node.name as name, 
                           coalesce(node.text, node.description, 'No textual description') as text, 
                           labels(node) as labels,
                           score
                """, emb=emb, folder_id=folder_id)
                return await res.data()
            except:
                # Fallback to 'entity_search' which is created in neo4j_utils.py
                res = await session.run("""
                    CALL db.index.fulltext.queryNodes('entity_search', $search_text)
                    YIELD node, score
                    WHERE ($folder_id IS NULL OR node.folder_id = $folder_id OR node.folderId = $folder_id)
                    RETURN node.name as name, 
                           coalesce(node.text, node.description, 'No textual description') as text, 
                           labels(node) as labels,
                           score LIMIT 5
                """, search_text=query, folder_id=folder_id)
                return await res.data()
