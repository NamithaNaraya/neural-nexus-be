import logging
from typing import List, Dict, Any, Optional
from app.services.gds_service import get_gds_service

logger = logging.getLogger(__name__)

class GDSCombinedService:
    def __init__(self):
        self.gds = get_gds_service()

    async def get_similarity_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Finds entities with similar connectivity patterns using PageRank."""
        # Using PageRank as a proxy for 'centrality similarity' in current folder
        return await self.gds.run_pagerank(folder_id=folder_id, top_k=8)

    async def get_centrality_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Identifies Hub nodes using ArticleRank."""
        return await self.gds.run_articlerank(folder_id=folder_id, top_k=8)

    async def get_community_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Discovers hidden groups using Louvain community detection."""
        return await self.gds.run_louvain(folder_id=folder_id)

    async def get_path_context(self, source_name: str, target_name: str, folder_id: str) -> str:
        """Traces paths between entities."""
        return f"Path analysis between {source_name} and {target_name} is in progress."
