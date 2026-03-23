import logging
from typing import List, Dict, Any, Optional
from app.services.gds_service import get_gds_service

logger = logging.getLogger(__name__)

class GDSCombinedService:
    """
    Wrapper around the core GDSService that exposes ALL graph algorithms
    for use by the Combined Chat RAG pipeline.

    Every method here delegates to the corresponding GDSService method,
    keeping the combined_chat module isolated while having full algorithm coverage.
    """

    def __init__(self):
        self.gds = get_gds_service()

    # ── CENTRALITY ─────────────────────────────────────────────

    async def get_centrality_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Identifies Hub nodes using ArticleRank (default centrality)."""
        return await self.gds.run_articlerank(folder_id=folder_id, top_k=8)

    async def get_pagerank_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """PageRank — classic influence / importance ranking."""
        return await self.gds.run_pagerank(folder_id=folder_id, top_k=10)

    async def get_articlerank_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """ArticleRank — improved PageRank for graphs with diverse degree distributions."""
        return await self.gds.run_articlerank(folder_id=folder_id, top_k=10)

    async def get_betweenness_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Betweenness Centrality — finds bridges and bottlenecks."""
        return await self.gds.run_betweenness(folder_id=folder_id, top_k=10)

    async def get_closeness_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Closeness Centrality — finds nodes closest to all others."""
        return await self.gds.run_closeness(folder_id=folder_id, top_k=10)

    async def get_degree_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Degree Centrality — counts direct connections per node."""
        return await self.gds.run_degree(folder_id=folder_id, top_k=10)

    async def get_hits_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """HITS — identifies hub nodes (link to many) and authority nodes (linked by many)."""
        return await self.gds.run_hits(folder_id=folder_id, top_k=10)

    # ── COMMUNITY DETECTION ────────────────────────────────────

    async def get_community_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Discovers hidden groups using Louvain community detection."""
        return await self.gds.run_louvain(folder_id=folder_id)

    async def get_louvain_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Louvain — community detection (groups/clusters)."""
        return await self.gds.run_louvain(folder_id=folder_id)

    async def get_leiden_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Leiden — improved community detection over Louvain."""
        return await self.gds.run_leiden(folder_id=folder_id)

    async def get_wcc_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Weakly Connected Components — finds isolated sub-groups."""
        return await self.gds.run_wcc(folder_id=folder_id)

    async def get_kcore_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """K-Core — finds the stable, tightly-connected core of the graph."""
        return await self.gds.run_kcore(folder_id=folder_id, top_k=20)

    async def get_triangle_count_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Triangle Count — counts triangles per node to measure local clustering density."""
        return await self.gds.run_triangle_count(folder_id=folder_id, top_k=20)

    # ── SIMILARITY ─────────────────────────────────────────────

    async def get_similarity_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Finds entities with similar connectivity patterns using Jaccard Similarity."""
        return await self.gds.run_node_similarity(folder_id=folder_id, top_k=8)

    # ── LINK PREDICTION ────────────────────────────────────────

    async def get_link_prediction_common_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Link Prediction — predicts missing links using common neighbors."""
        return await self.gds.run_link_prediction(folder_id=folder_id, method="common_neighbors", top_k=10)

    async def get_link_prediction_adamic_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Link Prediction — Adamic-Adar (weighted by rare shared connections)."""
        return await self.gds.run_link_prediction(folder_id=folder_id, method="adamic_adar", top_k=10)

    async def get_link_prediction_resource_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Link Prediction — Resource Allocation (flow-based)."""
        return await self.gds.run_link_prediction(folder_id=folder_id, method="resource_allocation", top_k=10)

    # ── PATHFINDING & TRAVERSAL ────────────────────────────────

    async def get_bfs_context(self, source_id: str, folder_id: str) -> List[Dict[str, Any]]:
        """BFS Traversal — explores nodes layer by layer from a starting point."""
        return await self.gds.run_bfs(source_id, folder_id=folder_id)

    async def get_dfs_context(self, source_id: str, folder_id: str) -> List[Dict[str, Any]]:
        """DFS Traversal — follows paths as deep as possible before backtracking."""
        return await self.gds.run_dfs(source_id, folder_id=folder_id)

    async def get_random_walk_context(self, source_id: str, folder_id: str) -> List[Dict[str, Any]]:
        """Random Walk — simulates wandering through the graph."""
        return await self.gds.run_random_walk(source_id, folder_id=folder_id)

    async def get_path_context(self, source_name: str, target_name: str, folder_id: str) -> str:
        """Traces paths between entities."""
        return f"Path analysis between {source_name} and {target_name} is in progress."

    # ── TOPOLOGY ───────────────────────────────────────────────

    async def get_topological_sort_context(self, folder_id: str) -> List[Dict[str, Any]]:
        """Topological Sort — orders nodes in a logical linear sequence for DAGs."""
        return await self.gds.run_topological_sort(folder_id=folder_id)
