import logging
from typing import Dict, Any, List, Optional
from app.services.gds_service import get_gds_service

logger = logging.getLogger(__name__)

class GraphAnalyticsAgent:
    """
    Agent responsible for selecting and executing graph algorithms based on user query intent.
    """
    
    def __init__(self, ai_service):
        self.ai = ai_service
        self.gds = get_gds_service()

    async def analyze_intent(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Determine if the user's query requires structural graph analytics.
        Returns the algorithm name and parameters if needed.
        """
        prompt = f"""
        Analyze the following user query and determine if it requires a structural graph analysis algorithm.
        
        AVAILABLE ALGORITHMS:

        — CENTRALITY (Who/What is most important?) —
        - pagerank: Influence, importance, popularity, "best", "most central", ranking, "key players".
        - articlerank: Like PageRank but better for diverse graphs. "Most authoritative", uneven connectivity.
        - betweenness: Bridges, bottlenecks, nodes connecting different groups, flow control, "critical pathways".
        - closeness: Reachability, fast communication, "closest to all others", central access point.
        - degree: Most directly connected, "most connections", "most active", "most linked".
        - hits: Hubs vs authorities. "Which are hubs" (link to many) vs "which are authorities" (linked BY many).

        — COMMUNITY DETECTION (How is the data grouped?) —
        - louvain: Communities, clusters, groups, "who belongs together", thematic grouping.
        - leiden: Same as louvain but higher quality. "Better clustering".
        - wcc: Islands, disconnected parts, "isolated groups", "what's not connected".
        - kcore: Core structure, "tight-knit core", "most stable central group", inner circle.
        - triangle_count: Local density, "tight clusters", "which nodes form triangles".

        — SIMILARITY —
        - node_similarity: "What is similar to X", "which share neighbors", "find similar entities".

        — LINK PREDICTION (What connections are missing?) —
        - link_prediction_common: Predict missing links by shared neighbors. "What should be connected?".
        - link_prediction_adamic: Advanced prediction weighted by rare shared connections. "Non-obvious links".
        - link_prediction_resource: Flow-based prediction. "Hidden links", "high-probability missing links".

        — PATHFINDING & TRAVERSAL —
        - path: "How are X and Y related", "connection between X and Y", shortest path.
        - bfs: Explore layer by layer from a starting point. "What's nearby X", "neighbors of".
        - dfs: Explore deep paths from a starting point. "Deep hierarchy from X", "trace lineage".
        - random_walk: Simulate wandering through the graph. "Explore randomly from X".

        — TOPOLOGY —
        - topological_sort: Order nodes in logical sequence (DAGs). "Process flow", "timeline order".
        
        QUERY: "{user_input}"
        
        If an algorithm is needed, return a JSON object: 
        {{"algorithm": "<algorithm_name>", "reason": "...", "entities": ["Name1", "Name2"]}}
        
        If no structural graph algorithm is needed (e.g., simple symptom/fact lookup), return NULL.
        
        ONLY return JSON or NULL.
        """
        
        try:
            result = await self.ai.chat_json([{"role": "user", "content": prompt}])
            if result and "algorithm" in result:
                return result
            return None
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return None

    async def execute_algorithm(self, selection: Dict[str, Any], folder_id: str) -> Dict[str, Any]:
        """Execute the selected algorithm and return the results."""
        algo = selection["algorithm"]
        entities = selection.get("entities", [])
        logger.info(f"Executing dynamic graph algorithm: {algo}")
        
        try:
            # --- Centrality ---
            if algo == "pagerank":
                results = await self.gds.run_pagerank(folder_id=folder_id, top_k=5)
            elif algo == "articlerank":
                results = await self.gds.run_articlerank(folder_id=folder_id, top_k=5)
            elif algo == "betweenness":
                results = await self.gds.run_betweenness(folder_id=folder_id, top_k=5)
            elif algo == "closeness":
                results = await self.gds.run_closeness(folder_id=folder_id, top_k=5)
            elif algo == "degree":
                results = await self.gds.run_degree(folder_id=folder_id, top_k=5)
            elif algo == "hits":
                results = await self.gds.run_hits(folder_id=folder_id, top_k=5)

            # --- Community Detection ---
            elif algo == "louvain":
                raw_results = await self.gds.run_louvain(folder_id=folder_id)
                communities = {}
                for r in raw_results:
                    cid = r["community_id"]
                    communities[cid] = communities.get(cid, 0) + 1
                results = sorted([{"community_id": cid, "size": size} for cid, size in communities.items()], key=lambda x: -x["size"])[:10]
            elif algo == "leiden":
                raw_results = await self.gds.run_leiden(folder_id=folder_id)
                communities = {}
                for r in raw_results:
                    cid = r["community_id"]
                    communities[cid] = communities.get(cid, 0) + 1
                results = sorted([{"community_id": cid, "size": size} for cid, size in communities.items()], key=lambda x: -x["size"])[:10]
            elif algo == "wcc":
                raw_results = await self.gds.run_wcc(folder_id=folder_id)
                communities = {}
                for r in raw_results:
                    cid = r["community_id"]
                    communities[cid] = communities.get(cid, 0) + 1
                results = sorted([{"community_id": cid, "size": size} for cid, size in communities.items()], key=lambda x: -x["size"])[:10]
            elif algo == "kcore":
                results = await self.gds.run_kcore(folder_id=folder_id, top_k=10)
            elif algo == "triangle_count":
                results = await self.gds.run_triangle_count(folder_id=folder_id, top_k=10)

            # --- Similarity ---
            elif algo == "node_similarity":
                results = await self.gds.run_node_similarity(folder_id=folder_id, top_k=5)

            # --- Link Prediction ---
            elif algo.startswith("link_prediction"):
                method_map = {
                    "link_prediction_common": "common_neighbors",
                    "link_prediction_adamic": "adamic_adar",
                    "link_prediction_resource": "resource_allocation",
                }
                lp_method = method_map.get(algo, "common_neighbors")
                results = await self.gds.run_link_prediction(folder_id=folder_id, method=lp_method, top_k=10)

            # --- Pathfinding & Traversal ---
            elif algo == "path":
                # Requires 2 resolved entity IDs — skip if not available
                results = []
                logger.warning("[GraphAnalyticsAgent] 'path' requires resolved entity IDs — use AnalyticChatService for this")
            elif algo == "bfs":
                if entities:
                    results = await self.gds.run_bfs(source_id=entities[0], folder_id=folder_id)
                else:
                    results = []
                    logger.warning("[GraphAnalyticsAgent] BFS requires a source entity")
            elif algo == "dfs":
                if entities:
                    results = await self.gds.run_dfs(source_id=entities[0], folder_id=folder_id)
                else:
                    results = []
                    logger.warning("[GraphAnalyticsAgent] DFS requires a source entity")
            elif algo == "random_walk":
                if entities:
                    results = await self.gds.run_random_walk(source_id=entities[0], folder_id=folder_id)
                else:
                    results = []
                    logger.warning("[GraphAnalyticsAgent] Random Walk requires a source entity")

            # --- Topology ---
            elif algo == "topological_sort":
                results = await self.gds.run_topological_sort(folder_id=folder_id)

            else:
                return {"error": f"Unsupported algorithm: {algo}"}
            
            return {
                "algorithm": algo,
                "reason": selection.get("reason"),
                "results": results
            }
        except Exception as e:
            logger.error(f"Algorithm execution failed: {e}")
            return {"error": str(e)}
