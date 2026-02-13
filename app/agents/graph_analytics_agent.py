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
        - pagerank: Use for influence, importance, "who are the key players", "most important entities".
        - betweenness: Use for "bridges", "bottlenecks", "critical pathways", "connectivity".
        - closeness: Use for "centrality", "distance", "fastest communication".
        - louvain: Use for "communities", "groups", "clusters", "thematic segments".
        
        QUERY: "{user_input}"
        
        If an algorithm is needed, return a JSON object: {{"algorithm": "pagerank|betweenness|closeness|louvain", "reason": "..."}}
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
        logger.info(f"Executing dynamic graph algorithm: {algo}")
        
        try:
            if algo == "pagerank":
                results = await self.gds.run_pagerank(folder_id=folder_id, top_k=5)
            elif algo == "betweenness":
                results = await self.gds.run_betweenness(folder_id=folder_id, top_k=5)
            elif algo == "closeness":
                results = await self.gds.run_closeness(folder_id=folder_id, top_k=5)
            elif algo == "louvain":
                # For Louvain, we'll return the community distribution
                raw_results = await self.gds.run_louvain(folder_id=folder_id)
                # Group and count for summary
                communities = {}
                for r in raw_results:
                    cid = r["community_id"]
                    communities[cid] = communities.get(cid, 0) + 1
                results = sorted([{"community_id": cid, "size": size} for cid, size in communities.items()], key=lambda x: -x["size"])[:5]
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
