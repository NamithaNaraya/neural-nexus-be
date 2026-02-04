"""
Cluster Comparison Service

Side-by-side analysis of node clusters.
Finds common entities, unique nodes, and bridge connections.
"""
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComparisonType(str, Enum):
    """Types of cluster comparisons."""
    FILE_VS_FILE = "file"
    FOLDER_VS_FOLDER = "folder"
    CLUSTER_VS_CLUSTER = "cluster"
    SELECTION_VS_SELECTION = "selection"


@dataclass
class ComparisonResult:
    """Result of cluster comparison."""
    left_nodes: List[Dict[str, Any]]
    right_nodes: List[Dict[str, Any]]
    common_entities: List[str]
    unique_left: List[str]
    unique_right: List[str]
    bridges: List[Dict[str, Any]]
    semantic_similarity: float
    structural_similarity: float


class ClusterComparisonService:
    """
    Service for comparing two clusters of nodes.
    
    Features:
    - Find common entities
    - Find unique entities per cluster
    - Identify bridge nodes connecting clusters
    - Calculate similarity metrics
    """
    
    def __init__(self, neo4j_driver):
        self.neo4j = neo4j_driver
    
    async def compare(
        self,
        left: Dict[str, Any],
        right: Dict[str, Any],
        include_bridges: bool = True,
        include_similarity: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare two clusters and return comprehensive analysis.
        
        Args:
            left: {"type": "file|folder|cluster|selection", "id": str, "node_ids": List[str]}
            right: Same structure as left
            include_bridges: Whether to find bridge nodes
            include_similarity: Whether to calculate similarity scores
        """
        try:
            # Get nodes for each cluster
            left_nodes = await self._get_cluster_nodes(left)
            right_nodes = await self._get_cluster_nodes(right)
            
            left_ids = {n["id"] for n in left_nodes}
            right_ids = {n["id"] for n in right_nodes}
            
            # Find common and unique
            common_ids = left_ids & right_ids
            unique_left_ids = left_ids - right_ids
            unique_right_ids = right_ids - left_ids
            
            # Map IDs to names for readability
            id_to_name = {}
            for n in left_nodes + right_nodes:
                id_to_name[n["id"]] = n.get("name", n["id"])
            
            common_entities = [id_to_name.get(id, id) for id in common_ids]
            unique_left = [id_to_name.get(id, id) for id in unique_left_ids]
            unique_right = [id_to_name.get(id, id) for id in unique_right_ids]
            
            # Find bridges if requested
            bridges = []
            if include_bridges:
                bridges = await self._find_bridges(left_ids, right_ids)
            
            # Calculate similarity if requested
            semantic_sim = 0.0
            structural_sim = 0.0
            if include_similarity:
                semantic_sim = await self._calculate_semantic_similarity(left_nodes, right_nodes)
                structural_sim = self._calculate_structural_similarity(
                    left_nodes, right_nodes, len(common_ids)
                )
            
            return {
                "left_nodes": left_nodes,
                "right_nodes": right_nodes,
                "common_entities": common_entities,
                "common_count": len(common_entities),
                "unique_left": unique_left,
                "unique_left_count": len(unique_left),
                "unique_right": unique_right,
                "unique_right_count": len(unique_right),
                "bridges": bridges,
                "bridge_count": len(bridges),
                "semantic_similarity": round(semantic_sim, 3),
                "structural_similarity": round(structural_sim, 3),
                "jaccard_index": self._calculate_jaccard(left_ids, right_ids),
            }
            
        except Exception as e:
            logger.error(f"Cluster comparison failed: {e}")
            return {
                "error": str(e),
                "left_nodes": [],
                "right_nodes": [],
                "common_entities": [],
                "unique_left": [],
                "unique_right": [],
                "bridges": [],
            }
    
    async def _get_cluster_nodes(self, cluster: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all nodes belonging to a cluster."""
        cluster_type = cluster.get("type", "selection")
        cluster_id = cluster.get("id")
        node_ids = cluster.get("node_ids", [])
        
        if cluster_type == "selection" and node_ids:
            # Direct node IDs
            query = """
            UNWIND $node_ids AS nodeId
            MATCH (n)
            WHERE n.id = nodeId OR elementId(n) = nodeId
            RETURN 
                COALESCE(n.id, n.entity_id, elementId(n)) as id,
                n.name as name,
                n.type as type,
                n.description as description
            """
            result = await self.neo4j.execute_query(query, {"node_ids": node_ids})
            
        elif cluster_type == "file" and cluster_id:
            query = """
            MATCH (n)
            WHERE n.file_id = $cluster_id OR n.fileId = $cluster_id
            RETURN 
                COALESCE(n.id, n.entity_id, elementId(n)) as id,
                n.name as name,
                n.type as type,
                n.description as description
            """
            result = await self.neo4j.execute_query(query, {"cluster_id": cluster_id})
            
        elif cluster_type == "folder" and cluster_id:
            query = """
            MATCH (n)
            WHERE n.folder_id = $cluster_id OR n.folderId = $cluster_id
            RETURN 
                COALESCE(n.id, n.entity_id, elementId(n)) as id,
                n.name as name,
                n.type as type,
                n.description as description
            """
            result = await self.neo4j.execute_query(query, {"cluster_id": cluster_id})
            
        elif cluster_type == "cluster" and cluster_id:
            # Community/cluster label
            query = """
            MATCH (n)
            WHERE n.community = $cluster_id OR n.cluster_id = $cluster_id
            RETURN 
                COALESCE(n.id, n.entity_id, elementId(n)) as id,
                n.name as name,
                n.type as type,
                n.description as description
            """
            result = await self.neo4j.execute_query(query, {"cluster_id": cluster_id})
        else:
            return []
        
        return [
            {
                "id": record["id"],
                "name": record["name"] or record["id"],
                "type": record["type"] or "Unknown",
                "description": record["description"] or "",
            }
            for record in result.records
        ]
    
    async def _find_bridges(
        self,
        left_ids: Set[str],
        right_ids: Set[str],
    ) -> List[Dict[str, Any]]:
        """
        Find bridge nodes that connect the two clusters.
        A bridge is a node connected to both clusters but not in either.
        """
        if not left_ids or not right_ids:
            return []
        
        # Convert to lists for Cypher
        left_list = list(left_ids)
        right_list = list(right_ids)
        
        # Find nodes connected to both clusters
        query = """
        UNWIND $left_ids AS leftId
        MATCH (left)-[r1]-(bridge)-[r2]-(right)
        WHERE (left.id = leftId OR elementId(left) = leftId)
          AND (right.id IN $right_ids OR elementId(right) IN $right_ids)
          AND NOT (bridge.id IN $left_ids OR elementId(bridge) IN $left_ids)
          AND NOT (bridge.id IN $right_ids OR elementId(bridge) IN $right_ids)
        WITH bridge, 
             count(DISTINCT left) as left_connections,
             count(DISTINCT right) as right_connections
        WHERE left_connections > 0 AND right_connections > 0
        RETURN 
            COALESCE(bridge.id, elementId(bridge)) as id,
            bridge.name as name,
            bridge.type as type,
            left_connections,
            right_connections,
            left_connections + right_connections as total_connections
        ORDER BY total_connections DESC
        LIMIT 20
        """
        
        try:
            result = await self.neo4j.execute_query(query, {
                "left_ids": left_list[:100],  # Limit for performance
                "right_ids": right_list[:100],
            })
            
            return [
                {
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                    "type": record["type"] or "Unknown",
                    "left_connections": record["left_connections"],
                    "right_connections": record["right_connections"],
                    "bridge_strength": record["total_connections"],
                }
                for record in result.records
            ]
        except Exception as e:
            logger.error(f"Bridge finding failed: {e}")
            return []
    
    async def _calculate_semantic_similarity(
        self,
        left_nodes: List[Dict[str, Any]],
        right_nodes: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate semantic similarity between clusters.
        Uses type distribution and description overlap.
        """
        if not left_nodes or not right_nodes:
            return 0.0
        
        # Type distribution similarity
        left_types = {}
        for n in left_nodes:
            t = n.get("type", "Unknown")
            left_types[t] = left_types.get(t, 0) + 1
        
        right_types = {}
        for n in right_nodes:
            t = n.get("type", "Unknown")
            right_types[t] = right_types.get(t, 0) + 1
        
        # Normalize
        left_total = sum(left_types.values())
        right_total = sum(right_types.values())
        
        left_dist = {k: v / left_total for k, v in left_types.items()}
        right_dist = {k: v / right_total for k, v in right_types.items()}
        
        # Cosine similarity of type distributions
        all_types = set(left_dist.keys()) | set(right_dist.keys())
        dot_product = sum(
            left_dist.get(t, 0) * right_dist.get(t, 0)
            for t in all_types
        )
        
        left_norm = sum(v ** 2 for v in left_dist.values()) ** 0.5
        right_norm = sum(v ** 2 for v in right_dist.values()) ** 0.5
        
        if left_norm == 0 or right_norm == 0:
            return 0.0
        
        return dot_product / (left_norm * right_norm)
    
    def _calculate_structural_similarity(
        self,
        left_nodes: List[Dict[str, Any]],
        right_nodes: List[Dict[str, Any]],
        common_count: int,
    ) -> float:
        """
        Calculate structural similarity based on size and overlap.
        """
        left_count = len(left_nodes)
        right_count = len(right_nodes)
        
        if left_count == 0 or right_count == 0:
            return 0.0
        
        # Size similarity (1 = same size, 0 = very different)
        size_ratio = min(left_count, right_count) / max(left_count, right_count)
        
        # Overlap ratio
        total = left_count + right_count - common_count
        overlap_ratio = common_count / total if total > 0 else 0.0
        
        # Combined score
        return (size_ratio + overlap_ratio) / 2
    
    def _calculate_jaccard(self, left_ids: Set[str], right_ids: Set[str]) -> float:
        """Calculate Jaccard similarity index."""
        if not left_ids and not right_ids:
            return 0.0
        
        intersection = len(left_ids & right_ids)
        union = len(left_ids | right_ids)
        
        return round(intersection / union, 3) if union > 0 else 0.0
    
    async def find_missing_in_b(
        self,
        cluster_a_nodes: List[Dict[str, Any]],
        cluster_b_nodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Find entities that exist in A and should likely exist in B.
        Uses type matching and relationship patterns.
        """
        a_types = {n.get("type") for n in cluster_a_nodes}
        b_ids = {n["id"] for n in cluster_b_nodes}
        b_types = {n.get("type") for n in cluster_b_nodes}
        
        # Find entities in A that match B's type distribution but aren't in B
        suggestions = []
        for node in cluster_a_nodes:
            if node["id"] not in b_ids and node.get("type") in b_types:
                suggestions.append({
                    "id": node["id"],
                    "name": node["name"],
                    "type": node["type"],
                    "reason": f"Entity of type '{node['type']}' exists in cluster A but not B",
                })
        
        return suggestions[:20]  # Limit suggestions


# Singleton instance
_comparison_service: Optional[ClusterComparisonService] = None


def get_comparison_service(neo4j) -> ClusterComparisonService:
    """Get or create the comparison service singleton."""
    global _comparison_service
    if _comparison_service is None:
        _comparison_service = ClusterComparisonService(neo4j)
    return _comparison_service
