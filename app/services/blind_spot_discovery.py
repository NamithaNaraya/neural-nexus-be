"""
Blind-Spot Discovery Service

Detects missing relationships that should exist but aren't documented.
Generates "Ghost Lines" for visualization.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class PredictionMethod(str, Enum):
    """Methods for predicting missing links."""
    COMMON_NEIGHBORS = "common_neighbors"
    JACCARD = "jaccard"
    ADAMIC_ADAR = "adamic_adar"
    PREFERENTIAL_ATTACHMENT = "preferential_attachment"
    RESOURCE_ALLOCATION = "resource_allocation"
    STRUCTURAL = "structural"


@dataclass
class GhostLine:
    """A predicted missing relationship."""
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    confidence: float
    predicted_type: str
    reason: str
    method: str


class BlindSpotDiscovery:
    """
    Service for discovering missing relationships (blind spots).
    
    Uses multiple link prediction algorithms to suggest
    relationships that should exist based on graph structure.
    """
    
    def __init__(self, neo4j_driver):
        self.neo4j = neo4j_driver
    
    async def discover(
        self,
        folder_id: Optional[str] = None,
        file_id: Optional[str] = None,
        min_confidence: float = 0.5,
        method: PredictionMethod = PredictionMethod.STRUCTURAL,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Discover blind spots (missing relationships).
        
        Returns ghost lines for visualization.
        """
        try:
            ghost_lines = []
            
            if method == PredictionMethod.STRUCTURAL:
                # Use multiple methods and combine
                cn_results = await self._common_neighbors(folder_id, file_id, limit)
                aa_results = await self._adamic_adar(folder_id, file_id, limit)
                pa_results = await self._preferential_attachment(folder_id, file_id, limit)
                
                # Combine and deduplicate
                seen = set()
                all_results = cn_results + aa_results + pa_results
                
                for result in all_results:
                    key = f"{result['source_id']}-{result['target_id']}"
                    reverse_key = f"{result['target_id']}-{result['source_id']}"
                    
                    if key not in seen and reverse_key not in seen:
                        if result["confidence"] >= min_confidence:
                            seen.add(key)
                            ghost_lines.append(result)
                
                # Sort by confidence
                ghost_lines.sort(key=lambda x: x["confidence"], reverse=True)
                ghost_lines = ghost_lines[:limit]
                
            else:
                # Single method
                method_map = {
                    PredictionMethod.COMMON_NEIGHBORS: self._common_neighbors,
                    PredictionMethod.ADAMIC_ADAR: self._adamic_adar,
                    PredictionMethod.PREFERENTIAL_ATTACHMENT: self._preferential_attachment,
                    PredictionMethod.JACCARD: self._jaccard_coefficient,
                    PredictionMethod.RESOURCE_ALLOCATION: self._resource_allocation,
                }
                
                method_func = method_map.get(method, self._common_neighbors)
                ghost_lines = await method_func(folder_id, file_id, limit)
                ghost_lines = [g for g in ghost_lines if g["confidence"] >= min_confidence]
            
            # Categorize by predicted relationship type
            categorized = self._categorize_predictions(ghost_lines)
            
            return {
                "ghost_lines": ghost_lines,
                "total_count": len(ghost_lines),
                "by_category": categorized,
                "method": method.value if isinstance(method, PredictionMethod) else method,
                "folder_id": folder_id,
                "file_id": file_id,
            }
            
        except Exception as e:
            logger.error(f"Blind spot discovery failed: {e}")
            return {
                "ghost_lines": [],
                "total_count": 0,
                "error": str(e),
            }
    
    async def _common_neighbors(
        self,
        folder_id: Optional[str],
        file_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Common Neighbors algorithm.
        Nodes with many shared neighbors are likely connected.
        """
        scope_filter = self._build_scope_filter(folder_id, file_id)
        
        query = f"""
        MATCH (a){scope_filter}
        MATCH (b){scope_filter}
        WHERE a <> b 
          AND NOT (a)--(b)
          AND id(a) < id(b)
        WITH a, b
        MATCH (a)--(common)--(b)
        WITH a, b, count(DISTINCT common) as commonNeighbors
        WHERE commonNeighbors > 1
        RETURN 
            COALESCE(a.id, elementId(a)) as source_id,
            a.name as source_name,
            a.type as source_type,
            COALESCE(b.id, elementId(b)) as target_id,
            b.name as target_name,
            b.type as target_type,
            commonNeighbors,
            toFloat(commonNeighbors) / 10.0 as confidence
        ORDER BY commonNeighbors DESC
        LIMIT $limit
        """
        
        try:
            result = self.neo4j.execute_query(query, {"limit": limit})
            
            return [
                {
                    "source_id": record["source_id"],
                    "source_name": record["source_name"] or record["source_id"],
                    "target_id": record["target_id"],
                    "target_name": record["target_name"] or record["target_id"],
                    "confidence": min(record["confidence"], 1.0),
                    "predicted_type": self._predict_relationship_type(
                        record["source_type"], record["target_type"]
                    ),
                    "reason": f"{record['commonNeighbors']} common neighbors",
                    "method": "common_neighbors",
                }
                for record in result.records
            ]
        except Exception as e:
            logger.error(f"Common neighbors query failed: {e}")
            return []
    
    async def _adamic_adar(
        self,
        folder_id: Optional[str],
        file_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Adamic-Adar algorithm.
        Weights common neighbors by their degree (rarer neighbors = stronger signal).
        """
        scope_filter = self._build_scope_filter(folder_id, file_id)
        
        query = f"""
        MATCH (a){scope_filter}
        MATCH (b){scope_filter}
        WHERE a <> b 
          AND NOT (a)--(b)
          AND id(a) < id(b)
        WITH a, b
        MATCH (a)--(common)--(b)
        WITH a, b, common
        MATCH (common)--()
        WITH a, b, common, count(*) as degree
        WHERE degree > 1
        WITH a, b, sum(1.0 / log(toFloat(degree))) as adamicAdar
        WHERE adamicAdar > 0.5
        RETURN 
            COALESCE(a.id, elementId(a)) as source_id,
            a.name as source_name,
            a.type as source_type,
            COALESCE(b.id, elementId(b)) as target_id,
            b.name as target_name,
            b.type as target_type,
            adamicAdar,
            CASE WHEN adamicAdar > 5 THEN 1.0 ELSE adamicAdar / 5.0 END as confidence
        ORDER BY adamicAdar DESC
        LIMIT $limit
        """
        
        try:
            result = self.neo4j.execute_query(query, {"limit": limit})
            
            return [
                {
                    "source_id": record["source_id"],
                    "source_name": record["source_name"] or record["source_id"],
                    "target_id": record["target_id"],
                    "target_name": record["target_name"] or record["target_id"],
                    "confidence": min(record["confidence"], 1.0),
                    "predicted_type": self._predict_relationship_type(
                        record["source_type"], record["target_type"]
                    ),
                    "reason": f"Adamic-Adar score: {record['adamicAdar']:.2f}",
                    "method": "adamic_adar",
                }
                for record in result.records
            ]
        except Exception as e:
            logger.error(f"Adamic-Adar query failed: {e}")
            return []
    
    async def _preferential_attachment(
        self,
        folder_id: Optional[str],
        file_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Preferential Attachment algorithm.
        High-degree nodes are more likely to form new connections.
        """
        scope_filter = self._build_scope_filter(folder_id, file_id)
        
        query = f"""
        MATCH (a){scope_filter}
        MATCH (b){scope_filter}
        WHERE a <> b 
          AND NOT (a)--(b)
          AND id(a) < id(b)
        WITH a, b
        OPTIONAL MATCH (a)--()
        WITH a, b, count(*) as degreeA
        OPTIONAL MATCH (b)--()
        WITH a, b, degreeA, count(*) as degreeB
        WHERE degreeA > 2 AND degreeB > 2
        WITH a, b, degreeA * degreeB as paScore
        WHERE paScore > 10
        RETURN 
            COALESCE(a.id, elementId(a)) as source_id,
            a.name as source_name,
            a.type as source_type,
            COALESCE(b.id, elementId(b)) as target_id,
            b.name as target_name,
            b.type as target_type,
            paScore,
            CASE WHEN paScore > 100 THEN 1.0 ELSE paScore / 100.0 END as confidence
        ORDER BY paScore DESC
        LIMIT $limit
        """
        
        try:
            result = self.neo4j.execute_query(query, {"limit": limit})
            
            return [
                {
                    "source_id": record["source_id"],
                    "source_name": record["source_name"] or record["source_id"],
                    "target_id": record["target_id"],
                    "target_name": record["target_name"] or record["target_id"],
                    "confidence": min(record["confidence"], 1.0),
                    "predicted_type": self._predict_relationship_type(
                        record["source_type"], record["target_type"]
                    ),
                    "reason": f"Preferential attachment: {record['paScore']}",
                    "method": "preferential_attachment",
                }
                for record in result.records
            ]
        except Exception as e:
            logger.error(f"Preferential attachment query failed: {e}")
            return []
    
    async def _jaccard_coefficient(
        self,
        folder_id: Optional[str],
        file_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Jaccard Coefficient algorithm.
        Ratio of common neighbors to total neighbors.
        """
        scope_filter = self._build_scope_filter(folder_id, file_id)
        
        query = f"""
        MATCH (a){scope_filter}
        MATCH (b){scope_filter}
        WHERE a <> b 
          AND NOT (a)--(b)
          AND id(a) < id(b)
        WITH a, b
        MATCH (a)--(neighborA)
        WITH a, b, collect(neighborA) as neighborsA
        MATCH (b)--(neighborB)
        WITH a, b, neighborsA, collect(neighborB) as neighborsB
        WITH a, b, 
             [n IN neighborsA WHERE n IN neighborsB] as intersection,
             neighborsA + [n IN neighborsB WHERE NOT n IN neighborsA] as union
        WITH a, b, 
             size(intersection) as common,
             size(union) as total
        WHERE total > 0 AND common > 1
        WITH a, b, common, total, toFloat(common) / total as jaccard
        WHERE jaccard > 0.1
        RETURN 
            COALESCE(a.id, elementId(a)) as source_id,
            a.name as source_name,
            a.type as source_type,
            COALESCE(b.id, elementId(b)) as target_id,
            b.name as target_name,
            b.type as target_type,
            jaccard as confidence,
            common
        ORDER BY jaccard DESC
        LIMIT $limit
        """
        
        try:
            result = self.neo4j.execute_query(query, {"limit": limit})
            
            return [
                {
                    "source_id": record["source_id"],
                    "source_name": record["source_name"] or record["source_id"],
                    "target_id": record["target_id"],
                    "target_name": record["target_name"] or record["target_id"],
                    "confidence": record["confidence"],
                    "predicted_type": self._predict_relationship_type(
                        record["source_type"], record["target_type"]
                    ),
                    "reason": f"Jaccard: {record['confidence']:.2f} ({record['common']} common)",
                    "method": "jaccard",
                }
                for record in result.records
            ]
        except Exception as e:
            logger.error(f"Jaccard query failed: {e}")
            return []
    
    async def _resource_allocation(
        self,
        folder_id: Optional[str],
        file_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Resource Allocation Index.
        Similar to Adamic-Adar but uses inverse degree directly.
        """
        # Simplified version using common neighbors with degree weighting
        return await self._adamic_adar(folder_id, file_id, limit)
    
    def _build_scope_filter(
        self,
        folder_id: Optional[str],
        file_id: Optional[str],
    ) -> str:
        """Build the WHERE clause for scope filtering."""
        # Note: This returns an empty string for now
        # The actual filtering would need parameters which complicates the query
        # For now, we'll filter post-query or rely on projected graphs
        return ""
    
    def _predict_relationship_type(
        self,
        source_type: Optional[str],
        target_type: Optional[str],
    ) -> str:
        """
        Predict the most likely relationship type based on node types.
        """
        type_map = {
            ("Person", "Person"): "KNOWS",
            ("Person", "Organization"): "WORKS_FOR",
            ("Person", "Place"): "LIVES_IN",
            ("Person", "Event"): "PARTICIPATED_IN",
            ("Organization", "Place"): "LOCATED_IN",
            ("Event", "Place"): "OCCURRED_AT",
            ("Concept", "Concept"): "RELATED_TO",
        }
        
        key = (source_type, target_type)
        reverse_key = (target_type, source_type)
        
        return type_map.get(key) or type_map.get(reverse_key) or "RELATED_TO"
    
    def _categorize_predictions(
        self,
        ghost_lines: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize predictions by relationship type."""
        categories = {}
        
        for line in ghost_lines:
            pred_type = line.get("predicted_type", "UNKNOWN")
            if pred_type not in categories:
                categories[pred_type] = []
            categories[pred_type].append(line)
        
        return categories
    
    async def get_cross_topic_bridges(
        self,
        folder_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Find potential cross-topic bridging opportunities.
        Entities that could connect different folders.
        """
        if len(folder_ids) < 2:
            return []
        
        # Find entities with similar names across folders
        query = """
        MATCH (a)
        WHERE a.folder_id = $folder1 OR a.folderId = $folder1
        WITH a
        MATCH (b)
        WHERE (b.folder_id = $folder2 OR b.folderId = $folder2)
          AND a.name = b.name
          AND a <> b
        RETURN 
            a.name as entity_name,
            COALESCE(a.id, elementId(a)) as id_in_folder1,
            COALESCE(b.id, elementId(b)) as id_in_folder2,
            a.type as type
        LIMIT 50
        """
        
        bridges = []
        
        # Check pairs of folders
        for i in range(len(folder_ids)):
            for j in range(i + 1, len(folder_ids)):
                try:
                    result = self.neo4j.execute_query(query, {
                        "folder1": folder_ids[i],
                        "folder2": folder_ids[j],
                    })
                    
                    for record in result.records:
                        bridges.append({
                            "entity_name": record["entity_name"],
                            "folder_1": folder_ids[i],
                            "id_in_folder_1": record["id_in_folder1"],
                            "folder_2": folder_ids[j],
                            "id_in_folder_2": record["id_in_folder2"],
                            "type": record["type"],
                        })
                except Exception as e:
                    logger.error(f"Cross-topic bridge query failed: {e}")
        
        return bridges


# Singleton
_discovery_service: Optional[BlindSpotDiscovery] = None


def get_discovery_service(neo4j) -> BlindSpotDiscovery:
    """Get or create the discovery service singleton."""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = BlindSpotDiscovery(neo4j)
    return _discovery_service
