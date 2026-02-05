"""
Link Prediction Algorithm

Predicts future relationships using Neo4j GDS link prediction functions.
Fully utilizes GDS for accurate similarity-based predictions.
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)


@dataclass
class PredictedLink:
    """A predicted relationship between two entities."""
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    predicted_type: str
    probability: float
    method: str


class LinkPrediction:
    """
    Predict missing links in the knowledge graph using Neo4j GDS.
    
    Available Methods (all use GDS):
    - common_neighbors: gds.linkPrediction.commonNeighbors
    - adamic_adar: gds.linkPrediction.adamicAdar (weighted by neighbor rarity)
    - resource_allocation: gds.linkPrediction.resourceAllocation
    - preferential_attachment: gds.linkPrediction.preferentialAttachment
    - total_neighbors: gds.linkPrediction.totalNeighbors
    """
    
    def __init__(self):
        self.driver = get_neo4j_driver()
    
    async def predict(
        self, 
        folder_id: Optional[str] = None,
        method: str = "adamic_adar",
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Predict missing links in the graph using GDS link prediction.
        
        Args:
            folder_id: Optional scope to specific folder
            method: Prediction method (common_neighbors, adamic_adar, 
                    resource_allocation, preferential_attachment)
            top_k: Number of predictions to return
            
        Returns:
            Dictionary with predicted links and scores
        """
        gds_functions = {
            "common_neighbors": "gds.linkPrediction.commonNeighbors",
            "adamic_adar": "gds.linkPrediction.adamicAdar",
            "resource_allocation": "gds.linkPrediction.resourceAllocation",
            "preferential_attachment": "gds.linkPrediction.preferentialAttachment",
            "total_neighbors": "gds.linkPrediction.totalNeighbors",
        }
        
        gds_func = gds_functions.get(method, "gds.linkPrediction.adamicAdar")
        
        async with self.driver.session() as session:
            try:
                folder_filter = "WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id" if folder_id else ""
                
                # Find unconnected node pairs and score them
                query = f"""
                    MATCH (a:Entity), (b:Entity)
                    {folder_filter}
                    WHERE a <> b 
                      AND id(a) < id(b)  // Avoid duplicates
                      AND NOT (a)--(b)   // Not already connected
                    WITH a, b, {gds_func}(a, b) AS score
                    WHERE score > 0
                    RETURN a.id AS source_id, 
                           a.name AS source_name, 
                           a.type AS source_type,
                           b.id AS target_id, 
                           b.name AS target_name,
                           b.type AS target_type,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
                
                params = {"top_k": top_k}
                if folder_id:
                    params["folder_id"] = folder_id
                
                result = await session.run(query, **params)
                records = await result.data()
                
                predictions = [
                    {
                        "source_id": r["source_id"],
                        "source_name": r["source_name"],
                        "source_type": r["source_type"],
                        "target_id": r["target_id"],
                        "target_name": r["target_name"],
                        "target_type": r["target_type"],
                        "score": r["score"],
                        "predicted_type": "RELATED_TO",
                    }
                    for r in records
                ]
                
                if predictions:
                    top = predictions[0]
                    insight = f"Strongest prediction: '{top['source_name']}' ↔ '{top['target_name']}' (score: {top['score']:.3f}). Found {len(predictions)} potential links."
                else:
                    insight = "No link predictions found. The graph may be fully connected or too sparse."
                
                return {
                    "algorithm": "link_prediction",
                    "engine": gds_func,
                    "method": method,
                    "folder_id": folder_id,
                    "predictions": predictions,
                    "count": len(predictions),
                    "insight": insight,
                }
                
            except Exception as e:
                logger.error(f"Link prediction failed: {e}")
                return {
                    "algorithm": "link_prediction",
                    "method": method,
                    "predictions": [],
                    "error": str(e),
                    "insight": f"Link prediction failed: {str(e)}",
                }
    
    async def score_pair(
        self, 
        source_id: str, 
        target_id: str,
        method: str = "adamic_adar"
    ) -> Dict[str, Any]:
        """
        Score a specific node pair using multiple GDS link prediction methods.
        """
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
                RETURN 
                    gds.linkPrediction.commonNeighbors(a, b) AS common_neighbors,
                    gds.linkPrediction.adamicAdar(a, b) AS adamic_adar,
                    gds.linkPrediction.resourceAllocation(a, b) AS resource_allocation,
                    gds.linkPrediction.preferentialAttachment(a, b) AS preferential_attachment,
                    gds.linkPrediction.totalNeighbors(a, b) AS total_neighbors,
                    EXISTS((a)--(b)) AS already_connected
            """, source_id=source_id, target_id=target_id)
            
            record = await result.single()
            
            if not record:
                return {"error": "One or both nodes not found"}
            
            return {
                "source_id": source_id,
                "target_id": target_id,
                "already_connected": record["already_connected"],
                "scores": {
                    "common_neighbors": record["common_neighbors"],
                    "adamic_adar": record["adamic_adar"],
                    "resource_allocation": record["resource_allocation"],
                    "preferential_attachment": record["preferential_attachment"],
                    "total_neighbors": record["total_neighbors"],
                },
            }
    
    async def get_ghost_lines(
        self, 
        folder_id: Optional[str] = None,
        threshold: float = 0.5,
        top_k: int = 50
    ) -> List[PredictedLink]:
        """
        Get predicted links above threshold for 'ghost line' visualization.
        These are dashed lines shown in the UI for likely connections.
        """
        result = await self.predict(folder_id, method="adamic_adar", top_k=top_k)
        
        predictions = result.get("predictions", [])
        
        # Filter by threshold and convert to PredictedLink objects
        ghost_lines = []
        for p in predictions:
            if p["score"] >= threshold:
                ghost_lines.append(PredictedLink(
                    source_id=p["source_id"],
                    source_name=p["source_name"],
                    target_id=p["target_id"],
                    target_name=p["target_name"],
                    predicted_type=p["predicted_type"],
                    probability=min(p["score"], 1.0),  # Normalize to 0-1
                    method="adamic_adar",
                ))
        
        return ghost_lines

    async def find_missing_between_types(
        self,
        source_type: str,
        target_type: str,
        folder_id: Optional[str] = None,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Find missing links specifically between two entity types.
        Useful for targeted analysis like "People ↔ Organizations".
        """
        async with self.driver.session() as session:
            folder_filter = "AND a.folder_id = $folder_id AND b.folder_id = $folder_id" if folder_id else ""
            
            query = f"""
                MATCH (a:Entity), (b:Entity)
                WHERE a.type = $source_type AND b.type = $target_type
                  AND a <> b AND NOT (a)--(b)
                  {folder_filter}
                WITH a, b, gds.linkPrediction.adamicAdar(a, b) AS score
                WHERE score > 0
                RETURN a.id AS source_id, a.name AS source_name,
                       b.id AS target_id, b.name AS target_name,
                       score
                ORDER BY score DESC
                LIMIT $top_k
            """
            
            params = {"source_type": source_type, "target_type": target_type, "top_k": top_k}
            if folder_id:
                params["folder_id"] = folder_id
            
            result = await session.run(query, **params)
            records = await result.data()
            
            return {
                "source_type": source_type,
                "target_type": target_type,
                "predictions": records,
                "count": len(records),
            }
