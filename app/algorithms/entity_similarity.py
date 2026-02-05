"""
Entity Similarity Algorithm

Deep comparison of nodes using Neo4j GDS similarity algorithms.
Finds entities that are similar based on their connections and properties.
"""
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)


class EntitySimilarity:
    """
    Calculate similarity between entities using Neo4j GDS.
    
    GDS Methods:
    - gds.nodeSimilarity: Jaccard/Overlap based on shared neighbors
    - gds.knn: K-Nearest Neighbors using node properties/embeddings
    - Alpha functions for pairwise similarity
    """
    
    def __init__(self):
        self.driver = get_neo4j_driver()
    
    async def calculate_similarity(
        self, 
        entity_a_id: str, 
        entity_b_id: str
    ) -> Dict[str, float]:
        """
        Calculate multiple similarity metrics between two entities using GDS.
        
        Returns:
            Dictionary with similarity scores for each metric
        """
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (a:Entity {id: $id_a}), (b:Entity {id: $id_b})
                
                // Get neighbors
                OPTIONAL MATCH (a)--(na)
                WITH a, b, collect(DISTINCT id(na)) AS neighbors_a
                OPTIONAL MATCH (b)--(nb)
                WITH a, b, neighbors_a, collect(DISTINCT id(nb)) AS neighbors_b
                
                // Calculate Jaccard (shared / union)
                WITH a, b, neighbors_a, neighbors_b,
                     gds.similarity.jaccard(neighbors_a, neighbors_b) AS jaccard,
                     gds.similarity.overlap(neighbors_a, neighbors_b) AS overlap
                
                // Cosine on embeddings if available
                WITH a, b, jaccard, overlap,
                     CASE WHEN a.embedding IS NOT NULL AND b.embedding IS NOT NULL
                          THEN gds.similarity.cosine(a.embedding, b.embedding)
                          ELSE null
                     END AS cosine
                
                RETURN jaccard, overlap, cosine
            """, id_a=entity_a_id, id_b=entity_b_id)
            
            record = await result.single()
            
            if not record:
                return {"jaccard": 0.0, "overlap": 0.0, "cosine": None, "combined": 0.0}
            
            jaccard = record["jaccard"] or 0.0
            overlap = record["overlap"] or 0.0
            cosine = record["cosine"]
            
            # Combined score (weighted average)
            if cosine is not None:
                combined = (jaccard * 0.3 + overlap * 0.3 + cosine * 0.4)
            else:
                combined = (jaccard + overlap) / 2
            
            return {
                "jaccard": jaccard,
                "overlap": overlap,
                "cosine": cosine,
                "combined": combined,
            }
    
    async def find_similar(
        self, 
        entity_id: str, 
        top_k: int = 10,
        folder_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find top K similar entities using GDS node similarity.
        Uses Jaccard similarity based on shared neighbors.
        """
        projection_name = f"similarity_{entity_id}"
        
        async with self.driver.session() as session:
            try:
                # Create projection
                folder_filter = f"WHERE n.folder_id = '{folder_id}'" if folder_id else ""
                
                await session.run(f"""
                    CALL gds.graph.project.cypher(
                        $name,
                        'MATCH (n:Entity) {folder_filter} RETURN id(n) AS id',
                        'MATCH (a:Entity)-[r]->(b:Entity) RETURN id(a) AS source, id(b) AS target'
                    )
                """, name=projection_name)
                
                # Run node similarity
                result = await session.run("""
                    MATCH (source:Entity {id: $entity_id})
                    CALL gds.nodeSimilarity.stream($graph, {
                        topK: $top_k
                    })
                    YIELD node1, node2, similarity
                    WHERE node1 = id(source) OR node2 = id(source)
                    WITH CASE WHEN node1 = id(source) 
                              THEN gds.util.asNode(node2) 
                              ELSE gds.util.asNode(node1) 
                         END AS similar_node, 
                         similarity
                    RETURN similar_node.id AS id, 
                           similar_node.name AS name, 
                           similar_node.type AS type,
                           similarity AS score
                    ORDER BY score DESC
                    LIMIT $top_k
                """, graph=projection_name, entity_id=entity_id, top_k=top_k)
                
                records = await result.data()
                return records
                
            except Exception as e:
                logger.error(f"Node similarity failed: {e}")
                # Fallback to simpler neighbor-based similarity
                return await self._fallback_similarity(entity_id, top_k, session)
            finally:
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except:
                    pass
    
    async def _fallback_similarity(
        self, 
        entity_id: str, 
        top_k: int,
        session
    ) -> List[Dict[str, Any]]:
        """Fallback to Cypher-based similarity if GDS fails."""
        result = await session.run("""
            MATCH (source:Entity {id: $entity_id})--(neighbor)--(similar:Entity)
            WHERE similar <> source
            WITH similar, count(DISTINCT neighbor) AS shared
            MATCH (source:Entity {id: $entity_id})--(n1)
            WITH similar, shared, count(DISTINCT n1) AS source_neighbors
            MATCH (similar)--(n2)
            WITH similar, shared, source_neighbors, count(DISTINCT n2) AS similar_neighbors
            WITH similar, 
                 toFloat(shared) / (source_neighbors + similar_neighbors - shared) AS jaccard
            RETURN similar.id AS id, similar.name AS name, similar.type AS type, jaccard AS score
            ORDER BY score DESC
            LIMIT $top_k
        """, entity_id=entity_id, top_k=top_k)
        
        return await result.data()
    
    async def find_all_similar_pairs(
        self,
        folder_id: Optional[str] = None,
        similarity_cutoff: float = 0.5,
        top_k: int = 100
    ) -> Dict[str, Any]:
        """
        Find all similar entity pairs in the graph using GDS.
        """
        projection_name = f"all_pairs_{folder_id or 'all'}"
        
        async with self.driver.session() as session:
            try:
                # Create projection
                if folder_id:
                    await session.run("""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n:Entity) WHERE n.folder_id = $folder_id RETURN id(n) AS id',
                            'MATCH (a:Entity)-[r]->(b:Entity) 
                             WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id
                             RETURN id(a) AS source, id(b) AS target',
                            {parameters: {folder_id: $folder_id}}
                        )
                    """, name=projection_name, folder_id=folder_id)
                else:
                    await session.run("""
                        CALL gds.graph.project($name, 'Entity', '*')
                    """, name=projection_name)
                
                # Run node similarity for all pairs
                result = await session.run("""
                    CALL gds.nodeSimilarity.stream($graph, {
                        similarityCutoff: $cutoff,
                        topK: 10
                    })
                    YIELD node1, node2, similarity
                    WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
                    RETURN n1.id AS source_id, n1.name AS source_name,
                           n2.id AS target_id, n2.name AS target_name,
                           similarity AS score
                    ORDER BY score DESC
                    LIMIT $top_k
                """, graph=projection_name, cutoff=similarity_cutoff, top_k=top_k)
                
                records = await result.data()
                
                return {
                    "algorithm": "node_similarity",
                    "engine": "gds.nodeSimilarity.stream",
                    "folder_id": folder_id,
                    "similarity_cutoff": similarity_cutoff,
                    "pairs": records,
                    "count": len(records),
                }
            finally:
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except:
                    pass
    
    async def knn_similarity(
        self,
        property_name: str = "embedding",
        folder_id: Optional[str] = None,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Find K-Nearest Neighbors using node embeddings with GDS KNN.
        """
        projection_name = f"knn_{folder_id or 'all'}"
        
        async with self.driver.session() as session:
            try:
                # Create projection with node properties
                if folder_id:
                    await session.run("""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n:Entity) WHERE n.folder_id = $folder_id AND n[$prop] IS NOT NULL 
                             RETURN id(n) AS id, n[$prop] AS embedding',
                            'MATCH (a:Entity)-[r]->(b:Entity) 
                             WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id
                             RETURN id(a) AS source, id(b) AS target',
                            {parameters: {folder_id: $folder_id, prop: $prop}}
                        )
                    """, name=projection_name, folder_id=folder_id, prop=property_name)
                else:
                    await session.run("""
                        CALL gds.graph.project(
                            $name,
                            {Entity: {properties: [$prop]}},
                            '*'
                        )
                    """, name=projection_name, prop=property_name)
                
                # Run KNN
                result = await session.run("""
                    CALL gds.knn.stream($graph, {
                        topK: $k,
                        nodeProperties: ['embedding'],
                        sampleRate: 1.0,
                        concurrency: 4
                    })
                    YIELD node1, node2, similarity
                    WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
                    RETURN n1.id AS source_id, n1.name AS source_name,
                           n2.id AS target_id, n2.name AS target_name,
                           similarity AS score
                    ORDER BY score DESC
                    LIMIT 100
                """, graph=projection_name, k=k)
                
                records = await result.data()
                
                return {
                    "algorithm": "knn",
                    "engine": "gds.knn.stream",
                    "property": property_name,
                    "k": k,
                    "pairs": records,
                    "count": len(records),
                }
            except Exception as e:
                logger.error(f"KNN failed: {e}")
                return {"algorithm": "knn", "error": str(e), "pairs": []}
            finally:
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except:
                    pass

    # Utility functions using GDS similarity functions
    @staticmethod
    def jaccard_similarity(set_a: List, set_b: List) -> float:
        """Calculate Jaccard similarity (local, no Neo4j)."""
        if not set_a and not set_b:
            return 0.0
        set_a, set_b = set(set_a), set(set_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity (local, no Neo4j)."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a * a for a in vec_a))
        magnitude_b = math.sqrt(sum(b * b for b in vec_b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)
