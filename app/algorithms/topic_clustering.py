"""
Topic Clustering Algorithm

Automatically groups nodes into logical thematic areas using Neo4j GDS.
Uses Louvain and Leiden community detection algorithms.
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)


@dataclass
class TopicCluster:
    """A cluster of related entities."""
    cluster_id: int
    label: str
    entities: List[Dict[str, str]]
    size: int
    coherence_score: float


class TopicClustering:
    """
    Cluster entities into thematic topics using Neo4j GDS.
    
    Methods (all use GDS):
    - Louvain: gds.louvain.stream - Fast community detection
    - Leiden: gds.leiden.stream - Higher quality communities
    - Label Propagation: gds.labelPropagation.stream - Fast, unstable
    - WCC: gds.wcc.stream - Weakly connected components
    """
    
    def __init__(self):
        self.driver = get_neo4j_driver()
    
    async def cluster(
        self,
        folder_id: Optional[str] = None,
        method: str = "louvain",
        resolution: float = 1.0
    ) -> Dict[str, Any]:
        """
        Cluster entities into topics using GDS community detection.
        
        Args:
            folder_id: Optional scope to specific folder
            method: Clustering method ('louvain', 'leiden', 'label_propagation', 'wcc')
            resolution: Resolution parameter for Louvain/Leiden (higher = more clusters)
            
        Returns:
            Dictionary with clusters and quality metrics
        """
        projection_name = f"clustering_{folder_id or 'all'}"
        
        async with self.driver.session() as session:
            try:
                # Clean up existing projection
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except:
                    pass
                
                # Create projection
                if folder_id:
                    await session.run("""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n:Entity) WHERE n.folder_id = $folder_id RETURN id(n) AS id, labels(n) AS labels',
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
                
                # Run clustering based on method
                if method == "louvain":
                    result = await session.run("""
                        CALL gds.louvain.stream($graph, {
                            includeIntermediateCommunities: false
                        })
                        YIELD nodeId, communityId
                        WITH gds.util.asNode(nodeId) AS node, communityId
                        RETURN communityId AS cluster_id,
                               collect({id: node.id, name: node.name, type: node.type}) AS members,
                               count(*) AS size
                        ORDER BY size DESC
                    """, graph=projection_name)
                    
                elif method == "leiden":
                    result = await session.run("""
                        CALL gds.leiden.stream($graph, {
                            gamma: $resolution
                        })
                        YIELD nodeId, communityId
                        WITH gds.util.asNode(nodeId) AS node, communityId
                        RETURN communityId AS cluster_id,
                               collect({id: node.id, name: node.name, type: node.type}) AS members,
                               count(*) AS size
                        ORDER BY size DESC
                    """, graph=projection_name, resolution=resolution)
                    
                elif method == "label_propagation":
                    result = await session.run("""
                        CALL gds.labelPropagation.stream($graph)
                        YIELD nodeId, communityId
                        WITH gds.util.asNode(nodeId) AS node, communityId
                        RETURN communityId AS cluster_id,
                               collect({id: node.id, name: node.name, type: node.type}) AS members,
                               count(*) AS size
                        ORDER BY size DESC
                    """, graph=projection_name)
                    
                elif method == "wcc":
                    result = await session.run("""
                        CALL gds.wcc.stream($graph)
                        YIELD nodeId, componentId
                        WITH gds.util.asNode(nodeId) AS node, componentId AS communityId
                        RETURN communityId AS cluster_id,
                               collect({id: node.id, name: node.name, type: node.type}) AS members,
                               count(*) AS size
                        ORDER BY size DESC
                    """, graph=projection_name)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                records = await result.data()
                
                clusters = {}
                total_nodes = 0
                for r in records:
                    cid = r["cluster_id"]
                    clusters[cid] = {
                        "id": cid,
                        "members": r["members"],
                        "size": r["size"],
                    }
                    total_nodes += r["size"]
                
                # Calculate modularity (for Louvain/Leiden)
                modularity = 0.0
                if method in ["louvain", "leiden"]:
                    try:
                        mod_result = await session.run(f"""
                            CALL gds.{method}.stats($graph)
                            YIELD modularity
                            RETURN modularity
                        """, graph=projection_name)
                        mod_record = await mod_result.single()
                        if mod_record:
                            modularity = mod_record["modularity"]
                    except:
                        pass
                
                # Generate insight
                if clusters:
                    largest = max(clusters.values(), key=lambda x: x["size"])
                    insight = f"Found {len(clusters)} communities. Largest has {largest['size']} members ({largest['size']*100//total_nodes}% of graph). Modularity: {modularity:.3f}."
                else:
                    insight = "No communities found."
                
                return {
                    "algorithm": "topic_clustering",
                    "engine": f"gds.{method}.stream",
                    "method": method,
                    "folder_id": folder_id,
                    "num_clusters": len(clusters),
                    "total_nodes": total_nodes,
                    "modularity": modularity,
                    "clusters": clusters,
                    "insight": insight,
                }
                
            except Exception as e:
                logger.error(f"Topic clustering failed: {e}")
                return {
                    "algorithm": "topic_clustering",
                    "method": method,
                    "clusters": {},
                    "error": str(e),
                    "insight": f"Clustering failed: {str(e)}",
                }
            finally:
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except:
                    pass
    
    async def get_cluster_bridges(
        self, 
        folder_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find nodes that bridge multiple communities.
        These are valuable connector entities.
        """
        projection_name = f"bridges_{folder_id or 'all'}"
        
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
                    await session.run("CALL gds.graph.project($name, 'Entity', '*')", name=projection_name)
                
                # Get communities and betweenness in one go
                result = await session.run("""
                    CALL gds.louvain.stream($graph)
                    YIELD nodeId, communityId
                    WITH nodeId, communityId
                    CALL gds.betweenness.stream($graph)
                    YIELD nodeId AS bNodeId, score
                    WHERE nodeId = bNodeId
                    WITH gds.util.asNode(nodeId) AS node, communityId, score
                    ORDER BY score DESC
                    LIMIT 20
                    RETURN node.id AS id, node.name AS name, node.type AS type,
                           communityId AS community, score AS betweenness
                """, graph=projection_name)
                
                records = await result.data()
                
                return {
                    "bridges": records,
                    "insight": f"Top {len(records)} bridge nodes that connect communities.",
                }
            finally:
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except:
                    pass

    async def compare_clusters(
        self,
        cluster_id_1: int,
        cluster_id_2: int,
        folder_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two clusters to find shared concepts and bridges.
        """
        # First run clustering to get cluster assignments
        clustering_result = await self.cluster(folder_id, method="louvain")
        clusters = clustering_result.get("clusters", {})
        
        c1 = clusters.get(cluster_id_1, {})
        c2 = clusters.get(cluster_id_2, {})
        
        c1_ids = {m["id"] for m in c1.get("members", [])}
        c2_ids = {m["id"] for m in c2.get("members", [])}
        
        # Find nodes connected between clusters
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (a:Entity)-[r]-(b:Entity)
                WHERE a.id IN $c1_ids AND b.id IN $c2_ids
                RETURN a.id AS source_id, a.name AS source_name,
                       b.id AS target_id, b.name AS target_name,
                       type(r) AS relationship
            """, c1_ids=list(c1_ids), c2_ids=list(c2_ids))
            
            bridges = await result.data()
        
        return {
            "cluster_1": {"id": cluster_id_1, "size": len(c1_ids)},
            "cluster_2": {"id": cluster_id_2, "size": len(c2_ids)},
            "connecting_relationships": bridges,
            "bridge_count": len(bridges),
        }
