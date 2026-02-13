"""
GDS Service

Manages Neo4j GDS (Graph Data Science) projections with a 'load once' strategy.
Handles graph projection creation, reuse, and invalidation.
"""
from typing import List, Optional, Dict, Any
import logging
import hashlib
from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)

class GDSService:
    def __init__(self):
        self.driver = get_neo4j_driver()
        self.active_projections = set()

    def _get_graph_name(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None) -> str:
        """Generate a consistent graph name for a given scope."""
        if node_ids:
            # Sort and hash node IDs for a unique identifier
            node_hash = hashlib.md5(",".join(sorted(node_ids)).encode()).hexdigest()[:8]
            return f"subgraph_{node_hash}"
        return f"neural_nexus_{folder_id or 'all'}"

    async def ensure_projection(
        self, 
        folder_id: Optional[str] = None, 
        node_ids: Optional[List[str]] = None,
        force_recreate: bool = False
    ) -> str:
        """
        Ensures a GDS graph projection exists for the given scope.
        Implements 'load once' - returns existing projection if it's still valid.
        """
        graph_name = self._get_graph_name(folder_id, node_ids)
        
        async with self.driver.session() as session:
            # Check if projection exists
            check_query = "CALL gds.graph.exists($name) YIELD exists RETURN exists"
            result = await session.run(check_query, name=graph_name)
            record = await result.single()
            
            if record and record["exists"] and not force_recreate:
                logger.debug(f"Reusing existing GDS projection: {graph_name}")
                return graph_name
            
            # Drop if force_recreate or if we need to refresh
            if record and record["exists"]:
                await session.run("CALL gds.graph.drop($name, false)", name=graph_name)
            
            # Create new projection
            logger.info(f"Creating new GDS projection: {graph_name}")
            
            if node_ids:
                # Cypher projection for specific nodes
                await session.run("""
                    CALL gds.graph.project.cypher(
                        $name,
                        'MATCH (n:Entity) WHERE n.id IN $node_ids RETURN id(n) AS id, labels(n) AS labels',
                        'MATCH (a:Entity)-[r]->(b:Entity) 
                         WHERE a.id IN $node_ids AND b.id IN $node_ids 
                         RETURN id(a) AS source, id(b) AS target, type(r) AS type, 
                                coalesce(r.strength, 1.0) AS weight',
                        {parameters: {node_ids: $node_ids}}
                    )
                """, name=graph_name, node_ids=node_ids)
            elif folder_id:
                # Cypher projection for folder scope
                await session.run("""
                    CALL gds.graph.project.cypher(
                        $name,
                        'MATCH (n:Entity) WHERE n.folder_id = $folder_id RETURN id(n) AS id, labels(n) AS labels',
                        'MATCH (a:Entity)-[r]->(b:Entity) 
                         WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id 
                         RETURN id(a) AS source, id(b) AS target, type(r) AS type, 
                                coalesce(r.strength, 1.0) AS weight',
                        {parameters: {folder_id: $folder_id}}
                    )
                """, name=graph_name, folder_id=folder_id)
            else:
                # Native projection for full graph (fastest)
                await session.run("""
                    CALL gds.graph.project(
                        $name,
                        'Entity',
                        {
                            _ALL_: {
                                type: '*',
                                orientation: 'UNDIRECTED',
                                properties: {weight: {property: 'strength', defaultValue: 1.0}}
                            }
                        }
                    )
                """, name=graph_name)
            
            self.active_projections.add(graph_name)
            return graph_name

    async def invalidate_all(self):
        """Drops all managed GDS projections. Call this when graph data changes significantly."""
        async with self.driver.session() as session:
            # We can also fetch all projections starting with our prefix
            result = await session.run("CALL gds.graph.list() YIELD graphName RETURN graphName")
            projections = await result.data()
            
            for p in projections:
                name = p["graphName"]
                if name.startswith("neural_nexus_") or name.startswith("subgraph_"):
                    try:
                        await session.run("CALL gds.graph.drop($name, false)", name=name)
                        logger.info(f"Invalidated GDS projection: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to drop projection {name}: {e}")
            
            self.active_projections.clear()

    async def drop_projection(self, graph_name: str):
        """Manually drop a specific projection."""
        async with self.driver.session() as session:
            try:
                await session.run("CALL gds.graph.drop($name, false)", name=graph_name)
                if graph_name in self.active_projections:
                    self.active_projections.remove(graph_name)
            except Exception as e:
                logger.warning(f"Failed to drop projection {graph_name}: {e}")

    # === GDS Algorithm Implementations ===

    async def run_pagerank(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        graph_name = await self.ensure_projection(folder_id, node_ids)
        query = """
            CALL gds.pageRank.stream($graph_name)
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            RETURN node.id AS id, node.name AS name, node.type AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k)
            return await result.data()

    async def run_betweenness(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        graph_name = await self.ensure_projection(folder_id, node_ids)
        query = """
            CALL gds.betweenness.stream($graph_name)
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            RETURN node.id AS id, node.name AS name, node.type AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k)
            return await result.data()

    async def run_closeness(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        graph_name = await self.ensure_projection(folder_id, node_ids)
        query = """
            CALL gds.closeness.stream($graph_name)
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE score > 0
            RETURN node.id AS id, node.name AS name, node.type AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k)
            return await result.data()

    async def run_louvain(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        graph_name = await self.ensure_projection(folder_id, node_ids)
        query = """
            CALL gds.louvain.stream($graph_name)
            YIELD nodeId, communityId
            WITH gds.util.asNode(nodeId) AS node, communityId
            RETURN node.id AS id, node.name AS name, node.type AS type, communityId AS community_id
            ORDER BY community_id ASC
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name)
            return await result.data()

# Dependency
_gds_service = None

def get_gds_service() -> GDSService:
    global _gds_service
    if _gds_service is None:
        _gds_service = GDSService()
    return _gds_service
