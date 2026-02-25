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
        self._needs_refresh = True  # Force refresh on first run to clear stale Entity-only projections

    def _get_graph_name(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, undirected: bool = False) -> str:
        """Generate a consistent graph name for a given scope."""
        suffix = "_undirected" if undirected else ""
        if node_ids:
            # Sort and hash node IDs for a unique identifier
            node_hash = hashlib.md5(",".join(sorted(node_ids)).encode()).hexdigest()[:8]
            return f"subgraph_{node_hash}{suffix}"
        return f"neural_nexus_{folder_id or 'all'}{suffix}"

    async def _ensure_global_undirected(self, session) -> str:
        """Ensure the global undirected native projection exists. Returns the graph name."""
        global_name = "neural_nexus_all_undirected_native"
        check = await session.run("CALL gds.graph.exists($name) YIELD exists RETURN exists", name=global_name)
        rec = await check.single()
        if rec and rec["exists"]:
            logger.info(f"Global undirected projection '{global_name}' already exists, reusing.")
            return global_name
        
        logger.info(f"Creating global undirected native projection: {global_name}")
        await session.run(f"""
            CALL gds.graph.project(
                $name,
                '*',
                {{
                    _ALL_: {{
                        type: '*',
                        orientation: 'UNDIRECTED',
                        properties: {{weight: {{property: 'strength', defaultValue: 1.0}}}}
                    }}
                }}
            )
        """, name=global_name)
        self.active_projections.add(global_name)
        return global_name

    async def ensure_projection(
        self, 
        folder_id: Optional[str] = None, 
        node_ids: Optional[List[str]] = None,
        force_recreate: bool = False,
        undirected: bool = False
    ) -> str:
        """
        Ensures a GDS graph projection exists for the given scope.
        Implements 'load once' - returns existing projection if it's still valid.
        
        For undirected + scoped (folder/nodes): returns the global native UNDIRECTED
        projection. Algorithm callers must filter results by folder_id/node_ids.
        """
        # On first run, clear all stale projections (they may use old Entity-only label)
        if self._needs_refresh:
            logger.info("[GDS] First run — invalidating all stale projections")
            await self.invalidate_all()
            self._needs_refresh = False
        
        graph_name = self._get_graph_name(folder_id, node_ids, undirected)
        
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
            logger.info(f"Creating new GDS projection: {graph_name} (undirected={undirected})")
            
            try:
                # --- Undirected + scoped: use global native undirected projection ---
                # Cypher projections always produce directed metadata in GDS.
                # gds.graph.filter() is unreliable (silently fails to create).
                # Solution: use the global undirected native projection and let
                # algorithm callers filter results by folder_id/node_ids in their queries.
                if undirected and (folder_id or node_ids):
                    global_name = await self._ensure_global_undirected(session)
                    # Return the global projection name — callers filter results
                    self.active_projections.add(graph_name)
                    logger.info(f"Using global undirected projection '{global_name}' for scoped query '{graph_name}'")
                    return global_name
                    
                elif node_ids:
                    # Directed Cypher projection for specific nodes (highest priority)
                    logger.info(f"Creating Cypher projection for {len(node_ids)} specific nodes")
                    await session.run("""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n) WHERE n.id IN $node_ids RETURN id(n) AS id, labels(n) AS labels',
                            'MATCH (a)-[r]->(b) 
                             WHERE a.id IN $node_ids AND b.id IN $node_ids 
                             RETURN id(a) AS source, id(b) AS target, type(r) AS type, 
                                    coalesce(r.strength, 1.0) AS weight',
                            {parameters: {node_ids: $node_ids}}
                        )
                    """, name=graph_name, node_ids=node_ids)
                elif folder_id:
                    # Directed Cypher projection for folder scope
                    logger.info(f"Creating Cypher projection for folder: {folder_id}")
                    await session.run("""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n) WHERE n.folder_id = $folder_id RETURN id(n) AS id, labels(n) AS labels',
                            'MATCH (a)-[r]->(b) 
                             WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id 
                             RETURN id(a) AS source, id(b) AS target, type(r) AS type, 
                                    coalesce(r.strength, 1.0) AS weight',
                            {parameters: {folder_id: $folder_id}}
                        )
                    """, name=graph_name, folder_id=folder_id)
                else:
                    # Native projection for full graph (fastest)
                    orientation = "UNDIRECTED" if undirected else "NATURAL"
                    logger.info(f"Creating native projection with orientation={orientation}")
                    await session.run(f"""
                        CALL gds.graph.project(
                            $name,
                            '*',
                            {{
                                _ALL_: {{
                                    type: '*',
                                    orientation: '{orientation}',
                                    properties: {{weight: {{property: 'strength', defaultValue: 1.0}}}}
                                }}
                            }}
                        )
                    """, name=graph_name)
                
                self.active_projections.add(graph_name)
                logger.info(f"Successfully created GDS projection: {graph_name}")
                return graph_name
            except Exception as e:
                logger.error(f"Failed to create GDS projection '{graph_name}': {e}")
                raise

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

    async def run_pagerank(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        logger.info(f"[GDS] run_pagerank: folder_id={folder_id}, node_ids={node_ids}, top_k={top_k}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False)
        logger.info(f"[GDS] Using projection: {graph_name}")
        query = """
            CALL gds.pageRank.stream($graph_name)
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) = toLower($target_type))
            RETURN node.id AS id, node.name AS name, coalesce(node.type, labels(node)[0]) AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, target_type=target_type, folder_id=folder_id)
            data = await result.data()
            logger.info(f"[GDS] PageRank returned {len(data)} results")
            return data

    async def run_betweenness(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        logger.info(f"[GDS] run_betweenness: folder_id={folder_id}, node_ids={node_ids}, top_k={top_k}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False)
        logger.info(f"[GDS] Using projection: {graph_name}")
        query = """
            CALL gds.betweenness.stream($graph_name)
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) = toLower($target_type))
            RETURN node.id AS id, node.name AS name, coalesce(node.type, labels(node)[0]) AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, target_type=target_type, folder_id=folder_id)
            data = await result.data()
            logger.info(f"[GDS] Betweenness returned {len(data)} results")
            return data

    async def run_closeness(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        logger.info(f"[GDS] run_closeness: folder_id={folder_id}, top_k={top_k}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False)
        query = """
            CALL gds.closeness.stream($graph_name)
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE score > 0
            AND ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) = toLower($target_type))
            RETURN node.id AS id, node.name AS name, coalesce(node.type, labels(node)[0]) AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, target_type=target_type, folder_id=folder_id)
            data = await result.data()
            logger.info(f"[GDS] Closeness returned {len(data)} results")
            return data

    async def run_louvain(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        logger.info(f"[GDS] run_louvain: folder_id={folder_id}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=True)
        query = """
            CALL gds.louvain.stream($graph_name)
            YIELD nodeId, communityId
            WITH gds.util.asNode(nodeId) AS node, communityId
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) = toLower($target_type))
            RETURN node.id AS id, node.name AS name, coalesce(node.type, labels(node)[0]) AS type, communityId AS community_id
            ORDER BY community_id ASC
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, folder_id=folder_id, target_type=target_type)
            data = await result.data()
            logger.info(f"[GDS] Louvain returned {len(data)} results")
            return data

    async def run_wcc(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        logger.info(f"[GDS] run_wcc: folder_id={folder_id}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=True)
        query = """
            CALL gds.wcc.stream($graph_name)
            YIELD nodeId, componentId
            WITH gds.util.asNode(nodeId) AS node, componentId
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) = toLower($target_type))
            RETURN node.id AS id, node.name AS name, coalesce(node.type, labels(node)[0]) AS type, componentId AS community_id
            ORDER BY community_id ASC
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, folder_id=folder_id, target_type=target_type)
            data = await result.data()
            logger.info(f"[GDS] WCC returned {len(data)} results")
            return data

# Dependency
_gds_service = None

def get_gds_service() -> GDSService:
    global _gds_service
    if _gds_service is None:
        _gds_service = GDSService()
    return _gds_service
