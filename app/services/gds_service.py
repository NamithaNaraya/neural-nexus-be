"""
GDS Service

Manages Neo4j GDS (Graph Data Science) projections with a 'load once' strategy.
Handles graph projection creation, reuse, and invalidation.
"""
from typing import List, Optional, Dict, Any
import logging
import hashlib
import json
from app.db.connections import get_neo4j_driver
from app.services.weight_service import WeightService

logger = logging.getLogger(__name__)

class GDSService:
    def __init__(self):
        self.driver = get_neo4j_driver()
        self.active_projections = set()
        self._needs_refresh = True  # Force refresh on first run to clear stale Entity-only projections

    def _filter_unnamed(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters out nodes that have no identifier.
        Previously forced 'name', now allows fallback identifiers.
        """
        return [
            r for r in data
            if r.get("name") or r.get("id") or (r.get("source_name") and r.get("target_name"))
        ]

    def _get_graph_name(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, undirected: bool = False, weight_formula: Optional[Dict[str, Any]] = None) -> str:
        """Generate a consistent graph name for a given scope."""
        suffix = "_undirected" if undirected else ""
        # Include weight formula hash so weighted projections don't conflict with unweighted
        if weight_formula:
            wh = hashlib.md5(json.dumps(weight_formula, sort_keys=True).encode()).hexdigest()[:6]
            suffix += f"_w{wh}"
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
        # Use native projection instead of Cypher to correctly set UNDIRECTED orientation
        # which is required by algorithms like K-Core.
        await session.run("""
            CALL gds.graph.project(
                $name,
                ['Entity'],
                {
                    ALL_RELATIONSHIPS: {
                        type: '*',
                        orientation: 'UNDIRECTED',
                        properties: {
                            weight: {
                                property: 'strength',
                                defaultValue: 1.0
                            }
                        }
                    }
                }
            )
        """, name=global_name)
        self.active_projections.add(global_name)
        return global_name

    async def ensure_projection(
        self, 
        folder_id: Optional[str] = None, 
        node_ids: Optional[List[str]] = None,
        force_recreate: bool = False,
        undirected: bool = False,
        weight_formula: Optional[Dict[str, Any]] = None
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
        
        graph_name = self._get_graph_name(folder_id, node_ids, undirected, weight_formula)
        
        # Compute the weight Cypher expression from formula (or use default)
        if weight_formula:
            weight_expr = WeightService.formula_to_cypher(weight_formula, rel_var="r")
            logger.info(f"[GDS] Using custom weight expression: {weight_expr}")
        else:
            weight_expr = "coalesce(r.strength, 1.0)"
        
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
                    await session.run(f"""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n) WHERE n.id IN $node_ids RETURN id(n) AS id, labels(n) AS labels',
                            'MATCH (a)-[r]->(b) 
                             WHERE a.id IN $node_ids AND b.id IN $node_ids
                             RETURN id(a) AS source, id(b) AS target, type(r) AS type, 
                                    {weight_expr} AS weight',
                            {{parameters: {{node_ids: $node_ids}}}}
                        )
                    """, name=graph_name, node_ids=node_ids)
                elif folder_id:
                    # Directed Cypher projection for folder scope
                    logger.info(f"Creating Cypher projection for folder: {folder_id}")
                    await session.run(f"""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n) WHERE n.folder_id = $folder_id RETURN id(n) AS id, labels(n) AS labels',
                            'MATCH (a)-[r]->(b) 
                             WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id
                             RETURN id(a) AS source, id(b) AS target, type(r) AS type, 
                                    {weight_expr} AS weight',
                            {{parameters: {{folder_id: $folder_id}}}}
                        )
                    """, name=graph_name, folder_id=folder_id)
                else:
                    orientation = "UNDIRECTED" if undirected else "NATURAL"
                    logger.info(f"Creating Cypher projection for full graph with orientation={orientation}")
                    rel_pattern = 'MATCH (a)-[r]-(b)' if undirected else 'MATCH (a)-[r]->(b)'
                    await session.run(f"""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n) RETURN id(n) AS id, labels(n) AS labels',
                            '{rel_pattern}
                             RETURN id(a) AS source, id(b) AS target, type(r) AS type,
                                    {weight_expr} AS weight'
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

    async def run_pagerank(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10, target_type: Optional[str] = None, weight_formula: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        logger.info(f"[GDS] run_pagerank: folder_id={folder_id}, node_ids={node_ids}, top_k={top_k}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False, weight_formula=weight_formula)
        logger.info(f"[GDS] Using projection: {graph_name}")
        query = """
            CALL gds.pageRank.stream($graph_name, {
                relationshipWeightProperty: 'weight'
            })
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, target_type=target_type, folder_id=folder_id)
            data = await result.data()
            logger.info(f"[GDS] PageRank returned {len(data)} results")
            return self._filter_unnamed(data)

    async def run_betweenness(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10, target_type: Optional[str] = None, weight_formula: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        logger.info(f"[GDS] run_betweenness: folder_id={folder_id}, node_ids={node_ids}, top_k={top_k}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False, weight_formula=weight_formula)
        logger.info(f"[GDS] Using projection: {graph_name}")
        query = """
            CALL gds.betweenness.stream($graph_name)
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, target_type=target_type, folder_id=folder_id)
            data = await result.data()
            logger.info(f"[GDS] Betweenness returned {len(data)} results")
            return self._filter_unnamed(data)

    async def run_closeness(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10, target_type: Optional[str] = None, weight_formula: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        logger.info(f"[GDS] run_closeness: folder_id={folder_id}, top_k={top_k}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False, weight_formula=weight_formula)
        query = """
            CALL gds.closeness.stream($graph_name)
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE score > 0
            AND ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, target_type=target_type, folder_id=folder_id)
            data = await result.data()
            logger.info(f"[GDS] Closeness returned {len(data)} results")
            return self._filter_unnamed(data)

    async def run_louvain(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        logger.info(f"[GDS] run_louvain: folder_id={folder_id}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=True)
        query = """
            CALL gds.louvain.stream($graph_name)
            YIELD nodeId, communityId
            WITH gds.util.asNode(nodeId) AS node, communityId
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type, communityId AS community_id
            ORDER BY community_id ASC
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, folder_id=folder_id, target_type=target_type)
            data = await result.data()
            logger.info(f"[GDS] Louvain returned {len(data)} results")
            return self._filter_unnamed(data)

    async def run_wcc(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        logger.info(f"[GDS] run_wcc: folder_id={folder_id}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=True)
        query = """
            CALL gds.wcc.stream($graph_name)
            YIELD nodeId, componentId
            WITH gds.util.asNode(nodeId) AS node, componentId
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type, componentId AS community_id
            ORDER BY community_id ASC
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, folder_id=folder_id, target_type=target_type)
            data = await result.data()
            logger.info(f"[GDS] WCC returned {len(data)} results")
            return self._filter_unnamed(data)

    async def run_articlerank(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10, target_type: Optional[str] = None, weight_formula: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ArticleRank — improved PageRank for graphs with diverse degree distributions."""
        logger.info(f"[GDS] run_articlerank: folder_id={folder_id}, top_k={top_k}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False, weight_formula=weight_formula)
        query = """
            CALL gds.articleRank.stream($graph_name, {
                relationshipWeightProperty: 'weight'
            })
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, target_type=target_type, folder_id=folder_id)
            data = await result.data()
            logger.info(f"[GDS] ArticleRank returned {len(data)} results")
            return self._filter_unnamed(data)

    async def run_hits(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10, target_type: Optional[str] = None, weight_formula: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """HITS — identifies hub nodes (link to many) and authority nodes (linked by many)."""
        logger.info(f"[GDS] run_hits: folder_id={folder_id}, top_k={top_k}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False, weight_formula=weight_formula)
        query = """
            CALL gds.hits.stream($graph_name, {
                relationshipWeightProperty: 'weight'
            })
            YIELD nodeId, values
            WITH gds.util.asNode(nodeId) AS node, values.hub AS hubScore, values.auth AS authScore
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type,
                   hubScore AS hub_score, authScore AS auth_score, (hubScore + authScore) AS score
            ORDER BY authScore DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, target_type=target_type, folder_id=folder_id)
            data = await result.data()
            logger.info(f"[GDS] HITS returned {len(data)} results")
            return self._filter_unnamed(data)

    async def run_leiden(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Leiden — improved community detection over Louvain, better quality communities."""
        logger.info(f"[GDS] run_leiden: folder_id={folder_id}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=True)
        query = """
            CALL gds.leiden.stream($graph_name, {
                gamma: 1.0
            })
            YIELD nodeId, communityId
            WITH gds.util.asNode(nodeId) AS node, communityId
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type, communityId AS community_id
            ORDER BY community_id ASC
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(query, graph_name=graph_name, folder_id=folder_id, target_type=target_type)
                data = await result.data()
                logger.info(f"[GDS] Leiden returned {len(data)} results")
                return self._filter_unnamed(data)
        except Exception as e:
            error_msg = str(e).lower()
            if "undirected" in error_msg or "orientation" in error_msg or "not found" in error_msg:
                logger.warning(f"[GDS] Leiden failed due to graph structure: {e}")
                return [{"error_message": "Leiden requires sufficient undirected connections. Your current data may not have enough relationships for this algorithm."}]
            raise

    async def run_kcore(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 50, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """K-Core — finds the stable, tightly-connected core of the graph."""
        logger.info(f"[GDS] run_kcore: folder_id={folder_id}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=True)
        query = """
            CALL gds.kcore.stream($graph_name)
            YIELD nodeId, coreValue
            WITH gds.util.asNode(nodeId) AS node, coreValue
            WHERE coreValue >= 2
            AND ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type, coreValue AS score
            ORDER BY coreValue DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, folder_id=folder_id, target_type=target_type)
            data = await result.data()
            logger.info(f"[GDS] K-Core returned {len(data)} results")
            return self._filter_unnamed(data)

    async def run_triangle_count(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 20, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Triangle Count — counts triangles per node to measure local clustering density."""
        logger.info(f"[GDS] run_triangle_count: folder_id={folder_id}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=True)
        query = """
            CALL gds.triangleCount.stream($graph_name)
            YIELD nodeId, triangleCount
            WITH gds.util.asNode(nodeId) AS node, triangleCount
            WHERE triangleCount > 0
            AND ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type, triangleCount AS score
            ORDER BY triangleCount DESC
            LIMIT $top_k
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(query, graph_name=graph_name, top_k=top_k, folder_id=folder_id, target_type=target_type)
                data = await result.data()
                logger.info(f"[GDS] Triangle Count returned {len(data)} results")
                if not data:
                    return [{"error_message": "No triangles found. Your data may not have enough interconnected nodes to form triangles (3 nodes all connected to each other)."}]
                return self._filter_unnamed(data)
        except Exception as e:
            error_msg = str(e).lower()
            if "undirected" in error_msg or "orientation" in error_msg:
                logger.warning(f"[GDS] Triangle Count failed due to graph structure: {e}")
                return [{"error_message": "Triangle Count requires undirected relationships. Your current graph projection may not support this algorithm."}]
            raise

    async def run_node_similarity(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Node Similarity — finds pairs of nodes that share similar neighborhoods (Jaccard)."""
        logger.info(f"[GDS] run_node_similarity: folder_id={folder_id}, top_k={top_k}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=True)
        query = """
            CALL gds.nodeSimilarity.stream($graph_name, {
                topK: $top_k,
                similarityCutoff: 0.01
            })
            YIELD node1, node2, similarity
            WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
            WHERE ($folder_id IS NULL OR n1.folder_id = $folder_id)
            RETURN n1.id AS source_id, coalesce(n1.name, n1.title, n1.questionId, n1.question_text, n1.studentId, n1.examId, n1.code, n1.text, n1.content, n1.label, n1.val, n1.value, n1.id) AS source_name, coalesce(n1.type, labels(n1)[0]) AS source_type,
                   n2.id AS target_id, coalesce(n2.name, n2.title, n2.questionId, n2.question_text, n2.studentId, n2.examId, n2.code, n2.text, n2.content, n2.label, n2.val, n2.value, n2.id) AS target_name, coalesce(n2.type, labels(n2)[0]) AS target_type,
                   similarity AS score
            ORDER BY similarity DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, folder_id=folder_id, target_type=target_type)
            data = await result.data()
            logger.info(f"[GDS] Node Similarity returned {len(data)} results")
            return self._filter_unnamed(data)

    async def run_degree(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, top_k: int = 10, target_type: Optional[str] = None, weight_formula: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Degree Centrality — counts direct connections per node."""
        logger.info(f"[GDS] run_degree: folder_id={folder_id}, top_k={top_k}, target_type={target_type}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False, weight_formula=weight_formula)
        query = """
            CALL gds.degree.stream($graph_name)
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
            AND ($target_type IS NULL OR toLower(coalesce(node.type, labels(node)[0])) STARTS WITH toLower($target_type))
            RETURN node.id AS id, coalesce(node.name, node.title, node.questionId, node.question_text, node.studentId, node.examId, node.code, node.text, node.content, node.label, node.val, node.value, node.id) AS name, coalesce(node.type, labels(node)[0]) AS type, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(query, graph_name=graph_name, top_k=top_k, target_type=target_type, folder_id=folder_id)
            data = await result.data()
            logger.info(f"[GDS] Degree returned {len(data)} results")
            return self._filter_unnamed(data)

    async def run_link_prediction(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, method: str = "common_neighbors", top_k: int = 20, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Link Prediction — predicts missing links using various methods (common_neighbors, adamic_adar, resource_allocation)."""
        logger.info(f"[GDS] run_link_prediction: folder_id={folder_id}, method={method}, top_k={top_k}")
        
        folder_filter = ""
        if folder_id:
            folder_filter = " {folder_id: $folder_id}"
        
        if method == "adamic_adar":
            query = f"""
                MATCH (a:Entity{folder_filter}), (b:Entity{folder_filter})
                WHERE a <> b AND NOT (a)--(b) AND id(a) < id(b)
                WITH a, b
                MATCH (a)--(neighbor)--(b)
                WITH a, b, neighbor, COUNT {{ (neighbor)--() }} AS degree
                WHERE degree > 1
                WITH a, b, sum(1.0 / log(toFloat(degree))) AS score
                WHERE score > 0
                RETURN a.id AS source_id, coalesce(a.name, a.title, a.question_text, a.text, a.content, a.label, a.code, a.questionId, a.studentId, a.examId, a.val, a.value, a.id) AS source_name,
                       b.id AS target_id, coalesce(b.name, b.title, b.question_text, b.text, b.content, b.label, b.code, b.questionId, b.studentId, b.examId, b.val, b.value, b.id) AS target_name,
                       round(score * 1000) / 1000.0 AS score
                ORDER BY score DESC
                LIMIT $top_k
            """
        elif method == "resource_allocation":
            query = f"""
                MATCH (a:Entity{folder_filter}), (b:Entity{folder_filter})
                WHERE a <> b AND NOT (a)--(b) AND id(a) < id(b)
                WITH a, b
                MATCH (a)--(neighbor)--(b)
                WITH a, b, neighbor, toFloat(COUNT {{ (neighbor)--() }}) AS degree
                WHERE degree > 0
                WITH a, b, sum(1.0 / degree) AS score
                WHERE score > 0
                RETURN a.id AS source_id, coalesce(a.name, a.title, a.question_text, a.text, a.content, a.label, a.code, a.questionId, a.studentId, a.examId, a.val, a.value, a.id) AS source_name,
                       b.id AS target_id, coalesce(b.name, b.title, b.question_text, b.text, b.content, b.label, b.code, b.questionId, b.studentId, b.examId, b.val, b.value, b.id) AS target_name,
                       round(score * 1000) / 1000.0 AS score
                ORDER BY score DESC
                LIMIT $top_k
            """
        else:  # common_neighbors (default)
            query = f"""
                MATCH (a:Entity{folder_filter}), (b:Entity{folder_filter})
                WHERE a <> b AND NOT (a)--(b) AND id(a) < id(b)
                WITH a, b
                OPTIONAL MATCH (a)--(neighbor)--(b)
                WITH a, b, count(DISTINCT neighbor) AS score
                WHERE score > 0
                RETURN a.id AS source_id, coalesce(a.name, a.title, a.question_text, a.text, a.content, a.label, a.code, a.questionId, a.studentId, a.examId, a.val, a.value, a.id) AS source_name,
                       b.id AS target_id, coalesce(b.name, b.title, b.question_text, b.text, b.content, b.label, b.code, b.questionId, b.studentId, b.examId, b.val, b.value, b.id) AS target_name,
                       score
                ORDER BY score DESC
                LIMIT $top_k
            """
        
        params = {"top_k": top_k}
        if folder_id:
            params["folder_id"] = folder_id
        
        async with self.driver.session() as session:
            result = await session.run(query, **params)
            data = await result.data()
            logger.info(f"[GDS] Link Prediction ({method}) returned {len(data)} results")
            return data

    async def run_bfs(self, source_id: str, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """BFS Traversal — explores nodes layer by layer from a starting point."""
        logger.info(f"[GDS] run_bfs: source_id={source_id}, folder_id={folder_id}")
        # Traversal works best on undirected graphs for discovery
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=True)
        
        async with self.driver.session() as session:
            # Get internal node ID
            id_res = await session.run("MATCH (n {id: $id}) RETURN id(n) AS neo_id", id=source_id)
            rec = await id_res.single()
            if not rec:
                logger.warning(f"[GDS] BFS source node not found: {source_id}")
                return []
            
            # Build scope filter
            scope_filter = ""
            params = {"graph_name": graph_name, "source": rec["neo_id"]}
            if node_ids:
                scope_filter = "WHERE n.id IN $node_ids"
                params["node_ids"] = node_ids
            elif folder_id:
                scope_filter = "WHERE n.folder_id = $folder_id"
                params["folder_id"] = folder_id

            result = await session.run(f"""
                CALL gds.bfs.stream($graph_name, {{
                    sourceNode: $source
                }})
                YIELD nodeIds
                UNWIND nodeIds AS nid
                WITH gds.util.asNode(nid) AS n
                {scope_filter}
                RETURN n.id AS id, coalesce(n.name, n.title, n.question_text, n.text, n.content, n.label, n.code, n.questionId, n.studentId, n.examId, n.val, n.value, n.id) AS name, coalesce(n.type, labels(n)[0]) AS type
            """, **params)
            
            data = await result.data()
            logger.info(f"[GDS] BFS returned {len(data)} results")
            return data

    async def run_dfs(self, source_id: str, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """DFS Traversal — follows paths as deep as possible before backtracking."""
        logger.info(f"[GDS] run_dfs: source_id={source_id}, folder_id={folder_id}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=True)
        
        async with self.driver.session() as session:
            id_res = await session.run("MATCH (n {id: $id}) RETURN id(n) AS neo_id", id=source_id)
            rec = await id_res.single()
            if not rec:
                logger.warning(f"[GDS] DFS source node not found: {source_id}")
                return []
            
            # Build scope filter
            scope_filter = ""
            params = {"graph_name": graph_name, "source": rec["neo_id"]}
            if node_ids:
                scope_filter = "WHERE n.id IN $node_id_list"
                params["node_id_list"] = node_ids
            elif folder_id:
                scope_filter = "WHERE n.folder_id = $folder_id"
                params["folder_id"] = folder_id

            result = await session.run(f"""
                CALL gds.dfs.stream($graph_name, {{
                    sourceNode: $source
                }})
                YIELD nodeIds
                UNWIND nodeIds AS nid
                WITH gds.util.asNode(nid) AS n
                {scope_filter}
                RETURN n.id AS id, coalesce(n.name, n.title, n.question_text, n.text, n.content, n.label, n.code, n.questionId, n.studentId, n.examId, n.val, n.value, n.id) AS name, coalesce(n.type, labels(n)[0]) AS type
            """, **params)
            
            data = await result.data()
            logger.info(f"[GDS] DFS returned {len(data)} results")
            return data

    async def run_random_walk(self, source_id: str, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None, walk_length: int = 10, walk_count: int = 1) -> List[Dict[str, Any]]:
        """Random Walk — simulates a user wandering through the graph randomly."""
        logger.info(f"[GDS] run_random_walk: source_id={source_id}, folder_id={folder_id}, walk_length={walk_length}, walk_count={walk_count}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False)
        
        async with self.driver.session() as session:
            id_res = await session.run("MATCH (n {id: $id}) RETURN id(n) AS neo_id", id=source_id)
            rec = await id_res.single()
            if not rec:
                logger.warning(f"[GDS] Random Walk source node not found: {source_id}")
                return []
            
            # Build scope filter
            scope_filter = ""
            params = {"graph_name": graph_name, "source": rec["neo_id"], "length": walk_length, "count": walk_count}
            if node_ids:
                scope_filter = "WHERE n.id IN $node_id_list"
                params["node_id_list"] = node_ids
            elif folder_id:
                scope_filter = "WHERE n.folder_id = $folder_id"
                params["folder_id"] = folder_id

            result = await session.run(f"""
                CALL gds.randomWalk.stream($graph_name, {{
                    sourceNodes: [$source],
                    walkLength: $length,
                    walksPerNode: $count
                }})
                YIELD nodeIds
                UNWIND nodeIds AS nid
                WITH DISTINCT gds.util.asNode(nid) AS n
                {scope_filter}
                RETURN n.id AS id, coalesce(n.name, n.title, n.question_text, n.text, n.content, n.label, n.code, n.questionId, n.studentId, n.examId, n.val, n.value, n.id) AS name, coalesce(n.type, labels(n)[0]) AS type
            """, **params)
            
            data = await result.data()
            logger.info(f"[GDS] Random Walk returned {len(data)} results")
            return data

    async def run_topological_sort(self, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Topological Sort — orders nodes in a logical linear sequence for DAGs."""
        logger.info(f"[GDS] run_topological_sort: folder_id={folder_id}")
        graph_name = await self.ensure_projection(folder_id, node_ids, undirected=False)
        
        async with self.driver.session() as session:
            result = await session.run("""
                CALL gds.dag.topologicalSort.stream($graph_name)
                YIELD nodeId
                WITH gds.util.asNode(nodeId) AS n
                WHERE ($folder_id IS NULL OR n.folder_id = $folder_id)
                RETURN n.id AS id, coalesce(n.name, n.title, n.question_text, n.text, n.content, n.label, n.code, n.questionId, n.studentId, n.examId, n.val, n.value, n.id) AS name, coalesce(n.type, labels(n)[0]) AS type
            """, graph_name=graph_name, folder_id=folder_id)
            
            data = await result.data()
            logger.info(f"[GDS] Topological Sort returned {len(data)} results")
            return self._filter_unnamed(data)

# Dependency
_gds_service = None

def get_gds_service() -> GDSService:
    global _gds_service
    if _gds_service is None:
        _gds_service = GDSService()
    return _gds_service
