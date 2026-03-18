"""
Combined Chat — FastRP Graph Structural Search Service

Provides topology-based retrieval using Neo4j GDS FastRP embeddings.
FastRP generates node embeddings from GRAPH STRUCTURE (not text),
capturing how nodes are connected — nodes with similar neighborhood
topology get similar embeddings.

Flow:
  1. Ensure FastRP embeddings exist on folder entities (generate if missing)
  2. Embed the query text → use it to find structurally relevant nodes
  3. Return top-K structurally similar entities for context fusion
"""
import logging
from typing import List, Dict, Any, Optional
from app.db.connections import get_neo4j_driver
from app.services.ai_service import get_ai_service
from app.core.config import settings

logger = logging.getLogger(__name__)

# FastRP configuration
_FASTRP_DIMENSION = 256
_FASTRP_PROJECTION_PREFIX = "fastrp_chat"
_FASTRP_PROPERTY = "fastrp_embedding"
_FASTRP_INDEX_NAME = "fastrp_vector_idx"


class FastRPService:
    """
    Graph Structure Search using FastRP topology embeddings.

    Unlike text embeddings (which capture semantic meaning of words),
    FastRP embeddings capture the STRUCTURAL ROLE of a node in the graph.
    Two nodes that are connected to similar types of neighbors will have
    similar FastRP embeddings — even if their names/descriptions are different.
    """

    def __init__(self):
        self.driver = get_neo4j_driver()
        self.ai = get_ai_service()
        self._gds_available: Optional[bool] = None

    # ────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────

    async def structural_search(
        self,
        query: str,
        folder_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find structurally relevant nodes for a query.

        Strategy:
          1. Check if GDS is available
          2. Ensure FastRP embeddings exist for this folder
          3. Use the entity names from text-vector search as anchor points
          4. Find structurally similar neighbors via FastRP cosine similarity
          5. Return enriched results with structural context

        Args:
            query: User's natural language question
            folder_id: Folder scope for isolation
            top_k: Maximum results to return

        Returns:
            List of structurally relevant node dicts with similarity scores
        """
        # Step 1: Check GDS availability
        if not await self._check_gds():
            logger.info("[FastRP] GDS not available — using degree-based structural fallback")
            return await self._structural_fallback(query, folder_id, top_k)

        # Step 2: Ensure FastRP embeddings exist
        has_embeddings = await self._ensure_fastrp_embeddings(folder_id)
        if not has_embeddings:
            logger.warning("[FastRP] Could not generate embeddings — using fallback")
            return await self._structural_fallback(query, folder_id, top_k)

        # Step 3: Find anchor nodes via text matching (quick keyword match)
        anchors = await self._find_anchor_nodes(query, folder_id)

        if not anchors:
            # No anchors found — get top structurally important nodes instead
            return await self._get_top_structural_nodes(folder_id, top_k)

        # Step 4: Find structurally similar nodes to anchors via FastRP
        results = await self._find_similar_by_fastrp(anchors, folder_id, top_k)

        logger.info(f"[FastRP] Structural search returned {len(results)} results for folder {folder_id}")
        return results

    # ────────────────────────────────────────────────────────────
    #  GDS Check
    # ────────────────────────────────────────────────────────────

    async def _check_gds(self) -> bool:
        """Check if GDS plugin is available (cached)."""
        if self._gds_available is not None:
            return self._gds_available

        async with self.driver.session() as session:
            try:
                result = await session.run("CALL gds.version() YIELD version RETURN version")
                record = await result.single()
                self._gds_available = record is not None
                if self._gds_available:
                    logger.info(f"[FastRP] GDS available: v{record['version']}")
            except Exception:
                self._gds_available = False
                logger.info("[FastRP] GDS plugin not installed")

        return self._gds_available

    # ────────────────────────────────────────────────────────────
    #  FastRP Embedding Generation
    # ────────────────────────────────────────────────────────────

    async def _ensure_fastrp_embeddings(self, folder_id: Optional[str]) -> bool:
        """
        Check if FastRP embeddings exist for this folder.
        If not, generate them using GDS FastRP.write.
        """
        async with self.driver.session() as session:
            # Check if any entities in this folder already have FastRP embeddings
            check_query = """
                MATCH (n:Entity)
                WHERE ($folder_id IS NULL OR n.folder_id = $folder_id)
                  AND n.fastrp_embedding IS NOT NULL
                RETURN count(n) AS count
            """
            result = await session.run(check_query, folder_id=folder_id)
            record = await result.single()
            existing_count = record["count"] if record else 0

            # Also check total entity count
            total_query = """
                MATCH (n:Entity)
                WHERE ($folder_id IS NULL OR n.folder_id = $folder_id)
                RETURN count(n) AS total
            """
            total_result = await session.run(total_query, folder_id=folder_id)
            total_record = await total_result.single()
            total_count = total_record["total"] if total_record else 0

            if total_count == 0:
                logger.info("[FastRP] No entities in folder — skipping FastRP")
                return False

            # If more than 50% already have embeddings, consider them valid
            if existing_count > total_count * 0.5:
                logger.info(f"[FastRP] Embeddings exist: {existing_count}/{total_count} entities")
                return True

            # Need to generate — run FastRP
            logger.info(f"[FastRP] Generating embeddings for {total_count} entities in folder {folder_id}")
            return await self._run_fastrp(folder_id)

    async def _run_fastrp(self, folder_id: Optional[str]) -> bool:
        """Execute GDS FastRP algorithm and write embeddings to nodes."""
        projection_name = f"{_FASTRP_PROJECTION_PREFIX}_{(folder_id or 'global').replace('-', '_')}"

        async with self.driver.session() as session:
            try:
                # 1. Drop stale projection
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except Exception:
                    pass

                # 2. Create graph projection scoped to folder
                if folder_id:
                    await session.run("""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n:Entity) WHERE n.folder_id = $folder_id RETURN id(n) AS id',
                            'MATCH (a:Entity)-[r]->(b:Entity)
                             WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id
                             RETURN id(a) AS source, id(b) AS target, type(r) AS type',
                            {parameters: {folder_id: $folder_id}}
                        )
                    """, name=projection_name, folder_id=folder_id)
                else:
                    await session.run("""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n:Entity) RETURN id(n) AS id',
                            'MATCH (a:Entity)-[r]->(b:Entity)
                             RETURN id(a) AS source, id(b) AS target, type(r) AS type'
                        )
                    """, name=projection_name)

                # 3. Run FastRP and write embeddings
                await session.run("""
                    CALL gds.fastRP.write(
                        $name,
                        {
                            writeProperty: $property,
                            embeddingDimension: $dimension,
                            iterationWeights: [0.0, 1.0, 1.0, 0.5],
                            randomSeed: 42
                        }
                    )
                """, name=projection_name, property=_FASTRP_PROPERTY, dimension=_FASTRP_DIMENSION)

                logger.info(f"[FastRP] ✅ Embeddings generated successfully for folder {folder_id}")
                return True

            except Exception as e:
                error_str = str(e)
                if "ProcedureNotFound" in error_str:
                    logger.warning("[FastRP] GDS FastRP procedure not available")
                    self._gds_available = False
                else:
                    logger.error(f"[FastRP] FastRP generation failed: {e}")
                return False

            finally:
                # Cleanup projection
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except Exception:
                    pass

    # ────────────────────────────────────────────────────────────
    #  Anchor Node Discovery
    # ────────────────────────────────────────────────────────────

    async def _find_anchor_nodes(
        self, query: str, folder_id: Optional[str], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find seed/anchor nodes by keyword matching.
        These serve as starting points for FastRP similarity expansion.
        """
        terms = [w.strip("?,.!()[]{}\"'").lower() for w in query.split() if len(w) > 2][:10]

        if not terms:
            return []

        async with self.driver.session() as session:
            try:
                result = await session.run("""
                    MATCH (n:Entity)
                    WHERE ($folder_id IS NULL OR n.folder_id = $folder_id)
                      AND n.fastrp_embedding IS NOT NULL
                      AND ANY(term IN $terms WHERE
                            toLower(n.name) CONTAINS term
                            OR toLower(coalesce(n.description, '')) CONTAINS term)
                    RETURN id(n)       AS neo_id,
                           n.name      AS name,
                           n.type      AS type,
                           coalesce(n.description, '') AS description,
                           n.fastrp_embedding AS embedding
                    LIMIT $limit
                """, folder_id=folder_id, terms=terms, limit=limit)
                return await result.data()
            except Exception as e:
                logger.warning(f"[FastRP] Anchor search failed: {e}")
                return []

    # ────────────────────────────────────────────────────────────
    #  FastRP Similarity Search
    # ────────────────────────────────────────────────────────────

    async def _find_similar_by_fastrp(
        self,
        anchors: List[Dict[str, Any]],
        folder_id: Optional[str],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Given anchor nodes, find structurally similar nodes using
        cosine similarity on FastRP embeddings.
        """
        if not anchors:
            return []

        anchor_names = {a["name"] for a in anchors}
        all_results = []
        seen_names = set()

        async with self.driver.session() as session:
            for anchor in anchors:
                anchor_embedding = anchor.get("embedding")
                if not anchor_embedding:
                    continue

                try:
                    # Use Neo4j's built-in cosine similarity between FastRP vectors
                    result = await session.run("""
                        MATCH (n:Entity)
                        WHERE ($folder_id IS NULL OR n.folder_id = $folder_id)
                          AND n.fastrp_embedding IS NOT NULL
                          AND n.name <> $anchor_name
                        WITH n,
                             gds.similarity.cosine(n.fastrp_embedding, $anchor_emb) AS similarity
                        WHERE similarity > 0.3
                        RETURN n.name        AS name,
                               n.type        AS type,
                               coalesce(n.description, '') AS description,
                               similarity    AS structural_score,
                               $anchor_name  AS similar_to
                        ORDER BY similarity DESC
                        LIMIT $limit
                    """,
                        folder_id=folder_id,
                        anchor_name=anchor["name"],
                        anchor_emb=anchor_embedding,
                        limit=top_k,
                    )
                    data = await result.data()

                    for r in data:
                        key = r["name"].lower()
                        if key not in seen_names and r["name"] not in anchor_names:
                            all_results.append(r)
                            seen_names.add(key)

                except Exception as e:
                    logger.warning(f"[FastRP] Similarity search failed for anchor '{anchor['name']}': {e}")
                    continue

        # Also include anchors themselves in context (they were directly matched)
        for a in anchors:
            key = a["name"].lower()
            if key not in seen_names:
                all_results.append({
                    "name": a["name"],
                    "type": a.get("type", ""),
                    "description": a.get("description", ""),
                    "structural_score": 1.0,
                    "similar_to": "query",
                })
                seen_names.add(key)

        # Sort by structural score
        all_results.sort(key=lambda x: x.get("structural_score", 0), reverse=True)
        return all_results[:top_k]

    # ────────────────────────────────────────────────────────────
    #  Top Structural Nodes (no anchor needed)
    # ────────────────────────────────────────────────────────────

    async def _get_top_structural_nodes(
        self, folder_id: Optional[str], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Return the most structurally important nodes (highest degree) as generic context."""
        async with self.driver.session() as session:
            try:
                result = await session.run("""
                    MATCH (n:Entity)
                    WHERE ($folder_id IS NULL OR n.folder_id = $folder_id)
                    WITH n, COUNT { (n)--() } AS degree
                    WHERE degree > 0
                    RETURN n.name      AS name,
                           n.type      AS type,
                           coalesce(n.description, '') AS description,
                           degree      AS structural_score,
                           'hub node'  AS similar_to
                    ORDER BY degree DESC
                    LIMIT $top_k
                """, folder_id=folder_id, top_k=top_k)
                return await result.data()
            except Exception as e:
                logger.warning(f"[FastRP] Top structural nodes failed: {e}")
                return []

    # ────────────────────────────────────────────────────────────
    #  Fallback (no GDS)
    # ────────────────────────────────────────────────────────────

    async def _structural_fallback(
        self, query: str, folder_id: Optional[str], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fallback structural search when GDS is not available.
        Uses 2-hop neighborhood expansion from keyword-matched nodes.
        """
        terms = [w.strip("?,.!()[]{}\"'").lower() for w in query.split() if len(w) > 2][:8]
        if not terms:
            return await self._get_top_structural_nodes(folder_id, top_k)

        async with self.driver.session() as session:
            try:
                result = await session.run("""
                    MATCH (anchor:Entity)
                    WHERE ($folder_id IS NULL OR anchor.folder_id = $folder_id)
                      AND ANY(term IN $terms WHERE
                            toLower(anchor.name) CONTAINS term
                            OR toLower(coalesce(anchor.description, '')) CONTAINS term)
                    WITH anchor LIMIT 3
                    MATCH (anchor)-[r1]-(neighbor:Entity)
                    WHERE ($folder_id IS NULL OR neighbor.folder_id = $folder_id)
                    OPTIONAL MATCH (neighbor)-[r2]-(hop2:Entity)
                    WHERE ($folder_id IS NULL OR hop2.folder_id = $folder_id)
                      AND hop2 <> anchor
                    WITH DISTINCT
                         coalesce(hop2.name, neighbor.name) AS name,
                         coalesce(hop2.type, neighbor.type) AS type,
                         coalesce(hop2.description, neighbor.description, '') AS description,
                         CASE WHEN hop2 IS NOT NULL THEN 0.6 ELSE 0.9 END AS structural_score,
                         anchor.name AS similar_to
                    WHERE name IS NOT NULL
                    RETURN name, type, description, structural_score, similar_to
                    ORDER BY structural_score DESC
                    LIMIT $top_k
                """, folder_id=folder_id, terms=terms, top_k=top_k)
                return await result.data()
            except Exception as e:
                logger.warning(f"[FastRP] Structural fallback failed: {e}")
                return []
