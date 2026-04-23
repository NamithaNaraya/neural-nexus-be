"""
Combined Chat — Embedding & Vector Search Service

Multi-layer retrieval with proper folder isolation:
  Layer 1: Vector search (over-fetch globally, post-filter by folder)
  Layer 2: Lexical keyword search (folder-scoped)
  Layer 3: Fulltext index fallback (folder-scoped)
  Layer 4: Direct folder scan (last resort)
"""
import logging
from typing import List, Dict, Any, Optional
from app.services.ai_service import get_ai_service
from app.db.connections import get_neo4j_driver
from app.core.config import settings

logger = logging.getLogger(__name__)

# How many candidates to fetch from the vector index BEFORE folder filtering.
# Must be large enough so that after post-filtering by folder, we still have results.
_VECTOR_OVER_FETCH = 50
# Final result count returned to the caller
_MAX_RESULTS = 15


class EmbeddingService:
    def __init__(self):
        self.ai = get_ai_service()
        self.neo4j = get_neo4j_driver()

    async def vector_search(self, query: str, folder_id: str = None) -> List[Dict[str, Any]]:
        """
        Multi-layer semantic + lexical search with strict folder isolation.

        Returns up to _MAX_RESULTS enriched node dicts, sorted by relevance.
        """
        # 1. Generate embedding for the query
        try:
            emb = await self.ai.embed(query)
        except Exception as e:
            logger.error(f"[EmbeddingService] Failed to embed query: {e}")
            emb = None

        # Extract search terms for keyword-based layers
        terms = [w.strip("?,.!()[]{}\"'").lower() for w in query.split() if len(w) > 2][:12]

        all_results: List[Dict[str, Any]] = []
        seen_names: set = set()

        async with self.neo4j.session() as session:

            # ── Layer 1: Vector Search (Over-fetch → Post-filter) ──────────
            if emb:
                try:
                    vector_query = f"""
                        CALL db.index.vector.queryNodes('{settings.VECTOR_INDEX_NAME}', $top_k, $emb)
                        YIELD node, score
                        WHERE node.name IS NOT NULL
                          AND ($folder_id IS NULL OR node.folder_id = $folder_id)
                        RETURN node.name       AS name,
                               coalesce(node.description, node.text, '') AS text,
                               labels(node)    AS labels,
                               score,
                               coalesce(node.id, elementId(node)) AS node_id
                        LIMIT {_MAX_RESULTS}
                    """
                    res = await session.run(vector_query, emb=emb, folder_id=folder_id, top_k=_VECTOR_OVER_FETCH)
                    vec_data = await res.data()
                    for r in vec_data:
                        key = r["name"].lower()
                        if key not in seen_names:
                            all_results.append(r)
                            seen_names.add(key)
                    logger.info(f"[EmbeddingService] Layer 1 (Vector): {len(vec_data)} results for folder {folder_id}")
                except Exception as e:
                    logger.warning(f"[EmbeddingService] Layer 1 (Vector) failed: {e}")

            # ── Layer 2: Lexical Keyword Search (ALL properties) ──────────
            if terms:
                try:
                    # Search across ALL string properties dynamically — not just name/description.
                    # This catches commonName, scientificName, synonyms, origin, family, etc.
                    _SKIP_PROPS = ['id', 'embedding', 'folder_id', 'file_id', 'fastrp_embedding',
                                   'created_at', 'updated_at', 'source_count']
                    lexical_query = """
                        MATCH (node:Entity)
                        WHERE node.name IS NOT NULL
                          AND ($folder_id IS NULL OR node.folder_id = $folder_id)
                          AND ANY(term IN $terms WHERE
                                toLower(node.name) CONTAINS term
                                OR toLower(coalesce(node.description, '')) CONTAINS term
                                OR ANY(key IN keys(node) WHERE
                                    NOT key IN $skip_props
                                    AND toLower(toString(node[key])) CONTAINS term
                                )
                          )
                        RETURN node.name       AS name,
                               coalesce(node.description, node.text, '') AS text,
                               labels(node)    AS labels,
                               0.75            AS score,
                               coalesce(node.id, elementId(node)) AS node_id
                        LIMIT 15
                    """
                    res = await session.run(lexical_query, folder_id=folder_id, terms=terms, skip_props=_SKIP_PROPS)
                    lex_data = await res.data()
                    for r in lex_data:
                        key = r["name"].lower()
                        if key not in seen_names:
                            all_results.append(r)
                            seen_names.add(key)
                    logger.info(f"[EmbeddingService] Layer 2 (Lexical All-Props): {len(lex_data)} results")
                except Exception as e:
                    logger.warning(f"[EmbeddingService] Layer 2 (Lexical) failed: {e}")

            # ── Layer 3: Fulltext Index Fallback ───────────────────────────
            if len(all_results) < 3:
                try:
                    res = await session.run("""
                        CALL db.index.fulltext.queryNodes('entity_search', $search_text)
                        YIELD node, score
                        WHERE ($folder_id IS NULL OR node.folder_id = $folder_id)
                        RETURN node.name       AS name,
                               coalesce(node.description, node.text, '') AS text,
                               labels(node)    AS labels,
                               score * 0.6     AS score,
                               coalesce(node.id, elementId(node)) AS node_id
                        LIMIT 10
                    """, search_text=query, folder_id=folder_id)
                    ft_data = await res.data()
                    for r in ft_data:
                        key = r["name"].lower()
                        if key not in seen_names:
                            all_results.append(r)
                            seen_names.add(key)
                    logger.info(f"[EmbeddingService] Layer 3 (Fulltext): {len(ft_data)} results")
                except Exception as e:
                    logger.warning(f"[EmbeddingService] Layer 3 (Fulltext) failed: {e}")

            # ── Layer 4: Direct Folder Scan (last resort) ──────────────────
            if not all_results and folder_id:
                try:
                    res = await session.run("""
                        MATCH (node:Entity {folder_id: $folder_id})
                        WHERE node.name IS NOT NULL
                        RETURN node.name       AS name,
                               coalesce(node.description, node.text, '') AS text,
                               labels(node)    AS labels,
                               0.5             AS score,
                               coalesce(node.id, elementId(node)) AS node_id
                        ORDER BY node.source_count DESC
                        LIMIT 20
                    """, folder_id=folder_id)
                    scan_data = await res.data()
                    all_results.extend(scan_data)
                    logger.info(f"[EmbeddingService] Layer 4 (Folder Scan): {len(scan_data)} results")
                except Exception as e:
                    logger.warning(f"[EmbeddingService] Layer 4 (Folder Scan) failed: {e}")

        # ── Sort by relevance score and return ─────────────────────────────
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        final = all_results[:_MAX_RESULTS]
        logger.info(f"[EmbeddingService] Final retrieval: {len(final)} unique nodes for folder {folder_id or 'global'}")
        return final
