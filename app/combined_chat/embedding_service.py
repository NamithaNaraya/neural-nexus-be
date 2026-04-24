"""
Combined Chat - Embedding & Vector Search Service

Hybrid retrieval for combined chat:
  1. Dense vector search for semantic recall
  2. Exact-name lookup for precise entity matches
  3. Lexical all-property search for keyword grounding
  4. Fulltext fallback for broader recall
  5. Folder scan as a last-resort recovery path
"""
import logging
import re
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.db.connections import get_neo4j_driver
from app.services.ai_service import get_ai_service

logger = logging.getLogger(__name__)

_VECTOR_OVER_FETCH = 50
_MAX_RESULTS = 15


class EmbeddingService:
    def __init__(self):
        self.ai = get_ai_service()
        self.neo4j = get_neo4j_driver()

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", (value or "").lower())).strip()

    def _tokenize(self, value: str) -> List[str]:
        tokens = []
        seen = set()
        for token in self._normalize_text(value).split():
            if len(token) < 3 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens[:12]

    async def vector_search(
        self,
        query: str,
        folder_id: Optional[str] = None,
        expanded_terms: Optional[List[str]] = None,
        exact_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Multi-layer semantic + lexical search with strict folder isolation.

        Returns up to _MAX_RESULTS enriched node dicts, sorted by a lightweight
        hybrid score rather than raw source score alone.
        """
        expanded_terms = [t.strip() for t in (expanded_terms or []) if t and str(t).strip()]
        exact_names = [t.strip() for t in (exact_names or []) if t and str(t).strip()]

        enriched_query = " ".join(
            part for part in [query.strip(), " ".join(expanded_terms[:6]), " ".join(exact_names[:4])] if part
        ).strip()

        try:
            emb = await self.ai.embed(enriched_query or query)
        except Exception as e:
            logger.error(f"[EmbeddingService] Failed to embed query: {e}")
            emb = None

        terms = self._tokenize(enriched_query or query)
        search_text = " ".join(exact_names[:3] + expanded_terms[:4] + terms[:8]).strip() or (query or "").strip()

        all_results: List[Dict[str, Any]] = []
        seen_names: set[str] = set()

        def add_results(rows: List[Dict[str, Any]], source: str) -> None:
            for row in rows:
                name = str(row.get("name", "")).strip()
                if not name:
                    continue
                key = name.lower()
                if key in seen_names:
                    continue
                row["source"] = source
                row["hybrid_score"] = float(row.get("score", 0.0))
                all_results.append(row)
                seen_names.add(key)

        async with self.neo4j.session() as session:
            if emb:
                try:
                    vector_query = f"""
                        CALL db.index.vector.queryNodes('{settings.VECTOR_INDEX_NAME}', $top_k, $emb)
                        YIELD node, score
                        WHERE node.name IS NOT NULL
                          AND ($folder_id IS NULL OR node.folder_id = $folder_id)
                        RETURN
                            node.name AS name,
                            coalesce(
                                node.description,
                                node.text,
                                node.summary,
                                node.commonName,
                                node.scientificName,
                                ''
                            ) AS text,
                            labels(node) AS labels,
                            score,
                            coalesce(node.id, elementId(node)) AS node_id
                        LIMIT {_MAX_RESULTS}
                    """
                    res = await session.run(vector_query, emb=emb, folder_id=folder_id, top_k=_VECTOR_OVER_FETCH)
                    vec_data = await res.data()
                    add_results(vec_data, "vector")
                    logger.info(f"[EmbeddingService] Layer 1 (Vector): {len(vec_data)} results")
                except Exception as e:
                    logger.warning(f"[EmbeddingService] Layer 1 (Vector) failed: {e}")

            if exact_names:
                try:
                    exact_query = """
                        MATCH (node)
                        WHERE node.name IS NOT NULL
                          AND ($folder_id IS NULL OR node.folder_id = $folder_id)
                          AND toLower(node.name) IN $exact_names
                        RETURN
                            node.name AS name,
                            coalesce(
                                node.description,
                                node.text,
                                node.summary,
                                node.commonName,
                                node.scientificName,
                                ''
                            ) AS text,
                            labels(node) AS labels,
                            0.98 AS score,
                            coalesce(node.id, elementId(node)) AS node_id
                        LIMIT 10
                    """
                    res = await session.run(
                        exact_query,
                        folder_id=folder_id,
                        exact_names=[name.lower() for name in exact_names[:8]],
                    )
                    exact_data = await res.data()
                    add_results(exact_data, "exact")
                    logger.info(f"[EmbeddingService] Layer 2 (Exact): {len(exact_data)} results")
                except Exception as e:
                    logger.warning(f"[EmbeddingService] Layer 2 (Exact) failed: {e}")

            if terms:
                try:
                    lexical_query = """
                        MATCH (node)
                        WHERE node.name IS NOT NULL
                          AND ($folder_id IS NULL OR node.folder_id = $folder_id)
                          AND ANY(term IN $terms WHERE
                                toLower(node.name) CONTAINS term
                                OR toLower(coalesce(node.description, '')) CONTAINS term
                                OR toLower(coalesce(node.text, '')) CONTAINS term
                                OR toLower(coalesce(node.summary, '')) CONTAINS term
                                OR toLower(coalesce(node.commonName, '')) CONTAINS term
                                OR toLower(coalesce(node.scientificName, '')) CONTAINS term
                                OR toLower(coalesce(node.family, '')) CONTAINS term
                                OR toLower(coalesce(node.origin, '')) CONTAINS term
                                OR toLower(coalesce(node.aliases, '')) CONTAINS term
                                OR toLower(coalesce(toString(node.type), '')) CONTAINS term
                          )
                        RETURN
                            node.name AS name,
                            coalesce(
                                node.description,
                                node.text,
                                node.summary,
                                node.commonName,
                                node.scientificName,
                                ''
                            ) AS text,
                            labels(node) AS labels,
                            0.78 AS score,
                            coalesce(node.id, elementId(node)) AS node_id
                        LIMIT 15
                    """
                    res = await session.run(lexical_query, folder_id=folder_id, terms=terms)
                    lex_data = await res.data()
                    add_results(lex_data, "lexical")
                    logger.info(f"[EmbeddingService] Layer 3 (Lexical): {len(lex_data)} results")
                except Exception as e:
                    logger.warning(f"[EmbeddingService] Layer 3 (Lexical) failed: {e}")

            if len(all_results) < 5 and search_text:
                try:
                    fulltext_query = """
                        CALL db.index.fulltext.queryNodes('entity_search', $search_text)
                        YIELD node, score
                        WHERE node.name IS NOT NULL
                          AND ($folder_id IS NULL OR node.folder_id = $folder_id)
                        RETURN
                            node.name AS name,
                            coalesce(
                                node.description,
                                node.text,
                                node.summary,
                                node.commonName,
                                node.scientificName,
                                ''
                            ) AS text,
                            labels(node) AS labels,
                            score * 0.65 AS score,
                            coalesce(node.id, elementId(node)) AS node_id
                        LIMIT 12
                    """
                    res = await session.run(fulltext_query, search_text=search_text, folder_id=folder_id)
                    ft_data = await res.data()
                    add_results(ft_data, "fulltext")
                    logger.info(f"[EmbeddingService] Layer 4 (Fulltext): {len(ft_data)} results")
                except Exception as e:
                    logger.warning(f"[EmbeddingService] Layer 4 (Fulltext) failed: {e}")

            if not all_results and folder_id:
                try:
                    scan_query = """
                        MATCH (node {folder_id: $folder_id})
                        WHERE node.name IS NOT NULL
                        RETURN
                            node.name AS name,
                            coalesce(
                                node.description,
                                node.text,
                                node.summary,
                                node.commonName,
                                node.scientificName,
                                ''
                            ) AS text,
                            labels(node) AS labels,
                            0.45 AS score,
                            coalesce(node.id, elementId(node)) AS node_id
                        ORDER BY coalesce(node.source_count, 0) DESC, node.name
                        LIMIT 20
                    """
                    res = await session.run(scan_query, folder_id=folder_id)
                    scan_data = await res.data()
                    add_results(scan_data, "scan")
                    logger.info(f"[EmbeddingService] Layer 5 (Folder Scan): {len(scan_data)} results")
                except Exception as e:
                    logger.warning(f"[EmbeddingService] Layer 5 (Folder Scan) failed: {e}")

        query_terms = set(terms)
        exact_term_set = {self._normalize_text(name) for name in exact_names}
        expanded_term_set = {self._normalize_text(term) for term in expanded_terms}

        for item in all_results:
            name_text = self._normalize_text(str(item.get("name", "")))
            body_text = self._normalize_text(str(item.get("text", "")))
            lexical_hits = sum(1 for term in query_terms if term and (term in name_text or term in body_text))
            exact_boost = 0.35 if name_text in exact_term_set else 0.0
            expanded_boost = 0.1 if any(term and term in name_text for term in expanded_term_set) else 0.0
            source_boost = {
                "exact": 0.25,
                "vector": 0.18,
                "lexical": 0.14,
                "fulltext": 0.08,
                "scan": 0.0,
            }.get(str(item.get("source", "")), 0.0)
            item["hybrid_score"] = float(item.get("score", 0.0)) + (lexical_hits * 0.05) + exact_boost + expanded_boost + source_boost

        all_results.sort(key=lambda row: row.get("hybrid_score", row.get("score", 0.0)), reverse=True)
        final = all_results[:_MAX_RESULTS]
        logger.info(f"[EmbeddingService] Final retrieval: {len(final)} unique nodes for folder {folder_id or 'global'}")
        return final
