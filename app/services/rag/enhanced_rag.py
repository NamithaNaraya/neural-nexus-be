"""
Enhanced RAG Service — LangGraph Orchestrator with 10 Intelligence Features

Pipeline:
  load_context → smart_scope → clarification_check
    → [if vague] → clarification_response → END
    → [if clear] → vector_search → ml_enrichment → strategic_scout
      → graph_expansion → prediction_injection → generate_answer → END

Features:
  1. Context Grounding       — Forces LLM to stick to database evidence
  2. Conversational Ask-Back — Asks clarifying questions for vague queries
  3. ML-Powered Retrieval    — Uses GDS Node Similarity for structural search
  4. Link Prediction Context — Injects trained LP predictions into answers
  5. Node Classification     — Auto-labels unknown nodes via trained NC model
  6. Similarity Expansion    — Expands context to structurally similar nodes
  7. Smart Scoping           — Auto-detects relevant scope from question
  8. Citation Reranking      — Ranks citations by relevance × centrality
  9. Multi-Turn Memory       — Compresses old history into summaries
  10. Answer Confidence      — Returns a grounding score for every answer
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from app.core.config import settings
from app.core.prompts import get_enhanced_rag_system_prompt, get_greeting_prompt
from app.services.gds_service import get_gds_service

logger = logging.getLogger(__name__)

SLIDING_WINDOW = 5


# ═══════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════

class EnhancedRAGState(TypedDict):
    # ── Input ──
    question: str
    session_id: str
    scope: Optional[Dict[str, Any]]
    history: List[Dict[str, str]]

    # ── Phase 1: Conversation ──
    enhanced_question: str
    conversation_summary: str       # Feature 9
    auto_scope: Optional[Dict]      # Feature 7
    needs_clarification: bool       # Feature 2
    clarification_question: str     # Feature 2

    # ── Phase 2: Retrieval ──
    vector_results: List[Dict[str, Any]]
    ml_similar_nodes: List[Dict[str, Any]]    # Feature 3 + 6

    # ── Phase 3: Reasoning ──
    strategic_results: List[Dict[str, Any]]
    graph_context: Dict[str, Any]
    prediction_context: List[Dict[str, Any]]  # Feature 4 + 5

    # ── Phase 4: Generation ──
    answer: str
    citations: List[Dict[str, Any]]
    related_nodes: List[str]
    grounding_score: float          # Feature 10
    schema: Optional[Dict[str, Any]]
    error: Optional[str]


# ═══════════════════════════════════════════════════════════════
#  SERVICE
# ═══════════════════════════════════════════════════════════════

class EnhancedRAGService:
    """
    Enhanced Hybrid RAG with 10 intelligence features,
    orchestrated by LangGraph.
    """

    def __init__(self, neo4j_driver, ai_service):
        self.neo4j = neo4j_driver
        self.ai = ai_service
        self.schema_cache = None
        self._gds_projection_name = None      # cached graph name
        self._model_catalog_cache = None      # cached model list
        self._model_catalog_ts = 0            # cache timestamp
        self.graph = self._build_graph()

    # ─── Pipeline ────────────────────────────────────────

    def _build_graph(self):
        wf = StateGraph(EnhancedRAGState)

        wf.add_node("load_context",          self._load_context_node)
        wf.add_node("smart_scope",           self._smart_scope_node)
        wf.add_node("clarification_check",   self._clarification_node)
        wf.add_node("clarification_response",self._clarification_response_node)
        wf.add_node("vector_search",         self._vector_search_node)
        wf.add_node("graph_expansion",       self._graph_expansion_node)
        wf.add_node("generate_answer",       self._answer_generation_node)

        wf.set_entry_point("load_context")
        wf.add_edge("load_context", "smart_scope")
        wf.add_edge("smart_scope", "clarification_check")

        # Feature 2: conditional — ask-back or continue
        wf.add_conditional_edges(
            "clarification_check",
            self._should_clarify,
            {"clarify": "clarification_response", "continue": "vector_search"},
        )
        wf.add_edge("clarification_response", END)

        # Optimized pipeline: vector_search → graph_expansion → generate_answer
        # (removed strategic_scout LLM call, merged ml/prediction into graph_expansion)
        wf.add_edge("vector_search", "graph_expansion")
        wf.add_edge("graph_expansion", "generate_answer")
        wf.add_edge("generate_answer", END)

        return wf.compile()

    def _should_clarify(self, state: EnhancedRAGState) -> str:
        return "clarify" if state.get("needs_clarification") else "continue"

    # ═══════════════════════════════════════════════════════
    #  PHASE 1: Context + Conversation
    # ═══════════════════════════════════════════════════════

    async def _load_context_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """Load history + Feature 9: compress old history into summary."""
        logger.info("[EnhancedRAG] Entering _load_context_node")
        history = state.get("history", [])
        summary = ""

        # Feature 9: Multi-Turn Memory Compression
        if len(history) > SLIDING_WINDOW * 2:
            old_messages = history[: -(SLIDING_WINDOW * 2)]
            recent_messages = history[-(SLIDING_WINDOW * 2):]

            old_text = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')[:200]}"
                for m in old_messages
            )
            try:
                summary = await self.ai.chat([
                    {"role": "system", "content": "Compress this conversation into 2-3 key facts. Be extremely brief."},
                    {"role": "user", "content": old_text},
                ])
                logger.info(f"[RAG] Compressed {len(old_messages)} old messages into summary")
            except Exception:
                summary = ""
            history = recent_messages

        # Build enhanced question with context
        context_parts = []
        if summary:
            context_parts.append(f"[Previous conversation summary: {summary}]")
        for m in history[-4:]:
            context_parts.append(f"{m.get('role', 'user')}: {m.get('content', '')[:150]}")
        context_parts.append(f"Current question: {state['question']}")
        enhanced = "\n".join(context_parts)

        return {
            "enhanced_question": enhanced,
            "conversation_summary": summary,
            "history": history,
        }

    async def _smart_scope_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """Feature 7: Auto-detect scope from question if not provided."""
        logger.info("[EnhancedRAG] Entering _smart_scope_node")
        if state.get("scope"):
            logger.info(f"[EnhancedRAG] Scope already provided: {state['scope']}")
            return {"auto_scope": state["scope"]}

        question_lower = state["question"].lower()

        # Check if question mentions a known folder name
        try:
            async with self.neo4j.session() as session:
                result = await session.run("""
                    MATCH (n:Entity)
                    WITH DISTINCT n.folder_id AS fid
                    WHERE fid IS NOT NULL
                    RETURN fid
                    LIMIT 10
                """)
                folders = await result.data()

                # If only one folder exists, auto-scope to it
                if len(folders) == 1:
                    auto = {"type": "folder", "id": folders[0]["fid"]}
                    logger.info(f"[RAG] Feature 7: Auto-scoped to single folder: {auto['id'][:8]}...")
                    return {"auto_scope": auto, "scope": auto}

                # If multiple folders, try to match question keywords
                if len(folders) > 1:
                    # Get folder entity counts to pick the most relevant
                    for f in folders:
                        res = await session.run(
                            "MATCH (n:Entity {folder_id: $fid}) "
                            "WHERE ANY(term IN $terms WHERE toLower(n.name) CONTAINS term) "
                            "RETURN count(n) AS hits",
                            fid=f["fid"],
                            terms=[w.strip("?,!.").lower() for w in state["question"].split() if len(w) > 2][:8],
                        )
                        data = await res.data()
                        f["hits"] = data[0]["hits"] if data else 0

                    best = max(folders, key=lambda x: x["hits"])
                    if best["hits"] > 0:
                        auto = {"type": "folder", "id": best["fid"]}
                        logger.info(f"[RAG] Feature 7: Auto-scoped to best folder ({best['hits']} hits)")
                        return {"auto_scope": auto, "scope": auto}

        except Exception as e:
            logger.warning(f"[RAG] Smart scope failed: {e}")

        return {"auto_scope": None}

    async def _clarification_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """Feature 2: Detect vague queries and ask for clarification."""
        logger.info("[EnhancedRAG] Entering _clarification_node")
        question = state["question"].strip()
        words = [w for w in question.split() if len(w) > 1]

        # Greetings — never need clarification
        greetings = {"hi", "hello", "hey", "howdy", "thanks", "thank"}
        if any(w.lower() in greetings for w in words):
            return {"needs_clarification": False, "clarification_question": ""}

        # Check if question is too vague (< 3 substantial words, no specific entity)
        is_vague = len(words) < 3

        if is_vague:
            # Find top entities to suggest
            try:
                async with self.neo4j.session() as session:
                    result = await session.run("""
                        MATCH (n:Entity)
                        WHERE n.name IS NOT NULL AND n.type IS NOT NULL
                        WITH n.type AS type, collect(DISTINCT n.name)[..5] AS examples
                        RETURN type, examples
                        LIMIT 5
                    """)
                    records = await result.data()

                    if records:
                        suggestions = []
                        for r in records:
                            names = ", ".join(r["examples"][:3])
                            suggestions.append(f"**{r['type']}**: {names}")

                        # Build a dynamic example using the first real entity name
                        example_name = records[0]["examples"][0] if records and records[0].get("examples") else "an entity"
                        example_type = records[0]["type"] if records else "type"
                        clarification = (
                            f"I'd love to help! Your question \"{question}\" is a bit broad. "
                            f"Could you be more specific? Here's what I know about:\n\n"
                            + "\n".join(f"- {s}" for s in suggestions)
                            + f"\n\nFor example, try: *\"What are the properties of {example_name}?\"*"
                        )
                        return {"needs_clarification": True, "clarification_question": clarification}
            except Exception as e:
                logger.warning(f"[RAG] Clarification check failed: {e}")

        return {"needs_clarification": False, "clarification_question": ""}

    async def _clarification_response_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """Return clarification question as the answer."""
        return {
            "answer": state.get("clarification_question", "Could you be more specific?"),
            "citations": [],
            "related_nodes": [],
            "grounding_score": 1.0,
        }

    # ═══════════════════════════════════════════════════════
    #  PHASE 2: Retrieval
    # ═══════════════════════════════════════════════════════

    async def _vector_search_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """
        Hybrid search: Lexical (fast) → Vector (semantic, only if needed) → Relationship-aware.
        Feature 8: Rerank by relevance × centrality.
        Optimized: Skips expensive embedding if lexical search finds enough results.
        """
        logger.info(f"[EnhancedRAG] Entering _vector_search_node for: '{state['question'][:60]}...'")

        params = {
            "terms": [w.strip("?,.!").lower() for w in state["question"].split() if len(w) > 2][:10],
            "top_k": 15,
        }

        # Scope filter
        scope = state.get("scope") or state.get("auto_scope")
        scope_filter = ""
        if scope:
            if scope.get("type") == "folder":
                scope_filter = "AND (node.folder_id IS NULL OR node.folder_id = $scope_id)"
                params["scope_id"] = scope["id"]
            elif scope.get("type") == "file":
                scope_filter = "AND (node.file_id IS NULL OR node.file_id = $scope_id)"
                params["scope_id"] = scope["id"]

        all_results = []
        seen = set()

        # ── A. Lexical search FIRST (fast, no embedding needed) ──
        try:
            async with self.neo4j.session() as session:
                res = await session.run(f"""
                    MATCH (node)
                    WHERE (node.name IS NOT NULL OR node.text IS NOT NULL OR node.title IS NOT NULL)
                    AND ANY(term IN $terms WHERE toLower(COALESCE(node.name, node.text, node.title, '')) CONTAINS term
                            OR toLower(COALESCE(node.description,'')) CONTAINS term)
                    {scope_filter}
                    RETURN COALESCE(node.id, elementId(node)) AS node_id,
                           COALESCE(node.name, node.text, node.title, '') AS name,
                           COALESCE(node.description, '') AS description,
                           COALESCE(node.type, labels(node)[0]) AS type,
                           0.85 AS score
                    LIMIT 20
                """, params)
                for r in await res.data():
                    if r["node_id"] not in seen:
                        all_results.append(r)
                        seen.add(r["node_id"])
        except Exception as e:
            logger.warning(f"[RAG] Lexical search failed: {e}")

        logger.info(f"[RAG] Lexical search found {len(all_results)} results")

        # ── B. Vector search ONLY if lexical didn't find enough ──
        if len(all_results) < 3:
            try:
                logger.info("[EnhancedRAG] Lexical insufficient, generating embedding via Ollama...")
                question_embedding = await self.ai.embed(state["question"])
                params["embedding"] = question_embedding
                logger.info(f"[EnhancedRAG] Embedding generated ({len(question_embedding)} dims)")

                async with self.neo4j.session() as session:
                    res = await session.run(f"""
                        CALL db.index.vector.queryNodes('{settings.VECTOR_INDEX_NAME}', $top_k, $embedding) YIELD node, score
                        WHERE (node.name IS NOT NULL OR node.text IS NOT NULL OR node.title IS NOT NULL) {scope_filter}
                        RETURN COALESCE(node.id, elementId(node)) AS node_id,
                               COALESCE(node.name, node.text, node.title, '') AS name,
                               COALESCE(node.description, '') AS description,
                               COALESCE(node.type, labels(node)[0]) AS type,
                               score
                    """, params)
                    for r in await res.data():
                        if r["node_id"] not in seen:
                            all_results.append(r)
                            seen.add(r["node_id"])
            except Exception as e:
                logger.warning(f"[RAG] Vector search failed: {e}")
        else:
            logger.info("[RAG] Skipping vector search — lexical found enough results")

        # ── C. Relationship-aware search (only if still need more) ──
        if len(all_results) < 8:
            try:
                async with self.neo4j.session() as session:
                    # REMOVED scope_filter_c: We want to find nodes connected to our terms
                    # regardless of the connected node's folder. If they are connected to
                    # something relevant, they are relevant.
                    res = await session.run(f"""
                        MATCH (node)-[r]-(connected)
                        WHERE node.name IS NOT NULL AND connected.name IS NOT NULL
                        AND ANY(term IN $terms WHERE toLower(connected.name) CONTAINS term
                                OR toLower(COALESCE(connected.description,'')) CONTAINS term)
                        WITH DISTINCT node, max(0.80) AS score
                        RETURN COALESCE(node.id, elementId(node)) AS node_id,
                               node.name AS name,
                               COALESCE(node.description, '') AS description,
                               COALESCE(node.type, labels(node)[0]) AS type,
                               score
                        LIMIT 15
                    """, params)
                    for r in await res.data():
                        if r["node_id"] not in seen:
                            all_results.append(r)
                            seen.add(r["node_id"])
            except Exception as e:
                logger.warning(f"[RAG] Relationship-aware search failed: {e}")

        # Feature 8: Rerank by relevance × graph centrality
        if all_results:
            try:
                node_ids = [r["node_id"] for r in all_results]
                async with self.neo4j.session() as session:
                    res = await session.run("""
                        UNWIND $ids AS nid
                        MATCH (n) WHERE n.id = nid OR elementId(n) = nid
                        OPTIONAL MATCH (n)-[r]-()
                        RETURN COALESCE(n.id, elementId(n)) AS node_id, count(r) AS degree
                    """, ids=node_ids)
                    degrees = {r["node_id"]: r["degree"] for r in await res.data()}

                max_deg = max(degrees.values()) if degrees else 1
                for r in all_results:
                    deg = degrees.get(r["node_id"], 0)
                    r["centrality"] = deg / max(max_deg, 1)
                    r["combined_score"] = r["score"] * 0.7 + r["centrality"] * 0.3

                all_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            except Exception as e:
                logger.warning(f"[RAG] Citation reranking failed: {e}")
                all_results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"[RAG] Retrieval: {len(all_results)} results (optimized pipeline)")
        return {"vector_results": all_results[:25]}

    # ═══════════════════════════════════════════════════════
    #  PHASE 3: Reasoning (Graph Expansion + ML + Predictions — single fast step)
    # ═══════════════════════════════════════════════════════

    async def _graph_expansion_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """
        Combined: Deep graph expansion + ML enrichment + prediction injection.
        All in one step to avoid sequential overhead.
        """
        logger.info(f"[EnhancedRAG] Entering _graph_expansion_node with {len(state.get('vector_results', []))} vector results")
        import asyncio
        import time as _time

        if not state.get("vector_results"):
            return {
                "graph_context": {"nodes": [], "relationships": [], "backbone_types": [], "entity_profiles": []},
                "ml_similar_nodes": [],
                "prediction_context": [],
            }

        node_ids = [r["node_id"] for r in state["vector_results"][:15]]
        seed_ids = [r["node_id"] for r in state["vector_results"][:5]]
        seed_names = [r["name"] for r in state["vector_results"][:10]]

        scope = state.get("scope") or state.get("auto_scope")
        scope_filter = ""
        scope_filter_2hop = ""
        params: Dict[str, Any] = {"node_ids": node_ids, "sid": None}
        if scope:
            params["sid"] = scope.get("id")
            if scope.get("type") == "folder":
                # Lenient filter: include nodes WITHOUT folder_id (they're shared/global nodes
                # like questions, exams, properties that are part of the graph but weren't
                # tagged with a folder). Seed nodes are already scoped via vector search.
                scope_filter = "AND (related.folder_id IS NULL OR related.folder_id = $sid)"
                scope_filter_2hop = "AND (hop2.folder_id IS NULL OR hop2.folder_id = $sid)"
            elif scope.get("type") == "file":
                scope_filter = "AND (related.file_id IS NULL OR $sid IN related.file_ids OR related.file_id = $sid)"
                scope_filter_2hop = "AND (hop2.file_id IS NULL OR $sid IN hop2.file_ids OR hop2.file_id = $sid)"

        # ─── Run entity profiles, ML enrichment, and prediction check CONCURRENTLY ───
        t0 = _time.time()

        async def _get_entity_profiles():
            """Deep 2-hop entity profiles."""
            nodes = set()
            rels = []
            entity_profiles = []
            try:
                async with self.neo4j.session() as session:
                    profile_res = await session.run(f"""
                        UNWIND $node_ids AS nodeId
                        MATCH (n) WHERE n.id = nodeId OR elementId(n) = nodeId
                        OPTIONAL MATCH (n)-[r1]-(hop1)
                        WHERE hop1 IS NOT NULL
                        WITH n, nodeId, hop1, r1,
                             COALESCE(hop1.name, hop1.text, hop1.title, hop1.label, elementId(hop1)) as hop1_title
                        OPTIONAL MATCH (hop1)-[r2]-(hop2)
                        WHERE hop2 IS NOT NULL AND hop2 <> n {scope_filter_2hop}
                        WITH n,
                             collect(DISTINCT {{
                               name: hop1_title,
                               type: COALESCE(hop1.type, labels(hop1)[0]),
                               rel: type(r1),
                               desc: COALESCE(hop1.description, '')
                             }}) AS direct_connections,
                             collect(DISTINCT {{
                               from_name: hop1_title,
                               from_rel: type(r1),
                               name: COALESCE(hop2.name, hop2.text, hop2.title, hop2.label, elementId(hop2)),
                               type: COALESCE(hop2.type, labels(hop2)[0]),
                               rel: type(r2),
                               desc: COALESCE(hop2.description, '')
                             }}) AS two_hop_connections
                        RETURN COALESCE(n.name, n.text, n.title, n.label, elementId(n)) AS entity_name,
                               COALESCE(n.type, labels(n)[0]) AS entity_type,
                               COALESCE(n.description, '') AS entity_desc,
                               direct_connections[..25] AS direct,
                               two_hop_connections[..20] AS two_hop
                    """, params)

                    for rec in await profile_res.data():
                        entity_name = rec.get("entity_name")
                        if not entity_name:
                            continue
                        nodes.add(entity_name)

                        profile_lines = [f"## {entity_name} [{rec.get('entity_type', 'Entity')}]"]
                        if rec.get("entity_desc"):
                            profile_lines.append(f"   Description: {rec['entity_desc']}")

                        direct_by_rel = {}
                        for conn in rec.get("direct", []):
                            if conn.get("name"):
                                rel_type = conn.get("rel", "RELATED_TO")
                                if rel_type not in direct_by_rel:
                                    direct_by_rel[rel_type] = []
                                entry = conn["name"]
                                if conn.get("desc"):
                                    entry += f" ({conn['desc'][:80]})"
                                direct_by_rel[rel_type].append(entry)
                                nodes.add(conn["name"])
                                rels.append(f"{entity_name} -[{rel_type}]-> {conn['name']}")

                        for rel_type, items in direct_by_rel.items():
                            profile_lines.append(f"   {rel_type}: {', '.join(items)}")

                        two_hop_chains = []
                        for conn2 in rec.get("two_hop", []):
                            if conn2.get("name") and conn2.get("from_name"):
                                chain = f"{conn2['from_name']} -[{conn2.get('rel', '?')}]-> {conn2['name']}"
                                if chain not in two_hop_chains:
                                    two_hop_chains.append(chain)
                                    nodes.add(conn2["name"])

                        if two_hop_chains:
                            profile_lines.append(f"   Extended connections: {'; '.join(two_hop_chains[:12])}")

                        entity_profiles.append("\n".join(profile_lines))

                    # Backbone types (fast query)
                    res3 = await session.run("""
                        MATCH ()-[r]->()
                        WHERE ($sid IS NULL) OR r.folder_id = $sid
                        WITH type(r) AS t, count(*) AS c ORDER BY c DESC LIMIT 5
                        RETURN t AS relationshipType
                    """, params)
                    backbone = [r["relationshipType"] for r in await res3.data()]

            except Exception as e:
                logger.error(f"[RAG] Entity profiles failed: {e}")
                backbone = []

            return nodes, rels, entity_profiles, backbone

        async def _get_ml_similar():
            """ML enrichment — only if GDS projection is already cached (don't create in RAG hot path)."""
            ml_nodes = []
            # Skip if no projection is cached — creating one takes 5-15s and isn't worth it for chat
            if self._gds_projection_name is None:
                logger.debug("[RAG] ML enrichment skipped — no cached GDS projection")
                return []
            try:
                async with self.neo4j.session() as session:
                    result = await session.run(f"""
                        CALL gds.nodeSimilarity.stream('{self._gds_projection_name}', {{
                            topK: 3, similarityCutoff: 0.3
                        }})
                        YIELD node1, node2, similarity
                        WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
                        WHERE n1.id IN $seeds OR n2.id IN $seeds
                        WITH CASE WHEN n1.id IN $seeds THEN n2 ELSE n1 END AS similar,
                             similarity
                        WHERE similar.id IS NOT NULL AND NOT similar.id IN $seeds
                        RETURN DISTINCT similar.id AS node_id, similar.name AS name,
                               similar.type AS type,
                               COALESCE(similar.description, '') AS description,
                               similarity
                        ORDER BY similarity DESC
                        LIMIT 5
                    """, seeds=seed_ids)
                    ml_nodes = await result.data()
            except Exception as e:
                logger.warning(f"[RAG] ML enrichment skipped: {e}")
            return ml_nodes

        async def _get_predictions():
            """Prediction context — only if we already have cached model catalog with LP models."""
            predictions = []
            # Skip the expensive gds.model.list() call if we haven't cached models yet
            # or if the last check found no LP models
            if self._model_catalog_cache is not None and len(self._model_catalog_cache) == 0:
                return []  # Previously checked, no models exist
            try:
                now = _time.time()
                if self._model_catalog_cache is None or (now - self._model_catalog_ts) > 300:
                    # First-time check or refresh every 5 minutes (was 60s)
                    async with self.neo4j.session() as session:
                        models = await session.run(
                            "CALL gds.model.list() YIELD modelName, modelType "
                            "RETURN modelName, modelType"
                        )
                        self._model_catalog_cache = await models.data()
                        self._model_catalog_ts = now

                model_data = self._model_catalog_cache or []
                lp_models = [m for m in model_data if "Link" in str(m.get("modelType", ""))]

                if lp_models:
                    async with self.neo4j.session() as session:
                        pred_res = await session.run("""
                            MATCH (a:Entity)-[r:PREDICTED_LINK]-(b:Entity)
                            WHERE a.name IN $names OR b.name IN $names
                            RETURN a.name AS source, b.name AS target,
                                   COALESCE(r.probability, 0.5) AS probability
                            LIMIT 5
                        """, names=seed_names)
                        for p in await pred_res.data():
                            predictions.append({
                                "type": "link_prediction",
                                "detail": f"AI predicts: {p['source']} ↔ {p['target']} ({p['probability']*100:.0f}% confidence)",
                            })
            except Exception as e:
                logger.debug(f"[RAG] Predictions skipped: {e}")
            return predictions

        # ─── Run entity profiles ALWAYS + ML/predictions only if fast ───
        (nodes, rels, entity_profiles, backbone), ml_nodes, predictions = await asyncio.gather(
            _get_entity_profiles(),
            _get_ml_similar(),
            _get_predictions(),
        )

        elapsed = _time.time() - t0
        logger.info(
            f"[RAG] Graph expansion: {len(entity_profiles)} profiles, "
            f"{len(ml_nodes)} ML nodes, {len(predictions)} predictions in {elapsed:.1f}s"
        )

        return {
            "graph_context": {
                "nodes": list(nodes),
                "relationships": rels,
                "backbone_types": backbone,
                "entity_profiles": entity_profiles,
            },
            "ml_similar_nodes": ml_nodes,
            "prediction_context": predictions,
        }


    # ═══════════════════════════════════════════════════════
    #  PHASE 4: Answer Generation
    # ═══════════════════════════════════════════════════════

    async def _answer_generation_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """
        Generate answer with:
          Feature 1: Strict context grounding (database-only)
          Feature 10: Grounding score calculation
        """
        logger.info(f"[EnhancedRAG] Entering _answer_generation_node with {len(state.get('vector_results', []))} vector results")
        # ── Build context ──
        context_parts = []

        # Conversation summary (Feature 9)
        if state.get("conversation_summary"):
            context_parts.append(f"[Previous conversation context: {state['conversation_summary']}]")

        # Entity Profiles (the most important part — full structured data per entity)
        entity_profiles = state.get("graph_context", {}).get("entity_profiles", [])
        if entity_profiles:
            context_parts.append("═══ COMPLETE ENTITY PROFILES FROM DATABASE ═══")
            for profile in entity_profiles:
                context_parts.append(profile)

        # Vector results (basic list)
        context_parts.append("\n═══ DATABASE EVIDENCE (Matched Entities) ═══")
        entity_names = []
        for r in state["vector_results"][:15]:
            name = r.get("name") or ""
            type_label = r.get("type") or "Entity"
            desc = r.get("description", "")[:500]
            centrality_note = f" (centrality: {r.get('centrality', 0):.2f})" if r.get("centrality") else ""
            context_parts.append(f"• {name} [{type_label}]{centrality_note}: {desc}")
            if name:
                entity_names.append(name.lower())

        # Also add entity names from profiles to the grounding check
        for profile in entity_profiles:
            for line in profile.split("\n"):
                if line.startswith("## "):
                    pname = line.replace("## ", "").split(" [")[0].strip().lower()
                    if pname and pname not in entity_names:
                        entity_names.append(pname)

        # ML similar nodes (Feature 3+6)
        if state.get("ml_similar_nodes"):
            context_parts.append("\n═══ STRUCTURALLY SIMILAR NODES (ML Discovery) ═══")
            for n in state["ml_similar_nodes"][:5]:
                ml_name = n.get("name") or ""
                context_parts.append(
                    f"• {ml_name} [{n.get('type', 'Entity')}] — "
                    f"{n.get('similarity', 0)*100:.0f}% structural similarity"
                )
                if ml_name:
                    entity_names.append(ml_name.lower())

        # Strategic insights
        if state.get("strategic_results"):
            context_parts.append("\n═══ STRATEGIC STRUCTURAL INSIGHTS ═══")
            context_parts.append(json.dumps(state["strategic_results"][:8], indent=2, default=str))

        # Graph relationships
        if state.get("graph_context", {}).get("relationships"):
            context_parts.append("\n═══ GRAPH CONNECTIONS (All Relationship Paths) ═══")
            # Deduplicate and show up to 40 relationships
            seen_rels = set()
            for rel in state["graph_context"]["relationships"][:40]:
                if rel not in seen_rels:
                    context_parts.append(f"• {rel}")
                    seen_rels.add(rel)

        # Backbone
        backbone = ", ".join(state.get("graph_context", {}).get("backbone_types", []))
        if backbone:
            context_parts.append(f"\n═══ DOMAIN STRUCTURE ═══\nKey relationship types: {backbone}")

        # Prediction insights (Feature 4+5)
        if state.get("prediction_context"):
            context_parts.append("\n═══ AI PREDICTIONS (Machine Learning) ═══")
            for p in state["prediction_context"]:
                context_parts.append(f"• {p['detail']}")

        full_context = "\n".join(context_parts)

        # Feature 1: Strict database grounding + natural language response
        system_prompt = get_enhanced_rag_system_prompt()

        user_prompt = f"Database Evidence & Knowledge Graph Data:\n{full_context}\n\nUser's Question: {state['question']}"

        try:
            answer = await self.ai.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

            # Feature 10: Compute grounding score
            answer = answer or ""
            answer_lower = answer.lower()
            mentioned = sum(1 for name in entity_names if name in answer_lower)
            grounding_score = min(mentioned / max(len(entity_names), 1), 1.0)
            grounding_score = round(grounding_score, 2)

            # ── ANTI-HALLUCINATION GATE ──
            # Only trigger when there's truly NO relevant data at all (<10%).
            if grounding_score < 0.10 and entity_names:
                available_items = ", ".join(
                    f"**{r['name']}**" for r in state["vector_results"][:8]
                )
                answer = (
                    f"I found some related entries in the knowledge graph, but couldn't build a "
                    f"detailed answer for *\"{state['question']}\"*.\n\n"
                    f"Here's what I found that might be relevant:\n"
                    f"- {available_items}\n\n"
                    f"Would you like me to explore any of these in detail?"
                )
                grounding_score = 1.0

            citations = [
                {
                    "node_id": r["node_id"],
                    "node_name": r["name"],
                    "chunk_text": r.get("description", "")[:200],
                    "confidence": r.get("combined_score", r.get("score", 0)),
                }
                for r in state["vector_results"][:15]
            ]

            return {
                "answer": answer,
                "citations": citations,
                "related_nodes": [r["node_id"] for r in state["vector_results"][:15]],
                "grounding_score": grounding_score,
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {e}",
                "citations": [],
                "grounding_score": 0.0,
            }

    # ═══════════════════════════════════════════════════════
    #  QUERY ENTRY POINT
    # ═══════════════════════════════════════════════════════

    async def query(
        self,
        question: str,
        session_id: str,
        history: List[Dict[str, str]] = None,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the enhanced LangGraph RAG pipeline."""
        initial_state: EnhancedRAGState = {
            "question": question,
            "session_id": session_id,
            "scope": scope,
            "history": history or [],
            "enhanced_question": question,
            "conversation_summary": "",
            "auto_scope": None,
            "needs_clarification": False,
            "clarification_question": "",
            "vector_results": [],
            "ml_similar_nodes": [],
            "strategic_results": [],
            "graph_context": {},
            "prediction_context": [],
            "answer": "",
            "citations": [],
            "related_nodes": [],
            "grounding_score": 0.0,
            "schema": None,
            "error": None,
        }

        # Quick-response for greetings and simple chat — no need for full RAG pipeline
        greeting_words = {"hi", "hello", "hey", "howdy", "thanks", "thank", "bye", "goodbye", "good morning", "good evening"}
        clean_q = question.strip().lower().rstrip("!.,?")
        if clean_q in greeting_words or len(clean_q.split()) <= 2 and any(g in clean_q for g in greeting_words):
            logger.info(f"[EnhancedRAG] Greeting detected: '{question}' — fast LLM response")
            try:
                answer = await self.ai.chat([
                    {"role": "system", "content": get_greeting_prompt()},
                    {"role": "user", "content": question}
                ])
                return {
                    "answer": answer,
                    "citations": [],
                    "related_nodes": [],
                    "grounding_score": 1.0,
                    "needs_clarification": False,
                    "ml_insights_count": 0,
                    "predictions_count": 0,
                }
            except Exception as e:
                logger.error(f"[EnhancedRAG] Greeting response failed: {e}")

        try:
            import asyncio
            logger.info(f"[EnhancedRAG] Starting LangGraph ainvoke for session {session_id}")
            result = await asyncio.wait_for(
                self.graph.ainvoke(initial_state),
                timeout=60.0  # 60 second timeout to prevent infinite hang
            )
            logger.info(f"[EnhancedRAG] LangGraph ainvoke completed for session {session_id}")
            return {
                "answer": result.get("answer", "No answer generated"),
                "citations": result.get("citations", []),
                "related_nodes": result.get("related_nodes", []),
                "grounding_score": result.get("grounding_score", 0.0),
                "needs_clarification": result.get("needs_clarification", False),
                "ml_insights_count": len(result.get("ml_similar_nodes", [])),
                "predictions_count": len(result.get("prediction_context", [])),
            }
        except asyncio.TimeoutError:
            logger.error(f"[EnhancedRAG] Pipeline timed out after 60s for session {session_id}")
            return {
                "answer": "I'm sorry, the query took longer than expected. Please try a simpler or more specific question.",
                "citations": [],
                "related_nodes": [],
                "grounding_score": 0.0,
            }
        except Exception as e:
            logger.error(f"Enhanced RAG pipeline failed: {e}", exc_info=True)
            return {
                "answer": f"Error: {e}",
                "citations": [],
                "related_nodes": [],
                "grounding_score": 0.0,
            }


# ═══════════════════════════════════════════════════════════════
#  SINGLETON
# ═══════════════════════════════════════════════════════════════

_service: Optional[EnhancedRAGService] = None


def get_enhanced_rag_service(neo4j, ai_service) -> EnhancedRAGService:
    global _service
    if _service is None:
        _service = EnhancedRAGService(neo4j, ai_service)
    return _service
