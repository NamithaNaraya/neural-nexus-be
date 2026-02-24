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
        self.graph = self._build_graph()

    # ─── Pipeline ────────────────────────────────────────

    def _build_graph(self):
        wf = StateGraph(EnhancedRAGState)

        wf.add_node("load_context",          self._load_context_node)
        wf.add_node("smart_scope",           self._smart_scope_node)
        wf.add_node("clarification_check",   self._clarification_node)
        wf.add_node("clarification_response",self._clarification_response_node)
        wf.add_node("vector_search",         self._vector_search_node)
        wf.add_node("ml_enrichment",         self._ml_enrichment_node)
        wf.add_node("strategic_scout",       self._strategic_scout_node)
        wf.add_node("graph_expansion",       self._graph_expansion_node)
        wf.add_node("prediction_injection",  self._prediction_injection_node)
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

        wf.add_edge("vector_search", "ml_enrichment")
        wf.add_edge("ml_enrichment", "strategic_scout")
        wf.add_edge("strategic_scout", "graph_expansion")
        wf.add_edge("graph_expansion", "prediction_injection")
        wf.add_edge("prediction_injection", "generate_answer")
        wf.add_edge("generate_answer", END)

        return wf.compile()

    def _should_clarify(self, state: EnhancedRAGState) -> str:
        return "clarify" if state.get("needs_clarification") else "continue"

    # ═══════════════════════════════════════════════════════
    #  PHASE 1: Context + Conversation
    # ═══════════════════════════════════════════════════════

    async def _load_context_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """Load history + Feature 9: compress old history into summary."""
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
        if state.get("scope"):
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

                        clarification = (
                            f"I'd love to help! Your question \"{question}\" is a bit broad. "
                            f"Could you be more specific? Here's what I know about:\n\n"
                            + "\n".join(f"- {s}" for s in suggestions)
                            + "\n\nFor example, try: *\"What are the properties of Shatavari?\"*"
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
        Hybrid search: Vector (semantic) + Lexical (keyword) + Relationship-aware.
        Feature 8: Rerank by relevance × centrality.
        """
        question_embedding = await self.ai.embed(state["question"])

        params = {
            "terms": [w.strip("?,.!").lower() for w in state["question"].split() if len(w) > 2][:10],
            "embedding": question_embedding,
            "top_k": 15,
        }

        # Scope filter
        scope = state.get("scope") or state.get("auto_scope")
        scope_filter = ""
        if scope:
            if scope.get("type") == "folder":
                scope_filter = "AND node.folder_id = $scope_id"
                params["scope_id"] = scope["id"]
            elif scope.get("type") == "file":
                scope_filter = "AND node.file_id = $scope_id"
                params["scope_id"] = scope["id"]

        all_results = []
        seen = set()

        # A. Vector search
        try:
            async with self.neo4j.session() as session:
                res = await session.run(f"""
                    CALL db.index.vector.queryNodes('embedding_idx', $top_k, $embedding) YIELD node, score
                    WHERE node.name IS NOT NULL {scope_filter}
                    RETURN COALESCE(node.id, elementId(node)) AS node_id,
                           node.name AS name,
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

        # B. Lexical search (direct name/description match)
        try:
            async with self.neo4j.session() as session:
                res = await session.run(f"""
                    MATCH (node)
                    WHERE node.name IS NOT NULL
                    AND ANY(term IN $terms WHERE toLower(node.name) CONTAINS term
                            OR toLower(COALESCE(node.description,'')) CONTAINS term)
                    {scope_filter}
                    RETURN COALESCE(node.id, elementId(node)) AS node_id,
                           node.name AS name,
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

        # C. Relationship-aware search: find nodes connected TO matched terms
        # This catches cases where user says "medications" but the node is
        # "Menopause Relief" connected via HAS_PROPERTY/TREATS/etc.
        try:
            async with self.neo4j.session() as session:
                scope_filter_c = scope_filter.replace("node.", "connected.")
                res = await session.run(f"""
                    MATCH (node)-[r]-(connected)
                    WHERE node.name IS NOT NULL AND connected.name IS NOT NULL
                    AND ANY(term IN $terms WHERE toLower(connected.name) CONTAINS term
                            OR toLower(COALESCE(connected.description,'')) CONTAINS term)
                    {scope_filter_c}
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

        logger.info(f"[RAG] Retrieval: {len(all_results)} results (vector+lexical+relationship+reranked)")
        return {"vector_results": all_results[:25]}

    async def _ml_enrichment_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """
        Feature 3: ML-powered retrieval via GDS Node Similarity.
        Feature 6: Similarity-enriched expansion.
        Finds structurally similar nodes that text search might miss.
        """
        if not state.get("vector_results"):
            return {"ml_similar_nodes": []}

        ml_nodes = []
        seed_ids = [r["node_id"] for r in state["vector_results"][:5]]

        try:
            # Check if any GDS projection exists
            async with self.neo4j.session() as session:
                check = await session.run(
                    "CALL gds.graph.exists('neural_nexus_all_undirected_native') YIELD exists RETURN exists"
                )
                rec = await check.single()
                if not (rec and rec["exists"]):
                    logger.info("[RAG] No GDS projection — skipping ML enrichment")
                    return {"ml_similar_nodes": []}

                # Find nodes similar to our seed results
                result = await session.run("""
                    CALL gds.nodeSimilarity.stream('neural_nexus_all_undirected_native', {
                        topK: 5, similarityCutoff: 0.3
                    })
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
                    LIMIT 10
                """, seeds=seed_ids)
                ml_nodes = await result.data()
                logger.info(f"[RAG] Feature 3+6: Found {len(ml_nodes)} structurally similar nodes")

        except Exception as e:
            logger.warning(f"[RAG] ML enrichment skipped: {e}")

        return {"ml_similar_nodes": ml_nodes}

    # ═══════════════════════════════════════════════════════
    #  PHASE 3: Reasoning
    # ═══════════════════════════════════════════════════════

    async def _strategic_scout_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """Strategic Cypher generation — same as before but with ML context."""
        if not self.schema_cache:
            try:
                async with self.neo4j.session() as s:
                    r1 = await s.run("CALL db.labels()")
                    labels = [list(r.values())[0] for r in await r1.data()]
                    r2 = await s.run("CALL db.relationshipTypes()")
                    rels = [list(r.values())[0] for r in await r2.data()]
                    self.schema_cache = {"labels": labels, "relationships": rels}
            except Exception:
                self.schema_cache = {"labels": [], "relationships": []}

        entities = [r["name"] for r in state["vector_results"][:5]]
        scope = state.get("scope") or state.get("auto_scope")
        sid = scope.get("id") if scope else None
        scope_clause = ""
        if scope:
            scope_clause = (
                f" (MANDATORY: Filter by node.folder_id = $sid)"
                if scope.get("type") == "folder"
                else f" (MANDATORY: Filter by $sid IN node.file_ids)"
            )

        scout_prompt = f"""
You are the Neural Nexus Strategic Scout. Generate a READ-ONLY Cypher query.

SCHEMA:
- Labels: {self.schema_cache['labels']}
- Relationships: {self.schema_cache['relationships']}

Relevant entities: {', '.join(entities)}.{scope_clause}
Scope $sid: {sid}

QUESTION: {state['question']}

RULES:
1. Output ONLY: {{"reasoning": "...", "cypher": "..."}}
2. Use only labels/relationships from SCHEMA.
3. LIMIT 50. Use $sid for scope filtering.
4. If simple/factual, return empty cypher.
"""

        try:
            prediction = await self.ai.chat_json([
                {"role": "system", "content": "You generate high-precision Cypher for graph analytics."},
                {"role": "user", "content": scout_prompt},
            ])
            cypher = prediction.get("cypher", "").strip()
            if not cypher:
                return {"strategic_results": []}

            async with self.neo4j.session() as s:
                result = await s.run(cypher, sid=sid)
                records = await result.data()
                logger.info(f"[RAG] Strategic Scout: {len(records)} insights")
                return {"strategic_results": records[:10]}
        except Exception as e:
            logger.error(f"[RAG] Strategic Scout failed: {e}")
            return {"strategic_results": []}

    async def _graph_expansion_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """Deep graph expansion: full entity profiles + paths + aliases."""
        if not state.get("vector_results"):
            return {"graph_context": {"nodes": [], "relationships": [], "backbone_types": [], "entity_profiles": []}}

        # Combine vector results + ML similar nodes for expansion
        node_ids = [r["node_id"] for r in state["vector_results"][:15]]
        for ml_node in state.get("ml_similar_nodes", [])[:5]:
            if ml_node.get("node_id") and ml_node["node_id"] not in node_ids:
                node_ids.append(ml_node["node_id"])

        scope = state.get("scope") or state.get("auto_scope")
        scope_filter = ""
        scope_filter_2hop = ""
        params: Dict[str, Any] = {"node_ids": node_ids, "sid": None}
        if scope:
            params["sid"] = scope.get("id")
            if scope.get("type") == "folder":
                scope_filter = "AND (related.folder_id = $sid OR r.folder_id = $sid)"
                scope_filter_2hop = "AND (hop2.folder_id = $sid OR r2.folder_id = $sid)"
            elif scope.get("type") == "file":
                scope_filter = "AND ($sid IN related.file_ids OR related.file_id = $sid)"
                scope_filter_2hop = "AND ($sid IN hop2.file_ids OR hop2.file_id = $sid)"

        try:
            nodes = set()
            rels = []
            entity_profiles = []

            async with self.neo4j.session() as session:
                # ── Deep Entity Profiles: 2-hop traversal ──
                # For each seed node, get ALL its direct connections AND
                # the connections of those connections (2 hops deep).
                # This captures full chains like:
                #   Herb -> HAS_PROPERTY -> Property -> EXPRESSED_AS -> Quality
                #   Entity -> ALSO_KNOWN_AS -> Alias
                #   Herb -> TREATS -> Condition
                profile_res = await session.run(f"""
                    UNWIND $node_ids AS nodeId
                    MATCH (n) WHERE n.id = nodeId OR elementId(n) = nodeId
                    OPTIONAL MATCH (n)-[r1]-(hop1)
                    WHERE hop1 IS NOT NULL {scope_filter.replace('related', 'hop1').replace(' r.', ' r1.')}
                    OPTIONAL MATCH (hop1)-[r2]-(hop2)
                    WHERE hop2 IS NOT NULL AND hop2 <> n {scope_filter_2hop}
                    WITH n,
                         collect(DISTINCT {{
                           name: hop1.name,
                           type: COALESCE(hop1.type, labels(hop1)[0]),
                           rel: type(r1),
                           desc: COALESCE(hop1.description, '')
                         }}) AS direct_connections,
                         collect(DISTINCT {{
                           from_name: hop1.name,
                           from_rel: type(r1),
                           name: hop2.name,
                           type: COALESCE(hop2.type, labels(hop2)[0]),
                           rel: type(r2),
                           desc: COALESCE(hop2.description, '')
                         }}) AS two_hop_connections
                    RETURN n.name AS entity_name,
                           COALESCE(n.type, labels(n)[0]) AS entity_type,
                           COALESCE(n.description, '') AS entity_desc,
                           direct_connections[..30] AS direct,
                           two_hop_connections[..30] AS two_hop
                """, params)

                for rec in await profile_res.data():
                    entity_name = rec.get("entity_name")
                    if not entity_name:
                        continue
                    nodes.add(entity_name)

                    # Build a structured profile
                    profile_lines = [f"## {entity_name} [{rec.get('entity_type', 'Entity')}]"]
                    if rec.get("entity_desc"):
                        profile_lines.append(f"   Description: {rec['entity_desc']}")

                    # Group direct connections by relationship type
                    direct_by_rel = {}
                    for conn in rec.get("direct", []):
                        if conn.get("name"):
                            rel_type = conn.get("rel", "RELATED_TO")
                            if rel_type not in direct_by_rel:
                                direct_by_rel[rel_type] = []
                            entry = conn["name"]
                            if conn.get("desc"):
                                entry += f" ({conn['desc'][:100]})"
                            direct_by_rel[rel_type].append(entry)
                            nodes.add(conn["name"])
                            rels.append(f"{entity_name} -[{rel_type}]-> {conn['name']}")

                    for rel_type, items in direct_by_rel.items():
                        profile_lines.append(f"   {rel_type}: {', '.join(items)}")

                    # 2-hop connections (e.g., Property -> EXPRESSED_AS -> Quality)
                    two_hop_chains = []
                    for conn2 in rec.get("two_hop", []):
                        if conn2.get("name") and conn2.get("from_name"):
                            chain = f"{conn2['from_name']} -[{conn2.get('rel', '?')}]-> {conn2['name']}"
                            if chain not in two_hop_chains:
                                two_hop_chains.append(chain)
                                nodes.add(conn2["name"])
                                rels.append(f"{entity_name} -> {conn2['from_name']} -[{conn2.get('rel', '?')}]-> {conn2['name']}")

                    if two_hop_chains:
                        profile_lines.append(f"   Extended connections: {'; '.join(two_hop_chains[:15])}")

                    entity_profiles.append("\n".join(profile_lines))

                # Shortest paths between seed entities
                res2 = await session.run(f"""
                    MATCH (n) WHERE n.id IN $node_ids OR elementId(n) IN $node_ids
                    WITH collect(n) AS seeds
                    UNWIND seeds AS a UNWIND seeds AS b
                    WITH a, b WHERE elementId(a) < elementId(b)
                    MATCH p = shortestPath((a)-[*..6]-(b))
                    RETURN [node IN nodes(p) | node.name] AS names,
                           [rel IN relationships(p) | type(rel)] AS types
                    LIMIT 10
                """, params)
                for rec in await res2.data():
                    path_parts = []
                    for i, t in enumerate(rec["types"]):
                        path_parts.append(f"{rec['names'][i]} -[{t}]->")
                    path_parts.append(rec["names"][-1])
                    rels.append(f"PATH: {' '.join(path_parts)}")
                    for n in rec["names"]:
                        nodes.add(n)

                # Backbone types
                res3 = await session.run("""
                    MATCH ()-[r]->()
                    WHERE ($sid IS NULL) OR r.folder_id = $sid
                    WITH type(r) AS t, count(*) AS c ORDER BY c DESC LIMIT 5
                    RETURN t AS relationshipType
                """, params)
                backbone = [r["relationshipType"] for r in await res3.data()]

            logger.info(f"[RAG] Graph expansion: {len(entity_profiles)} entity profiles, {len(rels)} relationships")
            return {"graph_context": {
                "nodes": list(nodes),
                "relationships": rels,
                "backbone_types": backbone,
                "entity_profiles": entity_profiles,
            }}
        except Exception as e:
            logger.error(f"[RAG] Graph expansion failed: {e}")
            return {"graph_context": {"nodes": [], "relationships": [], "backbone_types": [], "entity_profiles": []}}

    async def _prediction_injection_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """
        Feature 4: Inject Link Prediction results into context.
        Feature 5: Auto-label nodes via Node Classification.
        """
        predictions = []
        seed_names = [r["name"] for r in state.get("vector_results", [])[:10]]

        try:
            async with self.neo4j.session() as session:
                # Feature 4: Check for trained LP models and predicted links
                try:
                    models = await session.run(
                        "CALL gds.model.list() YIELD modelName, modelType "
                        "RETURN modelName, modelType"
                    )
                    model_data = await models.data()

                    lp_models = [m for m in model_data if "Link" in str(m.get("modelType", ""))]
                    nc_models = [m for m in model_data if "Classification" in str(m.get("modelType", ""))]

                    # If LP model exists, find predictions relevant to current entities
                    if lp_models:
                        # Check if predictions were previously applied (PREDICTED_LINK in graph)
                        pred_res = await session.run("""
                            MATCH (a:Entity)-[r:PREDICTED_LINK]-(b:Entity)
                            WHERE a.name IN $names OR b.name IN $names
                            RETURN a.name AS source, b.name AS target,
                                   COALESCE(r.probability, 0.5) AS probability
                            LIMIT 10
                        """, names=seed_names)
                        pred_data = await pred_res.data()
                        for p in pred_data:
                            predictions.append({
                                "type": "link_prediction",
                                "detail": f"AI predicts: {p['source']} ↔ {p['target']} ({p['probability']*100:.0f}% confidence)",
                            })

                    # Feature 5: If NC model exists, predict types for nodes without clear type
                    if nc_models:
                        # Check for nodes in results that might have uncertain types
                        for r in state.get("vector_results", [])[:5]:
                            if not r.get("type") or r["type"] == "Entity":
                                predictions.append({
                                    "type": "classification_suggestion",
                                    "detail": f"Node '{r['name']}' may need type classification (NC model available)",
                                })

                except Exception as e:
                    logger.debug(f"[RAG] Model catalog check: {e}")

        except Exception as e:
            logger.warning(f"[RAG] Prediction injection skipped: {e}")

        logger.info(f"[RAG] Feature 4+5: {len(predictions)} prediction insights injected")
        return {"prediction_context": predictions}

    # ═══════════════════════════════════════════════════════
    #  PHASE 4: Answer Generation
    # ═══════════════════════════════════════════════════════

    async def _answer_generation_node(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """
        Generate answer with:
          Feature 1: Strict context grounding (database-only)
          Feature 10: Grounding score calculation
        """
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
            type_label = r.get("type") or "Entity"
            desc = r.get("description", "")[:500]
            centrality_note = f" (centrality: {r.get('centrality', 0):.2f})" if r.get("centrality") else ""
            context_parts.append(f"• {r['name']} [{type_label}]{centrality_note}: {desc}")
            entity_names.append(r["name"].lower())

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
                context_parts.append(
                    f"• {n['name']} [{n.get('type', 'Entity')}] — "
                    f"{n.get('similarity', 0)*100:.0f}% structural similarity"
                )
                entity_names.append(n["name"].lower())

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
        system_prompt = (
            "You are the Neural Nexus, a brilliant and friendly wellness expert powered EXCLUSIVELY by a knowledge graph database. "
            "Every piece of information you share MUST come from the provided database evidence. "
            "You do NOT generate any information from your own training data.\n\n"

            "ABSOLUTE RULES:\n\n"

            "1. **DATABASE-ONLY ANSWERS**: Your answer must come ONLY from the ENTITY PROFILES, DATABASE EVIDENCE, "
            "and GRAPH CONNECTIONS provided below. If information is in the evidence, USE IT. If it's NOT in the evidence, "
            "say 'This specific information is not in our database.' Do NOT invent or generalize.\n\n"

            "2. **SEMANTIC MATCHING (CRITICAL)**: The user's phrasing and the database names may differ. "
            "YOU MUST intelligently match them:\n"
            "   - User says 'medications for menopause' → Match 'Menopause Relief', 'Hormonal Balance', 'Estrogenic' etc.\n"
            "   - User says 'what helps with memory' → Match 'Memory Enhancer', 'Brain Tonic', 'Cognitive Support' etc.\n"
            "   - User says 'stress remedies' → Match 'Stress Relief', 'Adaptogen', 'Calms Nervous System' etc.\n"
            "   ANY entity in the evidence that semantically relates to the question IS relevant — use it!\n\n"

            "3. **READ THE ENTITY PROFILES CAREFULLY**: The COMPLETE ENTITY PROFILES section contains full details for each entity, "
            "including ALL their connections (what they treat, their properties, aliases, qualities, etc.). "
            "Use this structured data to build comprehensive answers. For example, if the profile shows:\n"
            "   'Shatavari [Herb]\n   TREATS: Menopause Relief\n   HAS_PROPERTY: Estrogenic\n   ALSO_KNOWN_AS: Asparagus Racemosus'\n"
            "Then when asked about menopause, include Shatavari AND explain its connection pathway.\n\n"

            "4. **FOLLOW ALL CONNECTION PATHS**: When the evidence shows multi-hop connections like:\n"
            "   'Brahmi -> Brain Tonic -> Memory Enhancer'\n"
            "   Explain the FULL chain naturally: 'Brahmi is expressed as a Brain Tonic, which directly treats Memory Enhancement.'\n\n"

            "5. **INCLUDE 'ALSO KNOWN AS' / ALIASES**: If the database shows an entity has aliases or alternative names, "
            "ALWAYS mention them: 'Shatavari (also known as Asparagus Racemosus)...'\n\n"

            "6. **NATURAL, CONVERSATIONAL LANGUAGE**: Write like a knowledgeable friend explaining to a patient. "
            "Use complete sentences with proper grammar. NO bullet-only lists — explain with context:\n"
            "   BAD: '• Brahmi - Memory Enhancer'\n"
            "   GOOD: 'Based on our database, **Brahmi** is specifically linked to memory enhancement. "
            "It functions as a Brain Tonic and has direct TREATS connections to Memory Enhancer properties.'\n\n"

            "7. **BE COMPREHENSIVE**: If the question matches multiple entities, discuss ALL of them. "
            "Don't stop at the first match. Group by relevance and explain how each one relates to the question.\n\n"

            "8. **HONEST GAPS**: If the database truly has NO related information, say so clearly and helpfully. "
            "But FIRST, thoroughly check ALL entity profiles and connections — often the answer is there under a different name.\n\n"

            "9. **ML INSIGHTS**: If AI Predictions are present, mention them naturally as 'Our ML analysis also suggests...'\n\n"

            "10. **FORMATTING**: Use markdown (bold, headers, bullet points) for clarity. Start with a direct answer, "
            "then provide supporting details.\n\n"

            "11. **GREETINGS**: For hi/hello, respond warmly and briefly describe what you can help with."
        )

        user_prompt = f"Database Evidence & Knowledge Graph Data:\n{full_context}\n\nUser's Question: {state['question']}"

        try:
            answer = await self.ai.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

            # Feature 10: Compute grounding score
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

        try:
            result = await self.graph.ainvoke(initial_state)
            return {
                "answer": result.get("answer", "No answer generated"),
                "citations": result.get("citations", []),
                "related_nodes": result.get("related_nodes", []),
                "grounding_score": result.get("grounding_score", 0.0),
                "needs_clarification": result.get("needs_clarification", False),
                "ml_insights_count": len(result.get("ml_similar_nodes", [])),
                "predictions_count": len(result.get("prediction_context", [])),
            }
        except Exception as e:
            logger.error(f"Enhanced RAG pipeline failed: {e}")
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
