from typing import List, Dict, Any, Optional
import logging
import json
from app.services.ai_service import get_ai_service
from app.services.gds_service import get_gds_service
from app.services.weight_service import WeightService
from app.db.connections import get_neo4j_driver, get_postgres_session
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Algorithms that actually use the weight_formula for scoring
WEIGHT_AWARE_ALGOS = frozenset({
    "pagerank", "articlerank", "betweenness", "closeness", "degree", "hits"
})

class AnalyticChatService:
    def __init__(self):
        self.ai = get_ai_service()
        self.gds = get_gds_service()
        self.weights = WeightService()
        self.driver = get_neo4j_driver()

    async def process_query(self, query: str, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Processes a natural language query by deciding and running a graph algorithm.
        Always scoped to the active folder/selected nodes — never runs on the full database.
        """
        logger.info(f"[AnalyticChat] process_query: query='{query}', folder_id={folder_id}, node_ids={node_ids}")
        
        # 0. Check for active weight configurations
        weight_formula = None
        weight_desc = ""
        if folder_id:
            try:
                async with get_postgres_session() as session:
                    result = await session.execute(
                        text("SELECT formula, name FROM neural_nexus.weight_configs WHERE folder_id = :fid AND is_active = true LIMIT 1"),
                        {"fid": folder_id}
                    )
                    row = result.fetchone()
                    if row:
                        weight_formula = row.formula
                        weight_desc = f"Analyzing using custom weights: {row.name} ({self.weights.get_formula_description(weight_formula)})"
                        logger.info(f"[AnalyticChat] Found active weight: {row.name}")
            except Exception as e:
                logger.warning(f"[AnalyticChat] Could not fetch weight config: {e}")

        # 0.5 Gather scope context for the LLM
        scope_info, available_types = await self._get_scope_context(folder_id, node_ids)
        if weight_desc:
            scope_info += f"\nActive Weighting: {weight_desc}"
        
        logger.info(f"[AnalyticChat] Scope context: {scope_info}")

        # 1. Decide which algorithm to use (with scope awareness)
        decision = await self._decide_algorithm(query, scope_info, available_types)
        algo_name = decision.get("algorithm")
        params = decision.get("parameters", {})
        entities_to_resolve = decision.get("entities", [])
        
        # 2. Resolve entities if mentioned by name
        # IMPORTANT: Resolved entities are for CONTEXT only, not for scoping the projection.
        # The projection should always use the full folder_id scope (or user-selected nodes).
        # Only the "path" algorithm needs specific node IDs to find shortest path between them.
        resolved_entities = []
        resolved_node_ids = []
        if entities_to_resolve:
            resolved_entities = await self._resolve_entities(entities_to_resolve, folder_id)
            resolved_node_ids = [r["id"] for r in resolved_entities]
            logger.info(f"[AnalyticChat] Resolved entities: {entities_to_resolve} -> {resolved_node_ids}")

        # Keep original user-selected node_ids for projection scoping (from UI selection)
        scope_node_ids = node_ids  # These come from the frontend (user clicked on nodes)

        if not algo_name or algo_name == "none":
            # Fallback to standard chat
            answer = await self.ai.chat([
                {"role": "system", "content": "You are a graph analytics assistant. Answer the user's question accurately based ONLY on the selected dataset. If they mention specific entities, use the context provided."},
                {"role": "user", "content": f"Context: {scope_info}\nResolved Entities={resolved_node_ids}\nQuestion: {query}"}
            ])
            return {
                "answer": answer,
                "algorithm": None,
                "results": [],
                "resolved_entities": resolved_node_ids
            }

        # 3. Determine the graph scope for the algorithm
        target_type = decision.get("target_type")
        
        # Validate target_type: if it doesn't match any available type, don't filter
        if target_type and available_types:
            type_match = any(t.lower() == target_type.lower() for t in available_types)
            if not type_match:
                logger.warning(f"[AnalyticChat] target_type '{target_type}' not found in available types {available_types} — removing filter")
                target_type = None
        
        # LOGIC FIX:
        # - For "path" OR "node_similarity" between specific names, use resolved_node_ids.
        # - For analytical algorithms (pagerank, degree, community, etc.), use the FULL FOLDER scope
        #   even if specific nodes are selected in the UI. This allows comparative analysis.
        global_algos = ("pagerank", "articlerank", "betweenness", "closeness", "degree", "hits", "louvain", "leiden", "wcc", "kcore", "triangle_count", "link_prediction_common", "link_prediction_adamic", "link_prediction_resource", "topological_sort")
        
        if algo_name in ("path", "node_similarity") and resolved_node_ids and len(resolved_node_ids) >= 2:
            algo_node_ids = resolved_node_ids
        elif algo_name in global_algos:
            # For global/ranking algorithms, we ignore specific node selections to allow folder-wide comparison
            # Unless NO folder is selected, then we fallback to UI selection
            algo_node_ids = None if folder_id else node_ids
        else:
            # Fallback to UI selection
            algo_node_ids = node_ids
        
        results = await self._run_algorithm(algo_name, folder_id, algo_node_ids, params, target_type, resolved_node_ids=resolved_node_ids, weight_formula=weight_formula)
        
        # 4. Hybrid Logic: If they asked about specific entities but DIDN'T get a path, 
        # try to find a direct connection path to augment the algorithmic results.
        if algo_name != "path" and len(resolved_node_ids) >= 2:
            path_results = await self._run_shortest_path(resolved_node_ids[0], resolved_node_ids[1])
            if path_results:
                logger.info(f"[AnalyticChat] Found additional path for context: {len(path_results)} nodes")
                # Prepend or append path info to results for chip rendering
                # The interpretation logic will see these in the pre-summary
                results = path_results + results 

        # 5. Generate response (scope-aware, weight-aware)
        weight_name = None
        if weight_formula and folder_id:
            try:
                async with get_postgres_session() as session:
                    result = await session.execute(
                        text("SELECT name FROM neural_nexus.weight_configs WHERE folder_id = :fid AND is_active = true LIMIT 1"),
                        {"fid": folder_id}
                    )
                    wrow = result.fetchone()
                    if wrow:
                        weight_name = wrow.name
            except Exception:
                pass

        answer = await self._interpret_results(query, algo_name, results, scope_info, weight_formula=weight_formula, weight_name=weight_name, resolved_node_ids=resolved_node_ids)
        
        return {
            "answer": answer,
            "algorithm": algo_name,
            "results": results,
            "decision": decision,
            "resolved_entities": resolved_node_ids
        }

    async def _get_scope_context(self, folder_id: Optional[str], node_ids: Optional[List[str]]) -> tuple:
        """Builds a human-readable scope description for the LLM. Returns (scope_info_str, available_types_list)."""
        parts = []
        available_types = []
        async with self.driver.session() as session:
            if folder_id:
                # Count ALL nodes in this folder (not just Entity label)
                result = await session.run(
                    "MATCH (n) WHERE n.folder_id = $fid RETURN count(n) AS cnt",
                    fid=folder_id
                )
                rec = await result.single()
                count = rec["cnt"] if rec else 0
                # Get available types in this folder
                type_result = await session.run(
                    "MATCH (n) WHERE n.folder_id = $fid RETURN DISTINCT coalesce(n.type, labels(n)[0]) AS t, count(n) AS c ORDER BY c DESC LIMIT 10",
                    fid=folder_id
                )
                types_data = await type_result.data()
                # Strip folder suffix (e.g. Student_F_abc -> Student) to provide clean types to the LLM
                available_types = []
                type_summary_list = []
                for t in types_data:
                    if not t['t']: continue
                    clean_type = t['t'].split("_F_")[0]
                    available_types.append(clean_type)
                    type_summary_list.append(f"{clean_type}({t['c']})")
                
                type_summary = ", ".join(type_summary_list)
                parts.append(f"Active folder: {folder_id} with {count} nodes. Types: [{type_summary}]")
            elif node_ids:
                parts.append(f"Selected {len(node_ids)} specific nodes")
            else:
                parts.append("No folder selected — using the full database")
        scope_str = "; ".join(parts) if parts else "Full database"
        return scope_str, available_types

    async def _resolve_entities(self, names: List[str], folder_id: Optional[str]) -> List[str]:
        """Finds node IDs for names mentioned in a query, scoped to folder. Batched into a single query."""
        if not names:
            return []
        async with self.driver.session() as session:
            # Build a single query that resolves all names at once
            query = "UNWIND $names AS name MATCH (n) WHERE n.name =~ ('(?i).*' + name + '.*')"
            if folder_id:
                query += " AND n.folder_id = $folder_id"
            query += " RETURN n.id AS id, n.name AS name, coalesce(n.type, labels(n)[0]) AS type LIMIT 10"
            result = await session.run(query, names=names, folder_id=folder_id)
            data = await result.data()
            return data # Now returns full node dicts instead of just strings

    async def _decide_algorithm(self, query: str, scope_info: str = "", available_types: List[str] = None) -> Dict[str, Any]:
        """Uses LLM to decide which algorithm fits the query, with scope awareness."""
        
        # Build dynamic target types string for the prompt
        types_str = " | ".join([f'"{t}"' for t in available_types]) if available_types else '"TypeA" | "TypeB"'
        types_example = available_types[:2] if available_types and len(available_types) >= 2 else ["Category", "Type"]
        
        system_prompt = f"""
        You are a Graph Data Science expert. Based on the user's question, decide which graph algorithm should be used.
        
        IMPORTANT: The algorithms will run ONLY on the currently selected dataset, which is:
        {scope_info}
        
        Available Algorithms:
        
        — CENTRALITY (Who/What is most important?) —
        1. pagerank: Influence, importance, popularity, "best", "most central", ranking, or comparing qualities.
        2. articlerank: Like PageRank but better for diverse graphs. Use when "which is most authoritative" or data has very uneven connectivity.
        3. betweenness: Bridges, bottlenecks, nodes connecting different groups, flow control, "what connects X and Y groups".
        4. closeness: Reachability, fast communication, "closest to all others", central access point.
        5. degree: Most directly connected, "which has the most connections", "most active", "most linked".
        6. hits: Hubs vs authorities. "Which nodes are hubs" (link to many) vs "which are authorities" (linked BY many).
        
        — COMMUNITY DETECTION (How is the data grouped?) —
        7. louvain: Communities, clusters, groups, "who belongs together", thematic grouping.
        8. leiden: Same as louvain but higher quality communities. Use for "better clustering" or when louvain gives messy results.
        9. wcc: Islands, disconnected parts, "isolated groups", "what's not connected".
        10. kcore: Core structure, "tight-knit core", "most stable central group", inner circle vs periphery.
        11. triangle_count: Local density, "tight clusters", "which nodes form triangles", "tightly-knit groups".
        
        — SIMILARITY —
        12. node_similarity: "What is similar to X", "which nodes share neighbors", "find similar entities", Jaccard similarity.
        
        — LINK PREDICTION (What connections are missing?) —
        13. link_prediction_common: Predict missing links by counting shared neighbors. "What should be connected?", "predict connections", "common neighbors".
        14. link_prediction_adamic: Advanced link prediction weighted by rare shared connections. "Non-obvious relationships", "unique connections".
        15. link_prediction_resource: Flow-based link prediction. "Hidden links", "resource allocation", "high-probability missing links".
        
        — PATHFINDING & TRAVERSAL —
        16. path: "How are X and Y related", "find connection between X and Y", "path between...", shortest path.
        17. bfs: Explore nodes layer by layer from a starting point. "What's nearby X", "neighbors of", "within N hops".
        18. dfs: Explore deep paths from a starting point. "Deep hierarchy from X", "follow the chain from X", "trace the lineage".
        19. random_walk: Simulate wandering through the graph. "Explore from X randomly", "discover associations from X", "serendipitous connections".
        
        — TOPOLOGY —
        20. topological_sort: Order nodes in a logical sequence (for DAGs). "Process flow", "timeline order", "logical sequence", "dependency order".
        
        Entities & Types:
        - If the user mentions specific names list them in "entities".
        - For traversal algorithms (bfs, dfs, random_walk), the FIRST entity in "entities" will be used as the starting point.
        - If the user specifically asks for a type of result, specify that in "target_type" (e.g. "{types_example[0]}" or "{types_example[1]}").
        - For link prediction, set "link_method" in parameters to one of: "common_neighbors", "adamic_adar", "resource_allocation".
        
        Respond ONLY with a JSON object:
        {{
          "algorithm": "pagerank" | "articlerank" | "betweenness" | "closeness" | "degree" | "hits" | "louvain" | "leiden" | "wcc" | "kcore" | "triangle_count" | "node_similarity" | "link_prediction_common" | "link_prediction_adamic" | "link_prediction_resource" | "path" | "bfs" | "dfs" | "random_walk" | "topological_sort" | "none",
          "entities": ["Name1", "Name2"],
          "target_type": {types_str} | null,
          "parameters": {{ "top_k": 10 }},
          "reasoning": "Brief explanation"
        }}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            decision = await self.ai.chat_json(messages)
            return decision
        except Exception as e:
            logger.error(f"Failed to decide algorithm: {e}")
            return {"algorithm": "none"}

    async def _run_algorithm(self, algo: str, folder_id: Optional[str], node_ids: Optional[List[str]], params: Dict[str, Any], target_type: Optional[str] = None, resolved_node_ids: Optional[List[str]] = None, weight_formula: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Executes the selected algorithm via GDSService."""
        top_k = params.get("top_k", 10)
        logger.info(f"[AnalyticChat] Running algorithm '{algo}' | folder_id={folder_id} | node_ids={node_ids} | top_k={top_k} | target_type={target_type}")
        
        try:
            if algo == "pagerank":
                results = await self.gds.run_pagerank(folder_id, node_ids, top_k=top_k, target_type=target_type, weight_formula=weight_formula)
            elif algo == "articlerank":
                results = await self.gds.run_articlerank(folder_id, node_ids, top_k=top_k, target_type=target_type, weight_formula=weight_formula)
            elif algo == "betweenness":
                results = await self.gds.run_betweenness(folder_id, node_ids, top_k=top_k, target_type=target_type, weight_formula=weight_formula)
            elif algo == "closeness":
                results = await self.gds.run_closeness(folder_id, node_ids, top_k=top_k, target_type=target_type, weight_formula=weight_formula)
            elif algo == "degree":
                results = await self.gds.run_degree(folder_id, node_ids, top_k=top_k, target_type=target_type, weight_formula=weight_formula)
            elif algo == "hits":
                results = await self.gds.run_hits(folder_id, node_ids, top_k=top_k, target_type=target_type, weight_formula=weight_formula)
            elif algo == "louvain":
                results = await self.gds.run_louvain(folder_id, node_ids, target_type=target_type)
            elif algo == "leiden":
                results = await self.gds.run_leiden(folder_id, node_ids, target_type=target_type)
            elif algo == "wcc":
                results = await self.gds.run_wcc(folder_id, node_ids, target_type=target_type)
            elif algo == "kcore":
                results = await self.gds.run_kcore(folder_id, node_ids, top_k=top_k, target_type=target_type)
            elif algo == "triangle_count":
                results = await self.gds.run_triangle_count(folder_id, node_ids, top_k=top_k, target_type=target_type)
            elif algo == "node_similarity":
                results = await self.gds.run_node_similarity(folder_id, node_ids, top_k=top_k, target_type=target_type)
                # If the user mentioned specific nodes, filter results for those pairs
                if node_ids and len(node_ids) >= 2:
                    results = [r for r in results if r.get('source_id') in node_ids and r.get('target_id') in node_ids]
            elif algo.startswith("link_prediction"):
                method_map = {
                    "link_prediction_common": "common_neighbors",
                    "link_prediction_adamic": "adamic_adar",
                    "link_prediction_resource": "resource_allocation",
                }
                lp_method = method_map.get(algo, params.get("link_method", "common_neighbors"))
                results = await self.gds.run_link_prediction(folder_id, node_ids, method=lp_method, top_k=top_k, target_type=target_type)
            elif algo == "bfs":
                source_id = resolved_node_ids[0] if resolved_node_ids else (node_ids[0] if node_ids else None)
                if source_id:
                    results = await self.gds.run_bfs(source_id, folder_id, node_ids)
                else:
                    results = []
                    logger.warning("[AnalyticChat] BFS requires a source node — none provided")
            elif algo == "dfs":
                source_id = resolved_node_ids[0] if resolved_node_ids else (node_ids[0] if node_ids else None)
                if source_id:
                    results = await self.gds.run_dfs(source_id, folder_id, node_ids)
                else:
                    results = []
                    logger.warning("[AnalyticChat] DFS requires a source node — none provided")
            elif algo == "random_walk":
                source_id = resolved_node_ids[0] if resolved_node_ids else (node_ids[0] if node_ids else None)
                if source_id:
                    results = await self.gds.run_random_walk(source_id, folder_id, node_ids)
                else:
                    results = []
                    logger.warning("[AnalyticChat] Random Walk requires a source node — none provided")
            elif algo == "topological_sort":
                results = await self.gds.run_topological_sort(folder_id, node_ids)
            elif algo == "path":
                if node_ids and len(node_ids) >= 2:
                    results = await self._run_shortest_path(node_ids[0], node_ids[1])
                else:
                    results = []
            else:
                logger.warning(f"[AnalyticChat] Unknown algorithm: {algo}")
                results = []
            
            logger.info(f"[AnalyticChat] Algorithm '{algo}' returned {len(results)} results")
            if results:
                # Check if the result is a friendly error message from GDS
                if len(results) == 1 and results[0].get("error_message"):
                    logger.info(f"[AnalyticChat] GDS returned friendly message: {results[0]['error_message']}")
                else:
                    logger.info(f"[AnalyticChat] Sample result: {results[0]}")
            return results
        except Exception as e:
            logger.error(f"[AnalyticChat] Algorithm '{algo}' execution FAILED: {e}", exc_info=True)
            return [{"error_message": f"The algorithm could not run on your current data. This usually means the dataset is too small or doesn't have the right structure for this analysis."}]

    async def _run_shortest_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        """Finds shortest path between two specific nodes, including relationship types."""
        query = """
        MATCH (start {id: $start_id}), (end {id: $end_id})
        MATCH p = shortestPath((start)-[*..15]-(end))
        WITH nodes(p) AS ns, relationships(p) AS rs
        RETURN [i IN range(0, size(ns)-1) | {
            id: ns[i].id, 
            name: coalesce(ns[i].name, ns[i].title, ns[i].question_text, ns[i].text, ns[i].content, ns[i].label, ns[i].code, ns[i].id), 
            type: coalesce(ns[i].type, labels(ns[i])[0]),
            rel_to_next: CASE WHEN i < size(rs) THEN type(rs[i]) ELSE null END
        }] AS path_data
        """
        async with self.driver.session() as session:
            result = await session.run(query, start_id=start_id, end_id=end_id)
            record = await result.single()
            if record and record["path_data"]:
                return record["path_data"]
        return []

    async def _get_discovery_evidence(self, results: List[Dict[str, Any]], resolved_node_ids: List[str]) -> List[List[Dict[str, Any]]]:
        """
        For discovery queries (BFS, DFS, Random Walk, PageRank, ArticleRank),
        this method finds the paths from the resolved_node_ids to the top results.
        This helps the LLM understand the "why" behind the discovery.
        """
        if not resolved_node_ids or not results:
            return []

        # Limit to a reasonable number of paths to avoid overwhelming the LLM
        max_paths = 5
        evidence_paths = []

        # For each resolved node (potential starting point)
        for start_node_id in resolved_node_ids:
            # For each top result, try to find a path to it from the start_node_id
            for result_node in results[:max_paths]: # Limit results to check paths for
                end_node_id = result_node.get("id")
                if not end_node_id or start_node_id == end_node_id:
                    continue

                # Find a path between the start node and the result node
                query = """
                MATCH (start {id: $start_id}), (end {id: $end_id})
                MATCH p = shortestPath((start)-[*..5]-(end))
                WITH nodes(p) AS ns, relationships(p) AS rs
                RETURN [i IN range(0, size(ns)-1) | {
                    id: ns[i].id, 
                    name: coalesce(ns[i].name, ns[i].title, ns[i].question_text, ns[i].text, ns[i].content, ns[i].label, ns[i].code, ns[i].id), 
                    type: coalesce(ns[i].type, labels(ns[i])[0]),
                    rel: CASE WHEN i < size(rs) THEN type(rs[i]) ELSE null END
                }] AS path_data
                """
                async with self.driver.session() as session:
                    path_result = await session.run(query, start_id=start_node_id, end_id=end_node_id)
                    record = await path_result.single()
                    if record and record["path_data"]:
                        evidence_paths.append(record["path_data"])
                        if len(evidence_paths) >= max_paths:
                            return evidence_paths # Stop if we have enough paths

        return evidence_paths

    def _pre_summarize(self, algo: str, results: List[Dict[str, Any]], evidence: List[List[Dict[str, Any]]] = None) -> str:
        """Pre-process raw algorithm results into a human-readable summary for the LLM."""
        if not results:
            return "No results."

        # 1. Format Evidence/Mapping if available (This is what the user wants: What is connected to what)
        evidence_str = ""
        if evidence:
            segments = []
            for path in evidence:
                chain = []
                for step in path:
                    name = f"**{step['name']}**"
                    rel = f" --[{step['rel']}]--> " if step['rel'] else ""
                    chain.append(f"{name}{rel}")
                segments.append("".join(chain))
            evidence_str = "RELATIONSHIP MAPPINGS FOUND (Evidence):\n" + "\n".join(segments[:10]) + "\n\n"

        # 2. Detect if we have path data (nodes with rel_to_next)
        path_segments = [r for r in results if "rel_to_next" in r]
        path_str = ""
        if path_segments:
            segments = []
            for r in path_segments:
                name = rf"**{r.get('name')}** ({r.get('type')})"
                rel = f" --[{r.get('rel_to_next')}]--> " if r.get('rel_to_next') else ""
                segments.append(f"{name}{rel}")
            path_str = "PRIMARY CONNECTION PATHWAY:\n" + "".join(segments) + "\n\n"

        # Filter out path segments from the main results list to avoid redundancy
        effective_results = [r for r in results if r not in path_segments]
        
        algo_str = f"ALGORITHM '{algo.upper()}' RESULTS:\n"
        
        if algo in ("louvain", "leiden", "wcc"):
            communities = {}
            for r in [node for node in effective_results if node.get("community_id") is not None]:
                cid = r.get("community_id")
                name = r.get("name") or "?"
                node_type = r.get("type", "Unknown")
                if cid not in communities:
                    communities[cid] = {"members": [], "types": {}}
                
                communities[cid]["members"].append(name)
                communities[cid]["types"][node_type] = communities[cid]["types"].get(node_type, 0) + 1
            
            lines = []
            sorted_communities = sorted(communities.items(), key=lambda x: -len(x[1]["members"]))
            for i, (cid, data) in enumerate(sorted_communities, 1):
                members, type_counts = data["members"], data["types"]
                type_summary = ", ".join([f"{count} {t}" for t, count in type_counts.items()])
                lines.append(f"GROUP {i} ({type_summary}): {', '.join(members[:8])}")
            algo_str += "\n".join(lines[:10])
        
        elif algo in ("bfs", "dfs", "random_walk", "pagerank", "articlerank"):
            # For discovery or importance, show what types were found
            type_groups = {}
            for r in effective_results:
                t = r.get("type", "Unknown")
                if t not in type_groups: type_groups[t] = []
                type_groups[t].append(r.get("name", "?"))
            
            lines = []
            for t, members in type_groups.items():
                lines.append(f"- {t}s: {', '.join(members[:10])}")
            algo_str += "SCAN DISCOVERY:\n" + "\n".join(lines)
            
        elif algo == "node_similarity" or algo.startswith("link_prediction"):
            lines = []
            sim_results = [r for r in effective_results if r.get("source_name")]
            for r in sim_results[:15]:
                source, target, score = r.get('source_name','?'), r.get('target_name','?'), r.get('score', 0)
                lines.append(f"Match: {source} ↔ {target} (Value: {score})")
            algo_str += "\n".join(lines) if lines else "No pairs found."

        else:
            # Centrality / scoring algorithms
            lines = []
            centrality_results = [r for r in effective_results if r.get("score") is not None and not r.get("source_name")]
            for i, r in enumerate(centrality_results[:15], 1):
                name, score, node_type = r.get("name") or "?", r.get("score", 0), r.get("type", "")
                lines.append(f"{i}. {name} ({node_type}) - Value: {score:.4f}")
            algo_str += "\n".join(lines) if lines else "No rankings found."

        return evidence_str + path_str + algo_str

    async def _interpret_results(self, query: str, algo: str, results: List[Dict[str, Any]], scope_info: str = "", weight_formula: Optional[Dict[str, Any]] = None, weight_name: Optional[str] = None, resolved_node_ids: List[str] = None) -> str:
        """Uses LLM to explain the algorithm results in natural language."""
        if not results:
            return "The analysis was completed but no significant patterns were found in the current selection. This can happen if the data isn't interconnected enough for this type of calculation."

        # Fetch evidence mappings if this is a discovery query or if we have specific targets
        evidence = None
        if algo in ("bfs", "dfs", "random_walk", "pagerank", "articlerank") and resolved_node_ids:
            evidence = await self._get_discovery_evidence(results, resolved_node_ids)

        # Check for friendly error messages from GDS
        if len(results) == 1 and results[0].get("error_message"):
            return results[0]["error_message"]

        # Pre-summarize results into human-readable format
        summary = self._pre_summarize(algo, results, evidence=evidence)

        # ── Weight context for the LLM ──
        algo_uses_weights = algo in WEIGHT_AWARE_ALGOS
        weight_text = ""
        if algo_uses_weights and weight_formula:
            weight_text = f"NOTE: This analysis USED custom weights ('{weight_name}'). The results factor in structural connections AND numeric property values."
        elif algo_uses_weights:
            weight_text = "NOTE: This analysis did NOT use weights (only connection structure). Enabling weights could factor in numeric scores/marks."

        # Algorithm-specific explanation goals
        algo_goals = {
            "pagerank": "Identify the most influential or 'popular' entities in this specific context.",
            "articlerank": "Identify the most authoritative entities considering diverse connections.",
            "betweenness": "Find the key 'bridges' or bottlenecks that connect different parts of the data.",
            "closeness": "Find entities that are 'closest' to everything else (central access points).",
            "degree": "Simply count who has the most connections in the system.",
            "hits": "Distinguish between 'Hubs' (links to many) and 'Authorities' (referenced by many).",
            "louvain": "Find natural 'groups' or clusters. Give each group a name based on the members provided.",
            "leiden": "Find high-quality groups. Give each group a name based on the members provided.",
            "wcc": "Find isolated 'islands' or disconnected sub-groups.",
            "kcore": "Identify the 'inner circle' or most stable, tight-knit core of the data.",
            "triangle_count": "Identify areas of very high collaborative density.",
            "node_similarity": "Explain which entities are most similar to each other and why.",
            "link_prediction_common": "Predict which connections should exist but don't yet.",
            "path": "Trace the exact connection between two items.",
            "topological_sort": "Show the logical sequence or order of operations.",
        }

        goal = algo_goals.get(algo, "Explain the results clearly.")

        system_prompt = f"""You are a plain-English data analyst. Your job is to interpret graph analysis results and answer a user question.

USER QUESTION: "{query}"
ANALYSIS TYPE: {algo} ({goal})
DATA SCOPE: {scope_info}
{weight_text}

RAW RESULTS SUMMARY:
{summary}

CRITICAL RULES:
1. **Answer the Question FIRST**: Don't start by saying "The algorithm PageRank found...". Start by answering the user's question directly using the data.
2. **Dynamic Relationship Mapping**: For discovery questions, you MUST provide a table or clear list showing the UNIQUE relationship chain for each result found in the data.
   - DO NOT follow a fixed template. Use the exact sequence and names from the 'RELATIONSHIP MAPPINGS' summary.
   - Format: **[Entity 1]** --[RELATIONSHIP]--> **[Entity 2]** --[RELATIONSHIP]--> **[Entity 3]**
3. **RELATIONSHIP NAMES**: Use the exact relationship names (e.g., 'TREATS', 'CONNECTED_TO', 'CONTAINS') provided in the raw mappings.
4. **Plain English Only**: Use extremely simple, professional, and clear English. No "Nodes", "Edges", or "Centrality".
5. **Include Technical Scores**: Show relevant scores (score: 0.123) alongside descriptions for precision.
6. **Name the Groups**: If there are groups, give them a descriptive name based on the members.
7. **Be Brief & To The Point**: Use 2-4 short paragraphs maximum. Use bullet points for lists.
8. **Entity Formatting**: Bold all **Entity Names**.

Explain the connections clearly as a sequence so the user knows exactly HOW each result is relevant to their question."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Explain these results in plain English, show the mapping for each herb."}
        ]

        try:
            return await self.ai.chat(messages)
        except Exception as e:
            logger.error(f"Failed to interpret results: {e}")
            return f"Analysis complete. Most significant items: " + ", ".join([str(r.get('name', r.get('id'))) for r in results[:5]])

def get_analytic_chat_service() -> AnalyticChatService:
    return AnalyticChatService()
