from typing import List, Dict, Any, Optional
import logging
import json
from app.services.ai_service import get_ai_service
from app.services.gds_service import get_gds_service
from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)

class AnalyticChatService:
    def __init__(self):
        self.ai = get_ai_service()
        self.gds = get_gds_service()
        self.driver = get_neo4j_driver()

    async def process_query(self, query: str, folder_id: Optional[str] = None, node_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Processes a natural language query by deciding and running a graph algorithm.
        Always scoped to the active folder/selected nodes — never runs on the full database.
        """
        logger.info(f"[AnalyticChat] process_query: query='{query}', folder_id={folder_id}, node_ids={node_ids}")
        
        # 0. Gather scope context for the LLM
        scope_info, available_types = await self._get_scope_context(folder_id, node_ids)
        logger.info(f"[AnalyticChat] Scope context: {scope_info}")
        logger.info(f"[AnalyticChat] Available types: {available_types}")

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
        
        results = await self._run_algorithm(algo_name, folder_id, algo_node_ids, params, target_type, resolved_node_ids=resolved_node_ids)
        
        # 4. Hybrid Logic: If they asked about specific entities but DIDN'T get a path, 
        # try to find a direct connection path to augment the algorithmic results.
        if algo_name != "path" and len(resolved_node_ids) >= 2:
            path_results = await self._run_shortest_path(resolved_node_ids[0], resolved_node_ids[1])
            if path_results:
                logger.info(f"[AnalyticChat] Found additional path for context: {len(path_results)} nodes")
                # Prepend or append path info to results for chip rendering
                # The interpretation logic will see these in the pre-summary
                results = path_results + results 

        # 5. Generate response (scope-aware)
        answer = await self._interpret_results(query, algo_name, results, scope_info)
        
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

    async def _run_algorithm(self, algo: str, folder_id: Optional[str], node_ids: Optional[List[str]], params: Dict[str, Any], target_type: Optional[str] = None, resolved_node_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Executes the selected algorithm via GDSService."""
        top_k = params.get("top_k", 10)
        logger.info(f"[AnalyticChat] Running algorithm '{algo}' | folder_id={folder_id} | node_ids={node_ids} | top_k={top_k} | target_type={target_type}")
        
        try:
            if algo == "pagerank":
                results = await self.gds.run_pagerank(folder_id, node_ids, top_k=top_k, target_type=target_type)
            elif algo == "articlerank":
                results = await self.gds.run_articlerank(folder_id, node_ids, top_k=top_k, target_type=target_type)
            elif algo == "betweenness":
                results = await self.gds.run_betweenness(folder_id, node_ids, top_k=top_k, target_type=target_type)
            elif algo == "closeness":
                results = await self.gds.run_closeness(folder_id, node_ids, top_k=top_k, target_type=target_type)
            elif algo == "degree":
                results = await self.gds.run_degree(folder_id, node_ids, top_k=top_k, target_type=target_type)
            elif algo == "hits":
                results = await self.gds.run_hits(folder_id, node_ids, top_k=top_k, target_type=target_type)
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
        """Finds shortest path between two specific nodes."""
        query = """
        MATCH (start:Entity {id: $start_id}), (end:Entity {id: $end_id})
        MATCH p = shortestPath((start)-[*..15]-(end))
        RETURN [n IN nodes(p) | {id: n.id, name: n.name, type: coalesce(n.type, labels(n)[0])}] AS path_nodes
        """
        async with self.driver.session() as session:
            result = await session.run(query, start_id=start_id, end_id=end_id)
            record = await result.single()
            if record:
                return record["path_nodes"]
        return []

    def _pre_summarize(self, algo: str, results: List[Dict[str, Any]]) -> str:
        """Pre-process raw algorithm results into a human-readable summary for the LLM."""
        if not results:
            return "No results."

        # Separate path nodes from algorithm-specific results
        path_nodes = [r for r in results if not r.get("source_name") and r.get("name") and r.get("id") in [node.get("id") for node in results if not node.get("source_name")]]
        # Check if the list actually contains a path (sequence of nodes) or just centrality scores
        is_actual_path = algo == "path" or (len(path_nodes) >= 2 and any(" → " in str(r) for r in results)) # Rough check
        
        # If we have a path, format it separately
        path_str = ""
        if algo == "path" or (len(path_nodes) >= 2):
            path_names = [r.get("name", "?") for r in path_nodes]
            path_str = "DIRECT CONNECTION FOUND: " + " → ".join(path_names) + "\n\n"

        # Filter out path nodes for the algorithmic section (except for 'path' algo itself)
        effective_results = [r for r in results if r not in path_nodes] if algo != "path" else []
        
        algo_str = f"ALGORITHM '{algo.upper()}' RESULTS:\n"
        
        if algo in ("louvain", "leiden", "wcc"):
            communities = {}
            for r in [node for node in results if node.get("community_id")]:
                cid = r.get("community_id", "unknown")
                name = r.get("name", r.get("id", "?"))
                node_type = r.get("type", "")
                if cid not in communities:
                    communities[cid] = []
                communities[cid].append(f"{name} ({node_type})" if node_type else name)
            
            lines = []
            for i, (cid, members) in enumerate(sorted(communities.items(), key=lambda x: -len(x[1])), 1):
                member_str = ", ".join(members[:15])
                if len(members) > 15:
                    member_str += f" ... and {len(members) - 15} more"
                lines.append(f"Group {i} ({len(members)} members): {member_str}")
            algo_str += "\n".join(lines[:20])
        
        elif algo == "node_similarity" or algo.startswith("link_prediction"):
            lines = []
            # Look specifically for source/target style results
            sim_results = [r for r in results if r.get("source_name")]
            label = "Predicted link" if algo.startswith("link_prediction") else "Similarity"
            for r in sim_results[:15]:
                source = r.get('source_name','?')
                target = r.get('target_name','?')
                score = r.get('score', 0)
                if algo.startswith("link_prediction"):
                    lines.append(f"{source} ↔ {target} (Score: {score})")
                else:
                    lines.append(f"{source} ↔ {target} (Similarity: {score:.0%})")
            algo_str += "\n".join(lines) if lines else f"No {label.lower()} pairs found above threshold."

        elif algo == "path":
            # Already handled in path_str
            algo_str = ""
        
        elif algo in ("bfs", "dfs", "random_walk"):
            traversal_names = [r.get("name", "?") for r in results[:30]]
            algo_str += f"Traversed {len(results)} nodes: " + " → ".join(traversal_names)
            if len(results) > 30:
                algo_str += f" ... and {len(results) - 30} more"
        
        elif algo == "topological_sort":
            topo_names = [r.get("name", "?") for r in results[:30]]
            algo_str += f"Logical sequence ({len(results)} nodes): " + " → ".join(topo_names)
            if len(results) > 30:
                algo_str += f" ... and {len(results) - 30} more"
        
        else:
            # Centrality / scoring algorithms
            lines = []
            centrality_results = [r for r in results if r.get("score") and not r.get("source_name")]
            for i, r in enumerate(centrality_results[:15], 1):
                name = r.get("name", r.get("id", "?"))
                score = r.get("score", 0)
                node_type = r.get("type", "")
                type_label = f" ({node_type})" if node_type else ""
                lines.append(f"{i}. {name}{type_label} — Score: {score:.4f}")
            algo_str += "\n".join(lines) if lines else "No significant scores found."

        return path_str + algo_str

    async def _interpret_results(self, query: str, algo: str, results: List[Dict[str, Any]], scope_info: str = "") -> str:
        """Uses LLM to explain the algorithm results in natural language."""
        if not results:
            return "The algorithm was executed but returned no significant results for the current selection. Make sure you have a folder selected with data."

        # Check for friendly error messages from GDS
        if len(results) == 1 and results[0].get("error_message"):
            return results[0]["error_message"]

        # Pre-summarize results into human-readable format
        summary = self._pre_summarize(algo, results)

        # Algorithm-specific explanation instructions
        algo_instructions = {
            "pagerank": "PageRank measures influence/importance. Higher scores = more influential. Explain which entities are most influential and WHY (what makes them central hubs in this dataset).",
            "articlerank": "ArticleRank is like PageRank but better for diverse graphs. Explain which entities are most authoritative and what role they play.",
            "betweenness": "Betweenness finds bridge nodes that connect different groups. High scores = critical connectors. Explain which entities act as bridges and what groups they connect.",
            "closeness": "Closeness measures how quickly a node can reach all others. High scores = central access points. Explain which entities are most centrally positioned.",
            "degree": "Degree counts direct connections. High scores = most directly connected. Explain which entities have the most relationships and what that implies.",
            "hits": "HITS separates Hubs (nodes that LINK to many others) from Authorities (nodes that ARE LINKED by many). Explain both roles clearly.",
            "louvain": "Louvain found groups/communities of tightly connected entities. Explain what each group represents thematically — name each group based on its members (e.g., 'Health Benefits Group', 'Properties Group'). Don't use technical IDs like 'Community 44'.",
            "leiden": "Leiden found higher-quality communities than Louvain. Explain what each group represents thematically, name each group based on its members.",
            "wcc": "WCC found disconnected islands. Explain which groups are isolated from each other and what that means for the data completeness.",
            "kcore": "K-Core found the tightly-knit core of the graph. Entities in the core are the most stable and interconnected. Explain what makes this core group special.",
            "triangle_count": "Triangle count measures local clustering. Nodes with many triangles are part of very tight, collaborative groups. Explain which entities form the tightest clusters.",
            "node_similarity": "Node Similarity found pairs that share similar neighborhoods. Explain which entities are most similar and WHY (what shared connections make them alike).",
            "link_prediction_common": "Common Neighbors link prediction found pairs of currently unconnected entities that share many neighbors. Higher score = more shared connections. Explain which new connections are most likely and WHY.",
            "link_prediction_adamic": "Adamic-Adar link prediction found missing connections weighted by rare shared neighbors. Entities sharing uncommon connections are highlighted. Explain which hidden relationships are most significant.",
            "link_prediction_resource": "Resource Allocation link prediction simulated information flow to find hidden connections. Explain which potential new links are the strongest and what they would mean for the data.",
            "path": "Shortest path shows how two entities are connected through intermediaries. Walk through the chain step by step and explain each link.",
            "bfs": "BFS (Breadth-First Search) explored entities layer by layer from the starting node. The order shows proximity — entities listed first are closest neighbors. Explain what the traversal reveals about the local neighborhood structure.",
            "dfs": "DFS (Depth-First Search) followed paths as deep as possible before backtracking. This reveals long chains and hierarchies. Explain the deep connections and hierarchical structure found.",
            "random_walk": "Random Walk simulated wandering through the graph from a starting point. The entities encountered represent associative, serendipitous connections. Explain what surprising or non-obvious associations were discovered.",
            "topological_sort": "Topological Sort ordered entities in a logical dependency sequence. Earlier entities are prerequisites for later ones. Explain the logical flow, process order, or timeline revealed.",
        }

        specific_instruction = algo_instructions.get(algo, "Explain the results clearly.")

        system_prompt = f"""You are a friendly data analyst explaining graph algorithm results to a non-technical user.

ALGORITHM: {algo}
SCOPE: This ran ONLY on: {scope_info}
USER QUESTION: {query}

PRE-SUMMARIZED RESULTS:
{summary}

INSTRUCTIONS:
{specific_instruction}

RULES:
1. **Explain the Direct Connection FIRST**: If a "DIRECT CONNECTION FOUND" is provided in the results summary, walk through that path immediately. Explain how the entities are linked (e.g., "They both share the quality X").
2. **Context over Scores**: Don't say "Similarity is 5%". Instead say "They share a subtle connection through quality X, but are otherwise unique in their broad connections."
3. **Simple language**: No technical jargon. Don't say "Community ID 44" — instead say "Group 1: The Health Benefits cluster".
4. **Name the groups**: For community algorithms, give each group a descriptive theme name based on its members.
5. **Actionable insights**: Tell the user what they can DO with this information.
6. **Be brief**: 3-5 short paragraphs max.
7. **Bold key names**: Use **bold** for important entity names."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Explain these results simply."}
        ]

        try:
            return await self.ai.chat(messages)
        except Exception as e:
            logger.error(f"Failed to interpret results: {e}")
            return f"Algorithm {algo} finished. Top results: " + ", ".join([str(r.get('name', r.get('id'))) for r in results[:5]])

def get_analytic_chat_service() -> AnalyticChatService:
    return AnalyticChatService()
