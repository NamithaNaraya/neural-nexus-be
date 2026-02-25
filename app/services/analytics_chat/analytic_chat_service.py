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
        decision = await self._decide_algorithm(query, scope_info)
        algo_name = decision.get("algorithm")
        params = decision.get("parameters", {})
        entities_to_resolve = decision.get("entities", [])
        
        # 2. Resolve entities if mentioned by name
        # IMPORTANT: Resolved entities are for CONTEXT only, not for scoping the projection.
        # The projection should always use the full folder_id scope (or user-selected nodes).
        # Only the "path" algorithm needs specific node IDs to find shortest path between them.
        resolved_node_ids = []
        if entities_to_resolve:
            resolved_node_ids = await self._resolve_entities(entities_to_resolve, folder_id)
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

        # 3. Run the chosen algorithm (scoped to folder or user-selected nodes)
        target_type = decision.get("target_type")
        
        # Validate target_type: if it doesn't match any available type, don't filter
        # (e.g. LLM picks "Herb" but data only has "Entity" — filtering would return 0)
        if target_type and available_types:
            type_match = any(t.lower() == target_type.lower() for t in available_types)
            if not type_match:
                logger.warning(f"[AnalyticChat] target_type '{target_type}' not found in available types {available_types} — removing filter")
                target_type = None
        
        # For "path" algorithm, we need the resolved entity IDs as start/end points
        if algo_name == "path" and resolved_node_ids and len(resolved_node_ids) >= 2:
            algo_node_ids = resolved_node_ids
        else:
            algo_node_ids = scope_node_ids  # Use original UI selection, NOT resolved entities
        
        results = await self._run_algorithm(algo_name, folder_id, algo_node_ids, params, target_type)
        
        # 4. Generate response (scope-aware)
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
                available_types = [t['t'] for t in types_data if t['t']]
                type_summary = ", ".join([f"{t['t']}({t['c']})" for t in types_data])
                parts.append(f"Active folder: {folder_id} with {count} nodes. Types: [{type_summary}]")
            elif node_ids:
                parts.append(f"Selected {len(node_ids)} specific nodes")
            else:
                parts.append("No folder selected — using the full database")
        scope_str = "; ".join(parts) if parts else "Full database"
        return scope_str, available_types

    async def _resolve_entities(self, names: List[str], folder_id: Optional[str]) -> List[str]:
        """Finds node IDs for names mentioned in a query, scoped to folder."""
        ids = []
        async with self.driver.session() as session:
            for name in names:
                query = "MATCH (n) WHERE n.name =~ $regex"
                if folder_id:
                    query += " AND n.folder_id = $folder_id"
                query += " RETURN n.id AS id LIMIT 1"
                result = await session.run(query, regex="(?i).*" + name + ".*", folder_id=folder_id)
                record = await result.single()
                if record:
                    ids.append(record["id"])
        return ids

    async def _decide_algorithm(self, query: str, scope_info: str = "") -> Dict[str, Any]:
        """Uses LLM to decide which algorithm fits the query, with scope awareness."""
        system_prompt = f"""
        You are a Graph Data Science expert. Based on the user's question, decide which graph algorithm should be used.
        
        IMPORTANT: The algorithms will run ONLY on the currently selected dataset, which is:
        {scope_info}
        
        Available Algorithms:
        1. pagerank: Use for: influence, importance, popularity, "best", "most central", ranking, or comparing qualities.
        2. betweenness: Use for: bridges, bottlenecks, nodes connecting different groups, flow control.
        3. closeness: Use for: reachability, fast communication, "closest to all others".
        4. louvain: Use for: communities, clusters, groups, "who belongs together".
        5. wcc: Use for: islands, disconnected parts, "isolated groups".
        6. path: Use for: "how are X and Y related", "find connection between X and Y", "path between...".
        
        Entities & Types:
        - If the user mentions specific names (e.g. "Tulsi", "Shatavari"), list them in "entities".
        - If the user specifically asks for a type of result (e.g. "Which herb", "Which condition"), specify that in "target_type" (e.g. "Herb", "Condition").
        
        Respond ONLY with a JSON object:
        {{
          "algorithm": "pagerank" | "betweenness" | "closeness" | "louvain" | "wcc" | "path" | "none",
          "entities": ["Name1", "Name2"],
          "target_type": "Herb" | "Condition" | "Property" | null,
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

    async def _run_algorithm(self, algo: str, folder_id: Optional[str], node_ids: Optional[List[str]], params: Dict[str, Any], target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Executes the selected algorithm via GDSService."""
        top_k = params.get("top_k", 10)
        logger.info(f"[AnalyticChat] Running algorithm '{algo}' | folder_id={folder_id} | node_ids={node_ids} | top_k={top_k} | target_type={target_type}")
        
        try:
            if algo == "pagerank":
                results = await self.gds.run_pagerank(folder_id, node_ids, top_k=top_k, target_type=target_type)
            elif algo == "betweenness":
                results = await self.gds.run_betweenness(folder_id, node_ids, top_k=top_k, target_type=target_type)
            elif algo == "closeness":
                results = await self.gds.run_closeness(folder_id, node_ids, top_k=top_k, target_type=target_type)
            elif algo == "louvain":
                results = await self.gds.run_louvain(folder_id, node_ids, target_type=target_type)
            elif algo == "wcc":
                results = await self.gds.run_wcc(folder_id, node_ids, target_type=target_type)
            elif algo == "path":
                if node_ids and len(node_ids) >= 2:
                    results = await self._run_shortest_path(node_ids[0], node_ids[1])
                else:
                    logger.warning(f"[AnalyticChat] Path algorithm requires at least 2 node_ids, got: {node_ids}")
                    results = []
            else:
                logger.warning(f"[AnalyticChat] Unknown algorithm: {algo}")
                results = []
            
            logger.info(f"[AnalyticChat] Algorithm '{algo}' returned {len(results)} results")
            if results:
                logger.info(f"[AnalyticChat] Sample result: {results[0]}")
            return results
        except Exception as e:
            logger.error(f"[AnalyticChat] Algorithm '{algo}' execution FAILED: {e}", exc_info=True)
            return []

    async def _run_shortest_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        """Finds shortest path between two specific nodes."""
        query = """
        MATCH (start:Entity {id: $start_id}), (end:Entity {id: $end_id})
        MATCH p = shortestPath((start)-[*..10]-(end))
        RETURN [n IN nodes(p) | {id: n.id, name: n.name, type: n.type}] AS path_nodes
        """
        async with self.driver.session() as session:
            result = await session.run(query, start_id=start_id, end_id=end_id)
            record = await result.single()
            if record:
                return record["path_nodes"]
        return []

    async def _interpret_results(self, query: str, algo: str, results: List[Dict[str, Any]], scope_info: str = "") -> str:
        """Uses LLM to explain the algorithm results in natural language."""
        if not results:
            return "The algorithm was executed but returned no significant results for the current selection. Make sure you have a folder selected with data."

        system_prompt = f"""
        You are a graph analyst. You just ran the '{algo}' algorithm on a knowledge graph to answer a user's question.
        
        SCOPE: The algorithm ran ONLY on this dataset: {scope_info}
        USER QUESTION: {query}
        ALGORITHM RESULTS (JSON): {json.dumps(results[:15])}
        
        Instructions:
        1. Explain what the results mean in simple, natural language.
        2. Specifically mention the top entities found and why they matter.
        3. If it's a community detection (louvain), describe the groups found and what defines each cluster.
        4. Make it clear that these results are from the selected dataset, not the entire database.
        5. Maintain a professional, insightful tone.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Explain these results in the context of my question."}
        ]
        
        try:
            return await self.ai.chat(messages)
        except Exception as e:
            logger.error(f"Failed to interpret results: {e}")
            return f"Algorithm {algo} finished. Top results: " + ", ".join([str(r.get('name', r.get('id'))) for r in results[:5]])

def get_analytic_chat_service() -> AnalyticChatService:
    return AnalyticChatService()
