"""
Analytics Routes

Graph algorithms and statistical analysis endpoints.
Fully integrated with Neo4j GDS (Graph Data Science) and APOC plugins.
"""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import json

from app.core.security import get_current_user
from app.db.connections import get_neo4j_driver
from app.algorithms import (
    DegreeDistribution,
    EntitySimilarity,
    GirvanNewman,
    GraphHealth,
    HITS,
    IncompleteEntities,
    KCore,
    KnowledgeCompleteness,
    LinkPrediction,
    MissingRelationships,
    StatisticalTests,
    StructuralHoles,
    TopicClustering,
)
from app.services.gds_service import get_gds_service, GDSService

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Reusable COALESCE expressions for dynamic name/type resolution ──
# These ensure every node type gets its best human-readable display name
# regardless of which property stores the actual text.
_NAME_COALESCE = "coalesce(node.name, node.title, node.question_text, node.text, node.content, node.label, node.code, node.questionId, node.studentId, node.examId, node.val, node.value, node.id)"
_TYPE_COALESCE = "coalesce(node.type, labels(node)[0])"

# Same for arbitrary aliases (n1, n2, a, b, n, etc.)
def _name_col(alias: str) -> str:
    return f"coalesce({alias}.name, {alias}.title, {alias}.question_text, {alias}.text, {alias}.content, {alias}.label, {alias}.code, {alias}.questionId, {alias}.studentId, {alias}.examId, {alias}.val, {alias}.value, {alias}.id)"

def _type_col(alias: str) -> str:
    return f"coalesce({alias}.type, labels({alias})[0])"


# === Algorithm Availability ===
@router.get("/available")
async def list_available_algorithms() -> Dict[str, Any]:
    """List all algorithms with their implementation status."""
    return {
        "implemented": [
            {"name": "pagerank", "category": "centrality", "description": "Find influential nodes (GDS)", "engine": "gds"},
            {"name": "articlerank", "category": "centrality", "description": "Improved influence for diverse graphs (GDS)", "engine": "gds"},
            {"name": "betweenness", "category": "centrality", "description": "Find bridge nodes (GDS)", "engine": "gds"},
            {"name": "closeness", "category": "centrality", "description": "Find central nodes by distance (GDS)", "engine": "gds"},
            {"name": "degree", "category": "centrality", "description": "Count direct connections per node (GDS)", "engine": "gds"},
            {"name": "hits", "category": "centrality", "description": "Hub and authority scores (GDS)", "engine": "gds"},
            {"name": "louvain", "category": "community", "description": "Community detection (GDS)", "engine": "gds"},
            {"name": "leiden", "category": "community", "description": "Improved community detection (GDS)", "engine": "gds"},
            {"name": "wcc", "category": "community", "description": "Identify connected islands (GDS)", "engine": "gds"},
            {"name": "k_core", "category": "decomposition", "description": "Core structure analysis (GDS)", "engine": "gds"},
            {"name": "triangle_count", "category": "community", "description": "Count structural triangles (GDS)", "engine": "gds"},
            {"name": "node_similarity", "category": "similarity", "description": "Jaccard similarity (GDS)", "engine": "gds"},
            {"name": "link_prediction", "category": "prediction", "description": "Predict missing links (GDS)", "engine": "gds"},
            {"name": "shortest_path", "category": "pathfinding", "description": "Dijkstra shortest path (GDS)", "engine": "gds"},
            {"name": "bfs_dfs", "category": "pathfinding", "description": "Breadth/Depth first search (GDS)", "engine": "gds"},
            {"name": "random_walk", "category": "pathfinding", "description": "Simulate random paths (GDS)", "engine": "gds"},
            {"name": "topological_sort", "category": "topology", "description": "Sequence nodes in a DAG (GDS)", "engine": "gds"},
        ],
        "exports": [
            {"name": "json", "description": "Export graph as JSON (APOC)", "engine": "apoc"},
            {"name": "cypher", "description": "Export as Cypher statements (APOC)", "engine": "apoc"},
        ],
        "coming_soon": [
            {"name": "ml_training", "category": "ml", "description": "Train ML models on graph"},
            {"name": "node_classification", "category": "ml", "description": "Auto-categorize nodes"},
            {"name": "graph_health", "category": "analysis", "description": "Comprehensive graph audit"},
            {"name": "completeness", "category": "analysis", "description": "Knowledge completeness score"},
            {"name": "degree_distribution", "category": "analysis", "description": "Node connectivity analysis"},
        ]
    }


# === Analytics Routes ===


# === GDS Centrality Algorithms ===
@router.get("/centrality/pagerank")
async def run_pagerank(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    top_k: int = Query(default=10, le=100),
    damping_factor: float = Query(default=0.85, ge=0.1, le=0.99),
    max_iterations: int = Query(default=20, le=100),
    weight_formula: Optional[str] = Query(default=None, description="JSON weight formula"),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run PageRank algorithm using Neo4j GDS for accurate centrality scores."""
    driver = get_neo4j_driver()
    wf = json.loads(weight_formula) if weight_formula else None
    graph_name = await gds.ensure_projection(folder_id, node_ids, weight_formula=wf)
    
    try:
        async with driver.session() as session:
            # Run GDS PageRank stream
            result = await session.run(f"""
                CALL gds.pageRank.stream($graph_name, {{
                    dampingFactor: $damping,
                    maxIterations: $max_iter,
                    relationshipWeightProperty: 'weight'
                }})
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS node, score
                RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, score
                ORDER BY score DESC
                LIMIT $top_k
            """, graph_name=graph_name, damping=damping_factor, max_iter=max_iterations, top_k=top_k)
            
            records = await result.data()
            
            # Calculate insights
            if records:
                top_node = records[0]
                insight = f"'{top_node['name']}' is currently the most influential entity in this dataset. Its high connectivity indicates it serves as a primary hub, significantly anchoring the surrounding knowledge graph."
            else:
                insight = "No significant influence hubs were identified in the current selection."
            
            return {
                "algorithm": "pagerank",
                "engine": "gds.pageRank.stream",
                "folder_id": folder_id,
                "parameters": {"damping_factor": damping_factor, "max_iterations": max_iterations, "top_k": top_k},
                "results": records,
                "insight": insight,
            }
    except Exception as e:
        logger.error(f"GDS PageRank error: {e}")
        raise HTTPException(status_code=500, detail=f"PageRank failed: {str(e)}")


@router.get("/centrality/betweenness")
async def run_betweenness(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    top_k: int = Query(default=10, le=100),
    sampling_size: int = Query(default=0, ge=0, description="0 = exact, >0 = sample size for approximation"),
    weight_formula: Optional[str] = Query(default=None, description="JSON weight formula"),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run Betweenness Centrality using Neo4j GDS to find critical bridge nodes."""
    driver = get_neo4j_driver()
    wf = json.loads(weight_formula) if weight_formula else None
    graph_name = await gds.ensure_projection(folder_id, node_ids, weight_formula=wf)
    
    try:
        async with driver.session() as session:
            # Use sampled or exact betweenness
            if sampling_size > 0:
                query = f"""
                    CALL gds.betweenness.stream($graph_name, {{
                        samplingSize: $sampling
                    }})
                    YIELD nodeId, score
                    WITH gds.util.asNode(nodeId) AS node, score
                    RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
                result = await session.run(query, graph_name=graph_name, sampling=sampling_size, top_k=top_k)
            else:
                query = f"""
                    CALL gds.betweenness.stream($graph_name)
                    YIELD nodeId, score
                    WITH gds.util.asNode(nodeId) AS node, score
                    RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
                result = await session.run(query, graph_name=graph_name, top_k=top_k)
            
            records = await result.data()
            
            # Insights
            if records:
                high_betweenness = [r for r in records if r["score"] > 0]
                if high_betweenness:
                    top = high_betweenness[0]
                    insight = f"'{top['name']}' acts as a vital bridge between isolated groups in your data. It plays a critical role in information flow, serving as a necessary pathway for cross-module communication."
                else:
                    insight = "Your data appears to be very densely connected, with no single entity acting as a unique bottleneck or bridge."
            else:
                insight = "No clear bridge entities were detected."
            
            return {
                "algorithm": "betweenness",
                "engine": "gds.betweenness.stream",
                "folder_id": folder_id,
                "parameters": {"top_k": top_k, "sampling_size": sampling_size},
                "results": records,
                "insight": insight,
            }
    except Exception as e:
        logger.error(f"GDS Betweenness error: {e}")
        raise HTTPException(status_code=500, detail=f"Betweenness failed: {str(e)}")


@router.get("/centrality/closeness")
async def run_closeness(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    top_k: int = Query(default=10, le=100),
    use_wasserman_faust: bool = Query(default=True, description="Normalize for disconnected graphs"),
    weight_formula: Optional[str] = Query(default=None, description="JSON weight formula"),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run Closeness Centrality using Neo4j GDS to find nodes closest to all others."""
    driver = get_neo4j_driver()
    wf = json.loads(weight_formula) if weight_formula else None
    graph_name = await gds.ensure_projection(folder_id, node_ids, weight_formula=wf)
    
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.closeness.stream($graph_name, {{
                    useWassermanFaust: $wf
                }})
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS node, score
                WHERE score > 0
                RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, score
                ORDER BY score DESC
                LIMIT $top_k
            """, graph_name=graph_name, wf=use_wasserman_faust, top_k=top_k)
            
            records = await result.data()
            
            if records:
                top = records[0]
                insight = f"'{top['name']}' is the most centrally positioned entity. It has the shortest paths to all other information in this set, making it the most efficient point for broadcasting or gathering data."
            else:
                insight = "Could not identify a central focus point in this specific dataset fragment."
            
            return {
                "algorithm": "closeness",
                "engine": "gds.closeness.stream",
                "folder_id": folder_id,
                "parameters": {"top_k": top_k, "use_wasserman_faust": use_wasserman_faust},
                "results": records,
                "insight": insight,
            }
    except Exception as e:
        logger.error(f"GDS Closeness error: {e}")
        raise HTTPException(status_code=500, detail=f"Closeness failed: {str(e)}")


@router.get("/centrality/articlerank")
async def run_articlerank(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    damping_factor: float = Query(default=0.85, ge=0.0, le=1.0),
    max_iterations: int = Query(default=20, le=100),
    top_k: int = Query(default=10, le=100),
    weight_formula: Optional[str] = Query(default=None, description="JSON weight formula"),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run ArticleRank (influence variant for diverse degree distributions)."""
    driver = get_neo4j_driver()
    wf = json.loads(weight_formula) if weight_formula else None
    graph_name = await gds.ensure_projection(folder_id, node_ids, weight_formula=wf)
    
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.articleRank.stream($graph_name, {{
                    dampingFactor: $damping,
                    maxIterations: $max_iter
                }})
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS node, score
                RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, score
                ORDER BY score DESC
                LIMIT $top_k
            """, graph_name=graph_name, damping=damping_factor, max_iter=max_iterations, top_k=top_k)
            
            records = await result.data()
            insight = f"ArticleRank identified '{records[0]['name']}' as the primary authority." if records else "No results."
            
            return {
                "algorithm": "articlerank",
                "engine": "gds.articleRank.stream",
                "folder_id": folder_id,
                "parameters": {"damping_factor": damping_factor, "max_iterations": max_iterations},
                "results": records,
                "insight": insight,
            }
    except Exception as e:
        logger.error(f"GDS ArticleRank error: {e}")
        raise HTTPException(status_code=500, detail=f"ArticleRank failed: {str(e)}")


@router.get("/centrality/degree")
async def run_degree_centrality(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    top_k: int = Query(default=10, le=100),
    weight_formula: Optional[str] = Query(default=None, description="JSON weight formula"),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run Degree Centrality to find the most directly connected nodes."""
    driver = get_neo4j_driver()
    wf = json.loads(weight_formula) if weight_formula else None
    graph_name = await gds.ensure_projection(folder_id, node_ids, weight_formula=wf)
    
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.degree.stream($graph_name)
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS node, score
                RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, score
                ORDER BY score DESC
                LIMIT $top_k
            """, graph_name=graph_name, top_k=top_k)
            
            records = await result.data()
            
            if records:
                top = records[0]
                insight = f"'{top['name']}' has the most direct connections ({top['score']:.0f}) in this dataset, making it the most actively linked entity."
            else:
                insight = "No degree data found."
            
            return {
                "algorithm": "degree",
                "engine": "gds.degree.stream",
                "folder_id": folder_id,
                "parameters": {"top_k": top_k},
                "results": records,
                "insight": insight,
            }
    except Exception as e:
        logger.error(f"GDS Degree error: {e}")
        raise HTTPException(status_code=500, detail=f"Degree failed: {str(e)}")

@router.get("/centrality/hits")
async def run_hits_gds(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    max_iterations: int = Query(default=20, le=100),
    top_k: int = Query(default=10, le=100),
    weight_formula: Optional[str] = Query(default=None, description="JSON weight formula"),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run HITS (Hubs and Authorities) algorithm using Neo4j GDS."""
    driver = get_neo4j_driver()
    wf = json.loads(weight_formula) if weight_formula else None
    graph_name = await gds.ensure_projection(folder_id, node_ids, weight_formula=wf)
    
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.hits.stream($graph_name, {{}})
                YIELD nodeId, values
                WITH gds.util.asNode(nodeId) AS node, values.hub AS hubScore, values.auth AS authScore
                RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, 
                       hubScore AS hub_score, authScore AS auth_score
                ORDER BY authScore DESC
                LIMIT $top_k
            """, graph_name=graph_name, max_iter=max_iterations, top_k=top_k)
            
            records = await result.data()
            
            if records:
                top = records[0]
                insight = f"'{top['name']}' is the strongest authority in this set. Hub scores indicate '{records[0]['name']}' also serves as a key information aggregator."
            else:
                insight = "No clear hubs or authorities found."
            
            return {
                "algorithm": "hits",
                "engine": "gds.hits.stream",
                "folder_id": folder_id,
                "parameters": {"max_iterations": max_iterations, "top_k": top_k},
                "results": records,
                "insight": insight,
            }
    except Exception as e:
        logger.error(f"GDS HITS error: {e}")
        raise HTTPException(status_code=500, detail=f"HITS failed: {str(e)}")


# === GDS Community Detection ===
@router.get("/community/louvain")
async def run_louvain(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    include_intermediate: bool = Query(default=False),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run Louvain community detection using Neo4j GDS."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id, node_ids, undirected=True)
    
    # Build scope filter — node_ids takes priority (more specific scope)
    scope_filter = ""
    params = {"graph_name": graph_name, "intermediate": include_intermediate}
    if node_ids:
        scope_filter = "WHERE node.id IN $node_ids"
        params["node_ids"] = node_ids
    elif folder_id:
        scope_filter = "WHERE node.folder_id = $folder_id"
        params["folder_id"] = folder_id
    
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.louvain.stream($graph_name, {{
                    includeIntermediateCommunities: $intermediate
                }})
                YIELD nodeId, communityId, intermediateCommunityIds
                WITH gds.util.asNode(nodeId) AS node, communityId, intermediateCommunityIds
                {scope_filter}
                RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, 
                       communityId AS community_id,
                       intermediateCommunityIds AS hierarchy
                ORDER BY communityId, {_NAME_COALESCE}
            """, **params)
            
            records = await result.data()
            
            # Group by community
            communities = {}
            for r in records:
                cid = r["community_id"]
                if cid not in communities:
                    communities[cid] = []
                communities[cid].append({"id": r["id"], "name": r["name"], "type": r["type"]})
            
            community_sizes = sorted([(cid, len(members)) for cid, members in communities.items()], 
                                    key=lambda x: -x[1])
            
            insight = f"The engine successfully mapped {len(communities)} distinct thematic clusters. The largest group represents a dense hub of {community_sizes[0][1]} related entities working in unison." if communities else "No distinct community structures were identified in this selection."
            
            return {
                "algorithm": "louvain",
                "engine": "gds.louvain.stream",
                "folder_id": folder_id,
                "parameters": {"include_intermediate": include_intermediate},
                "community_count": len(communities),
                "communities": communities,
                "results": records[:50],  # Limit for response size
                "insight": insight,
            }
    except Exception as e:
        logger.error(f"GDS Louvain error: {e}")
        raise HTTPException(status_code=500, detail=f"Louvain failed: {str(e)}")


@router.get("/community/leiden")
async def run_leiden(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    gamma: float = Query(default=1.0, ge=0.1, le=10.0, description="Resolution parameter"),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run Leiden community detection (improved Louvain) using Neo4j GDS."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id, node_ids, undirected=True)
    
    scope_filter = ""
    params = {"graph_name": graph_name, "gamma": gamma}
    if node_ids:
        scope_filter = "WHERE node.id IN $node_ids"
        params["node_ids"] = node_ids
    elif folder_id:
        scope_filter = "WHERE node.folder_id = $folder_id"
        params["folder_id"] = folder_id
    
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.leiden.stream($graph_name, {{
                    gamma: $gamma
                }})
                YIELD nodeId, communityId
                WITH gds.util.asNode(nodeId) AS node, communityId
                {scope_filter}
                RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, 
                       communityId AS community_id
                ORDER BY communityId, {_NAME_COALESCE}
            """, **params)
            
            records = await result.data()
            
            # Group by community
            communities = {}
            for r in records:
                cid = r["community_id"]
                if cid not in communities:
                    communities[cid] = []
                communities[cid].append({"id": r["id"], "name": r["name"], "type": r["type"]})
            
            insight = f"Leiden found {len(communities)} communities with gamma={gamma}." if communities else "No communities found."
            
            return {
                "algorithm": "leiden",
                "engine": "gds.leiden.stream",
                "folder_id": folder_id,
                "parameters": {"gamma": gamma},
                "community_count": len(communities),
                "communities": communities,
                "results": records[:50],
                "insight": insight,
            }
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"GDS Leiden error: {e}")
        if "undirected" in error_msg or "orientation" in error_msg or "not found" in error_msg or "weight" in error_msg:
            return {
                "algorithm": "leiden",
                "engine": "gds.leiden.stream",
                "folder_id": folder_id,
                "results": [],
                "communities": {},
                "community_count": 0,
                "insight": "Leiden could not run on this dataset. The graph may not have enough relationships or the right structure for community detection. Try Louvain instead.",
            }
        raise HTTPException(status_code=500, detail=f"Leiden failed: {str(e)}")


@router.get("/community/wcc")
async def run_wcc_gds(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run Weakly Connected Components (WCC) to identify disconnected islands."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id, node_ids, undirected=True)
    
    scope_filter = ""
    params = {"graph_name": graph_name}
    if node_ids:
        scope_filter = "WHERE node.id IN $node_ids"
        params["node_ids"] = node_ids
    elif folder_id:
        scope_filter = "WHERE node.folder_id = $folder_id"
        params["folder_id"] = folder_id
    
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.wcc.stream($graph_name)
                YIELD nodeId, componentId
                WITH gds.util.asNode(nodeId) AS node, componentId
                {scope_filter}
                RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, 
                       componentId AS community_id
                ORDER BY componentId
            """, **params)
            
            records = await result.data()
            
            communities = {}
            for r in records:
                cid = r["community_id"]
                if cid not in communities:
                    communities[cid] = []
                communities[cid].append({"id": r["id"], "name": r["name"], "type": r["type"]})
            
            # Sort communities by size (descending) to find the main cluster
            sorted_comps = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
            
            if len(sorted_comps) <= 1:
                insight = "All nodes are connected in a single component. There are no isolated islands in this current view."
            else:
                giant_id, giant_nodes = sorted_comps[0]
                islands = sorted_comps[1:]
                island_node_count = sum(len(c[1]) for c in islands)
                
                # Get names from the smaller components
                sample_isolated = []
                for _, nodes in islands[:3]:
                    sample_isolated.append(nodes[0]['name'])
                
                names_str = ", ".join(sample_isolated)
                if len(islands) > 3:
                    names_str += ", and others"
                
                insight = (f"Found {len(sorted_comps)} disconnected islands. The main cluster has {len(giant_nodes)} nodes. "
                          f"There are {len(islands)} isolated islands (total {island_node_count} nodes), including: {names_str}.")
            
            return {
                "algorithm": "wcc",
                "engine": "gds.wcc.stream",
                "folder_id": folder_id,
                "community_count": len(communities),
                "communities": communities,
                "results": records[:50],
                "insight": insight,
            }
    except Exception as e:
        logger.error(f"GDS WCC error: {e}")
        raise HTTPException(status_code=500, detail=f"WCC failed: {str(e)}")


@router.get("/community/kcore")
async def run_kcore_gds(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    k: int = Query(default=3, ge=1),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run K-Core decomposition to find the stable core of the graph."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id, node_ids, undirected=True)
    
    scope_filter = "WHERE coreValue >= $k"
    params = {"graph_name": graph_name, "k": k}
    if node_ids:
        scope_filter = "WHERE coreValue >= $k AND node.id IN $node_ids"
        params["node_ids"] = node_ids
    elif folder_id:
        scope_filter = "WHERE coreValue >= $k AND node.folder_id = $folder_id"
        params["folder_id"] = folder_id
    
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.kcore.stream($graph_name)
                YIELD nodeId, coreValue
                WITH gds.util.asNode(nodeId) AS node, coreValue
                {scope_filter}
                RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, 
                       coreValue AS score
                ORDER BY coreValue DESC
            """, **params)
            
            records = await result.data()
            
            insight = f"{len(records)} nodes belong to the {k}-core (the most stable, highly-connected center of the graph)." if records else f"No nodes found in the {k}-core."
            
            return {
                "algorithm": "kcore",
                "engine": "gds.kcore.stream",
                "folder_id": folder_id,
                "parameters": {"k": k},
                "results": records,
                "insight": insight,
            }
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"GDS K-Core error: {e}")
        if "undirected" in error_msg or "orientation" in error_msg:
            return {
                "algorithm": "kcore",
                "engine": "gds.kcore.stream",
                "folder_id": folder_id,
                "results": [],
                "insight": "K-Core requires undirected relationships. Your graph might be directed or have a structure that doesn't support core decomposition yet.",
            }
        raise HTTPException(status_code=500, detail=f"K-Core failed: {str(e)}")


@router.get("/community/triangles")
async def run_triangle_count_gds(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run Triangle Counting to measure local group density."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id, node_ids, undirected=True)
    
    scope_filter = ""
    params = {"graph_name": graph_name}
    if node_ids:
        scope_filter = "WHERE node.id IN $node_ids"
        params["node_ids"] = node_ids
    elif folder_id:
        scope_filter = "WHERE node.folder_id = $folder_id"
        params["folder_id"] = folder_id
    
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.triangleCount.stream($graph_name)
                YIELD nodeId, triangleCount
                WITH gds.util.asNode(nodeId) AS node, triangleCount
                {scope_filter}
                RETURN node.id AS id, {_NAME_COALESCE} AS name, {_TYPE_COALESCE} AS type, 
                       triangleCount AS score
                ORDER BY triangleCount DESC
                LIMIT 50
            """, **params)
            
            records = await result.data()
            total_triangles = sum(r["score"] for r in records)
            
            insight = f"Found {total_triangles} structural triangles. Entities with high triangle counts are part of very tight-knit, collaborative groups."
            
            return {
                "algorithm": "triangle_count",
                "engine": "gds.triangleCount.stream",
                "folder_id": folder_id,
                "results": records,
                "insight": insight,
            }
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"GDS Triangle Count error: {e}")
        if "undirected" in error_msg or "orientation" in error_msg:
            return {
                "algorithm": "triangle_count",
                "engine": "gds.triangleCount.stream",
                "folder_id": folder_id,
                "results": [],
                "insight": "Triangle Count requires undirected relationships. Your current data may not have the right structure for this algorithm. Try adding more interconnected entities.",
            }
        raise HTTPException(status_code=500, detail=f"Triangle count failed: {str(e)}")


# === GDS Similarity ===
@router.get("/similarity/nodes")
async def run_node_similarity(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    top_k: int = Query(default=10, le=100),
    similarity_cutoff: float = Query(default=0.1, ge=0.0, le=1.0),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Find similar node pairs using GDS Node Similarity (Jaccard)."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id, node_ids, undirected=True)
    
    scope_filter = ""
    params = {"graph_name": graph_name, "top_k": top_k, "cutoff": similarity_cutoff}
    if node_ids:
        scope_filter = "WHERE n1.id IN $node_ids AND n2.id IN $node_ids"
        params["node_ids"] = node_ids
    elif folder_id:
        scope_filter = "WHERE n1.folder_id = $folder_id AND n2.folder_id = $folder_id"
        params["folder_id"] = folder_id
    
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.nodeSimilarity.stream($graph_name, {{
                    topK: $top_k,
                    similarityCutoff: $cutoff
                }})
                YIELD node1, node2, similarity
                WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
                {scope_filter}
                RETURN n1.id AS source_id, {_name_col('n1')} AS source_name, {_type_col('n1')} AS source_type,
                       n2.id AS target_id, {_name_col('n2')} AS target_name, {_type_col('n2')} AS target_type,
                       similarity AS score
                ORDER BY score DESC
                LIMIT $top_k
            """, **params)
            
            records = await result.data()
            
            if records:
                top = records[0]
                insight = f"'{top['source_name']}' and '{top['target_name']}' are most similar ({top['score']:.2%} Jaccard similarity)."
            else:
                insight = "No similar node pairs found above the cutoff."
            
            return {
                "algorithm": "node_similarity",
                "engine": "gds.nodeSimilarity.stream",
                "folder_id": folder_id,
                "parameters": {"top_k": top_k, "similarity_cutoff": similarity_cutoff},
                "results": records,
                "insight": insight,
            }
    except Exception as e:
        logger.error(f"GDS Node Similarity error: {e}")
        raise HTTPException(status_code=500, detail=f"Node Similarity failed: {str(e)}")


@router.get("/similarity/knn")
async def run_knn(
    node_id: str,
    top_k: int = Query(default=5, le=50),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Find K nearest neighbors using similarity."""
    similarity = EntitySimilarity()
    results = await similarity.find_similar(node_id, top_k)
    return {
        "algorithm": "knn",
        "source_node": node_id,
        "neighbors": results,
    }


# === GDS Path Finding ===
@router.get("/path/shortest")
async def find_shortest_path(
    source_id: str,
    target_id: str,
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Find shortest path between two nodes using GDS Dijkstra."""
    driver = get_neo4j_driver()
    # Ensure source and target are in the node_ids list for the projection
    projection_ids = list(node_ids) if node_ids else None
    if projection_ids is not None:
        if source_id not in projection_ids: projection_ids.append(source_id)
        if target_id not in projection_ids: projection_ids.append(target_id)

    graph_name = await gds.ensure_projection(folder_id, projection_ids)
    
    try:
        async with driver.session() as session:
            # Get internal node IDs (allow any label as long as ID matches)
            id_result = await session.run("""
                MATCH (source {id: $source_id}), (target {id: $target_id})
                RETURN id(source) AS source_neo_id, id(target) AS target_neo_id
            """, source_id=source_id, target_id=target_id)
            
            id_record = await id_result.single()
            if not id_record:
                raise HTTPException(status_code=404, detail="Source or target node not found")
            
            source_neo_id = id_record["source_neo_id"]
            target_neo_id = id_record["target_neo_id"]
            
            # Run Dijkstra
            result = await session.run("""
                CALL gds.shortestPath.dijkstra.stream($graph_name, {
                    sourceNode: $source,
                    targetNode: $target,
                    relationshipWeightProperty: 'weight'
                })
                YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
                WITH nodeIds, totalCost
                UNWIND nodeIds AS nodeId
                WITH collect(gds.util.asNode(nodeId)) AS nodes, totalCost
                RETURN [n IN nodes | {id: n.id, name: coalesce(n.name, n.title, n.question_text, n.text, n.content, n.label, n.code, n.questionId, n.studentId, n.examId, n.val, n.value, n.id), type: coalesce(n.type, labels(n)[0])}] AS path, 
                       totalCost AS total_cost,
                       size(nodes) AS path_length
            """, graph_name=graph_name, source=source_neo_id, target=target_neo_id)
            
            record = await result.single()
            
            if record:
                path = record["path"]
                cost = record["total_cost"]
                length = record["path_length"]
                insight = f"Found path with {length} nodes and total cost {cost:.2f}."
            else:
                path = []
                cost = -1
                length = 0
                insight = "No path exists between these nodes."
            
            return {
                "algorithm": "shortest_path",
                "engine": "gds.shortestPath.dijkstra.stream",
                "source_id": source_id,
                "target_id": target_id,
                "path": path,
                "results": path,  # Consistent with other algorithms for table display
                "total_cost": cost,
                "path_length": length,
                "insight": insight,
            }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"GDS Shortest Path error: {e}")
        if "not found" in error_msg or "projection" in error_msg or "in-memory" in error_msg or "not exist" in error_msg:
            return {
                "algorithm": "shortest_path",
                "engine": "gds.shortestPath.dijkstra.stream",
                "results": [],
                "insight": "Could not find a path. One or both nodes might not be present in the current view or folder."
            }
        raise HTTPException(status_code=500, detail=f"Shortest path failed: {str(e)}")


@router.get("/path/traversal")
async def run_traversal(
    source_id: str,
    method: str = Query(default="bfs", enum=["bfs", "dfs"]),
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run BFS or DFS traversal from a source node."""
    driver = get_neo4j_driver()
    # BFS/DFS often work best on undirected graphs to find all conceptual neighbors
    graph_name = await gds.ensure_projection(folder_id, node_ids, undirected=True)
    
    try:
        async with driver.session() as session:
            # Get internal node ID
            id_res = await session.run("MATCH (n:Entity {id: $id}) RETURN id(n) AS neo_id", id=source_id)
            rec = await id_res.single()
            if not rec: raise HTTPException(status_code=404, detail="Start node not found")
            
            proc = "bfs" if method == "bfs" else "dfs"
            if method == "bfs":
                records = await gds.run_bfs(source_id, folder_id, node_ids)
            else:
                records = await gds.run_dfs(source_id, folder_id, node_ids)

            return {
                "algorithm": method,
                "engine": f"gds.{proc}.stream",
                "source_id": source_id,
                "results": records,
                "insight": f"{method.upper()} traversal explored {len(records)} nodes starting from '{source_id}'."
            }
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"GDS Traversal error: {e}")
        if "not found" in error_msg or "projection" in error_msg:
            return {
                "algorithm": method,
                "engine": f"gds.{method}.stream",
                "source_id": source_id,
                "results": [],
                "insight": f"Could not start {method.upper()} traversal. The starting node might not be present in the current view or folder."
            }
        raise HTTPException(status_code=500, detail=f"Traversal failed: {str(e)}")


@router.get("/path/random-walk")
async def run_random_walk(
    source_id: str,
    walk_length: int = Query(default=10, le=50),
    walk_count: int = Query(default=1),
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Simulate random walks from a source node."""
    driver = get_neo4j_driver()
    # Ensure source node is in the node_ids list for the projection
    projection_ids = list(node_ids) if node_ids else None
    if projection_ids is not None and source_id not in projection_ids:
        projection_ids.append(source_id)

    graph_name = await gds.ensure_projection(folder_id, projection_ids)
    
    try:
        async with driver.session() as session:
            # Allow any label for starting node
            id_res = await session.run("MATCH (n {id: $id}) RETURN id(n) AS neo_id", id=source_id)
            rec = await id_res.single()
            if not rec: raise HTTPException(status_code=404, detail="Start node not found")
            
            records = await gds.run_random_walk(source_id, folder_id, projection_ids, walk_length, walk_count)
            
            return {
                "algorithm": "random_walk",
                "engine": "gds.randomWalk.stream",
                "results": records,
                "insight": f"Completed {walk_count} random walks of length {walk_length}."
            }
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"GDS Random Walk error: {e}")
        if "not found" in error_msg or "projection" in error_msg or "in-memory" in error_msg or "not exist" in error_msg:
            return {
                "algorithm": "random_walk",
                "engine": "gds.randomWalk.stream",
                "results": [],
                "insight": "Could not start random walk. The starting node might not be present in the current view or folder."
            }
        raise HTTPException(status_code=500, detail=f"Random walk failed: {str(e)}")


@router.get("/topology/topological-sort")
async def run_topological_sort(
    folder_id: Optional[str] = None,
    node_ids: Optional[List[str]] = Query(default=None),
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Sequence nodes in a Directed Acyclic Graph (DAG)."""
    driver = get_neo4j_driver()
    # Note: Topological sort requires directed relationships, so we use undirected=False (default)
    graph_name = await gds.ensure_projection(folder_id, node_ids)
    
    scope_filter = ""
    params = {"graph_name": graph_name}
    if node_ids:
        scope_filter = "WHERE n.id IN $node_ids"
        params["node_ids"] = node_ids
    elif folder_id:
        scope_filter = "WHERE n.folder_id = $folder_id"
        params["folder_id"] = folder_id
        
    try:
        async with driver.session() as session:
            result = await session.run(f"""
                CALL gds.dag.topologicalSort.stream($graph_name)
                YIELD nodeId
                WITH gds.util.asNode(nodeId) AS n
                {scope_filter}
                RETURN n.id AS id, {_name_col('n')} AS name, {_type_col('n')} AS type
            """, **params)
            
            records = await result.data()
            return {
                "algorithm": "topological_sort",
                "engine": "gds.dag.topologicalSort.stream",
                "folder_id": folder_id,
                "results": records,
                "insight": f"Topological sort successful for {len(records)} nodes. This defines a valid logical sequence."
            }
    except Exception as e:
        logger.error(f"GDS TopoSort error: {e}")
        raise HTTPException(status_code=500, detail="Topological sort failed. Ensure your graph is a DAG (Directed Acyclic Graph).")


# === GDS Link Prediction ===
@router.get("/link-prediction")
async def run_link_prediction_gds(
    folder_id: Optional[str] = None,
    method: str = Query(default="common_neighbors", enum=[
        "common_neighbors", 
        "adamic_adar", 
        "preferential_attachment",
        "resource_allocation",
        "total_neighbors",
        "same_community"
    ]),
    top_k: int = Query(default=20, le=100),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Predict missing links using Cypher-based link prediction algorithms."""
    driver = get_neo4j_driver()
    
    try:
        async with driver.session() as session:
            # Build the folder filter for node matching
            folder_filter = ""
            if folder_id:
                folder_filter = " {folder_id: $folder_id}"
            
            if method == "common_neighbors":
                # Count of shared neighbors between two non-connected nodes
                query = f"""
                    MATCH (a:Entity{folder_filter}), (b:Entity{folder_filter})
                    WHERE a <> b AND NOT (a)--(b) AND id(a) < id(b)
                    WITH a, b
                    OPTIONAL MATCH (a)--(neighbor)--(b)
                    WITH a, b, count(DISTINCT neighbor) AS score
                    WHERE score > 0
                    RETURN a.id AS source_id, {_name_col('a')} AS source_name,
                           b.id AS target_id, {_name_col('b')} AS target_name,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
            elif method == "adamic_adar":
                # Sum of 1/log(degree) for each shared neighbor
                query = f"""
                    MATCH (a:Entity{folder_filter}), (b:Entity{folder_filter})
                    WHERE a <> b AND NOT (a)--(b) AND id(a) < id(b)
                    WITH a, b
                    MATCH (a)--(neighbor)--(b)
                    WITH a, b, neighbor, COUNT {{ (neighbor)--() }} AS degree
                    WHERE degree > 1
                    WITH a, b, sum(1.0 / log(toFloat(degree))) AS score
                    WHERE score > 0
                    RETURN a.id AS source_id, {_name_col('a')} AS source_name,
                           b.id AS target_id, {_name_col('b')} AS target_name,
                           round(score * 1000) / 1000.0 AS score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
            elif method == "preferential_attachment":
                # Product of degrees of the two nodes
                query = f"""
                    MATCH (a:Entity{folder_filter}), (b:Entity{folder_filter})
                    WHERE a <> b AND NOT (a)--(b) AND id(a) < id(b)
                    WITH a, b, COUNT {{ (a)--() }} * COUNT {{ (b)--() }} AS score
                    WHERE score > 0
                    RETURN a.id AS source_id, {_name_col('a')} AS source_name,
                           b.id AS target_id, {_name_col('b')} AS target_name,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
            elif method == "resource_allocation":
                # Sum of 1/degree for each shared neighbor
                query = f"""
                    MATCH (a:Entity{folder_filter}), (b:Entity{folder_filter})
                    WHERE a <> b AND NOT (a)--(b) AND id(a) < id(b)
                    WITH a, b
                    MATCH (a)--(neighbor)--(b)
                    WITH a, b, neighbor, toFloat(COUNT {{ (neighbor)--() }}) AS degree
                    WHERE degree > 0
                    WITH a, b, sum(1.0 / degree) AS score
                    WHERE score > 0
                    RETURN a.id AS source_id, {_name_col('a')} AS source_name,
                           b.id AS target_id, {_name_col('b')} AS target_name,
                           round(score * 1000) / 1000.0 AS score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
            elif method == "total_neighbors":
                # Total unique neighbors of either node (union)
                query = f"""
                    MATCH (a:Entity{folder_filter}), (b:Entity{folder_filter})
                    WHERE a <> b AND NOT (a)--(b) AND id(a) < id(b)
                    WITH a, b
                    OPTIONAL MATCH (a)--(na)
                    WITH a, b, collect(DISTINCT na) AS aNeighbors
                    OPTIONAL MATCH (b)--(nb)
                    WITH a, b, aNeighbors, collect(DISTINCT nb) AS bNeighbors
                    WITH a, b, size(apoc.coll.union(aNeighbors, bNeighbors)) AS score
                    WHERE score > 0
                    RETURN a.id AS source_id, {_name_col('a')} AS source_name,
                           b.id AS target_id, {_name_col('b')} AS target_name,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
            else:  # same_community — check for shared type
                query = f"""
                    MATCH (a:Entity{folder_filter}), (b:Entity{folder_filter})
                    WHERE a <> b AND NOT (a)--(b) AND id(a) < id(b)
                      AND a.type IS NOT NULL AND a.type = b.type
                    WITH a, b, 
                         CASE WHEN a.type = b.type THEN 1.0 ELSE 0.0 END AS score
                    WHERE score > 0
                    RETURN a.id AS source_id, {_name_col('a')} AS source_name,
                           b.id AS target_id, {_name_col('b')} AS target_name,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
            
            params = {"top_k": top_k}
            if folder_id:
                params["folder_id"] = folder_id
            
            result = await session.run(query, **params)
            records = await result.data()
            
            if records:
                top = records[0]
                insight = f"Strongest new link prediction: '{top['source_name']}' ↔ '{top['target_name']}' (score: {top['score']:.2f}). Identified {len(records)} potential new connections."
            else:
                insight = "No new link predictions above threshold."
            
            return {
                "algorithm": "link_prediction",
                "engine": f"cypher.linkPrediction.{method}",
                "folder_id": folder_id,
                "parameters": {"method": method, "top_k": top_k},
                "results": records,
                "insight": insight,
            }
    except Exception as e:
        logger.error(f"GDS Link Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Link prediction failed: {str(e)}")


# === APOC Export Endpoints ===
@router.get("/export/json")
async def export_graph_json(
    folder_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Export graph data as JSON using APOC."""
    driver = get_neo4j_driver()
    
    try:
        async with driver.session() as session:
            if folder_id:
                result = await session.run("""
                    MATCH (n:Entity {folder_id: $folder_id})
                    OPTIONAL MATCH (n)-[r]-(m:Entity {folder_id: $folder_id})
                    WITH collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS rels
                    CALL apoc.export.json.data(nodes, rels, null, {stream: true})
                    YIELD data
                    RETURN data
                """, folder_id=folder_id)
            else:
                result = await session.run("""
                    MATCH (n:Entity)
                    OPTIONAL MATCH (n)-[r]-(m:Entity)
                    WITH collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS rels
                    CALL apoc.export.json.data(nodes, rels, null, {stream: true})
                    YIELD data
                    RETURN data
                """)
            
            record = await result.single()
            data = record["data"] if record else "{}"
            
            return {
                "format": "json",
                "engine": "apoc.export.json.data",
                "folder_id": folder_id,
                "data": json.loads(data) if isinstance(data, str) else data,
            }
    except Exception as e:
        logger.error(f"APOC JSON export error: {e}")
        raise HTTPException(status_code=500, detail=f"JSON export failed: {str(e)}")


@router.get("/export/cypher")
async def export_graph_cypher(
    folder_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Export graph as Cypher statements using APOC."""
    driver = get_neo4j_driver()
    
    try:
        async with driver.session() as session:
            if folder_id:
                result = await session.run("""
                    MATCH (n:Entity {folder_id: $folder_id})
                    OPTIONAL MATCH (n)-[r]-(m:Entity {folder_id: $folder_id})
                    WITH collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS rels
                    CALL apoc.export.cypher.data(nodes, rels, null, {stream: true, format: 'cypher-shell'})
                    YIELD cypherStatements
                    RETURN cypherStatements
                """, folder_id=folder_id)
            else:
                result = await session.run("""
                    MATCH (n:Entity)
                    OPTIONAL MATCH (n)-[r]-(m:Entity)
                    WITH collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS rels
                    CALL apoc.export.cypher.data(nodes, rels, null, {stream: true, format: 'cypher-shell'})
                    YIELD cypherStatements
                    RETURN cypherStatements
                """)
            
            record = await result.single()
            statements = record["cypherStatements"] if record else ""
            
            return {
                "format": "cypher",
                "engine": "apoc.export.cypher.data",
                "folder_id": folder_id,
                "statements": statements,
                "line_count": len(statements.split('\n')) if statements else 0,
            }
    except Exception as e:
        logger.error(f"APOC Cypher export error: {e}")
        raise HTTPException(status_code=500, detail=f"Cypher export failed: {str(e)}")


# === Custom Algorithms (kept for compatibility) ===
@router.get("/health")
async def run_graph_health(
    folder_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Run comprehensive graph health audit."""
    health = GraphHealth()
    return await health.audit(folder_id)


@router.get("/completeness")
async def run_knowledge_completeness(
    folder_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Analyze knowledge completeness score."""
    completeness = KnowledgeCompleteness()
    return await completeness.analyze(folder_id)


@router.get("/degree-distribution")
async def run_degree_distribution(
    folder_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Analyze degree distribution and hub nodes."""
    dd = DegreeDistribution()
    return await dd.analyze(folder_id)


@router.get("/missing-relationships")
async def run_missing_relationships(
    folder_id: Optional[str] = None,
    min_confidence: float = Query(default=0.7, ge=0.0, le=1.0),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Find definite missing relationships."""
    mr = MissingRelationships()
    return await mr.find_missing(folder_id, min_confidence)


@router.get("/incomplete-entities")
async def run_incomplete_entities(
    folder_id: Optional[str] = None,
    threshold: float = Query(default=0.7, ge=0.0, le=1.0),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Find entities with missing properties."""
    ie = IncompleteEntities()
    return await ie.find_incomplete(folder_id, threshold)


@router.get("/structural-holes")
async def run_structural_holes(
    folder_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Find structural holes for bridging opportunities."""
    sh = StructuralHoles()
    return await sh.find_holes(folder_id)


@router.get("/hits")
async def run_hits(
    folder_id: Optional[str] = None,
    iterations: int = Query(default=20, le=100),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Run HITS algorithm for hubs and authorities."""
    hits = HITS()
    return await hits.analyze(folder_id, iterations)


@router.get("/k-core")
async def run_k_core(
    folder_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Run K-Core decomposition."""
    kcore = KCore()
    return await kcore.decompose(folder_id)


# === ML Pipelines (Coming Soon) ===
@router.post("/ml/train/{pipeline_type}")
async def train_ml_pipeline(
    pipeline_type: str,
    folder_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Train an ML pipeline (link-prediction, node-classification)."""
    return {
        "status": "coming_soon",
        "pipeline_type": pipeline_type,
        "message": "ML pipelines will be available in a future release. Use Link Prediction or Graph Health for similar insights.",
    }


@router.get("/ml/models")
async def list_ml_models(
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """List trained ML models."""
    return {
        "models": [],
        "message": "ML model training coming soon.",
    }


@router.post("/ml/predict/{model_name}")
async def run_ml_prediction(
    model_name: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Run prediction using a trained model."""
    return {
        "model": model_name,
        "predictions": [],
        "status": "coming_soon",
        "message": "ML predictions coming soon.",
    }
