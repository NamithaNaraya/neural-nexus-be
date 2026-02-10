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


# === Algorithm Availability ===
@router.get("/available")
async def list_available_algorithms() -> Dict[str, Any]:
    """List all algorithms with their implementation status."""
    return {
        "implemented": [
            {"name": "pagerank", "category": "centrality", "description": "Find influential nodes (GDS)", "engine": "gds"},
            {"name": "betweenness", "category": "centrality", "description": "Find bridge nodes (GDS)", "engine": "gds"},
            {"name": "closeness", "category": "centrality", "description": "Find central nodes by distance (GDS)", "engine": "gds"},
            {"name": "louvain", "category": "community", "description": "Community detection (GDS)", "engine": "gds"},
            {"name": "leiden", "category": "community", "description": "Improved community detection (GDS)", "engine": "gds"},
            {"name": "node_similarity", "category": "similarity", "description": "Jaccard similarity (GDS)", "engine": "gds"},
            {"name": "shortest_path", "category": "pathfinding", "description": "Dijkstra shortest path (GDS)", "engine": "gds"},
            {"name": "graph_health", "category": "analysis", "description": "Comprehensive graph audit"},
            {"name": "completeness", "category": "analysis", "description": "Knowledge completeness score"},
            {"name": "degree_distribution", "category": "analysis", "description": "Node connectivity analysis"},
            {"name": "link_prediction", "category": "prediction", "description": "Predict missing links (GDS)", "engine": "gds"},
            {"name": "missing_relationships", "category": "prediction", "description": "Find definite gaps"},
            {"name": "incomplete_entities", "category": "analysis", "description": "Find incomplete nodes"},
            {"name": "structural_holes", "category": "analysis", "description": "Bridging opportunities"},
            {"name": "hits", "category": "centrality", "description": "Hub and authority scores"},
            {"name": "k_core", "category": "decomposition", "description": "Core structure analysis"},
        ],
        "exports": [
            {"name": "json", "description": "Export graph as JSON (APOC)", "engine": "apoc"},
            {"name": "cypher", "description": "Export as Cypher statements (APOC)", "engine": "apoc"},
        ],
        "coming_soon": [
            {"name": "ml_training", "category": "ml", "description": "Train ML models on graph"},
            {"name": "node_classification", "category": "ml", "description": "Auto-categorize nodes"},
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
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run PageRank algorithm using Neo4j GDS for accurate centrality scores."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id, node_ids)
    
    try:
        async with driver.session() as session:
            # Run GDS PageRank stream
            result = await session.run("""
                CALL gds.pageRank.stream($graph_name, {
                    dampingFactor: $damping,
                    maxIterations: $max_iter
                })
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS node, score
                RETURN node.id AS id, node.name AS name, node.type AS type, score
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
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run Betweenness Centrality using Neo4j GDS to find critical bridge nodes."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id, node_ids)
    
    try:
        async with driver.session() as session:
            # Use sampled or exact betweenness
            if sampling_size > 0:
                query = """
                    CALL gds.betweenness.stream($graph_name, {
                        samplingSize: $sampling
                    })
                    YIELD nodeId, score
                    WITH gds.util.asNode(nodeId) AS node, score
                    RETURN node.id AS id, node.name AS name, node.type AS type, score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
                result = await session.run(query, graph_name=graph_name, sampling=sampling_size, top_k=top_k)
            else:
                query = """
                    CALL gds.betweenness.stream($graph_name)
                    YIELD nodeId, score
                    WITH gds.util.asNode(nodeId) AS node, score
                    RETURN node.id AS id, node.name AS name, node.type AS type, score
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
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Run Closeness Centrality using Neo4j GDS to find nodes closest to all others."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id, node_ids)
    
    try:
        async with driver.session() as session:
            result = await session.run("""
                CALL gds.closeness.stream($graph_name, {
                    useWassermanFaust: $wf
                })
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS node, score
                WHERE score > 0
                RETURN node.id AS id, node.name AS name, node.type AS type, score
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
    graph_name = await gds.ensure_projection(folder_id, node_ids)
    
    try:
        async with driver.session() as session:
            result = await session.run("""
                CALL gds.louvain.stream($graph_name, {
                    includeIntermediateCommunities: $intermediate
                })
                YIELD nodeId, communityId, intermediateCommunityIds
                WITH gds.util.asNode(nodeId) AS node, communityId, intermediateCommunityIds
                RETURN node.id AS id, node.name AS name, node.type AS type, 
                       communityId AS community_id,
                       intermediateCommunityIds AS hierarchy
                ORDER BY communityId, node.name
            """, graph_name=graph_name, intermediate=include_intermediate)
            
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
    graph_name = await gds.ensure_projection(folder_id, node_ids)
    
    try:
        async with driver.session() as session:
            result = await session.run("""
                CALL gds.leiden.stream($graph_name, {
                    gamma: $gamma
                })
                YIELD nodeId, communityId
                WITH gds.util.asNode(nodeId) AS node, communityId
                RETURN node.id AS id, node.name AS name, node.type AS type, 
                       communityId AS community_id
                ORDER BY communityId, node.name
            """, graph_name=graph_name, gamma=gamma)
            
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
        logger.error(f"GDS Leiden error: {e}")
        raise HTTPException(status_code=500, detail=f"Leiden failed: {str(e)}")


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
    """Find similar node pairs using GDS Node Similarity (Jaccard).."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id, node_ids)
    
    try:
        async with driver.session() as session:
            result = await session.run("""
                CALL gds.nodeSimilarity.stream($graph_name, {
                    topK: $top_k,
                    similarityCutoff: $cutoff
                })
                YIELD node1, node2, similarity
                WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
                RETURN n1.id AS source_id, n1.name AS source_name, n1.type AS source_type,
                       n2.id AS target_id, n2.name AS target_name, n2.type AS target_type,
                       similarity AS score
                ORDER BY score DESC
                LIMIT $top_k
            """, graph_name=graph_name, top_k=top_k, cutoff=similarity_cutoff)
            
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
    current_user: dict = Depends(get_current_user),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Find shortest path between two nodes using GDS Dijkstra."""
    driver = get_neo4j_driver()
    graph_name = await gds.ensure_projection(folder_id)
    
    try:
        async with driver.session() as session:
            # Get internal node IDs first
            id_result = await session.run("""
                MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
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
                RETURN [n IN nodes | {id: n.id, name: n.name, type: n.type}] AS path, 
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
                "total_cost": cost,
                "path_length": length,
                "insight": insight,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GDS Shortest Path error: {e}")
        raise HTTPException(status_code=500, detail=f"Shortest path failed: {str(e)}")


# === GDS Link Prediction ===
@router.get("/link-prediction")
async def run_link_prediction_gds(
    folder_id: Optional[str] = None,
    method: str = Query(default="common_neighbors", enum=["common_neighbors", "adamic_adar", "preferential_attachment"]),
    top_k: int = Query(default=20, le=100),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Predict missing links using GDS Link Prediction algorithms."""
    driver = get_neo4j_driver()
    
    try:
        async with driver.session() as session:
            # Get node pairs that aren't connected
            folder_filter = "WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id" if folder_id else ""
            
            if method == "common_neighbors":
                query = f"""
                    MATCH (a:Entity), (b:Entity)
                    {folder_filter}
                    WHERE a <> b AND NOT (a)--(b)
                    WITH a, b, gds.linkPrediction.commonNeighbors(a, b) AS score
                    WHERE score > 0
                    RETURN a.id AS source_id, a.name AS source_name,
                           b.id AS target_id, b.name AS target_name,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
            elif method == "adamic_adar":
                query = f"""
                    MATCH (a:Entity), (b:Entity)
                    {folder_filter}
                    WHERE a <> b AND NOT (a)--(b)
                    WITH a, b, gds.linkPrediction.adamicAdar(a, b) AS score
                    WHERE score > 0
                    RETURN a.id AS source_id, a.name AS source_name,
                           b.id AS target_id, b.name AS target_name,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
            else:  # preferential_attachment
                query = f"""
                    MATCH (a:Entity), (b:Entity)
                    {folder_filter}
                    WHERE a <> b AND NOT (a)--(b)
                    WITH a, b, gds.linkPrediction.preferentialAttachment(a, b) AS score
                    WHERE score > 0
                    RETURN a.id AS source_id, a.name AS source_name,
                           b.id AS target_id, b.name AS target_name,
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
                insight = f"Strongest predicted link: '{top['source_name']}' ↔ '{top['target_name']}' (score: {top['score']:.2f}). Found {len(records)} potential connections."
            else:
                insight = "No link predictions above threshold."
            
            return {
                "algorithm": "link_prediction",
                "engine": f"gds.linkPrediction.{method}",
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
