"""
Graph Routes

CRUD operations and retrieval for knowledge graph data.
Supports scoped queries by folder, file, and chunk.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import logging

from app.core.security import get_current_user
from app.db.connections import get_neo4j

router = APIRouter()
logger = logging.getLogger(__name__)


# === Pydantic Models ===
class NodeResponse(BaseModel):
    """Graph node representation."""
    id: str
    name: str
    type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = {}
    degree: int = 0
    file_id: Optional[str] = None
    folder_id: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class LinkResponse(BaseModel):
    """Graph link/relationship representation."""
    source: str
    target: str
    type: str
    strength: float = 1.0
    properties: Dict[str, Any] = {}


class GraphResponse(BaseModel):
    """Full graph data response."""
    nodes: List[NodeResponse]
    links: List[LinkResponse]
    total_nodes: int
    total_links: int


class NodeDetailsResponse(BaseModel):
    """Detailed node information (lazy-loaded)."""
    id: str
    name: str
    type: str
    description: str
    properties: Dict[str, Any]
    source_files: List[str]
    created_at: str
    created_by: str
    connection_count: int


# === Routes ===
@router.get("/all", response_model=GraphResponse)
async def get_all_graph(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(default=10000, le=100000),
    neo4j = Depends(get_neo4j),
) -> GraphResponse:
    """
    Get all nodes across all folders (Floating Island view).
    
    Limited to prevent browser overload.
    """
    try:
        # Query nodes with degree
        nodes_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(DISTINCT r) as degree
        RETURN n, degree
        LIMIT $limit
        """
        nodes_result = await neo4j.execute_query(nodes_query, {"limit": limit})
        
        nodes = []
        node_ids = set()
        
        for record in nodes_result.records:
            node = record["n"]
            props = dict(node)
            node_id = props.get("id") or props.get("entity_id") or str(node.element_id)
            
            if node_id not in node_ids:
                node_ids.add(node_id)
                labels = list(node.labels) if node.labels else ["Unknown"]
                # Extract and parse conflicts if present
                node_properties = {k: v for k, v in props.items() if k not in ["id", "name", "type", "description", "folder_id", "file_id", "conflicts"]}
                conflicts = props.get("conflicts")
                if conflicts and isinstance(conflicts, str):
                    import json
                    try:
                        node_properties["conflicts"] = json.loads(conflicts)
                    except:
                        node_properties["conflicts"] = {}
                elif conflicts:
                    node_properties["conflicts"] = conflicts

                nodes.append(NodeResponse(
                    id=node_id,
                    name=props.get("name", props.get("label", "Unknown")),
                    type=props.get("type", labels[0] if labels else "Unknown"),
                    description=props.get("description"),
                    properties=node_properties,
                    degree=record["degree"],
                    file_id=props.get("file_id"),
                    folder_id=props.get("folder_id"),
                ))
        
        # Query relationships
        links_query = """
        MATCH (a)-[r]->(b)
        RETURN COALESCE(a.id, a.entity_id, elementId(a)) as source,
               COALESCE(b.id, b.entity_id, elementId(b)) as target,
               type(r) as rel_type,
               r.weight as weight
        LIMIT $limit
        """
        links_result = await neo4j.execute_query(links_query, {"limit": limit * 2})
        
        links = []
        for record in links_result.records:
            source_id = str(record["source"])
            target_id = str(record["target"])
            if source_id in node_ids and target_id in node_ids:
                links.append(LinkResponse(
                    source=source_id,
                    target=target_id,
                    type=record["rel_type"],
                    strength=record["weight"] or 1.0,
                ))
        
        return GraphResponse(nodes=nodes, links=links, total_nodes=len(nodes), total_links=len(links))
    except Exception as e:
        logger.error(f"Error fetching all graph: {e}")
        return GraphResponse(nodes=[], links=[], total_nodes=0, total_links=0)


@router.get("/folder/{folder_id}", response_model=GraphResponse)
async def get_folder_graph(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
    node_type: Optional[str] = Query(default=None),
    min_connections: int = Query(default=0),
    limit: int = Query(default=1000, le=10000),
    neo4j = Depends(get_neo4j),
) -> GraphResponse:
    """Get graph data for all files in a folder."""
    try:
        # Build type filter
        type_filter = ""
        if node_type:
            type_filter = f"AND (n.type = '{node_type}' OR '{node_type}' IN labels(n))"
        
        # Query nodes
        nodes_query = f"""
        MATCH (n)
        WHERE n.folder_id = $folder_id OR n.folderId = $folder_id
        {type_filter}
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(DISTINCT r) as degree
        WHERE degree >= $min_connections
        RETURN n, degree
        LIMIT $limit
        """
        
        nodes_result = await neo4j.execute_query(nodes_query, {
            "folder_id": folder_id,
            "min_connections": min_connections,
            "limit": limit,
        })
        
        nodes = []
        node_ids = set()
        
        for record in nodes_result.records:
            node = record["n"]
            props = dict(node)
            node_id = props.get("id") or props.get("entity_id") or str(node.element_id)
            
            if node_id not in node_ids:
                node_ids.add(node_id)
                labels = list(node.labels) if node.labels else ["Unknown"]
                # Extract and parse conflicts if present
                node_properties = {k: v for k, v in props.items() if k not in ["id", "name", "type", "description", "folder_id", "file_id", "conflicts"]}
                conflicts = props.get("conflicts")
                if conflicts and isinstance(conflicts, str):
                    import json
                    try:
                        node_properties["conflicts"] = json.loads(conflicts)
                    except:
                        node_properties["conflicts"] = {}
                elif conflicts:
                    node_properties["conflicts"] = conflicts

                nodes.append(NodeResponse(
                    id=node_id,
                    name=props.get("name", props.get("label", "Unknown")),
                    type=props.get("type", labels[0] if labels else "Unknown"),
                    description=props.get("description"),
                    properties=node_properties,
                    degree=record["degree"],
                    file_id=props.get("file_id"),
                    folder_id=props.get("folder_id"),
                ))
        
        # Query relationships within folder
        links_query = """
        MATCH (a)-[r]->(b)
        WHERE (a.folder_id = $folder_id OR a.folderId = $folder_id)
          AND (b.folder_id = $folder_id OR b.folderId = $folder_id)
        RETURN COALESCE(a.id, a.entity_id, elementId(a)) as source,
               COALESCE(b.id, b.entity_id, elementId(b)) as target,
               type(r) as rel_type,
               r.weight as weight,
               r.description as description
        LIMIT $limit
        """
        
        links_result = await neo4j.execute_query(links_query, {
            "folder_id": folder_id,
            "limit": limit * 2,
        })
        
        links = []
        for record in links_result.records:
            source_id = str(record["source"])
            target_id = str(record["target"])
            if source_id in node_ids and target_id in node_ids:
                links.append(LinkResponse(
                    source=source_id,
                    target=target_id,
                    type=record["rel_type"],
                    strength=record["weight"] or 1.0,
                ))
        
        return GraphResponse(nodes=nodes, links=links, total_nodes=len(nodes), total_links=len(links))
    except Exception as e:
        logger.error(f"Error fetching folder graph: {e}")
        return GraphResponse(nodes=[], links=[], total_nodes=0, total_links=0)


@router.get("/file/{file_id}", response_model=GraphResponse)
async def get_file_graph(
    file_id: str,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> GraphResponse:
    """Get graph data for a specific file."""
    try:
        # Query nodes
        nodes_query = """
        MATCH (n)
        WHERE n.file_id = $file_id OR n.fileId = $file_id
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(DISTINCT r) as degree
        RETURN n, degree
        """
        nodes_result = await neo4j.execute_query(nodes_query, {"file_id": file_id})
        
        nodes = []
        node_ids = set()
        for record in nodes_result.records:
            node = record["n"]
            props = dict(node)
            node_id = props.get("id") or props.get("entity_id") or str(node.element_id)
            if node_id not in node_ids:
                node_ids.add(node_id)
                labels = list(node.labels) if node.labels else ["Unknown"]
                nodes.append(NodeResponse(
                    id=node_id,
                    name=props.get("name", props.get("label", "Unknown")),
                    type=props.get("type", labels[0] if labels else "Unknown"),
                    description=props.get("description"),
                    properties={k: v for k, v in props.items() if k not in ["id", "name", "type", "description", "folder_id", "file_id"]},
                    degree=record["degree"],
                    file_id=props.get("file_id"),
                    folder_id=props.get("folder_id"),
                ))
        
        # Query relationships
        links_query = """
        MATCH (a)-[r]->(b)
        WHERE (a.file_id = $file_id OR a.fileId = $file_id)
          AND (b.file_id = $file_id OR b.fileId = $file_id)
        RETURN COALESCE(a.id, a.entity_id, elementId(a)) as source,
               COALESCE(b.id, b.entity_id, elementId(b)) as target,
               type(r) as rel_type,
               r.weight as weight
        """
        links_result = await neo4j.execute_query(links_query, {"file_id": file_id})
        
        links = []
        for record in links_result.records:
            source_id = str(record["source"])
            target_id = str(record["target"])
            if source_id in node_ids and target_id in node_ids:
                links.append(LinkResponse(
                    source=source_id,
                    target=target_id,
                    type=record["rel_type"],
                    strength=record["weight"] or 1.0,
                ))
                
        return GraphResponse(nodes=nodes, links=links, total_nodes=len(nodes), total_links=len(links))
    except Exception as e:
        logger.error(f"Error fetching file graph: {e}")
        return GraphResponse(nodes=[], links=[], total_nodes=0, total_links=0)


@router.get("/node/{node_id}/details", response_model=NodeDetailsResponse)
async def get_node_details(
    node_id: str,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> NodeDetailsResponse:
    """
    Get detailed information for a specific node (lazy-loaded).
    """
    try:
        query = """
        MATCH (n)
        WHERE n.id = $node_id OR n.entity_id = $node_id OR elementId(n) = $node_id
        OPTIONAL MATCH (n)-[r]-()
        RETURN n, count(DISTINCT r) as degree
        LIMIT 1
        """
        result = await neo4j.execute_query(query, {"node_id": node_id})
        if not result.records:
            raise HTTPException(status_code=404, detail="Node not found")
            
        record = result.records[0]
        node = record["n"]
        props = dict(node)
        labels = list(node.labels)
        
        # Get source files (if tracking enabled)
        # This assumes we have a relationship or property tracking source
        source_files = [props.get("file_id")] if props.get("file_id") else []
        
        return NodeDetailsResponse(
            id=node_id,
            name=props.get("name", "Unknown"),
            type=labels[0] if labels else "Unknown",
            description=props.get("description", "No description available"),
            properties={k: v for k, v in props.items() if k not in ["id", "name", "type", "description"]},
            source_files=source_files,
            created_at=props.get("created_at", ""),
            created_by=props.get("created_by", "system"),
            connection_count=record["degree"],
        )
    except Exception as e:
        logger.error(f"Error fetching node details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/expand/{node_id}", response_model=GraphResponse)
async def expand_node(
    node_id: str,
    depth: int = Query(default=1, le=3),
    relationship_types: Optional[str] = Query(default=None),
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> GraphResponse:
    """
    Expand a node to show its connections (Neo4j Browser style).
    """
    try:
        rel_clause = ""
        if relationship_types:
            types = relationship_types.split(",")
            rel_clause = ":" + "|".join(types)
            
        query = f"""
        MATCH (center)
        WHERE center.id = $node_id OR center.entity_id = $node_id OR elementId(center) = $node_id
        MATCH p = (center)-[r{rel_clause}*1..{depth}]-(neighbor)
        WITH neighbor, relationships(p) as rels
        OPTIONAL MATCH (neighbor)-[r2]-()
        RETURN neighbor, rels, count(DISTINCT r2) as degree
        LIMIT 100
        """
        
        result = await neo4j.execute_query(query, {"node_id": node_id})
        
        nodes = []
        links = []
        node_ids = set()
        
        for record in result.records:
            neighbor = record["neighbor"]
            props = dict(neighbor)
            n_id = props.get("id") or props.get("entity_id") or str(neighbor.element_id)
            
            if n_id not in node_ids:
                node_ids.add(n_id)
                nodes.append(NodeResponse(
                    id=n_id,
                    name=props.get("name", "Unknown"),
                    type=list(neighbor.labels)[0] if neighbor.labels else "Unknown",
                    degree=record["degree"],
                    properties={k: v for k, v in props.items() if k not in ["id", "name", "type"]}
                ))
            
            for rel in record["rels"]:
                links.append(LinkResponse(
                    source=props.get("id", str(rel.start_node.id)),
                    target=props.get("id", str(rel.end_node.id)),
                    type=rel.type
                ))
        
        # Dedup links
        unique_links = {f"{l.source}-{l.target}-{l.type}": l for l in links}.values()
        
        return GraphResponse(nodes=nodes, links=list(unique_links), total_nodes=len(nodes), total_links=len(unique_links))
    except Exception as e:
        logger.error(f"Error expanding node: {e}")
        return GraphResponse(nodes=[], links=[], total_nodes=0, total_links=0)


@router.get("/path/{source_id}/{target_id}")
async def get_shortest_path(
    source_id: str,
    target_id: str,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """Find shortest path between two nodes."""
    try:
        query = """
        MATCH p = shortestPath((a)-[*..10]-(b))
        WHERE (a.id = $source_id OR elementId(a) = $source_id)
          AND (b.id = $target_id OR elementId(b) = $target_id)
        RETURN 
            [n IN nodes(p) | COALESCE(n.id, elementId(n))] as node_ids,
            [r IN relationships(p) | elementId(r)] as link_ids,
            length(p) as hops
        """
        result = await neo4j.execute_query(query, {"source_id": source_id, "target_id": target_id})
        
        if not result.records:
            return {"path_exists": False, "node_ids": [], "link_ids": [], "length": 0}
            
        record = result.records[0]
        return {
            "path_exists": True,
            "node_ids": record["node_ids"],
            "link_ids": record["link_ids"],
            "length": record["hops"]
        }
    except Exception as e:
        logger.error(f"Error finding path: {e}")
        return {"path_exists": False, "node_ids": [], "link_ids": [], "length": 0}


@router.get("/layout/{folder_id}")
async def get_layout(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
    algorithm: str = Query(default="forceAtlas2", description="Layout algorithm"),
    iterations: int = Query(default=100, ge=10, le=500),
    scale: float = Query(default=500.0, ge=100, le=2000),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """
    Calculate server-side layout for a folder's graph.
    
    Uses Neo4j GDS ForceAtlas2 when available, with intelligent fallback
    to deterministic type-clustered positioning.
    """
    try:
        from app.services.graph_layout import get_layout_service
        
        layout_service = get_layout_service(neo4j)
        positions = await layout_service.calculate_layout(
            folder_id=folder_id,
            algorithm=algorithm,
            iterations=iterations,
            scale=scale,
        )
        
        # Convert to serializable format
        result = {
            node_id: {"x": pos.x, "y": pos.y, "z": pos.z}
            for node_id, pos in positions.items()
        }
        
        return {
            "folder_id": folder_id,
            "algorithm": algorithm,
            "node_count": len(result),
            "positions": result,
        }
    except Exception as e:
        logger.error(f"Error calculating layout: {e}")
        return {
            "folder_id": folder_id,
            "algorithm": algorithm,
            "node_count": 0,
            "positions": {},
            "error": str(e),
        }


# === Phase 6: Advanced Intelligence ===

class CompareRequest(BaseModel):
    """Request for cluster comparison."""
    left: Dict[str, Any]  # {"type": "file|folder|cluster|selection", "id": str, "node_ids": List[str]}
    right: Dict[str, Any]
    include_bridges: bool = True
    include_similarity: bool = True


@router.post("/compare")
async def compare_clusters(
    request: CompareRequest,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """
    Compare two clusters of nodes (file vs file, folder vs folder, etc.).
    
    Returns common entities, unique nodes, bridge nodes, and similarity metrics.
    """
    from app.services.cluster_comparison import get_comparison_service
    
    try:
        comparison = get_comparison_service(neo4j)
        result = await comparison.compare(
            left=request.left,
            right=request.right,
            include_bridges=request.include_bridges,
            include_similarity=request.include_similarity,
        )
        return result
    except Exception as e:
        logger.error(f"Cluster comparison failed: {e}")
        return {"error": str(e)}


@router.get("/blind-spots/{folder_id}")
async def discover_blind_spots(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
    min_confidence: float = Query(default=0.5, ge=0.0, le=1.0),
    method: str = Query(default="structural"),
    limit: int = Query(default=50, le=200),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """
    Discover blind spots (missing relationships) in the graph.
    
    Returns "ghost lines" - predicted relationships that should exist
    but aren't documented.
    """
    from app.services.blind_spot_discovery import get_discovery_service, PredictionMethod
    
    try:
        discovery = get_discovery_service(neo4j)
        
        # Map string to enum
        method_map = {
            "structural": PredictionMethod.STRUCTURAL,
            "common_neighbors": PredictionMethod.COMMON_NEIGHBORS,
            "jaccard": PredictionMethod.JACCARD,
            "adamic_adar": PredictionMethod.ADAMIC_ADAR,
            "preferential_attachment": PredictionMethod.PREFERENTIAL_ATTACHMENT,
        }
        pred_method = method_map.get(method, PredictionMethod.STRUCTURAL)
        
        result = await discovery.discover(
            folder_id=folder_id,
            min_confidence=min_confidence,
            method=pred_method,
            limit=limit,
        )
        return result
    except Exception as e:
        logger.error(f"Blind spot discovery failed: {e}")
        return {"error": str(e), "ghost_lines": []}


@router.get("/blind-spots/cross-topic")
async def cross_topic_bridges(
    folder_ids: str = Query(..., description="Comma-separated folder IDs"),
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """
    Find cross-topic bridging opportunities between folders.
    
    Identifies entities that could connect different topic areas.
    """
    from app.services.blind_spot_discovery import get_discovery_service
    
    try:
        folder_list = [f.strip() for f in folder_ids.split(",") if f.strip()]
        
        if len(folder_list) < 2:
            return {"error": "At least 2 folder IDs required", "bridges": []}
        
        discovery = get_discovery_service(neo4j)
        bridges = await discovery.get_cross_topic_bridges(folder_list)
        
        return {
            "folder_count": len(folder_list),
            "folders": folder_list,
            "bridge_count": len(bridges),
            "bridges": bridges,
        }
    except Exception as e:
        logger.error(f"Cross-topic bridge discovery failed: {e}")
        return {"error": str(e), "bridges": []}


@router.get("/export/{folder_id}")
async def export_analytics(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
    format: str = Query(default="pdf", description="Export format: pdf, json, csv"),
    include_centrality: bool = Query(default=True),
    include_clustering: bool = Query(default=True),
    include_ghost_lines: bool = Query(default=True),
    include_health: bool = Query(default=True),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """
    Export graph analytics with analytics supplement tables.
    
    Available formats:
    - pdf: Structured data for frontend PDF rendering
    - json: Full analytics data
    - csv: Summary metrics table
    """
    from app.services.analytics_export import get_export_service, ExportConfig
    
    try:
        export_service = get_export_service(neo4j)
        
        config = ExportConfig(
            include_centrality=include_centrality,
            include_clustering=include_clustering,
            include_ghost_lines=include_ghost_lines,
            include_health_score=include_health,
            format=format,
        )
        
        result = await export_service.export(folder_id, config)
        return result
    except Exception as e:
        logger.error(f"Analytics export failed: {e}")
        return {"error": str(e), "folder_id": folder_id}
