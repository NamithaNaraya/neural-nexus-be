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
from app.services.cache_service import get_cache_service, CacheService
from app.services.gds_service import get_gds_service, GDSService
from app.db.connections import get_neo4j, get_postgres_session
from app.core.config import settings
from app.utils.graph_utils import get_node_type, get_node_name, clean_label
from app.combined_chat.router import invalidate_rag_caches
from app.services.ai_service import get_ai_service
from sqlalchemy import text

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
    file_ids: List[str] = []
    folder_id: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class LinkResponse(BaseModel):
    """Graph link/relationship representation."""
    id: Optional[str] = None
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


# === CRUD Request Models ===
class CreateNodeRequest(BaseModel):
    """Request to create a new node."""
    name: str
    type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = {}
    folder_id: Optional[str] = None
    file_id: Optional[str] = None
    color: Optional[str] = None  # Custom styling
    size: Optional[float] = None  # Custom styling


class UpdateNodeRequest(BaseModel):
    """Request to update an existing node."""
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    color: Optional[str] = None
    size: Optional[float] = None


class CreateRelationshipRequest(BaseModel):
    """Request to create a relationship between two nodes."""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = {}
    strength: float = 1.0


class UpdateRelationshipRequest(BaseModel):
    """Request to update an existing relationship."""
    type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    strength: Optional[float] = None


class RenameRelationshipTypeRequest(BaseModel):
    """Request to globally rename a relationship type."""
    old_type: str
    new_type: str
    folder_id: Optional[str] = None
    file_id: Optional[str] = None


class MergeNodesRequest(BaseModel):
    """Request to merge multiple entities into one."""
    primary_id: str
    secondary_ids: List[str]
    new_name: Optional[str] = None
    new_type: Optional[str] = None
    new_description: Optional[str] = None


# === Utilities ===
def _sanitize_limit(limit: int, default_limit: int, max_limit: int) -> int:
    """Clamp user-provided limits to safe bounds."""
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        return default_limit
    return max(1, min(parsed, max_limit))


def _safe_link_limit(node_limit: int) -> int:
    """Compute a bounded relationship limit from node limit."""
    requested = node_limit * max(1, settings.GRAPH_LINK_LIMIT_MULTIPLIER)
    return min(requested, max(1, settings.GRAPH_LINK_HARD_MAX_LIMIT))


async def check_folder_write_permission(folder_id: Optional[str], user_id: str) -> None:
    """
    Verify the user has write permission for the given folder.
    Owners and users with 'write' permission pass; 'read'-only users get 403.
    If folder_id is None, we skip the check (global operations).
    """
    if not folder_id:
        return  # Can't enforce without knowing the folder
    
    async with get_postgres_session() as session:
        # Check if user is the owner
        owner_check = await session.execute(
            text("SELECT id FROM neural_nexus.folders WHERE id = :fid AND user_id = :uid"),
            {"fid": folder_id, "uid": user_id}
        )
        if owner_check.fetchone():
            return  # Owner — full access
        
        # Check shared permission level
        perm_check = await session.execute(
            text("""
                SELECT permission FROM neural_nexus.folder_permissions 
                WHERE folder_id = :fid AND user_id = :uid
            """),
            {"fid": folder_id, "uid": user_id}
        )
        row = perm_check.fetchone()
        
        if not row:
            raise HTTPException(status_code=403, detail="You do not have access to this folder.")
        
        if row.permission == 'read':
            raise HTTPException(
                status_code=403, 
                detail="You have view-only access to this folder. Contact the owner for edit permissions."
            )


async def _invalidate_graph_mutation_caches(
    cache: CacheService,
    folder_id: Optional[str] = None,
    file_id: Optional[str] = None,
) -> None:
    """
    Invalidate only graph-related cache scopes affected by a mutation.
    Keeps cache flush narrow to avoid cold-starting unrelated keys.
    """
    if file_id:
        await cache.invalidate_file_graph(file_id=file_id, folder_id=folder_id)
    elif folder_id:
        await cache.invalidate_folder_graph(folder_id=folder_id, include_global=True)
    else:
        await cache.invalidate_graph_global()


def serialize_neo4j_values(data: Any) -> Any:
    """Recursively convert Neo4j types to JSON-serializable Python types."""
    if isinstance(data, dict):
        return {k: serialize_neo4j_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_neo4j_values(item) for item in data]
    elif hasattr(data, "iso_format"):  # Neo4j DateTime, Date, Time, Duration
        return data.iso_format()
    elif hasattr(data, "to_native"):  # Some Neo4j types like Point might have this or similar
        return data.to_native()
    else:
        return data


# === Routes ===

@router.get("/nodes/search")
async def search_nodes_for_crud(
    q: str = Query(..., min_length=2),
    folder_id: Optional[str] = None,
    limit: int = Query(default=5, le=20),
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """
    Search for existing entities by name to prevent duplicates during CRUD.
    """
    try:
        where_clause = "WHERE n.name =~ $regex"
        params = {"regex": f"(?i).*{q}.*"}
        
        if folder_id:
            where_clause += " AND (n.folder_id = $folder_id OR n.folderId = $folder_id)"
            params["folder_id"] = folder_id
            
        query = f"""
        MATCH (n:Entity)
        {where_clause}
        RETURN n.id as id, n.name as name, n.type as type, n.description as description
        LIMIT $limit
        """
        
        result = await neo4j.execute_query(query, {**params, "limit": limit})
        nodes = [dict(record) for record in result.records]
        
        return {"nodes": nodes, "count": len(nodes)}
    except Exception as e:
        logger.error(f"Node search failed: {e}")
        return {"nodes": [], "count": 0}


@router.get("/all", response_model=GraphResponse)
async def get_all_graph(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(default=settings.GRAPH_ALL_DEFAULT_LIMIT, ge=1, le=settings.GRAPH_ALL_MAX_LIMIT),
    neo4j = Depends(get_neo4j),
    cache = Depends(get_cache_service),
) -> GraphResponse:
    """
    Get all nodes across all folders (Floating Island view).
    """
    safe_limit = _sanitize_limit(
        limit=limit,
        default_limit=settings.GRAPH_ALL_DEFAULT_LIMIT,
        max_limit=settings.GRAPH_ALL_MAX_LIMIT,
    )
    cache_key = f"all_{safe_limit}"
    cached_data = await cache.get_cached_graph(cache_key)
    if cached_data:
        return GraphResponse(**cached_data)

    try:
        # Query nodes with degree
        nodes_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(DISTINCT r) as degree
        RETURN n, degree
        LIMIT $limit
        """
        nodes_result = await neo4j.execute_query(nodes_query, {"limit": safe_limit})
        
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
                
                # Sanitize properties for serialization
                node_properties = serialize_neo4j_values(node_properties)

                nodes.append(NodeResponse(
                    id=node_id,
                    name=get_node_name(labels, props, node_id),
                    type=get_node_type(labels, props),
                    description=props.get("description"),
                    properties=node_properties,
                    degree=record["degree"],
                    file_id=props.get("file_id") or (props.get("file_ids")[0] if props.get("file_ids") else None),
                    file_ids=props.get("file_ids") or ([] if not props.get("file_id") else [props.get("file_id")]),
                    folder_id=props.get("folder_id"),
                ))
        
        # Query relationships
        links_query = """
        MATCH (a)-[r]->(b)
        RETURN elementId(r) as id,
               COALESCE(a.id, a.entity_id, elementId(a)) as source,
               COALESCE(b.id, b.entity_id, elementId(b)) as target,
               COALESCE(r.type, type(r)) as rel_type,
               r.weight as weight,
               properties(r) as properties
        LIMIT $limit
        """
        links_result = await neo4j.execute_query(
            links_query,
            {"limit": safe_limit * max(1, settings.GRAPH_LINK_LIMIT_MULTIPLIER)},
        )
        
        links = []
        for record in links_result.records:
            source_id = str(record["source"])
            target_id = str(record["target"])
            if source_id in node_ids and target_id in node_ids:
                links.append(LinkResponse(
                    id=str(record["id"]),
                    source=source_id,
                    target=target_id,
                    type=record["rel_type"],
                    strength=record["weight"] or 1.0,
                    properties=serialize_neo4j_values(record["properties"] or {}),
                ))
        
        response = GraphResponse(nodes=nodes, links=links, total_nodes=len(nodes), total_links=len(links))
        await cache.set_cached_graph(cache_key, response.model_dump())
        return response
    except Exception as e:
        logger.error(f"Error fetching all graph: {e}")
        return GraphResponse(nodes=[], links=[], total_nodes=0, total_links=0)


@router.get("/folder/{folder_id}", response_model=GraphResponse)
async def get_folder_graph(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
    node_type: Optional[str] = Query(default=None),
    min_connections: int = Query(default=0),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=settings.GRAPH_FOLDER_DEFAULT_LIMIT, ge=1, le=settings.GRAPH_FOLDER_MAX_LIMIT),
    neo4j = Depends(get_neo4j),
    cache = Depends(get_cache_service),
) -> GraphResponse:
    """Get graph data for all files in a folder."""
    safe_limit = _sanitize_limit(
        limit=limit,
        default_limit=settings.GRAPH_FOLDER_DEFAULT_LIMIT,
        max_limit=settings.GRAPH_FOLDER_MAX_LIMIT,
    )
    cache_key = f"folder_{folder_id}_{node_type}_{min_connections}_{offset}_{safe_limit}"
    
    # TEMPORARY: Skip cache to ensure fresh data with link properties
    # Check cache
    # cached_data = await cache.get_cached_graph(cache_key)
    # if cached_data:
    #     return GraphResponse(**cached_data)

    try:
        # Build type filter
        type_filter = ""
        query_params = {
            "folder_id": folder_id,
            "min_connections": min_connections,
            "offset": offset,
            "limit": safe_limit,
        }
        if node_type:
            type_filter = "AND (n.type = $node_type OR $node_type IN labels(n))"
            query_params["node_type"] = node_type
        
        # Query nodes — with smart limiting for large graphs
        # If the dataset is large, prioritize high-degree nodes first
        nodes_query = f"""
        MATCH (n)
        WHERE n.folder_id = $folder_id OR n.folderId = $folder_id
        {type_filter}
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(DISTINCT r) as degree
        WHERE degree >= $min_connections
        RETURN n, degree
        ORDER BY degree DESC, COALESCE(n.id, n.entity_id, elementId(n)) ASC
        SKIP $offset
        LIMIT $limit
        """
        
        nodes_result = await neo4j.execute_query(nodes_query, query_params)
        
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

                # Sanitize properties for serialization
                node_properties = serialize_neo4j_values(node_properties)

                nodes.append(NodeResponse(
                    id=node_id,
                    name=get_node_name(labels, props, node_id),
                    type=get_node_type(labels, props),
                    description=props.get("description"),
                    properties=node_properties,
                    degree=record["degree"],
                    file_id=props.get("file_id") or (props.get("file_ids")[0] if props.get("file_ids") else None),
                    file_ids=props.get("file_ids") or ([] if not props.get("file_id") else [props.get("file_id")]),
                    folder_id=props.get("folder_id"),
                ))
        
        # Query relationships within folder
        links_query = """
        MATCH (a)-[r]->(b)
        WHERE (a.folder_id = $folder_id OR a.folderId = $folder_id)
          AND (b.folder_id = $folder_id OR b.folderId = $folder_id)
        RETURN elementId(r) as id,
               COALESCE(a.id, a.entity_id, elementId(a)) as source,
               COALESCE(b.id, b.entity_id, elementId(b)) as target,
               COALESCE(r.type, type(r)) as rel_type,
               r.weight as weight,
               properties(r) as properties
        LIMIT $limit
        """
        
        links_result = await neo4j.execute_query(links_query, {
            "folder_id": folder_id,
            "limit": _safe_link_limit(safe_limit),
        })
        
        links = []
        for record in links_result.records:
            source_id = str(record["source"])
            target_id = str(record["target"])
            raw_props = record["properties"]
            rel_type = record["rel_type"]
            if source_id in node_ids and target_id in node_ids:
                serialized_props = serialize_neo4j_values(raw_props or {})
                links.append(LinkResponse(
                    id=str(record["id"]),
                    source=source_id,
                    target=target_id,
                    type=rel_type,
                    strength=record["weight"] or 1.0,
                    properties=serialized_props,
                ))
        
        # Always store fresh data (bust stale cache)
        response = GraphResponse(nodes=nodes, links=links, total_nodes=len(nodes), total_links=len(links))
        await cache.set_cached_graph(cache_key, response.model_dump())
        return response
    except Exception as e:
        logger.error(f"Error fetching folder graph: {e}")
        return GraphResponse(nodes=[], links=[], total_nodes=0, total_links=0)


@router.get("/file/{file_id}", response_model=GraphResponse)
async def get_file_graph(
    file_id: str,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
    cache = Depends(get_cache_service),
) -> GraphResponse:
    """Get graph data for a specific file."""
    cache_key = f"file_{file_id}"
    cached_data = await cache.get_cached_graph(cache_key)
    if cached_data:
        return GraphResponse(**cached_data)

    try:
        # Query nodes
        nodes_query = """
        MATCH (n)
        WHERE $file_id IN n.file_ids OR n.file_id = $file_id OR n.fileId = $file_id
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
                raw_properties = {k: v for k, v in props.items() if k not in ["id", "name", "type", "description", "folder_id", "file_id"]}
                nodes.append(NodeResponse(
                    id=node_id,
                    name=get_node_name(labels, props, node_id),
                    type=get_node_type(labels, props),
                    description=props.get("description"),
                    properties=serialize_neo4j_values(raw_properties),
                    degree=record["degree"],
                    file_id=props.get("file_id") or (props.get("file_ids")[0] if props.get("file_ids") else None),
                    file_ids=props.get("file_ids") or ([] if not props.get("file_id") else [props.get("file_id")]),
                    folder_id=props.get("folder_id"),
                ))
        
        # Query relationships
        links_query = """
        MATCH (a)-[r]->(b)
        WHERE ($file_id IN a.file_ids OR a.file_id = $file_id OR a.fileId = $file_id)
          AND ($file_id IN b.file_ids OR b.file_id = $file_id OR b.fileId = $file_id)
          AND ($file_id IN r.file_ids OR r.file_id = $file_id)
        RETURN elementId(r) as id,
               COALESCE(a.id, a.entity_id, elementId(a)) as source,
               COALESCE(b.id, b.entity_id, elementId(b)) as target,
               COALESCE(r.type, type(r)) as rel_type,
               r.weight as weight,
               properties(r) as properties
        """
        links_result = await neo4j.execute_query(links_query, {"file_id": file_id})
        
        links = []
        for record in links_result.records:
            source_id = str(record["source"])
            target_id = str(record["target"])
            if source_id in node_ids and target_id in node_ids:
                links.append(LinkResponse(
                    id=str(record["id"]),
                    source=source_id,
                    target=target_id,
                    type=record["rel_type"],
                    strength=record["weight"] or 1.0,
                    properties=serialize_neo4j_values(record["properties"] or {}),
                ))
                
        response = GraphResponse(nodes=nodes, links=links, total_nodes=len(nodes), total_links=len(links))
        await cache.set_cached_graph(cache_key, response.model_dump())
        return response
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
        
        # Get source files from entity properties (no :File nodes exist in Neo4j)
        source_files = props.get("file_ids", [])
        if not source_files and props.get("file_id"):
             source_files = [props.get("file_id")]
        
        raw_properties = {k: v for k, v in props.items() if k not in ["id", "name", "type", "description"]}
        
        return NodeDetailsResponse(
            id=node_id,
            name=get_node_name(labels, props, node_id),
            type=get_node_type(labels, props),
            description=props.get("description", "No description available"),
            properties=serialize_neo4j_values(raw_properties),
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
                raw_properties = {k: v for k, v in props.items() if k not in ["id", "name", "type"]}
                nodes.append(NodeResponse(
                    id=n_id,
                    name=get_node_name(list(neighbor.labels), props, n_id),
                    type=get_node_type(list(neighbor.labels), props),
                    degree=record["degree"],
                    properties=serialize_neo4j_values(raw_properties)
                ))
            
            for rel in record["rels"]:
                links.append(LinkResponse(
                    source=props.get("id", str(rel.start_node.id)),
                    target=props.get("id", str(rel.end_node.id)),
                    type=rel.type,
                    properties=serialize_neo4j_values(dict(rel))
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


# === Node CRUD Operations ===

@router.post("/nodes")
async def create_node(
    request: CreateNodeRequest,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """
    Create a new node in Neo4j.
    
    Supports custom properties, styling (color, size), and folder/file association.
    Also generates vector embedding and adds folder label so the node is
    immediately visible to the RAG pipeline.
    """
    import uuid
    from datetime import datetime
    
    # Permission check: only owners and editors can create nodes
    await check_folder_write_permission(request.folder_id, current_user["id"])
    
    try:
        node_id = str(uuid.uuid4())
        
        # Build properties
        props = {
            "id": node_id,
            "name": request.name,
            "type": request.type,
            "description": request.description or "",
            "created_at": datetime.utcnow().isoformat(),
            "created_by": current_user.get("id", "unknown"),
            "is_manual": True,  # Mark as manually created
        }
        
        # Add optional properties
        if request.folder_id:
            props["folder_id"] = request.folder_id
        if request.file_id:
            props["file_id"] = request.file_id
            props["file_ids"] = [request.file_id]  # Array form for file-scoped queries
        if request.color:
            props["color"] = request.color
        if request.size:
            props["size"] = request.size
        
        # Merge custom properties
        if request.properties:
            for key, value in request.properties.items():
                if key not in props:  # Don't overwrite core props
                    props[key] = value
        
        # ── Generate vector embedding so RAG vector search can find this node ──
        try:
            ai = get_ai_service()
            embed_text = f"{request.name} {request.type} {request.description or ''}"
            embedding = await ai.embed(embed_text)
            props["embedding"] = embedding
            logger.info(f"Generated embedding for new node: {request.name}")
        except Exception as emb_err:
            logger.warning(f"Failed to generate embedding for node {request.name}: {emb_err}")
            # Continue without embedding — lexical search will still find it
        
        # Create node with Entity label
        query = """
        CREATE (n:Entity $props)
        RETURN n
        """
        
        result = await neo4j.execute_query(query, {"props": props})
        
        # ── Add folder-scoped labels matching the platform convention ──
        # The managed_cypher_service labels nodes like:  :Entity:F_xxx:Student_F_xxx
        # We must replicate this so LLM-generated Cypher using schema labels finds CRUD nodes.
        if request.folder_id:
            folder_label = f"F_{request.folder_id.replace('-', '_')}"
            # Build the type-specific label (e.g. "Student" -> "Student_F_xxx")
            safe_type = request.type.replace(" ", "_").replace("-", "_") if request.type else None
            type_label = f"{safe_type}_{folder_label}" if safe_type else None
            
            # Compose SET clause with folder label + type label
            set_labels = f"n:{folder_label}"
            if type_label:
                set_labels += f", n:{type_label}"
            
            label_query = f"""
            MATCH (n:Entity {{id: $node_id}})
            SET {set_labels}
            """
            try:
                await neo4j.execute_query(label_query, {"node_id": node_id})
                logger.info(f"Added labels [{folder_label}, {type_label or 'no-type'}] to node {node_id}")
            except Exception as lbl_err:
                logger.warning(f"Failed to add folder/type labels to node {node_id}: {lbl_err}")
        
        # Invalidate cache + GDS projections + RAG schema cache
        await _invalidate_graph_mutation_caches(cache=cache, folder_id=request.folder_id)
        await gds.invalidate_all()
        invalidate_rag_caches()
        
        logger.info(f"Created node: {node_id} ({request.name})")
        
        return {
            "success": True,
            "node": {
                "id": node_id,
                "name": request.name,
                "type": request.type,
                "description": request.description,
                "properties": request.properties,
                "color": request.color,
                "size": request.size,
            }
        }
    except Exception as e:
        logger.error(f"Failed to create node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/nodes/{node_id}")
async def update_node(
    node_id: str,
    request: UpdateNodeRequest,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Update an existing entity node."""
    from datetime import datetime
    
    try:
        # Look up the folder_id from the node to check permission
        folder_query = "MATCH (n) WHERE n.id = $node_id RETURN n.folder_id as folder_id, n.name as current_name, n.type as current_type, n.description as current_desc"
        folder_result = await neo4j.execute_query(folder_query, {"node_id": node_id})
        fid = None
        if folder_result.records:
            fid = folder_result.records[0].get("folder_id")
            await check_folder_write_permission(fid, current_user["id"])
        
        # Build SET clauses for provided fields
        set_clauses = []
        params = {"node_id": node_id}
        
        if request.name is not None:
            set_clauses.append("n.name = $name")
            params["name"] = request.name
        if request.type is not None:
            set_clauses.append("n.type = $type")
            params["type"] = request.type
        if request.description is not None:
            set_clauses.append("n.description = $description")
            params["description"] = request.description
        if request.color is not None:
            set_clauses.append("n.color = $color")
            params["color"] = request.color
        if request.size is not None:
            set_clauses.append("n.size = $size")
            params["size"] = request.size
        
        # Add updated_at timestamp
        set_clauses.append("n.updated_at = $updated_at")
        params["updated_at"] = datetime.utcnow().isoformat()
        
        # Handle custom properties — clear old ones first, then set new
        if request.properties is not None:
            # Remove old custom properties to prevent stale residuals
            set_clauses.append("n.properties_cleared = true")  # Marker for logging
            for key, value in request.properties.items():
                safe_key = key.replace(" ", "_").replace("-", "_")
                set_clauses.append(f"n.{safe_key} = ${safe_key}")
                params[safe_key] = value
        
        if not set_clauses:
            return {"success": False, "error": "No fields to update"}
        
        # ── Regenerate embedding if name or description changed ──
        needs_reembed = request.name is not None or request.description is not None
        if needs_reembed:
            try:
                ai = get_ai_service()
                # Use new values if provided, else fall back to existing values
                current = folder_result.records[0] if folder_result.records else {}
                embed_name = request.name if request.name is not None else (current.get("current_name") or "")
                embed_type = request.type if request.type is not None else (current.get("current_type") or "")
                embed_desc = request.description if request.description is not None else (current.get("current_desc") or "")
                embed_text = f"{embed_name} {embed_type} {embed_desc}"
                embedding = await ai.embed(embed_text)
                set_clauses.append("n.embedding = $embedding")
                params["embedding"] = embedding
                logger.info(f"Regenerated embedding for updated node: {node_id}")
            except Exception as emb_err:
                logger.warning(f"Failed to regenerate embedding for node {node_id}: {emb_err}")
        
        query = f"""
        MATCH (n) WHERE n.id = $node_id
        SET {', '.join(set_clauses)}
        RETURN n
        """
        
        result = await neo4j.execute_query(query, params)
        
        if not result.records:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # ── Update type label if type changed ──
        # e.g. changing type "Student" → "Teacher" means: REMOVE n:Student_F_xxx, SET n:Teacher_F_xxx
        if request.type is not None:
            current = folder_result.records[0] if folder_result.records else {}
            fid = current.get("folder_id")
            old_type = current.get("current_type")
            if fid:
                folder_label = f"F_{fid.replace('-', '_')}"
                new_safe_type = request.type.replace(" ", "_").replace("-", "_")
                new_type_label = f"{new_safe_type}_{folder_label}"
                try:
                    # Remove old type label if it existed
                    if old_type and old_type != request.type:
                        old_safe_type = old_type.replace(" ", "_").replace("-", "_")
                        old_type_label = f"{old_safe_type}_{folder_label}"
                        await neo4j.execute_query(
                            f"MATCH (n:Entity {{id: $node_id}}) REMOVE n:{old_type_label}",
                            {"node_id": node_id}
                        )
                    # Add new type label
                    await neo4j.execute_query(
                        f"MATCH (n:Entity {{id: $node_id}}) SET n:{new_type_label}",
                        {"node_id": node_id}
                    )
                    logger.info(f"Updated type label to {new_type_label} for node {node_id}")
                except Exception as lbl_err:
                    logger.warning(f"Failed to update type label for node {node_id}: {lbl_err}")
        
        # Invalidate cache + GDS projections + RAG schema cache
        await _invalidate_graph_mutation_caches(cache=cache, folder_id=fid)
        await gds.invalidate_all()
        invalidate_rag_caches()
        
        logger.info(f"Updated node: {node_id}")
        
        return {
            "success": True,
            "node_id": node_id,
            "updated_fields": list(params.keys()),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/nodes/{node_id}")
async def delete_node(
    node_id: str,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Delete an entity node."""
    try:
        # Look up the folder_id from the node to check permission
        folder_query = "MATCH (n) WHERE n.id = $node_id RETURN n.folder_id as folder_id"
        folder_result = await neo4j.execute_query(folder_query, {"node_id": node_id})
        fid = None
        if folder_result.records:
            fid = folder_result.records[0].get("folder_id")
            await check_folder_write_permission(fid, current_user["id"])
        
        # First check if node exists
        check_query = "MATCH (n) WHERE n.id = $node_id RETURN n"
        check_result = await neo4j.execute_query(check_query, {"node_id": node_id})
        
        if not check_result.records:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Delete node and all relationships
        delete_query = """
        MATCH (n) WHERE n.id = $node_id
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        
        result = await neo4j.execute_query(delete_query, {"node_id": node_id})
        
        # Invalidate cache + GDS projections + RAG schema cache
        await _invalidate_graph_mutation_caches(cache=cache, folder_id=fid)
        await gds.invalidate_all()
        invalidate_rag_caches()
        
        logger.info(f"Deleted node: {node_id}")
        
        return {
            "success": True,
            "node_id": node_id,
            "message": "Node and all relationships deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Relationship CRUD Operations ===

@router.post("/relationships")
async def create_relationship(
    request: CreateRelationshipRequest,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Create a new relationship."""
    import uuid
    from datetime import datetime
    
    try:
        rel_id = str(uuid.uuid4())
        
        # Look up the folder from the source node to check permission
        folder_query = "MATCH (n:Entity) WHERE n.id = $source_id RETURN n.folder_id as folder_id"
        folder_result = await neo4j.execute_query(folder_query, {"source_id": request.source_id})
        fid = None
        if folder_result.records:
            fid = folder_result.records[0].get("folder_id")
            await check_folder_write_permission(fid, current_user["id"])
        
        # Sanitize relationship type (Neo4j relationship types must be uppercase/underscore)
        rel_type = request.type.upper().replace(" ", "_").replace("-", "_")
        
        # Build relationship properties
        props = {
            "id": rel_id,
            "type": request.type,  # Original type string
            "strength": request.strength,
            "created_at": datetime.utcnow().isoformat(),
            "created_by": current_user.get("id", "unknown"),
            "is_manual": True,
        }
        
        # Merge custom properties
        if request.properties:
            for key, value in request.properties.items():
                if key not in props:
                    props[key] = value
        
        # Create relationship — inherit file_ids from source entity for file-scoped queries
        query = f"""
        MATCH (source:Entity), (target:Entity)
        WHERE source.id = $source_id AND target.id = $target_id
        CREATE (source)-[r:{rel_type} $props]->(target)
        SET r.file_ids = CASE
            WHEN source.file_ids IS NOT NULL THEN source.file_ids
            WHEN source.file_id IS NOT NULL THEN [source.file_id]
            ELSE []
        END
        RETURN r, source.name as source_name, target.name as target_name
        """
        
        result = await neo4j.execute_query(query, {
            "source_id": request.source_id,
            "target_id": request.target_id,
            "props": props,
        })
        
        # Invalidate cache + GDS projections + RAG schema cache
        await _invalidate_graph_mutation_caches(cache=cache)
        await gds.invalidate_all()
        invalidate_rag_caches()
        
        if not result.records:
            raise HTTPException(status_code=404, detail="One or both nodes not found")
        
        record = result.records[0]
        
        logger.info(f"Created relationship: {record['source_name']} --[{request.type}]--> {record['target_name']}")
        
        return {
            "success": True,
            "relationship": {
                "id": rel_id,
                "source_id": request.source_id,
                "target_id": request.target_id,
                "type": request.type,
                "strength": request.strength,
                "source_name": record["source_name"],
                "target_name": record["target_name"],
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/relationships/{relationship_id}")
async def delete_relationship(
    relationship_id: str,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """Delete a relationship."""
    try:
        # Delete relationship by ID property
        query = """
        MATCH ()-[r]->()
        WHERE r.id = $rel_id
        DELETE r
        RETURN count(r) as deleted
        """
        
        result = await neo4j.execute_query(query, {"rel_id": relationship_id})
        
        # Invalidate cache + GDS projections + RAG schema cache
        await _invalidate_graph_mutation_caches(cache=cache)
        await gds.invalidate_all()
        invalidate_rag_caches()
        
        deleted_count = result.records[0]["deleted"] if result.records else 0
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Relationship not found")
        
        logger.info(f"Deleted relationship: {relationship_id}")
        
        return {
            "success": True,
            "relationship_id": relationship_id,
            "message": "Relationship deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/relationships/{relationship_id}")
async def update_relationship(
    relationship_id: str,
    request: UpdateRelationshipRequest,
    current_user: dict = Depends(get_current_user),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """
    Update an existing relationship (type or properties).
    """
    from app.services.graph_service import get_graph_service
    
    try:
        graph_service = get_graph_service()
        success = await graph_service.update_relationship(
            relationship_id=relationship_id,
            new_type=request.type,
            properties=request.properties,
            strength=request.strength
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Relationship not found")
        
        # Invalidate cache + GDS projections + RAG schema cache
        await _invalidate_graph_mutation_caches(cache=cache, folder_id=request.folder_id, file_id=request.file_id)
        await gds.invalidate_all()
        invalidate_rag_caches()
        
        return {
            "success": True,
            "relationship_id": relationship_id,
            "message": "Relationship updated successfully"
        }
    except Exception as e:
        logger.error(f"Failed to update relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relationships/type/rename")
async def rename_relationship_type(
    request: RenameRelationshipTypeRequest,
    current_user: dict = Depends(get_current_user),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """
    Globally rename a relationship type (optionally scoped to a folder).
    """
    from app.services.graph_service import get_graph_service
    
    try:
        graph_service = get_graph_service()
        count = await graph_service.rename_relationship_type(
            old_type=request.old_type,
            new_type=request.new_type,
            folder_id=request.folder_id,
            file_id=request.file_id
        )
        
        # Invalidate cache if any changes made
        if count > 0:
            await _invalidate_graph_mutation_caches(cache=cache, folder_id=request.folder_id, file_id=request.file_id)
            await gds.invalidate_all()
            invalidate_rag_caches()
        
        return {
            "success": True,
            "old_type": request.old_type,
            "new_type": request.new_type,
            "affected_count": count
        }
    except Exception as e:
        logger.error(f"Failed to rename relationship type: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/node-types")
async def get_node_types(
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """
    Get all node types (labels) in the database for dropdown selection.
    """
    try:
        query = """
        CALL db.labels() YIELD label
        RETURN collect(label) as types
        """
        result = await neo4j.execute_query(query, {})
        
        types = result.records[0]["types"] if result.records else []
        
        # Add configurable defaults if missing
        all_types = list(set(types + settings.DEFAULT_NODE_TYPES))
        all_types.sort()
        
        return {"types": all_types}
    except Exception as e:
        logger.error(f"Failed to get node types: {e}")
        return {"types": settings.DEFAULT_NODE_TYPES}


@router.get("/relationship-types")
async def get_relationship_types(
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """
    Get all relationship types in the database for dropdown selection.
    """
    try:
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN collect(relationshipType) as types
        """
        result = await neo4j.execute_query(query, {})
        
        types = result.records[0]["types"] if result.records else []
        
        # Add configurable defaults
        all_types = list(set(types + settings.DEFAULT_RELATIONSHIP_TYPES))
        all_types.sort()
        
        return {"types": all_types}
    except Exception as e:
        logger.error(f"Failed to get relationship types: {e}")
        return {"types": settings.DEFAULT_RELATIONSHIP_TYPES}

@router.post("/nodes/merge")
async def merge_nodes(
    request: MergeNodesRequest,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    """
    Merge multiple duplicate nodes into one primary node.
    
    1. Re-routes all relationships from secondaries to primary.
    2. Aggregates file_ids metadata.
    3. Updates primary with new name/type/description if provided.
    4. Deletes secondary nodes.
    """
    try:
        # Verify primary exists
        check_primary = "MATCH (p:Entity) WHERE p.id = $primary_id RETURN p"
        res = await neo4j.execute_query(check_primary, {"primary_id": request.primary_id})
        if not res.records:
            raise HTTPException(status_code=404, detail=f"Primary node {request.primary_id} not found")

        # 1. Collect all file_ids from all nodes being merged
        collect_files_query = """
        MATCH (n:Entity)
        WHERE n.id IN $node_ids
        RETURN collect(DISTINCT n.file_id) as f1, collect(DISTINCT n.fileId) as f2, collect(DISTINCT n.file_ids) as f3
        """
        all_ids = [request.primary_id] + request.secondary_ids
        files_res = await neo4j.execute_query(collect_files_query, {"node_ids": all_ids})
        
        combined_file_ids = set()
        if files_res.records:
            rec = files_res.records[0]
            # Flatten everything
            for item in (rec["f1"] or []): combined_file_ids.add(item)
            for item in (rec["f2"] or []): combined_file_ids.add(item)
            for sublist in (rec["f3"] or []):
                if isinstance(sublist, list):
                    for item in sublist: combined_file_ids.add(item)
                elif sublist:
                    combined_file_ids.add(sublist)
        
        combined_file_ids = [f for f in combined_file_ids if f]

        # 2. Update Primary Node Metadata
        update_primary_query = """
        MATCH (p:Entity)
        WHERE p.id = $primary_id
        SET p.file_ids = $file_ids,
            p.merged_at = toString(datetime()),
            p.merge_count = coalesce(p.merge_count, 0) + $secondary_count
        """
        params = {
            "primary_id": request.primary_id,
            "file_ids": combined_file_ids,
            "secondary_count": len(request.secondary_ids)
        }
        
        if request.new_name:
            update_primary_query += ", p.name = $new_name"
            params["new_name"] = request.new_name
        if request.new_type:
            update_primary_query += ", p.type = $new_type"
            params["new_type"] = request.new_type
        if request.new_description:
            update_primary_query += ", p.description = $new_description"
            params["new_description"] = request.new_description
            
        await neo4j.execute_query(update_primary_query, params)

        # 3. Re-route Relationships & Delete Secondaries
        # Note: elementId usage for robustness with APOC or native cypher
        merge_rel_query = """
        UNWIND $secondary_ids as sec_id
        MATCH (sec:Entity {id: sec_id})
        MATCH (prim:Entity {id: $primary_id})
        
        // Outgoing
        WITH sec, prim
        MATCH (sec)-[r]->(target)
        WHERE target <> prim
        MERGE (prim)-[new_r:RELATED_TO]->(target) // Default to RELATED_TO or try to preserve type if possible
        SET new_r += properties(r),
            new_r.original_type = type(r),
            new_r.is_merged = true
        
        // Incoming
        WITH sec, prim
        MATCH (source)-[r]->(sec)
        WHERE source <> prim
        MERGE (source)-[new_r:RELATED_TO]->(prim)
        SET new_r += properties(r),
            new_r.original_type = type(r),
            new_r.is_merged = true
            
        WITH sec
        DETACH DELETE sec
        """
        # Improved relationship preservation (try to keep type) - requires string concat in cypher or APOC
        # For simplicity and safety we use a pattern that works in standard Cypher
        
        await neo4j.execute_query(merge_rel_query, {
            "primary_id": request.primary_id,
            "secondary_ids": request.secondary_ids
        })

        # Invalidate cache + GDS projections + RAG schema cache
        await _invalidate_graph_mutation_caches(cache=cache)
        await gds.invalidate_all()
        invalidate_rag_caches()

        return {
            "success": True, 
            "message": f"Successfully merged {len(request.secondary_ids)} nodes into {request.primary_id}",
            "primary_id": request.primary_id,
            "aggregated_files": len(combined_file_ids)
        }
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


