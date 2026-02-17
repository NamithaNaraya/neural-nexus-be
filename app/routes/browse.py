"""
Browse Routes

Endpoints for high-level database exploration and tabular data views.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import logging

from app.core.security import get_current_user
from app.db.connections import get_neo4j

router = APIRouter()
logger = logging.getLogger(__name__)

class BrowseNodeResponse(BaseModel):
    id: str
    name: str
    type: str
    folder_id: Optional[str] = None
    properties: Dict[str, Any] = {}
    connections: Dict[str, int] = {}  # Type name -> count

@router.get("/types")
async def get_all_node_types(
    folder_id: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """Retrieve all unique node types and their counts. Optional: Filter by folder_id."""
    try:
        where_clause = ""
        params = {}
        if folder_id:
            where_clause = "WHERE n.folder_id = $folder_id"
            params["folder_id"] = folder_id

        query = f"""
        MATCH (n:Entity)
        {where_clause}
        RETURN n.type as type, count(n) as count
        ORDER BY count DESC
        """
        result = await neo4j.execute_query(query, params)
        types = [dict(record) for record in result.records if record["type"]]
        return {"types": types, "total_types": len(types)}
    except Exception as e:
        logger.error(f"Failed to fetch node types: {e}")
        return {"types": [], "total_types": 0}

@router.get("/nodes/{node_type}")
async def get_nodes_by_type(
    node_type: str,
    folder_id: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """
    Get paginated nodes of a specific type with connection statistics.
    Optional: Filter by name (q) or folder_id.
    """
    try:
        skip = (page - 1) * page_size
        
        # Base filter
        where_clause = "WHERE n.type = $node_type"
        params = {"node_type": node_type}
        
        if folder_id:
            where_clause += " AND n.folder_id = $folder_id"
            params["folder_id"] = folder_id
        
        if q:
            where_clause += " AND n.name =~ $regex"
            params["regex"] = f"(?i).*{q}.*"
            
        # 1. Get total count for pagination
        count_query = f"MATCH (n:Entity) {where_clause} RETURN count(n) as total"
        count_result = await neo4j.execute_query(count_query, params)
        total_nodes = count_result.records[0]["total"] if count_result.records else 0
        
        # 2. Get nodes and their connection distribution
        nodes_query = f"""
        MATCH (n:Entity)
        {where_clause}
        WITH n ORDER BY n.name ASC SKIP $skip LIMIT $limit
        OPTIONAL MATCH (n)-[]-(neighbor:Entity)
        WITH n, neighbor.type as nt, count(neighbor) as c
        WITH n, collect({{type: nt, count: c}}) as raw_conn_list
        WITH n, [x IN raw_conn_list WHERE x.type IS NOT NULL] as conn_list
        RETURN n, conn_list
        """
        
        result = await neo4j.execute_query(nodes_query, {**params, "skip": skip, "limit": page_size})
        
        nodes = []
        for record in result.records:
            node_data = dict(record["n"])
            connections = {c["type"]: c["count"] for c in record["conn_list"]}
            
            # Map Neo4j properties
            node_id = node_data.get("id") or str(record["n"].element_id)
            
            nodes.append({
                "id": node_id,
                "name": node_data.get("name", "Unknown"),
                "type": node_data.get("type", node_type),
                "folder_id": node_data.get("folder_id"),
                "properties": {k: v for k, v in node_data.items() if k not in ["id", "name", "type", "folder_id"]},
                "connections": connections
            })
            
        return {
            "nodes": nodes,
            "total": total_nodes,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_nodes + page_size - 1) // page_size if total_nodes > 0 else 0
        }
    except Exception as e:
        logger.error(f"Failed to fetch nodes for type {node_type}: {e}")
        return {"nodes": [], "total": 0, "error": str(e)}
