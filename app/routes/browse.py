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
from app.core.config import settings
from app.utils.graph_utils import get_node_name, get_node_type, clean_label
from app.services.cache_service import CacheService, get_cache_service

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
    cache: CacheService = Depends(get_cache_service),
) -> Dict[str, Any]:
    """Retrieve all unique node types and their counts. Optional: Filter by folder_id."""
    try:
        scope = folder_id or "global"
        cache_key = f"browse_types:{scope}"
        cached = await cache.get_cached_value(f"{cache.analytics_prefix}{cache_key}")
        if cached is not None:
            return cached

        where_clause = ""
        params = {}
        if folder_id:
            where_clause = "WHERE n.folder_id = $folder_id"
            params["folder_id"] = folder_id

        query = f"""
        MATCH (n:Entity)
        {where_clause}
        WITH n, 
             CASE WHEN n.type IS NOT NULL THEN n.type 
             ELSE [l IN labels(n) WHERE NOT l IN $system_labels AND NOT l STARTS WITH 'F_'][0] 
             END as node_type
        WHERE node_type IS NOT NULL
        RETURN node_type as type, count(n) as count
        ORDER BY count DESC
        """
        result = await neo4j.execute_query(query, {**params, "system_labels": settings.GRAPH_SYSTEM_LABELS})
        types = []
        for record in result.records:
            t = record["type"]
            if t:
                types.append({
                    "type": clean_label(t),
                    "count": record["count"]
                })
        
        # Merge counts if cleaning led to same names (e.g. Herb_F1 and Herb_F2 both become Herb)
        merged_types = {}
        for item in types:
            name = item["type"]
            merged_types[name] = merged_types.get(name, 0) + item["count"]
            
        final_types = [{"type": k, "count": v} for k, v in merged_types.items()]
        final_types.sort(key=lambda x: x["count"], reverse=True)
        
        payload = {"types": final_types, "total_types": len(final_types)}
        await cache.set_cached_value(f"{cache.analytics_prefix}{cache_key}", payload, ttl=180)
        return payload
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
    cache: CacheService = Depends(get_cache_service),
) -> Dict[str, Any]:
    """
    Get paginated nodes of a specific type with connection statistics.
    Optional: Filter by name (q) or folder_id.
    """
    try:
        skip = (page - 1) * page_size
        scope = folder_id or "global"
        query_part = (q or "").strip().lower()
        cache_key = f"browse_nodes:{scope}:{node_type}:{page}:{page_size}:{query_part}"
        cached = await cache.get_cached_value(f"{cache.analytics_prefix}{cache_key}")
        if cached is not None:
            return cached
        
        # Use a more robust check that mirrors get_all_node_types and handles suffixes
        # We first identify the 'raw' type, then clean it, then compare
        where_clause = """
        WHERE n.folder_id = $folder_id
        AND (
          (CASE WHEN n.type IS NOT NULL THEN (CASE WHEN n.type CONTAINS '_F_' THEN split(n.type, '_F_')[0] ELSE n.type END) ELSE "" END) = $node_type
          OR
          any(l IN labels(n) WHERE NOT l IN $system_labels AND NOT l STARTS WITH 'F_' AND (CASE WHEN l CONTAINS '_F_' THEN split(l, '_F_')[0] ELSE l END) = $node_type)
        )
        """
        if not folder_id:
            where_clause = where_clause.replace("n.folder_id = $folder_id AND", "")

        params = {
            "node_type": node_type,
            "folder_id": folder_id,
            "system_labels": settings.GRAPH_SYSTEM_LABELS
        }
        
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
        RETURN n, conn_list, 
               CASE WHEN n.type IS NOT NULL THEN n.type 
               ELSE [l IN labels(n) WHERE NOT l IN $system_labels AND NOT l STARTS WITH 'F_'][0] 
               END as effective_type
        """
        
        result = await neo4j.execute_query(nodes_query, {
            **params, 
            "skip": skip, 
            "limit": page_size,
            "system_labels": settings.GRAPH_SYSTEM_LABELS
        })
        
        nodes = []
        for record in result.records:
            node_data = dict(record["n"])
            connections = {c["type"]: c["count"] for c in record["conn_list"]}
            
            # Map Neo4j properties
            node_id = node_data.get("id") or str(record["n"].element_id)
            
            labels = list(record["n"].labels)
            nodes.append({
                "id": node_id,
                "name": get_node_name(labels, node_data, node_id),
                "type": get_node_type(labels, node_data),
                "folder_id": node_data.get("folder_id"),
                "properties": {k: v for k, v in node_data.items() if k not in ["id", "name", "type", "folder_id"]},
                "connections": connections
            })
            
        payload = {
            "nodes": nodes,
            "total": total_nodes,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_nodes + page_size - 1) // page_size if total_nodes > 0 else 0
        }
        await cache.set_cached_value(f"{cache.analytics_prefix}{cache_key}", payload, ttl=120)
        return payload
    except Exception as e:
        logger.error(f"Failed to fetch nodes for type {node_type}: {e}")
        return {"nodes": [], "total": 0, "error": str(e)}
