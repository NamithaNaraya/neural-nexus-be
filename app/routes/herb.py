"""
Herb Domain-Specific Routes

Specialized routes for the Ayurvedic herb knowledge graph schema.
These encode the domain-specific traversal:
  Herb → HAS_PROPERTY → Property → HAS_QUALITY → Quality

Separated from the generic graph.py to keep the core engine domain-agnostic.
"""
from typing import Dict, Any, List
from fastapi import APIRouter, Depends
import logging

from app.core.security import get_current_user
from app.db.connections import get_neo4j

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/herb/{herb_name}/properties")
async def get_herb_properties(
    herb_name: str,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, List[str]]:
    """Get all properties for a specific herb."""
    try:
        query = """
        MATCH (h:Herb {name: $herb_name})-[:HAS_PROPERTY]->(p:Property)
        RETURN p.name AS property
        """
        result = await neo4j.execute_query(query, {"herb_name": herb_name})
        properties = [record["property"] for record in result.records]
        return {"properties": properties}
    except Exception as e:
        logger.error(f"Failed to get herb properties: {e}")
        return {"properties": []}


@router.get("/herb/{herb_name}/property/{property_name}/qualities")
async def get_herb_property_qualities(
    herb_name: str,
    property_name: str,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, List[str]]:
    """Get qualities for a specific herb property."""
    try:
        query = """
        MATCH (h:Herb {name: $herb_name})-[:HAS_PROPERTY]->(p:Property {name: $property_name})
              -[r:HAS_QUALITY {herb: $herb_name}]->(q:Quality)
        RETURN q.name AS quality
        """
        result = await neo4j.execute_query(query, {
            "herb_name": herb_name, 
            "property_name": property_name
        })
        qualities = [record["quality"] for record in result.records]
        
        return {"qualities": qualities}
    except Exception as e:
        logger.error(f"Failed to get herb qualities: {e}")
        return {"qualities": []}


@router.get("/herb/{herb_name}/full")
async def get_herb_full_profile(
    herb_name: str,
    current_user: dict = Depends(get_current_user),
    neo4j = Depends(get_neo4j),
) -> Dict[str, Any]:
    """Get full herb profile."""
    try:
        query = """
        MATCH (h:Herb {name: $herb_name})-[:HAS_PROPERTY]->(p:Property)
        OPTIONAL MATCH (h)-[:HAS_PROPERTY]->(p)-[r:HAS_QUALITY {herb: $herb_name}]->(q:Quality)
        RETURN p.name AS property, collect(q.name) AS qualities
        ORDER BY property
        """
        result = await neo4j.execute_query(query, {"herb_name": herb_name})
        
        # Format: { "Rasa": ["Madhura"], "Guna": ["Guru", "Snigdha"], ... }
        profile = {}
        for record in result.records:
            profile[record["property"]] = record["qualities"]
            
        return {"profile": profile}
    except Exception as e:
        logger.error(f"Failed to get full herb profile: {e}")
        return {"profile": {}}
