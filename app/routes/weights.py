"""
Weight Config Routes

CRUD endpoints for managing dynamic weight configurations per folder.
Supports:
  - List all weight configs for a folder
  - Create a new weight config
  - Update an existing weight config
  - Delete a weight config
  - Toggle a weight config ON/OFF (activates/deactivates)
  - Discover available numeric properties in a folder's graph data
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import logging
import uuid
import json

from app.core.security import get_current_user
from app.db.connections import get_postgres_session
from sqlalchemy import text

router = APIRouter()
logger = logging.getLogger(__name__)


# === Request/Response Models ===

class WeightFormulaRequest(BaseModel):
    """Formula definition for weight computation."""
    type: str  # "property" | "ratio" | "weighted_sum" | "expression"
    property: Optional[str] = None          # For type="property"
    numerator: Optional[str] = None         # For type="ratio"
    denominator: Optional[str] = None       # For type="ratio"
    label: Optional[str] = None             # For type="ratio" (human label)
    terms: Optional[List[Dict[str, Any]]] = None  # For type="weighted_sum"
    expr: Optional[str] = None              # For type="expression"
    properties: Optional[List[str]] = None  # For type="expression"


class CreateWeightConfigRequest(BaseModel):
    """Request to create a new weight configuration."""
    name: str
    folder_id: str
    formula: WeightFormulaRequest
    description: Optional[str] = None


class UpdateWeightConfigRequest(BaseModel):
    """Request to update a weight configuration."""
    name: Optional[str] = None
    formula: Optional[WeightFormulaRequest] = None
    description: Optional[str] = None


# === Routes ===

@router.get("/folder/{folder_id}")
async def list_weight_configs(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all weight configurations for a folder."""
    try:
        async with get_postgres_session() as session:
            result = await session.execute(
                text("""
                    SELECT id, name, formula, description, is_active, created_at, updated_at
                    FROM neural_nexus.weight_configs
                    WHERE folder_id = :folder_id AND user_id = :user_id
                    ORDER BY created_at DESC
                """),
                {"folder_id": folder_id, "user_id": str(current_user["id"])},
            )
            rows = result.fetchall()

            configs = []
            for row in rows:
                configs.append({
                    "id": str(row.id),
                    "name": row.name,
                    "formula": row.formula,
                    "description": row.description,
                    "is_active": row.is_active,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                })

            return {"configs": configs, "count": len(configs)}
    except Exception as e:
        logger.error(f"Failed to list weight configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_weight_config(
    request: CreateWeightConfigRequest,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Create a new weight configuration for a folder."""
    try:
        config_id = str(uuid.uuid4())
        formula_dict = request.formula.model_dump(exclude_none=True)

        async with get_postgres_session() as session:
            await session.execute(
                text("""
                    INSERT INTO neural_nexus.weight_configs
                        (id, folder_id, user_id, name, formula, description, is_active)
                    VALUES (:id, :folder_id, :user_id, :name, CAST(:formula AS jsonb), :description, false)
                """),
                {
                    "id": config_id,
                    "folder_id": request.folder_id,
                    "user_id": str(current_user["id"]),
                    "name": request.name,
                    "formula": json.dumps(formula_dict),
                    "description": request.description,
                },
            )
            await session.commit()

        return {
            "id": config_id,
            "name": request.name,
            "formula": formula_dict,
            "is_active": False,
            "message": "Weight config created successfully",
        }
    except Exception as e:
        logger.error(f"Failed to create weight config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{config_id}")
async def update_weight_config(
    config_id: str,
    request: UpdateWeightConfigRequest,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Update an existing weight configuration."""
    try:
        updates = []
        params: Dict[str, Any] = {
            "config_id": config_id,
            "user_id": str(current_user["id"]),
        }

        if request.name is not None:
            updates.append("name = :name")
            params["name"] = request.name
        if request.formula is not None:
            updates.append("formula = CAST(:formula AS jsonb)")
            formula_dict = request.formula.model_dump(exclude_none=True)
            params["formula"] = json.dumps(formula_dict)
        if request.description is not None:
            updates.append("description = :description")
            params["description"] = request.description

        updates.append("updated_at = NOW()")

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        async with get_postgres_session() as session:
            result = await session.execute(
                text(f"""
                    UPDATE neural_nexus.weight_configs
                    SET {', '.join(updates)}
                    WHERE id = :config_id AND user_id = :user_id
                    RETURNING id
                """),
                params,
            )
            row = result.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Weight config not found")
            await session.commit()

        return {"id": config_id, "message": "Weight config updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update weight config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{config_id}")
async def delete_weight_config(
    config_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete a weight configuration."""
    try:
        async with get_postgres_session() as session:
            result = await session.execute(
                text("""
                    DELETE FROM neural_nexus.weight_configs
                    WHERE id = :config_id AND user_id = :user_id
                    RETURNING id
                """),
                {"config_id": config_id, "user_id": str(current_user["id"])},
            )
            row = result.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Weight config not found")
            await session.commit()

        return {"id": config_id, "message": "Weight config deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete weight config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{config_id}/activate")
async def activate_weight_config(
    config_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Activate a weight configuration (and deactivate all others for the same folder).
    This is the global ON switch for weights.
    """
    try:
        async with get_postgres_session() as session:
            # Get the folder_id for this config
            check = await session.execute(
                text("""
                    SELECT folder_id FROM neural_nexus.weight_configs
                    WHERE id = :config_id AND user_id = :user_id
                """),
                {"config_id": config_id, "user_id": str(current_user["id"])},
            )
            row = check.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Weight config not found")
            folder_id = row.folder_id

            # Deactivate all configs for this folder
            await session.execute(
                text("""
                    UPDATE neural_nexus.weight_configs
                    SET is_active = false
                    WHERE folder_id = :folder_id AND user_id = :user_id
                """),
                {"folder_id": str(folder_id), "user_id": str(current_user["id"])},
            )

            # Activate the specified config
            await session.execute(
                text("""
                    UPDATE neural_nexus.weight_configs
                    SET is_active = true, updated_at = NOW()
                    WHERE id = :config_id
                """),
                {"config_id": config_id},
            )
            await session.commit()

        return {"id": config_id, "is_active": True, "message": "Weight config activated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate weight config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{config_id}/deactivate")
async def deactivate_weight_config(
    config_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Deactivate a weight configuration.
    This is the global OFF switch — no weights will be applied.
    """
    try:
        async with get_postgres_session() as session:
            result = await session.execute(
                text("""
                    UPDATE neural_nexus.weight_configs
                    SET is_active = false, updated_at = NOW()
                    WHERE id = :config_id AND user_id = :user_id
                    RETURNING id
                """),
                {"config_id": config_id, "user_id": str(current_user["id"])},
            )
            row = result.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Weight config not found")
            await session.commit()

        return {"id": config_id, "is_active": False, "message": "Weight config deactivated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deactivate weight config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active/{folder_id}")
async def get_active_weight_config(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get the currently active weight config for a folder.
    Returns null if no weight is active (weights OFF).
    """
    try:
        async with get_postgres_session() as session:
            result = await session.execute(
                text("""
                    SELECT id, name, formula, description, is_active
                    FROM neural_nexus.weight_configs
                    WHERE folder_id = :folder_id AND user_id = :user_id AND is_active = true
                    LIMIT 1
                """),
                {"folder_id": folder_id, "user_id": str(current_user["id"])},
            )
            row = result.fetchone()

            if row:
                return {
                    "active": True,
                    "config": {
                        "id": str(row.id),
                        "name": row.name,
                        "formula": row.formula,
                        "description": row.description,
                    },
                }
            else:
                return {"active": False, "config": None}
    except Exception as e:
        logger.error(f"Failed to get active weight config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/properties/{folder_id}")
async def discover_numeric_properties(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Discover all numeric properties available on nodes and relationships
    in a folder. Used by the UI to populate property dropdowns.
    """
    from app.db.connections import get_neo4j_driver

    driver = get_neo4j_driver()
    node_props = {}
    rel_props = {}

    try:
        async with driver.session() as session:
            # 1. Discover node properties
            # Filter: must NOT be a list/map, must NOT be a known non-weight property
            result = await session.run("""
                MATCH (n)
                WHERE n.folder_id = $folder_id
                UNWIND keys(n) AS key
                WITH key, n[key] AS val
                WHERE val IS NOT NULL 
                  AND NOT key IN ['x', 'y', 'z', 'embedding', 'id', 'folder_id', 'file_id', 'file_ids', 'created_at', 'name', 'type', 'description', 'fastrp_embedding']
                  AND NOT key CONTAINS 'embedding'
                  AND NOT toString(val) STARTS WITH '['
                  AND NOT toString(val) STARTS WITH '{'
                  AND NOT toString(val) IN ['true', 'false']
                WITH key, toFloat(val) as f_val
                WHERE f_val IS NOT NULL
                RETURN key, count(*) AS occurrences, avg(f_val) AS avg_val,
                     min(f_val) AS min_val, max(f_val) AS max_val
                ORDER BY occurrences DESC
            """, folder_id=folder_id)

            records = await result.data()
            for rec in records:
                node_props[rec["key"]] = {
                    "occurrences": rec["occurrences"],
                    "avg": round(rec["avg_val"], 2) if rec["avg_val"] is not None else None,
                    "min": round(rec["min_val"], 2) if rec["min_val"] is not None else None,
                    "max": round(rec["max_val"], 2) if rec["max_val"] is not None else None,
                }

            # 2. Discover relationship properties
            result2 = await session.run("""
                MATCH (a)-[r]->(b)
                WHERE (a.folder_id = $folder_id OR r.folder_id = $folder_id)
                UNWIND keys(r) AS key
                WITH key, r[key] AS val
                WHERE val IS NOT NULL
                  AND NOT key IN ['folder_id', 'file_id', 'file_ids', 'id', 'created_at']
                  AND NOT key CONTAINS 'embedding'
                  AND NOT toString(val) STARTS WITH '['
                  AND NOT toString(val) STARTS WITH '{'
                  AND NOT toString(val) IN ['true', 'false']
                WITH key, toFloat(val) as f_val
                WHERE f_val IS NOT NULL
                RETURN key, count(*) AS occurrences, avg(f_val) AS avg_val,
                     min(f_val) AS min_val, max(f_val) AS max_val
                ORDER BY occurrences DESC
            """, folder_id=folder_id)

            records2 = await result2.data()
            for rec in records2:
                rel_props[rec["key"]] = {
                    "occurrences": rec["occurrences"],
                    "avg": round(rec["avg_val"], 2) if rec["avg_val"] is not None else None,
                    "min": round(rec["min_val"], 2) if rec["min_val"] is not None else None,
                    "max": round(rec["max_val"], 2) if rec["max_val"] is not None else None,
                }

        logger.info(f"Discovered properties for folder {folder_id}: nodes={list(node_props.keys())}, rels={list(rel_props.keys())}")

        return {
            "folder_id": folder_id,
            "node_properties": node_props,
            "relationship_properties": rel_props,
        }
    except Exception as e:
        logger.error(f"Failed to discover properties for folder {folder_id}: {e}")
        return {"folder_id": folder_id, "node_properties": {}, "relationship_properties": {}}




