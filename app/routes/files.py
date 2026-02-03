"""
File Routes

File management endpoints (status, details, deletion).
Upload is handled in upload.py
"""
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import logging

from sqlalchemy import text

from app.core.security import get_current_user
from app.db.connections import get_postgres_session, get_neo4j_driver

router = APIRouter()
logger = logging.getLogger(__name__)


class FileResponse(BaseModel):
    """Complete file data response."""
    id: str
    folder_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    node_count: int
    relationship_count: int
    error_message: Optional[str]
    created_at: str
    processed_at: Optional[str]


class FileStatusResponse(BaseModel):
    """File status response for polling."""
    id: str
    status: str
    node_count: int
    relationship_count: int
    error_message: Optional[str]


@router.get("/{file_id}", response_model=FileResponse)
async def get_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
) -> FileResponse:
    """Get detailed information about a file."""
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        result = await session.execute(
            text("""
                SELECT f.id, f.folder_id, f.filename, f.file_type, f.file_size,
                       f.status, f.node_count, f.relationship_count, 
                       f.error_message, f.created_at, f.processed_at
                FROM neural_nexus.files f
                JOIN neural_nexus.folders fo ON fo.id = f.folder_id
                WHERE f.id = :file_id AND fo.user_id = :user_id
            """),
            {"file_id": file_id, "user_id": user_id}
        )
        row = result.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        id=str(row.id),
        folder_id=str(row.folder_id),
        filename=row.filename,
        file_type=row.file_type or "unknown",
        file_size=row.file_size or 0,
        status=row.status,
        node_count=row.node_count or 0,
        relationship_count=row.relationship_count or 0,
        error_message=row.error_message,
        created_at=row.created_at.isoformat() if row.created_at else "",
        processed_at=row.processed_at.isoformat() if row.processed_at else None,
    )


@router.get("/{file_id}/status", response_model=FileStatusResponse)
async def get_file_status(
    file_id: str,
    current_user: dict = Depends(get_current_user),
) -> FileStatusResponse:
    """Get the processing status of a file (for polling)."""
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        result = await session.execute(
            text("""
                SELECT f.id, f.status, f.node_count, f.relationship_count, f.error_message
                FROM neural_nexus.files f
                JOIN neural_nexus.folders fo ON fo.id = f.folder_id
                WHERE f.id = :file_id AND fo.user_id = :user_id
            """),
            {"file_id": file_id, "user_id": user_id}
        )
        row = result.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileStatusResponse(
        id=str(row.id),
        status=row.status,
        node_count=row.node_count or 0,
        relationship_count=row.relationship_count or 0,
        error_message=row.error_message,
    )


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Delete a file and its associated graph data.
    Uses reference counting for shared entities.
    """
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        # Verify file belongs to user
        result = await session.execute(
            text("""
                SELECT f.id, f.folder_id
                FROM neural_nexus.files f
                JOIN neural_nexus.folders fo ON fo.id = f.folder_id
                WHERE f.id = :file_id AND fo.user_id = :user_id
            """),
            {"file_id": file_id, "user_id": user_id}
        )
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="File not found")
        
        folder_id = str(row.folder_id)
        
        # Delete from PostgreSQL (CASCADE handles related records)
        await session.execute(
            text("DELETE FROM neural_nexus.files WHERE id = :file_id"),
            {"file_id": file_id}
        )
        await session.commit()
    
    # Clean up Neo4j with reference counting
    try:
        driver = get_neo4j_driver()
        async with driver.session() as neo_session:
            # Remove file_id from entity file_ids array
            # Delete entities that no longer have any file references
            await neo_session.run("""
                MATCH (e:Entity {folder_id: $folder_id})
                WHERE $file_id IN e.file_ids
                SET e.file_ids = [x IN e.file_ids WHERE x <> $file_id]
                WITH e
                WHERE size(e.file_ids) = 0
                DETACH DELETE e
            """, folder_id=folder_id, file_id=file_id)
            
            # Delete chunks for this file
            await neo_session.run(
                "MATCH (c:Chunk {file_id: $file_id}) DELETE c",
                file_id=file_id
            )
            
    except Exception as e:
        logger.warning(f"Failed to clean Neo4j for file {file_id}: {e}")
    
    logger.info(f"Deleted file {file_id}")
    
    return {"message": "File deleted successfully"}
