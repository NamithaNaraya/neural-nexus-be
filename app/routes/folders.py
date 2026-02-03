"""
Folder Routes

CRUD operations for topic folders.
Folders organize files and their associated knowledge graphs.
"""
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import uuid
import logging

from sqlalchemy import text

from app.core.security import get_current_user
from app.db.connections import get_postgres_session, get_neo4j_driver

router = APIRouter()
logger = logging.getLogger(__name__)


# === Request/Response Models ===

class FolderCreate(BaseModel):
    """Request to create a new folder."""
    name: str
    description: Optional[str] = None


class FolderUpdate(BaseModel):
    """Request to update a folder."""
    name: Optional[str] = None
    description: Optional[str] = None


class FolderResponse(BaseModel):
    """Folder data response."""
    id: str
    name: str
    description: Optional[str]
    file_count: int
    node_count: int
    created_at: str
    updated_at: str


class FileInFolder(BaseModel):
    """File summary within a folder."""
    id: str
    filename: str
    file_type: str
    status: str
    node_count: int
    relationship_count: int
    created_at: str


# === Routes ===

@router.get("", response_model=List[FolderResponse])
async def list_folders(
    current_user: dict = Depends(get_current_user),
) -> List[FolderResponse]:
    """
    List all folders for the current user.
    Includes file and node counts for each folder.
    """
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        result = await session.execute(
            text("""
                SELECT 
                    f.id, 
                    f.name, 
                    f.description,
                    f.created_at,
                    f.updated_at,
                    COUNT(DISTINCT fi.id) as file_count,
                    COALESCE(SUM(fi.node_count), 0) as node_count
                FROM neural_nexus.folders f
                LEFT JOIN neural_nexus.files fi ON fi.folder_id = f.id
                WHERE f.user_id = :user_id
                GROUP BY f.id
                ORDER BY f.updated_at DESC
            """),
            {"user_id": user_id}
        )
        rows = result.fetchall()
    
    return [
        FolderResponse(
            id=str(row.id),
            name=row.name,
            description=row.description,
            file_count=row.file_count or 0,
            node_count=row.node_count or 0,
            created_at=row.created_at.isoformat() if row.created_at else "",
            updated_at=row.updated_at.isoformat() if row.updated_at else "",
        )
        for row in rows
    ]


@router.post("", response_model=FolderResponse)
async def create_folder(
    data: FolderCreate,
    current_user: dict = Depends(get_current_user),
) -> FolderResponse:
    """Create a new topic folder."""
    user_id = current_user["id"]
    folder_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    async with get_postgres_session() as session:
        # Check if folder name already exists for user
        existing = await session.execute(
            text("""
                SELECT id FROM neural_nexus.folders 
                WHERE user_id = :user_id AND name = :name
            """),
            {"user_id": user_id, "name": data.name}
        )
        if existing.fetchone():
            raise HTTPException(
                status_code=400,
                detail=f"Folder '{data.name}' already exists"
            )
        
        # Create folder
        await session.execute(
            text("""
                INSERT INTO neural_nexus.folders 
                (id, user_id, name, description, created_at, updated_at)
                VALUES (:id, :user_id, :name, :description, :created_at, :updated_at)
            """),
            {
                "id": folder_id,
                "user_id": user_id,
                "name": data.name,
                "description": data.description,
                "created_at": now,
                "updated_at": now,
            }
        )
        await session.commit()
    
    logger.info(f"Created folder '{data.name}' for user {user_id}")
    
    return FolderResponse(
        id=folder_id,
        name=data.name,
        description=data.description,
        file_count=0,
        node_count=0,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )


@router.get("/{folder_id}", response_model=FolderResponse)
async def get_folder(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
) -> FolderResponse:
    """Get a specific folder with its stats."""
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        result = await session.execute(
            text("""
                SELECT 
                    f.id, 
                    f.name, 
                    f.description,
                    f.created_at,
                    f.updated_at,
                    COUNT(DISTINCT fi.id) as file_count,
                    COALESCE(SUM(fi.node_count), 0) as node_count
                FROM neural_nexus.folders f
                LEFT JOIN neural_nexus.files fi ON fi.folder_id = f.id
                WHERE f.id = :folder_id AND f.user_id = :user_id
                GROUP BY f.id
            """),
            {"folder_id": folder_id, "user_id": user_id}
        )
        row = result.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Folder not found")
    
    return FolderResponse(
        id=str(row.id),
        name=row.name,
        description=row.description,
        file_count=row.file_count or 0,
        node_count=row.node_count or 0,
        created_at=row.created_at.isoformat() if row.created_at else "",
        updated_at=row.updated_at.isoformat() if row.updated_at else "",
    )


@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: str,
    data: FolderUpdate,
    current_user: dict = Depends(get_current_user),
) -> FolderResponse:
    """Update a folder's name or description."""
    user_id = current_user["id"]
    now = datetime.utcnow()
    
    async with get_postgres_session() as session:
        # Check folder exists and belongs to user
        existing = await session.execute(
            text("""
                SELECT id, name, description FROM neural_nexus.folders 
                WHERE id = :folder_id AND user_id = :user_id
            """),
            {"folder_id": folder_id, "user_id": user_id}
        )
        row = existing.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # Update fields
        new_name = data.name if data.name is not None else row.name
        new_desc = data.description if data.description is not None else row.description
        
        await session.execute(
            text("""
                UPDATE neural_nexus.folders 
                SET name = :name, description = :description, updated_at = :updated_at
                WHERE id = :folder_id
            """),
            {
                "folder_id": folder_id,
                "name": new_name,
                "description": new_desc,
                "updated_at": now,
            }
        )
        await session.commit()
    
    # Return updated folder
    return await get_folder(folder_id, current_user)


@router.delete("/{folder_id}")
async def delete_folder(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Delete a folder and all its associated data.
    This includes all files and their graph nodes.
    """
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        # Check folder exists and belongs to user
        existing = await session.execute(
            text("""
                SELECT id FROM neural_nexus.folders 
                WHERE id = :folder_id AND user_id = :user_id
            """),
            {"folder_id": folder_id, "user_id": user_id}
        )
        if not existing.fetchone():
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # Delete folder (CASCADE will delete files)
        await session.execute(
            text("DELETE FROM neural_nexus.folders WHERE id = :folder_id"),
            {"folder_id": folder_id}
        )
        await session.commit()
    
    # Also delete from Neo4j
    try:
        driver = get_neo4j_driver()
        async with driver.session() as neo_session:
            await neo_session.run(
                "MATCH (e:Entity {folder_id: $folder_id}) DETACH DELETE e",
                folder_id=folder_id
            )
            await neo_session.run(
                "MATCH (c:Chunk {folder_id: $folder_id}) DELETE c",
                folder_id=folder_id
            )
    except Exception as e:
        logger.warning(f"Failed to clean Neo4j for folder {folder_id}: {e}")
    
    logger.info(f"Deleted folder {folder_id} for user {user_id}")
    
    return {"message": "Folder deleted successfully"}


@router.get("/{folder_id}/files", response_model=List[FileInFolder])
async def list_folder_files(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
) -> List[FileInFolder]:
    """List all files in a folder."""
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        # Verify folder belongs to user
        folder_check = await session.execute(
            text("""
                SELECT id FROM neural_nexus.folders 
                WHERE id = :folder_id AND user_id = :user_id
            """),
            {"folder_id": folder_id, "user_id": user_id}
        )
        if not folder_check.fetchone():
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # Get files
        result = await session.execute(
            text("""
                SELECT id, filename, file_type, status, node_count, 
                       relationship_count, created_at
                FROM neural_nexus.files
                WHERE folder_id = :folder_id
                ORDER BY created_at DESC
            """),
            {"folder_id": folder_id}
        )
        rows = result.fetchall()
    
    return [
        FileInFolder(
            id=str(row.id),
            filename=row.filename,
            file_type=row.file_type or "unknown",
            status=row.status,
            node_count=row.node_count or 0,
            relationship_count=row.relationship_count or 0,
            created_at=row.created_at.isoformat() if row.created_at else "",
        )
        for row in rows
    ]
