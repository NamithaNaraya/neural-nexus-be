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
    permission: str = "owner"  # 'owner', 'write', 'read'


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
    List all folders the user owns or has been shared with.
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
                    COALESCE(SUM(fi.node_count), 0) as node_count,
                    'owner' as permission
                FROM neural_nexus.folders f
                LEFT JOIN neural_nexus.files fi ON fi.folder_id = f.id
                WHERE f.user_id = :user_id
                GROUP BY f.id
                
                UNION ALL
                
                SELECT 
                    f.id, 
                    f.name, 
                    f.description,
                    f.created_at,
                    f.updated_at,
                    COUNT(DISTINCT fi.id) as file_count,
                    COALESCE(SUM(fi.node_count), 0) as node_count,
                    fp.permission as permission
                FROM neural_nexus.folders f
                INNER JOIN neural_nexus.folder_permissions fp 
                    ON fp.folder_id = f.id AND fp.user_id = :user_id
                LEFT JOIN neural_nexus.files fi ON fi.folder_id = f.id
                GROUP BY f.id, fp.permission
                
                ORDER BY updated_at DESC
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
            permission=row.permission,
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
    """Get a specific folder with its stats. Works for owners and shared users."""
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        # First try as owner
        result = await session.execute(
            text("""
                SELECT 
                    f.id, 
                    f.name, 
                    f.description,
                    f.created_at,
                    f.updated_at,
                    COUNT(DISTINCT fi.id) as file_count,
                    COALESCE(SUM(fi.node_count), 0) as node_count,
                    'owner' as permission
                FROM neural_nexus.folders f
                LEFT JOIN neural_nexus.files fi ON fi.folder_id = f.id
                WHERE f.id = :folder_id AND f.user_id = :user_id
                GROUP BY f.id
            """),
            {"folder_id": folder_id, "user_id": user_id}
        )
        row = result.fetchone()
        
        # If not owner, try as shared user
        if not row:
            result = await session.execute(
                text("""
                    SELECT 
                        f.id, 
                        f.name, 
                        f.description,
                        f.created_at,
                        f.updated_at,
                        COUNT(DISTINCT fi.id) as file_count,
                        COALESCE(SUM(fi.node_count), 0) as node_count,
                        fp.permission as permission
                    FROM neural_nexus.folders f
                    INNER JOIN neural_nexus.folder_permissions fp 
                        ON fp.folder_id = f.id AND fp.user_id = :user_id
                    LEFT JOIN neural_nexus.files fi ON fi.folder_id = f.id
                    WHERE f.id = :folder_id
                    GROUP BY f.id, fp.permission
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
        permission=row.permission,
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
    
    # Invalidate all caches so UI shows fresh data
    try:
        from app.services.cache_service import get_cache_service
        cache = get_cache_service()
        await cache.invalidate_all()
    except Exception as e:
        logger.warning(f"Cache invalidation failed after folder delete: {e}")
    
    logger.info(f"Deleted folder {folder_id} for user {user_id}")
    
    return {"message": "Folder deleted successfully"}


@router.get("/{folder_id}/files", response_model=List[FileInFolder])
async def list_folder_files(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
) -> List[FileInFolder]:
    """List all files in a folder. Works for owners and shared users."""
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        # Verify folder belongs to user OR is shared with them
        folder_check = await session.execute(
            text("""
                SELECT f.id FROM neural_nexus.folders f
                WHERE f.id = :folder_id AND (
                    f.user_id = :user_id
                    OR EXISTS (
                        SELECT 1 FROM neural_nexus.folder_permissions fp 
                        WHERE fp.folder_id = f.id AND fp.user_id = :user_id
                    )
                )
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


# === Permission Management Routes ===

class FolderPermission(BaseModel):
    """Permission entry for a folder."""
    user_id: str
    user_email: Optional[str] = None
    permission: str  # 'read', 'write', 'admin'
    granted_at: Optional[str] = None


class PermissionGrant(BaseModel):
    """Request to grant a permission."""
    user_email: str
    permission: str = "read"


class PermissionsResponse(BaseModel):
    """Response with all folder permissions."""
    folder_id: str
    owner_id: str
    permissions: List[FolderPermission]


@router.get("/{folder_id}/permissions", response_model=PermissionsResponse)
async def get_folder_permissions(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
) -> PermissionsResponse:
    """
    Get all permissions for a folder.
    Only the folder owner can view permissions.
    """
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        # Check folder exists and user is owner
        folder = await session.execute(
            text("""
                SELECT id, user_id FROM neural_nexus.folders 
                WHERE id = :folder_id
            """),
            {"folder_id": folder_id}
        )
        row = folder.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        if str(row.user_id) != user_id:
            # Check if user has admin permission
            perm = await session.execute(
                text("""
                    SELECT permission FROM neural_nexus.folder_permissions 
                    WHERE folder_id = :folder_id AND user_id = :user_id
                """),
                {"folder_id": folder_id, "user_id": user_id}
            )
            perm_row = perm.fetchone()
            if not perm_row or perm_row.permission != 'admin':
                raise HTTPException(status_code=403, detail="Not authorized to view permissions")
        
        # Get all permissions
        perms = await session.execute(
            text("""
                SELECT 
                    fp.user_id, 
                    u.email as user_email,
                    fp.permission,
                    fp.granted_at
                FROM neural_nexus.folder_permissions fp
                LEFT JOIN neural_nexus.users u ON u.id = fp.user_id
                WHERE fp.folder_id = :folder_id
                ORDER BY fp.granted_at DESC
            """),
            {"folder_id": folder_id}
        )
        perm_rows = perms.fetchall()
    
    return PermissionsResponse(
        folder_id=folder_id,
        owner_id=str(row.user_id),
        permissions=[
            FolderPermission(
                user_id=str(p.user_id),
                user_email=p.user_email,
                permission=p.permission,
                granted_at=p.granted_at.isoformat() if p.granted_at else None,
            )
            for p in perm_rows
        ]
    )


@router.post("/{folder_id}/permissions")
async def grant_folder_permission(
    folder_id: str,
    data: PermissionGrant,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Grant a permission to a user for a folder.
    Only the folder owner or users with admin permission can grant.
    """
    user_id = current_user["id"]
    now = datetime.utcnow()
    
    if data.permission not in ['read', 'write']:
        raise HTTPException(status_code=400, detail="Invalid permission level. Must be 'read' or 'write'.")
    
    async with get_postgres_session() as session:
        # Check folder exists and user is owner or admin
        folder = await session.execute(
            text("""
                SELECT id, user_id FROM neural_nexus.folders 
                WHERE id = :folder_id
            """),
            {"folder_id": folder_id}
        )
        row = folder.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        is_owner = str(row.user_id) == user_id
        if not is_owner:
            perm = await session.execute(
                text("""
                    SELECT permission FROM neural_nexus.folder_permissions 
                    WHERE folder_id = :folder_id AND user_id = :user_id
                """),
                {"folder_id": folder_id, "user_id": user_id}
            )
            perm_row = perm.fetchone()
            if not perm_row or perm_row.permission != 'admin':
                raise HTTPException(status_code=403, detail="Not authorized to grant permissions")
        
        # Find target user by email
        target = await session.execute(
            text("SELECT id FROM neural_nexus.users WHERE email = :email"),
            {"email": data.user_email}
        )
        target_row = target.fetchone()
        
        if not target_row:
            raise HTTPException(
                status_code=404, 
                detail=f"No user found with email '{data.user_email}'. They must create an account first."
            )
        
        target_user_id = str(target_row.id)
        
        # Prevent self-share
        if target_user_id == user_id:
            raise HTTPException(
                status_code=400, 
                detail="You cannot share a folder with yourself."
            )
        
        # Upsert permission
        await session.execute(
            text("""
                INSERT INTO neural_nexus.folder_permissions 
                (folder_id, user_id, permission, granted_at, granted_by)
                VALUES (:folder_id, :user_id, :permission, :granted_at, :granted_by)
                ON CONFLICT (folder_id, user_id) 
                DO UPDATE SET permission = :permission, granted_at = :granted_at
            """),
            {
                "folder_id": folder_id,
                "user_id": target_user_id,
                "permission": data.permission,
                "granted_at": now,
                "granted_by": user_id,
            }
        )
        await session.commit()
    
    logger.info(f"Granted {data.permission} permission on folder {folder_id} to {data.user_email}")
    
    return {
        "message": f"Permission '{data.permission}' granted to {data.user_email}",
        "folder_id": folder_id,
        "user_email": data.user_email,
        "permission": data.permission,
    }


@router.delete("/{folder_id}/permissions/{target_user_id}")
async def revoke_folder_permission(
    folder_id: str,
    target_user_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Revoke a user's permission for a folder.
    Only the folder owner or users with admin permission can revoke.
    """
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        # Check folder exists and user is owner or admin
        folder = await session.execute(
            text("""
                SELECT id, user_id FROM neural_nexus.folders 
                WHERE id = :folder_id
            """),
            {"folder_id": folder_id}
        )
        row = folder.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        is_owner = str(row.user_id) == user_id
        if not is_owner:
            perm = await session.execute(
                text("""
                    SELECT permission FROM neural_nexus.folder_permissions 
                    WHERE folder_id = :folder_id AND user_id = :user_id
                """),
                {"folder_id": folder_id, "user_id": user_id}
            )
            perm_row = perm.fetchone()
            if not perm_row or perm_row.permission != 'admin':
                raise HTTPException(status_code=403, detail="Not authorized to revoke permissions")
        
        # Delete permission
        await session.execute(
            text("""
                DELETE FROM neural_nexus.folder_permissions 
                WHERE folder_id = :folder_id AND user_id = :target_user_id
            """),
            {"folder_id": folder_id, "target_user_id": target_user_id}
        )
        await session.commit()
    
    logger.info(f"Revoked permission on folder {folder_id} from user {target_user_id}")
    
    return {
        "message": "Permission revoked",
        "folder_id": folder_id,
        "user_id": target_user_id,
    }

