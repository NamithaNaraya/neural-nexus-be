"""
Deletion API Routes

Background deletion with reference counting.
Features:
- File and folder deletion with job tracking
- Reference-counted shared node preservation
- Job status monitoring
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List
from pydantic import BaseModel
import logging

from app.db.connections import get_neo4j, get_postgres_session
from app.core.security import get_current_user
from app.services.deletion_service import get_deletion_service, DeletionStatus
from app.services.permission_service import get_permission_service, PermissionLevel
from app.services.cache_service import get_cache_service, CacheService
from app.services.gds_service import get_gds_service, GDSService

router = APIRouter(prefix="/deletion", tags=["deletion"])
logger = logging.getLogger(__name__)


class DeletionRequest(BaseModel):
    """Request to delete a file or folder."""
    target_id: str
    target_type: str = "file"  # "file" or "folder"
    background: bool = True


class DeletionJobResponse(BaseModel):
    """Response with deletion job details."""
    job_id: str
    target_type: str
    target_id: str
    status: str
    progress: float
    message: str
    stats: Dict[str, int]


@router.post("/delete")
async def initiate_deletion(
    request: DeletionRequest,
    background_tasks: BackgroundTasks,
    neo4j=Depends(get_neo4j),
    db=Depends(get_postgres_session),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
    current_user=Depends(get_current_user),
) -> DeletionJobResponse:
    """
    Initiate deletion of a file or folder.
    
    Args:
        request: Deletion request with target details
    
    Returns:
        Job ID and initial status for tracking
    """
    # Check permissions
    permission_service = get_permission_service(db)
    
    # For files, we'd need to look up the parent folder
    # For now, assume folder-level check
    if request.target_type == "folder":
        has_permission = await permission_service.check_permission(
            str(current_user["id"]),
            request.target_id,
            PermissionLevel.OWNER,  # Only owners can delete
        )
        if not has_permission:
            raise HTTPException(
                status_code=403,
                detail="Only folder owners can delete folders"
            )
    
    deletion_service = get_deletion_service(neo4j, db, cache, gds)
    
    if request.target_type == "folder":
        job = await deletion_service.delete_folder(
            folder_id=request.target_id,
            user_id=str(current_user["id"]),
            background=request.background,
        )
    else:
        job = await deletion_service.delete_file(
            file_id=request.target_id,
            user_id=str(current_user["id"]),
            background=request.background,
        )
    
    return DeletionJobResponse(
        job_id=job.job_id,
        target_type=job.target_type,
        target_id=job.target_id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        stats=job.stats,
    )


@router.get("/job/{job_id}")
async def get_deletion_status(
    job_id: str,
    neo4j=Depends(get_neo4j),
    db=Depends(get_postgres_session),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
    current_user=Depends(get_current_user),
) -> DeletionJobResponse:
    """
    Get status of a deletion job.
    
    Args:
        job_id: ID of the deletion job
    
    Returns:
        Current job status and progress
    """
    deletion_service = get_deletion_service(neo4j, db, cache, gds)
    job = deletion_service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail="Deletion job not found"
        )
    
    # Only allow users to see their own jobs
    if job.user_id != str(current_user["id"]):
        raise HTTPException(
            status_code=403,
            detail="Access denied to this deletion job"
        )
    
    return DeletionJobResponse(
        job_id=job.job_id,
        target_type=job.target_type,
        target_id=job.target_id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        stats=job.stats,
    )


@router.get("/jobs")
async def list_deletion_jobs(
    neo4j=Depends(get_neo4j),
    db=Depends(get_postgres_session),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
    current_user=Depends(get_current_user),
) -> List[DeletionJobResponse]:
    """
    List all deletion jobs for the current user.
    
    Returns:
        List of active and recent deletion jobs
    """
    deletion_service = get_deletion_service(neo4j, db, cache, gds)
    jobs = deletion_service.get_active_jobs(user_id=str(current_user["id"]))
    
    return [
        DeletionJobResponse(
            job_id=job.job_id,
            target_type=job.target_type,
            target_id=job.target_id,
            status=job.status.value,
            progress=job.progress,
            message=job.message,
            stats=job.stats,
        )
        for job in jobs
    ]


@router.get("/references/{entity_id}")
async def get_entity_references(
    entity_id: str,
    neo4j=Depends(get_neo4j),
    db=Depends(get_postgres_session),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get reference count for an entity.
    
    Shows how many files reference this entity.
    Entity will not be deleted until all referencing files are removed.
    
    Args:
        entity_id: ID of the entity to check
    
    Returns:
        Reference count and list of referencing files
    """
    deletion_service = get_deletion_service(neo4j, db, cache, gds)
    ref_count = await deletion_service.ref_counter.get_reference_count(entity_id)
    
    # Get list of files referencing this entity
    query = """
    MATCH (e:Entity {id: $entity_id})<-[:CONTAINS|MENTIONS]-(f:File)
    RETURN f.id as file_id, f.name as file_name
    LIMIT 10
    """
    async with neo4j.session() as session:
        result = await session.run(query, entity_id=entity_id)
        records = await result.data()
    
    return {
        "entity_id": entity_id,
        "reference_count": ref_count,
        "is_shared": ref_count > 1,
        "referencing_files": records,
        "note": (
            "This entity is shared across multiple files and will be preserved "
            "until all referencing files are deleted."
            if ref_count > 1
            else "This entity is only in one file and will be deleted with it."
        ),
    }


@router.delete("/file/{file_id}")
async def delete_file(
    file_id: str,
    background: bool = True,
    neo4j=Depends(get_neo4j),
    db=Depends(get_postgres_session),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
    current_user=Depends(get_current_user),
) -> DeletionJobResponse:
    """
    Quick endpoint to delete a specific file.
    
    Args:
        file_id: ID of the file to delete
        background: If True, run deletion in background
    
    Returns:
        Deletion job details
    """
    deletion_service = get_deletion_service(neo4j, db, cache, gds)
    
    job = await deletion_service.delete_file(
        file_id=file_id,
        user_id=str(current_user["id"]),
        background=background,
    )
    
    return DeletionJobResponse(
        job_id=job.job_id,
        target_type=job.target_type,
        target_id=job.target_id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        stats=job.stats,
    )


@router.delete("/folder/{folder_id}")
async def delete_folder(
    folder_id: str,
    background: bool = True,
    neo4j=Depends(get_neo4j),
    db=Depends(get_postgres_session),
    cache: CacheService = Depends(get_cache_service),
    gds: GDSService = Depends(get_gds_service),
    current_user=Depends(get_current_user),
) -> DeletionJobResponse:
    """
    Quick endpoint to delete a specific folder and all its contents.
    
    Args:
        folder_id: ID of the folder to delete
        background: If True, run deletion in background
    
    Returns:
        Deletion job details
    """
    # Check permissions
    permission_service = get_permission_service(db)
    has_permission = await permission_service.check_permission(
        str(current_user["id"]),
        folder_id,
        PermissionLevel.OWNER,
    )
    if not has_permission:
        raise HTTPException(
            status_code=403,
            detail="Only folder owners can delete folders"
        )
    
    deletion_service = get_deletion_service(neo4j, db)
    
    job = await deletion_service.delete_folder(
        folder_id=folder_id,
        user_id=str(current_user["id"]),
        background=background,
    )
    
    return DeletionJobResponse(
        job_id=job.job_id,
        target_type=job.target_type,
        target_id=job.target_id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        stats=job.stats,
    )
