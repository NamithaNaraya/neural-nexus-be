"""
File Routes

File management endpoints (status, details, deletion).
Upload is handled in upload.py
"""
from typing import Optional, List
from datetime import datetime
import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import logging

from sqlalchemy import text

from app.core.security import get_current_user
from app.db.connections import get_postgres_session, get_neo4j_driver

router = APIRouter()
logger = logging.getLogger(__name__)


# === Review Inbox Models & Endpoints ===

class PendingFileResponse(BaseModel):
    """File pending review."""
    id: str
    folder_id: str
    folder_name: str
    filename: str
    file_type: str
    file_size: int
    node_count: int
    relationship_count: int
    created_at: str


class ExtractionPreview(BaseModel):
    """Preview of extracted entities and relationships."""
    file_id: str
    filename: str
    entities: list
    relationships: list
    summary: dict


@router.get("/pending", response_model=List[PendingFileResponse])
async def list_pending_files(
    current_user: dict = Depends(get_current_user),
) -> List[PendingFileResponse]:
    """
    List all files pending review (ready_for_review status).
    These are files that have been processed but not yet approved.
    """
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        result = await session.execute(
            text("""
                SELECT 
                    f.id, f.folder_id, fo.name as folder_name,
                    f.filename, f.file_type, f.file_size,
                    f.node_count, f.relationship_count, f.created_at
                FROM neural_nexus.files f
                JOIN neural_nexus.folders fo ON fo.id = f.folder_id
                WHERE fo.user_id = :user_id 
                AND f.status = 'ready_for_review'
                ORDER BY f.created_at DESC
            """),
            {"user_id": user_id}
        )
        rows = result.fetchall()
    
    return [
        PendingFileResponse(
            id=str(row.id),
            folder_id=str(row.folder_id),
            folder_name=row.folder_name,
            filename=row.filename,
            file_type=row.file_type or "unknown",
            file_size=row.file_size or 0,
            node_count=row.node_count or 0,
            relationship_count=row.relationship_count or 0,
            created_at=row.created_at.isoformat() if row.created_at else "",
        )
        for row in rows
    ]


@router.get("/{file_id}/extraction-preview", response_model=ExtractionPreview)
async def get_extraction_preview(
    file_id: str,
    current_user: dict = Depends(get_current_user),
) -> ExtractionPreview:
    """
    Get a preview of extracted entities and relationships for review.
    Used in the Review Inbox to show what will be added to the graph.
    """
    user_id = current_user["id"]
    
    async with get_postgres_session() as session:
        # Verify file belongs to user and is pending review
        result = await session.execute(
            text("""
                SELECT f.id, f.filename, f.status, f.folder_id
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
        filename = row.filename
    
    # Fetch entities from staging (Postgres) first, then fallback to Neo4j
    entities = []
    relationships = []
    
    try:
        from app.agents.storage_agent import StorageAgent
        storage = StorageAgent()
        staging_data = await storage.get_staging(file_id)
        
        if staging_data["entities"]:
            entities = staging_data["entities"]
            relationships = staging_data["relationships"]
        else:
            # Fallback to Neo4j for already processed/legacy files
            driver = get_neo4j_driver()
            async with driver.session() as neo_session:
                # Get entities for this file
                entity_result = await neo_session.run("""
                    MATCH (e:Entity)
                    WHERE $file_id IN e.file_ids
                    RETURN 
                        e.id as id,
                        e.name as name,
                        e.type as type,
                        e.description as description,
                        e.properties as properties
                    LIMIT 200
                """, file_id=file_id)
                
                async for record in entity_result:
                    # Parse properties JSON string if needed
                    props = record["properties"]
                    if isinstance(props, str):
                        try:
                            props = json.loads(props)
                        except:
                            props = {}
                    entities.append({
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"] or "Unknown",
                        "description": record["description"],
                        "properties": props or {},
                    })
                
                # Get relationships for this file
                rel_result = await neo_session.run("""
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE $file_id IN a.file_ids AND $file_id IN b.file_ids
                    RETURN 
                        COALESCE(a.id, elementId(a)) as source_id,
                        a.name as source_name,
                        type(r) as relationship_type,
                        COALESCE(b.id, elementId(b)) as target_id,
                        b.name as target_name
                    LIMIT 200
                """, file_id=file_id)
                
                async for record in rel_result:
                    relationships.append({
                        "source_id": record["source_id"],
                        "source_name": record["source_name"],
                        "type": record["relationship_type"],
                        "target_id": record["target_id"],
                        "target_name": record["target_name"],
                    })
                
    except Exception as e:
        logger.error(f"Failed to fetch extraction preview: {e}")
    
    # Build summary
    type_counts = {}
    for entity in entities:
        t = entity.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    rel_type_counts = {}
    for rel in relationships:
        t = rel.get("type", "RELATED_TO")
        rel_type_counts[t] = rel_type_counts.get(t, 0) + 1
    
    return ExtractionPreview(
        file_id=file_id,
        filename=filename,
        entities=entities,
        relationships=relationships,
        summary={
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "entity_types": type_counts,
            "relationship_types": rel_type_counts,
        }
    )


@router.post("/{file_id}/approve")
async def approve_file_ingestion(
    file_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Approve a file's extracted data to be included in the graph.
    Changes status from 'ready_for_review' to 'completed'.
    """
    user_id = current_user["id"]
    from datetime import datetime
    now = datetime.utcnow()
    
    async with get_postgres_session() as session:
        # Verify file belongs to user and is pending review
        result = await session.execute(
            text("""
                SELECT f.id, f.status, f.filename, f.folder_id, fo.user_id
                FROM neural_nexus.files f
                JOIN neural_nexus.folders fo ON fo.id = f.folder_id
                WHERE f.id = :file_id AND fo.user_id = :user_id
            """),
            {"file_id": file_id, "user_id": user_id}
        )
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="File not found")
        
        if row.status != 'ready_for_review':
            raise HTTPException(
                status_code=400, 
                detail=f"File is not pending review (status: {row.status})"
            )
        
        folder_id = str(row.folder_id)
        filename = row.filename

    # 1. Fetch staging data
    from app.agents.storage_agent import StorageAgent
    from app.agents.embedding_agent import EmbeddingAgent
    from app.services.graph_service import get_graph_service
    
    storage = StorageAgent()
    embedding_agent = EmbeddingAgent()
    graph_service = get_graph_service()
    
    staging_data = await storage.get_staging(file_id)
    entities = staging_data.get("entities", [])
    relationships = staging_data.get("relationships", [])
    
    logger.info(f"[APPROVAL DEBUG] File {file_id}: Found {len(entities)} entities and {len(relationships)} relationships in staging")
    
    if not entities:
        logger.warning(f"No entities found in staging for file {file_id}. This could mean:")
        logger.warning("  1. The extraction phase failed to produce entities")
        logger.warning("  2. The staging table was not populated during the pipeline")
        logger.warning("  3. The AI service returned empty results")
        # We still update the status to completed, but with 0 counts
        async with get_postgres_session() as session:
            await session.execute(
                text("""
                    UPDATE neural_nexus.files 
                    SET status = 'completed', 
                        node_count = 0,
                        relationship_count = 0,
                        processed_at = :now,
                        error_message = 'No entities extracted from document'
                    WHERE id = :file_id
                """),
                {"file_id": file_id, "now": now}
            )
            await session.execute(
                text("DELETE FROM neural_nexus.entity_staging WHERE file_id = :file_id"),
                {"file_id": file_id}
            )
            await session.commit()
        
        return {
            "message": "File approved but no entities were extracted",
            "file_id": file_id,
            "filename": filename,
            "status": "completed",
            "entities_stored": 0,
            "relationships_stored": 0,
            "warning": "The AI extraction did not find any entities in this document."
        }
    
    try:
        # 2. Embedding (optional - skip if Ollama not available)
        try:
            logger.info(f"[APPROVAL DEBUG] Generating embeddings for {len(entities)} entities")
            embedded_entities = await embedding_agent.embed_entities(entities)
            logger.info(f"[APPROVAL DEBUG] Embeddings generated successfully")
        except Exception as e:
            logger.warning(f"Embedding failed (continuing without embeddings): {e}")
            embedded_entities = entities  # Use entities without embeddings
        
        # 3. Final Storage in Neo4j
        logger.info(f"[APPROVAL DEBUG] Storing {len(embedded_entities)} entities to Neo4j")
        entity_id_map = await storage.store_entities(
            embedded_entities, file_id, folder_id, user_id
        )
        logger.info(f"[APPROVAL DEBUG] stored {len(entity_id_map)} entities, entity_id_map: {list(entity_id_map.keys())[:5]}...")
        
        logger.info(f"[APPROVAL DEBUG] Storing {len(relationships)} relationships to Neo4j")
        rel_count = await storage.store_relationships(
            relationships, entity_id_map, file_id, folder_id
        )
        logger.info(f"[APPROVAL DEBUG] Stored {rel_count} relationships")
        
        # 4. Update status and counts in Postgres
        async with get_postgres_session() as session:
            await session.execute(
                text("""
                    UPDATE neural_nexus.files 
                    SET status = 'completed', 
                        node_count = :node_count,
                        relationship_count = :relationship_count,
                        processed_at = :now
                    WHERE id = :file_id
                """),
                {
                    "file_id": file_id, 
                    "node_count": len(entity_id_map),
                    "relationship_count": rel_count,
                    "now": now,
                }
            )
            
            # 5. Clean up staging
            await session.execute(
                text("DELETE FROM neural_nexus.entity_staging WHERE file_id = :file_id"),
                {"file_id": file_id}
            )
            
            await session.commit()
            
        # 6. Run FastRP or other analytics
        try:
            await graph_service.run_fastrp_node_embeddings(folder_id)
        except Exception as e:
            logger.warning(f"FastRP enrichment failed during approval: {e}")

    except Exception as e:
        logger.error(f"Approval commit failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to commit data to graph: {str(e)}")
    
    logger.info(f"Successfully finalized ingestion for file {file_id}")
    
    return {
        "message": "File approved and 100% committed to knowledge graph",
        "file_id": file_id,
        "filename": filename,
        "status": "completed",
        "entities_stored": len(entity_id_map),
        "relationships_stored": rel_count
    }


@router.post("/{file_id}/reject")
async def reject_file_ingestion(
    file_id: str,
    reason: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Reject a file's extracted data.
    Removes the pending data from Neo4j and marks file as rejected.
    """
    user_id = current_user["id"]
    from datetime import datetime
    now = datetime.utcnow()
    
    async with get_postgres_session() as session:
        # Verify file belongs to user and is pending review
        result = await session.execute(
            text("""
                SELECT f.id, f.status, f.filename, f.folder_id
                FROM neural_nexus.files f
                JOIN neural_nexus.folders fo ON fo.id = f.folder_id
                WHERE f.id = :file_id AND fo.user_id = :user_id
            """),
            {"file_id": file_id, "user_id": user_id}
        )
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="File not found")
        
        if row.status != 'ready_for_review':
            raise HTTPException(
                status_code=400, 
                detail=f"File is not pending review (status: {row.status})"
            )
        
        folder_id = str(row.folder_id)
        
        # Update status to rejected
        await session.execute(
            text("""
                UPDATE neural_nexus.files 
                SET status = 'rejected', 
                    processed_at = :processed_at,
                    error_message = :reason
                WHERE id = :file_id
            """),
            {
                "file_id": file_id, 
                "processed_at": now,
                "reason": reason or "Rejected by user",
            }
        )
        await session.commit()
    
    # Remove the pending data from Neo4j
    try:
        driver = get_neo4j_driver()
        async with driver.session() as neo_session:
            # Remove entities that only belong to this file
            await neo_session.run("""
                MATCH (e:Entity)
                WHERE $file_id IN e.file_ids
                SET e.file_ids = [x IN e.file_ids WHERE x <> $file_id]
                WITH e
                WHERE size(e.file_ids) = 0
                DETACH DELETE e
            """, file_id=file_id)
            
            # Remove chunks
            await neo_session.run(
                "MATCH (c:Chunk {file_id: $file_id}) DELETE c",
                file_id=file_id
            )
            
    except Exception as e:
        logger.warning(f"Failed to clean Neo4j for rejected file {file_id}: {e}")
    
    logger.info(f"Rejected file {file_id} ({row.filename})")
    
    return {
        "message": "File rejected and removed from pending data",
        "file_id": file_id,
        "filename": row.filename,
        "status": "rejected",
        "reason": reason,
    }


# === Standard File Endpoints ===

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
