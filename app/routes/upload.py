"""
Upload Routes

File upload handling with support for PDF, CSV, TSV, TXT, and other formats.
Triggers the 7-phase ingestion pipeline.
"""
import asyncio
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
import uuid
import logging
from datetime import datetime

from sqlalchemy import text

from app.core.security import get_current_user
from app.db.connections import get_postgres_session
from app.agents.pipeline import PipelineResult
from app.agents.langgraph_pipeline import run_langgraph_pipeline

router = APIRouter()
logger = logging.getLogger(__name__)


class UploadResponse(BaseModel):
    """Response after successful file upload."""
    file_id: str
    filename: str
    folder_id: str
    status: str
    message: str


class UploadStatus(BaseModel):
    """Status of an upload/ingestion task."""
    file_id: str
    filename: str
    status: str  # 'pending', 'processing', 'ready_for_review', 'completed', 'failed'
    progress: int  # 0-100
    current_phase: str
    node_count: int
    relationship_count: int
    message: str
    error_message: str = None


class UploadTextRequest(BaseModel):
    """Request for direct text ingestion."""
    filename: str
    content: str
    folder_id: str


class UploadCypherRequest(BaseModel):
    """Request for direct Cypher ingestion."""
    query: str
    folder_id: str
    filename: str = "Direct Cypher Ingestion"
    file_id: str = None  # Optional: append to existing file context


async def _create_file_record(
    file_id: str,
    filename: str,
    folder_id: str,
    user_id: str,
    file_type: str,
    file_size: int,
) -> None:
    """Create file record in PostgreSQL with conflict handling for retries."""
    async with get_postgres_session() as session:
        try:
            await session.execute(
                text("""
                    INSERT INTO neural_nexus.files 
                    (id, folder_id, filename, file_type, file_size, status, created_at)
                    VALUES (:id, :folder_id, :filename, :file_type, :file_size, :status, :created_at)
                    ON CONFLICT (id) DO UPDATE SET
                        status = 'pending',
                        error_message = NULL,
                        file_size = :file_size
                """),
                {
                    "id": file_id,
                    "folder_id": folder_id,
                    "filename": filename,
                    "file_type": file_type,
                    "file_size": file_size,
                    "status": "pending",
                    "created_at": datetime.utcnow(),
                }
            )
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to create file record for {file_id}: {e}")
            await session.rollback()
            raise


async def _process_file_async(
    content: str,
    file_id: str,
    folder_id: str,
    user_id: str,
    file_type: str,
) -> None:
    """Background task to process file through the pipeline with SSE progress updates."""
    from app.routes.sse import publish_ingestion_progress
    
    async def progress_callback(progress):
        """Send progress updates via SSE."""
        try:
            # Update database status for UI polling
            async with get_postgres_session() as session:
                await session.execute(
                    text("UPDATE neural_nexus.files SET progress = :progress, status = :status WHERE id = :file_id"),
                    {"progress": progress.progress_percent, "status": "processing", "file_id": file_id}
                )
                await session.commit()

            await publish_ingestion_progress(
                user_id=user_id,
                file_id=file_id,
                phase=progress.phase.value,
                progress=progress.progress_percent,
                message=progress.message,
            )
        except Exception as e:
            logger.warning(f"Failed to publish progress: {e}")
    
    try:
        final_state = await run_langgraph_pipeline(
            content=content,
            file_id=file_id,
            folder_id=folder_id,
            user_id=user_id,
            file_type=file_type,
            auto_approve=False,  # Pause for review before committing to Neo4j
            progress_callback=progress_callback,
        )
        
        # Transform LangGraph state to PipelineResult for the completion event
        success = final_state.get("status") in ["completed", "paused"]
        result = PipelineResult(
            success=success,
            file_id=file_id,
            folder_id=folder_id,
            entity_count=len(final_state.get("entity_id_map", {})),
            relationship_count=len(final_state.get("relationships", [])),
            chunk_count=len(final_state.get("chunks", [])),
            requires_review=final_state.get("requires_review", False),
            validation_issues=final_state.get("validation_issues", []),
            error_message=final_state.get("error"),
            duration_seconds=(datetime.utcnow() - final_state.get("start_time")).total_seconds()
        )
        
        # Update database with final result
        from app.agents.storage_agent import StorageAgent
        storage = StorageAgent()
        await storage.update_file_status(
            file_id=file_id,
            status="ready_for_review" if result.requires_review else ("completed" if result.success else "failed"),
            node_count=result.entity_count,
            relationship_count=result.relationship_count,
            error_message=result.error_message
        )
        
        # Explicitly set progress to 100 on completion
        if result.success:
            async with get_postgres_session() as session:
                await session.execute(
                    text("UPDATE neural_nexus.files SET progress = 100 WHERE id = :file_id"),
                    {"file_id": file_id}
                )
                await session.commit()
        
        logger.info(f"Pipeline finished for file {file_id}: success={result.success}")
        
        if result.success:
            logger.info(f"Pipeline completed for file {file_id}: "
                       f"{result.entity_count} entities, {result.relationship_count} relationships")
        else:
            logger.error(f"Pipeline failed for file {file_id}: {result.error_message}")
            
    except Exception as e:
        logger.error(f"Background processing failed for file {file_id}: {e}")
        
        # Send error event
        await publish_ingestion_progress(
            user_id=user_id,
            file_id=file_id,
            phase="failed",
            progress=0,
            message=f"Error: {str(e)}",
        )
        
        # Update file status to failed
        from app.agents.storage_agent import StorageAgent
        storage = StorageAgent()
        await storage.update_file_status(file_id, "failed", error_message=str(e))


def _extract_file_content(file_bytes: bytes, file_type: str) -> str:
    """Extract text content from file bytes."""
    # For now, handle text-based files
    # TODO: Add PDF extraction, DOCX parsing, etc.
    
    if file_type in ['txt', 'csv', 'tsv']:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                content = file_bytes.decode(encoding)
                if file_type in ['csv', 'tsv']:
                    # Simple validation for CSV/TSV
                    import io
                    import pandas as pd
                    sep = ',' if file_type == 'csv' else '\t'
                    df = pd.read_csv(io.StringIO(content), sep=sep)
                    return df.to_string(index=False)
                return content
            except Exception:
                continue
        raise ValueError("Could not decode or parse file content")
    
    elif file_type == 'xlsx':
        try:
            import io
            import pandas as pd
            # Use pandas to read Excel
            df = pd.read_excel(io.BytesIO(file_bytes))
            # Convert to string for the extraction pipeline
            # Note: For multi-sheet, we only take the first sheet by default
            return df.to_string(index=False)
        except ImportError:
            raise ValueError("Excel support requires 'openpyxl'. Please install it.")
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {e}")
    
    elif file_type == 'pdf':
        # TODO: Implement PDF extraction
        # For now, return placeholder
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_content = []
            for page in doc:
                text_content.append(page.get_text())
            return '\n'.join(text_content)
        except ImportError:
            raise ValueError("PDF support requires PyMuPDF. Install with: pip install pymupdf")
        except Exception as e:
            raise ValueError(f"Failed to extract PDF content: {e}")
    
    else:
        # Try to decode as text
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported file type: {file_type}")


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    folder_id: str = Form(...),
    current_user: dict = Depends(get_current_user),
) -> UploadResponse:
    """
    Upload a file for knowledge extraction.
    
    Supported formats: PDF, CSV, TSV, TXT, DOCX, etc.
    Files are processed through the 7-phase agentic pipeline.
    """
    # Validate file type
    allowed_extensions = {".pdf", ".csv", ".tsv", ".txt", ".docx", ".xlsx", ".md", ".cypher"}
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}",
        )
    
    # Read file content
    file_bytes = await file.read()
    file_size = len(file_bytes)
    file_type = file_ext[1:]  # Remove leading dot
    
    # Generate file ID
    file_id = str(uuid.uuid4())
    user_id = current_user['id']
    
    try:
        # Special handling for Cypher files - bypass extraction pipeline
        if file_type == 'cypher':
            try:
                content = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Try latin-1 as fallback for non-UTF-8 cypher files
                content = file_bytes.decode('latin-1')
                logger.warning(f"Cypher file {file.filename} decoded with latin-1 fallback")
            
            if not content.strip():
                raise HTTPException(status_code=400, detail="Cypher file is empty")
            
            # Create file record
            await _create_file_record(
                file_id=file_id,
                filename=file.filename,
                folder_id=folder_id,
                user_id=user_id,
                file_type="cypher",
                file_size=file_size,
            )
            
            # Validate the service can be imported before starting background task
            try:
                from app.services.managed_cypher_service import get_managed_cypher_service
                managed_service = get_managed_cypher_service()
            except Exception as import_err:
                logger.error(f"Failed to initialize ManagedCypherService: {import_err}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Cypher ingestion service initialization failed: {import_err}"
                )
            
            # Run in background to avoid timeout
            async def run_managed_cypher():
                from app.routes.sse import publish_ingestion_progress
                try:
                    logger.info(f"Starting background Cypher ingestion for file {file_id} ({file.filename})")
                    
                    # Update status to processing
                    from app.agents.storage_agent import StorageAgent
                    storage = StorageAgent()
                    await storage.update_file_status(file_id, "processing")
                    
                    result = await managed_service.execute_managed_query(
                        query=content,
                        file_id=file_id,
                        folder_id=folder_id,
                        user_id=user_id
                    )
                    
                    await storage.update_file_status(
                        file_id=file_id,
                        status="completed",
                        node_count=result["node_count"],
                        relationship_count=result.get("relationship_count", 0),
                    )
                    
                    logger.info(
                        f"Cypher ingestion completed for {file_id}: "
                        f"{result['node_count']} nodes, {result.get('relationship_count', 0)} rels"
                    )
                    
                    # Run FastRP Embeddings (non-fatal)
                    try:
                        from app.services.graph_service import get_graph_service
                        graph_service = get_graph_service()
                        await graph_service.run_fastrp_node_embeddings(folder_id)
                    except Exception as e:
                        logger.warning(f"FastRP failed (non-fatal): {e}")
                    
                    # Publish completion via SSE
                    try:
                        await publish_ingestion_progress(
                            user_id=user_id,
                            file_id=file_id,
                            phase="completed",
                            progress=100,
                            message=f"Cypher ingestion complete: {result['node_count']} nodes",
                        )
                    except Exception:
                        pass
                        
                except Exception as e:
                    logger.error(f"Background Cypher ingestion failed for {file_id}: {e}", exc_info=True)
                    from app.agents.storage_agent import StorageAgent
                    storage = StorageAgent()
                    await storage.update_file_status(file_id, "failed", error_message=str(e))
                    
                    # Publish failure via SSE
                    try:
                        await publish_ingestion_progress(
                            user_id=user_id,
                            file_id=file_id,
                            phase="failed",
                            progress=0,
                            message=f"Cypher ingestion failed: {str(e)}",
                        )
                    except Exception:
                        pass

            background_tasks.add_task(run_managed_cypher)
            
            return UploadResponse(
                file_id=file_id,
                filename=file.filename,
                folder_id=folder_id,
                status="pending",
                message="Cypher file uploaded. Ingestion started in background.",
            )

        # Extract text content for other files
        content = _extract_file_content(file_bytes, file_type)
        
        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail="File is empty or could not be read",
            )
        
        # Create file record
        await _create_file_record(
            file_id=file_id,
            filename=file.filename,
            folder_id=folder_id,
            user_id=user_id,
            file_type=file_type,
            file_size=file_size,
        )
        
        # Start background processing for AI pipeline
        background_tasks.add_task(
            _process_file_async,
            content,
            file_id,
            folder_id,
            user_id,
            file_type,
        )
        
        logger.info(f"File uploaded: {file.filename} by user {user_id}, starting pipeline")
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            folder_id=folder_id,
            status="pending",
            message="File uploaded successfully. Processing started.",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process upload: {str(e)}",
        )


@router.post("/upload/text", response_model=UploadResponse)
async def upload_text(
    background_tasks: BackgroundTasks,
    request: UploadTextRequest,
    current_user: dict = Depends(get_current_user),
) -> UploadResponse:
    """
    Ingest text content directly.
    Similar to file upload, but content is provided as a string.
    """
    content = request.content
    filename = request.filename
    folder_id = request.folder_id
    user_id = current_user['id']
    
    if not content.strip():
        raise HTTPException(
            status_code=400,
            detail="Content cannot be empty",
        )
    
    # Generate file ID
    file_id = str(uuid.uuid4())
    file_size = len(content.encode('utf-8'))
    file_type = "txt"
    
    try:
        # Create file record
        await _create_file_record(
            file_id=file_id,
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            file_type=file_type,
            file_size=file_size,
        )
        
        # Start background processing
        background_tasks.add_task(
            _process_file_async,
            content,
            file_id,
            folder_id,
            user_id,
            file_type,
        )
        
        logger.info(f"Text ingested: {filename} by user {user_id}, starting pipeline")
        
        return UploadResponse(
            file_id=file_id,
            filename=filename,
            folder_id=folder_id,
            status="pending",
            message="Text ingested successfully. Processing started.",
        )
        
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process text ingestion: {str(e)}",
        )


class CypherPreviewRequest(BaseModel):
    """Request to preview a Cypher query transformation."""
    query: str
    folder_id: str


@router.post("/upload/cypher/preview")
async def preview_cypher(
    request: CypherPreviewRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Preview what the platform will do with a Cypher query BEFORE running it.
    Shows: how many statements, which are skipped, what labels will change.
    No data is modified.
    """
    from app.services.managed_cypher_service import get_managed_cypher_service
    
    managed_service = get_managed_cypher_service()
    preview = managed_service.preview_query(request.query, request.folder_id)
    return preview


@router.post("/upload/cypher", response_model=UploadResponse)
async def upload_cypher(
    background_tasks: BackgroundTasks,
    request: UploadCypherRequest,
    current_user: dict = Depends(get_current_user),
) -> UploadResponse:
    """
    Ingest data directly via Cypher query.
    Bypasses AI extraction but automatically generates embeddings for new nodes.
    Normalizes labels, syncs to PostgreSQL, and runs FastRP.
    """
    from app.services.managed_cypher_service import get_managed_cypher_service
    
    query = request.query
    folder_id = request.folder_id
    filename = request.filename
    user_id = current_user['id']
    file_id = request.file_id or str(uuid.uuid4())
    
    try:
        # 1. Create file record if it's a new ingestion
        if not request.file_id:
            await _create_file_record(
                file_id=file_id,
                filename=filename,
                folder_id=folder_id,
                user_id=user_id,
                file_type="cypher",
                file_size=len(query.encode('utf-8')),
            )
        
        # 2. Execute via ManagedCypherService (handles label normalization + PostgreSQL staging sync)
        managed_service = get_managed_cypher_service()
        result = await managed_service.execute_managed_query(
            query=query,
            file_id=file_id,
            folder_id=folder_id,
            user_id=user_id
        )
        
        # 3. Update file status with both node and relationship counts
        from app.agents.storage_agent import StorageAgent
        storage = StorageAgent()
        await storage.update_file_status(
            file_id=file_id,
            status="completed",
            node_count=result["node_count"],
            relationship_count=result.get("relationship_count", 0),
        )
        
        # 4. Run FastRP Embeddings (for structure)
        try:
            from app.services.graph_service import get_graph_service
            graph_service = get_graph_service()
            await graph_service.run_fastrp_node_embeddings(folder_id)
        except Exception as e:
            logger.warning(f"FastRP failed for Cypher ingestion (optional): {e}")
            
        logger.info(f"Managed Cypher ingestion successful: {file_id}")
        
        return UploadResponse(
            file_id=file_id,
            filename=filename,
            folder_id=folder_id,
            status="completed",
            message=result["message"] + " FastRP updated.",
        )
        
    except Exception as e:
        logger.error(f"Cypher ingestion failed: {e}")
        # Update file status to failed if record was created
        try:
            storage = StorageAgent()
            await storage.update_file_status(file_id, "failed", error_message=str(e))
        except:
            pass
            
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process Cypher ingestion: {str(e)}",
        )


@router.get("/upload/status/{file_id}", response_model=UploadStatus)
async def get_upload_status(
    file_id: str,
    current_user: dict = Depends(get_current_user),
) -> UploadStatus:
    """Get the status of a file upload/ingestion."""
    
    async with get_postgres_session() as session:
        result = await session.execute(
            text("""
                SELECT id, filename, status, node_count, relationship_count, error_message
                FROM neural_nexus.files
                WHERE id = :file_id
            """),
            {"file_id": file_id}
        )
        row = result.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Map status to progress
    status_progress = {
        "pending": (10, "Queued for processing"),
        "processing": (50, "Extracting knowledge..."),
        "ready_for_review": (80, "Ready for human review"),
        "completed": (100, "Processing complete"),
        "failed": (0, "Processing failed"),
    }
    
    progress, message = status_progress.get(row.status, (0, "Unknown status"))
    
    return UploadStatus(
        file_id=row.id,
        filename=row.filename,
        status=row.status,
        progress=progress,
        current_phase=row.status,
        node_count=row.node_count or 0,
        relationship_count=row.relationship_count or 0,
        message=message,
        error_message=row.error_message,
    )


@router.delete("/upload/{file_id}")
async def cancel_upload(
    file_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, str]:
    """Cancel an in-progress upload/ingestion."""
    
    async with get_postgres_session() as session:
        # Update status to cancelled
        await session.execute(
            text("""
                UPDATE neural_nexus.files 
                SET status = 'failed', error_message = 'Cancelled by user'
                WHERE id = :file_id
            """),
            {"file_id": file_id}
        )
        await session.commit()
    
    return {"message": f"Upload {file_id} cancelled"}
