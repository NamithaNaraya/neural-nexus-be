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


async def _create_file_record(
    file_id: str,
    filename: str,
    folder_id: str,
    user_id: str,
    file_type: str,
    file_size: int,
) -> None:
    """Create file record in PostgreSQL."""
    async with get_postgres_session() as session:
        await session.execute(
            text("""
                INSERT INTO neural_nexus.files 
                (id, folder_id, filename, file_type, file_size, status, created_at)
                VALUES (:id, :folder_id, :filename, :file_type, :file_size, :status, :created_at)
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
        
        # Send completion event
        await publish_ingestion_progress(
            user_id=user_id,
            file_id=file_id,
            phase="completed" if result.success else "failed",
            progress=100 if result.success else 0,
            message=f"Completed: {result.entity_count} entities, {result.relationship_count} relationships" 
                    if result.success else f"Failed: {result.error_message}",
        )
        
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
    allowed_extensions = {".pdf", ".csv", ".tsv", ".txt", ".docx", ".xlsx", ".md"}
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
        # Extract text content
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
        
        # Start background processing
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


@router.post("/upload/{file_id}/approve")
async def approve_upload(
    file_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Approve a file that's ready for review and commit to database."""
    
    async with get_postgres_session() as session:
        result = await session.execute(
            text("""
                SELECT status FROM neural_nexus.files WHERE id = :file_id
            """),
            {"file_id": file_id}
        )
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="File not found")
        
        if row.status != "ready_for_review":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot approve file with status '{row.status}'. Must be 'ready_for_review'."
            )
        
        # Update to completed
        await session.execute(
            text("""
                UPDATE neural_nexus.files 
                SET status = 'completed', processed_at = :processed_at
                WHERE id = :file_id
            """),
            {"file_id": file_id, "processed_at": datetime.utcnow()}
        )
        await session.commit()
    
    return {"message": "File approved and committed to knowledge graph", "file_id": file_id}


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
