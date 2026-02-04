"""
Pipeline Orchestrator - Coordinates the 7-Phase Agentic Ingestion

This is the main entry point for document processing.
Orchestrates all agents and handles SSE progress updates.
"""
import logging
import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from app.agents.layout_agent import LayoutAgent
from app.agents.chunking_agent import ChunkingAgent
from app.agents.ontology_agent import OntologyAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.deduplication_agent import DeduplicationAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.embedding_agent import EmbeddingAgent
from app.agents.storage_agent import StorageAgent
from app.services.graph_service import get_graph_service

logger = logging.getLogger(__name__)


class PipelinePhase(Enum):
    """Enumeration of pipeline phases."""
    LAYOUT = "layout"
    CHUNKING = "chunking"
    ONTOLOGY = "ontology"
    EXTRACTION = "extraction"
    DEDUPLICATION = "deduplication"
    VALIDATION = "validation"
    HUMAN_REVIEW = "human_review"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineProgress:
    """Represents current pipeline progress."""
    phase: PipelinePhase
    phase_number: int
    total_phases: int
    progress_percent: int
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = None


@dataclass
class PipelineResult:
    """Final result of pipeline execution."""
    success: bool
    file_id: str
    folder_id: str
    entity_count: int
    relationship_count: int
    chunk_count: int
    requires_review: bool
    validation_issues: List[Any] = field(default_factory=list)
    error_message: str = None
    duration_seconds: float = 0.0


class PipelineOrchestrator:
    """
    7-Phase Agentic Ingestion Pipeline Orchestrator.
    
    Phases:
    1. Layout Analysis
    2. Semantic Chunking
    3. Ontology Definition
    4. Entity/Relationship Extraction
    5. Deduplication (Neural Reconciliation)
    6. Validation
    7. Embedding Generation
    + Storage (commits to Neo4j/PostgreSQL)
    """
    
    def __init__(
        self,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            progress_callback: Optional callback for progress updates (e.g., SSE)
        """
        # Initialize agents
        self.layout_agent = LayoutAgent()
        self.chunking_agent = ChunkingAgent()
        self.ontology_agent = OntologyAgent()
        self.extraction_agent = ExtractionAgent()
        self.deduplication_agent = DeduplicationAgent()
        self.validation_agent = ValidationAgent()
        self.embedding_agent = EmbeddingAgent()
        self.storage_agent = StorageAgent()
        self.graph_service = get_graph_service()
        
        self.progress_callback = progress_callback
        self.total_phases = 8  # Including storage
        
    async def process(
        self,
        content: str,
        file_id: str,
        folder_id: str,
        user_id: str,
        file_type: str,
        auto_approve: bool = False,
    ) -> PipelineResult:
        """
        Run the complete ingestion pipeline.
        
        Args:
            content: Document text content
            file_id: File ID in database
            folder_id: Folder ID for scoping
            user_id: User ID for ownership
            file_type: Document type (pdf, csv, txt, etc.)
            auto_approve: If True, skip human review phase
            
        Returns:
            PipelineResult with processing outcome
        """
        start_time = datetime.utcnow()
        
        try:
            # Update file status to processing
            await self.storage_agent.update_file_status(file_id, "processing")
            
            # Phase 1: Layout Analysis
            await self._report_progress(PipelinePhase.LAYOUT, 1, "Analyzing document structure...")
            layout_result = await self.layout_agent.analyze(content, file_type)
            
            if not layout_result.get("sections"):
                # No sections found, create a single section from content
                layout_result = {
                    "sections": [{
                        "section_type": "paragraph",
                        "content": content,
                        "start_position": 0,
                        "end_position": len(content),
                        "level": 0,
                    }],
                    "hierarchy": {},
                    "metadata": {},
                    "document_type": file_type,
                }
            
            await self._report_progress(
                PipelinePhase.LAYOUT, 1,
                f"Found {len(layout_result['sections'])} sections",
                {"sections": len(layout_result["sections"])}
            )
            
            # Phase 2: Chunking
            await self._report_progress(PipelinePhase.CHUNKING, 2, "Creating semantic chunks...")
            chunks = await self.chunking_agent.chunk(layout_result)
            
            if not chunks:
                logger.warning("No chunks created, aborting pipeline")
                raise ValueError("Document produced no chunks")
            
            await self._report_progress(
                PipelinePhase.CHUNKING, 2,
                f"Created {len(chunks)} chunks",
                {"chunks": len(chunks)}
            )
            
            # Phase 3: Ontology
            await self._report_progress(PipelinePhase.ONTOLOGY, 3, "Defining schema...")
            schema = await self.ontology_agent.define_schema(chunks, folder_id)
            
            await self._report_progress(
                PipelinePhase.ONTOLOGY, 3,
                f"Schema: {len(schema.get('entity_types', []))} entity types, "
                f"{len(schema.get('relationship_types', []))} relationship types",
                {"entity_types": len(schema.get("entity_types", [])),
                 "relationship_types": len(schema.get("relationship_types", []))}
            )
            
            # Phase 4: Extraction
            await self._report_progress(PipelinePhase.EXTRACTION, 4, "Extracting entities and relationships...")
            extraction_result = await self.extraction_agent.extract(chunks, schema)
            
            entities = extraction_result.get("entities", [])
            relationships = extraction_result.get("relationships", [])
            
            await self._report_progress(
                PipelinePhase.EXTRACTION, 4,
                f"Extracted {len(entities)} entities, {len(relationships)} relationships",
                {"entities": len(entities), "relationships": len(relationships)}
            )
            
            # Phase 5: Deduplication
            await self._report_progress(PipelinePhase.DEDUPLICATION, 5, "Performing neural reconciliation...")
            dedup_result = await self.deduplication_agent.deduplicate(
                entities, relationships, folder_id
            )
            
            deduped_entities = dedup_result.get("new_entities", [])
            deduped_relationships = dedup_result.get("relationships", [])
            
            await self._report_progress(
                PipelinePhase.DEDUPLICATION, 5,
                f"After dedup: {len(deduped_entities)} unique entities",
                dedup_result.get("stats", {})
            )
            
            # Phase 6: Validation
            await self._report_progress(PipelinePhase.VALIDATION, 6, "Validating extractions...")
            validation_result = await self.validation_agent.validate(
                deduped_entities, deduped_relationships, chunks
            )
            
            validated_entities = validation_result.validated_entities
            validated_relationships = validation_result.validated_relationships
            
            await self._report_progress(
                PipelinePhase.VALIDATION, 6,
                f"Validated: {len(validated_entities)} entities, "
                f"{len(validated_relationships)} relationships "
                f"({validation_result.removed_count} removed, {validation_result.flagged_count} flagged)",
                {"removed": validation_result.removed_count, 
                 "flagged": validation_result.flagged_count}
            )
            
            # Check if human review is required
            requires_review = (
                not auto_approve and 
                (validation_result.flagged_count > 0 or 
                 len(validation_result.issues) > 0)
            )
            
            if requires_review:
                await self._report_progress(
                    PipelinePhase.HUMAN_REVIEW, 7,
                    "Ready for human review",
                    {"issues": len(validation_result.issues)}
                )
                
                # Update file status to ready_for_review
                await self.storage_agent.update_file_status(
                    file_id, 
                    "ready_for_review",
                    node_count=len(validated_entities),
                    relationship_count=len(validated_relationships),
                )
                
                # Store in temporary state for review
                await self.storage_agent.store_staging(
                    file_id=file_id,
                    entities=validated_entities,
                    relationships=validated_relationships,
                )
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                return PipelineResult(
                    success=True,
                    file_id=file_id,
                    folder_id=folder_id,
                    entity_count=len(validated_entities),
                    relationship_count=len(validated_relationships),
                    chunk_count=len(chunks),
                    requires_review=True,
                    validation_issues=validation_result.issues,
                    duration_seconds=duration,
                )
            
            # Phase 7: Embedding
            await self._report_progress(PipelinePhase.EMBEDDING, 7, "Generating embeddings...")
            embedded_entities = await self.embedding_agent.embed_entities(validated_entities)
            embedded_chunks = await self.embedding_agent.embed_chunks(chunks)
            
            await self._report_progress(
                PipelinePhase.EMBEDDING, 7,
                f"Generated embeddings for {len(embedded_entities)} entities, {len(embedded_chunks)} chunks"
            )
            
            # Phase 8: Storage
            await self._report_progress(PipelinePhase.STORAGE, 8, "Committing to database...")
            
            # Store entities
            entity_id_map = await self.storage_agent.store_entities(
                embedded_entities, file_id, folder_id, user_id
            )
            
            # Store relationships
            rel_count = await self.storage_agent.store_relationships(
                validated_relationships, entity_id_map, file_id, folder_id
            )
            
            # Store chunks
            chunk_count = await self.storage_agent.store_chunks(
                embedded_chunks, file_id, folder_id
            )
            
            # Update file status to completed
            await self.storage_agent.update_file_status(
                file_id,
                "completed",
                node_count=len(entity_id_map),
                relationship_count=rel_count,
            )
            
            # Create audit log
            await self.storage_agent.create_audit_log(
                user_id=user_id,
                action="ingest",
                target_type="file",
                target_id=file_id,
                details={
                    "entities_created": len(entity_id_map),
                    "relationships_created": rel_count,
                    "chunks_stored": chunk_count,
                }
            )
            
            # Phase 9: Graph Structural Enrichment (FastRP)
            await self._report_progress(PipelinePhase.STORAGE, 8, "Generating graph structural embeddings (FastRP)...")
            await self.graph_service.run_fastrp_node_embeddings(folder_id)
            
            await self._report_progress(
                PipelinePhase.COMPLETED, 8,
                f"Completed! {len(entity_id_map)} entities, {rel_count} relationships stored"
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return PipelineResult(
                success=True,
                file_id=file_id,
                folder_id=folder_id,
                entity_count=len(entity_id_map),
                relationship_count=rel_count,
                chunk_count=chunk_count,
                requires_review=False,
                duration_seconds=duration,
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            
            # Update file status to failed
            await self.storage_agent.update_file_status(
                file_id, "failed", error_message=str(e)
            )
            
            await self._report_progress(
                PipelinePhase.FAILED, 0,
                f"Pipeline failed: {str(e)}",
                error=str(e)
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return PipelineResult(
                success=False,
                file_id=file_id,
                folder_id=folder_id,
                entity_count=0,
                relationship_count=0,
                chunk_count=0,
                requires_review=False,
                error_message=str(e),
                duration_seconds=duration,
            )
    
    async def _report_progress(
        self,
        phase: PipelinePhase,
        phase_number: int,
        message: str,
        details: Dict[str, Any] = None,
        error: str = None,
    ) -> None:
        """Report progress to callback if available."""
        progress = PipelineProgress(
            phase=phase,
            phase_number=phase_number,
            total_phases=self.total_phases,
            progress_percent=int((phase_number / self.total_phases) * 100),
            message=message,
            details=details or {},
            error=error,
        )
        
        logger.info(f"[{phase.value.upper()}] {message}")
        
        if self.progress_callback:
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(progress)
                else:
                    self.progress_callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")


async def run_pipeline(
    content: str,
    file_id: str,
    folder_id: str,
    user_id: str,
    file_type: str,
    auto_approve: bool = False,
    progress_callback: Optional[Callable] = None,
) -> PipelineResult:
    """
    Convenience function to run the ingestion pipeline.
    
    Args:
        content: Document content
        file_id: File ID
        folder_id: Folder ID
        user_id: User ID
        file_type: File type (pdf, csv, txt, etc.)
        auto_approve: Skip human review
        progress_callback: Optional progress callback
        
    Returns:
        PipelineResult
    """
    orchestrator = PipelineOrchestrator(progress_callback)
    return await orchestrator.process(
        content=content,
        file_id=file_id,
        folder_id=folder_id,
        user_id=user_id,
        file_type=file_type,
        auto_approve=auto_approve,
    )
