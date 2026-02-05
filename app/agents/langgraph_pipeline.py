"""
LangGraph Pipeline Orchestrator

Refactored ingestion pipeline using LangGraph for multi-step AI reasoning.
Coordinates Layout, Chunking, Ontology, Extraction, Deduplication, Validation, 
Embedding, and Storage agents as a stateful graph.
"""
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Annotated, TypedDict, Union, Callable

from langgraph.graph import StateGraph, END

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

# Define the State for the LangGraph pipeline
class GraphState(TypedDict):
    """
    Represents the state of the ingestion pipeline.
    """
    # Input data
    content: str
    file_id: str
    folder_id: str
    user_id: str
    file_type: str
    auto_approve: bool
    
    # Processed data (passed between nodes)
    layout_result: Dict[str, Any]
    chunks: List[Any]
    schema: Dict[str, Any]
    entities: List[Any]
    relationships: List[Any]
    
    # Validation & Review
    requires_review: bool
    validation_issues: List[Any]
    
    # Final mappings
    entity_id_map: Dict[str, str]
    
    # Metadata
    status: str
    error: Optional[str]
    progress_callback: Optional[Callable]
    start_time: datetime

# Node Implementations
async def layout_node(state: GraphState) -> Dict[str, Any]:
    """Phase 1: Layout Analysis"""
    logger.info("LangGraph: Entering Layout Phase")
    agent = LayoutAgent()
    
    if state.get("progress_callback"):
        await _report_progress(state, "layout", 1, "Analyzing document structure...")
        
    layout_result = await agent.analyze(state["content"], state["file_type"])
    
    if not layout_result.get("sections"):
        layout_result = {
            "sections": [{
                "section_type": "paragraph",
                "content": state["content"],
                "start_position": 0,
                "end_position": len(state["content"]),
                "level": 0,
            }],
            "hierarchy": {},
            "metadata": {},
            "document_type": state["file_type"],
        }
    
    return {"layout_result": layout_result}

async def chunking_node(state: GraphState) -> Dict[str, Any]:
    """Phase 2: Semantic Chunking"""
    logger.info("LangGraph: Entering Chunking Phase")
    agent = ChunkingAgent()
    
    if state.get("progress_callback"):
        await _report_progress(state, "chunking", 2, "Creating semantic chunks...")
        
    chunks = await agent.chunk(state["layout_result"])
    
    if not chunks:
        raise ValueError("Document produced no chunks")
        
    return {"chunks": chunks}

async def ontology_node(state: GraphState) -> Dict[str, Any]:
    """Phase 3: Ontology Definition"""
    logger.info("LangGraph: Entering Ontology Phase")
    agent = OntologyAgent()
    
    if state.get("progress_callback"):
        await _report_progress(state, "ontology", 3, "Defining schema...")
        
    schema = await agent.define_schema(state["chunks"], state["folder_id"])
    
    return {"schema": schema}

async def extraction_node(state: GraphState) -> Dict[str, Any]:
    """Phase 4: Entity & Relationship Extraction"""
    logger.info("LangGraph: Entering Extraction Phase")
    agent = ExtractionAgent()
    
    if state.get("progress_callback"):
        await _report_progress(state, "extraction", 4, "Extracting entities and relationships...")
    
    logger.info(f"[EXTRACTION DEBUG] Processing {len(state.get('chunks', []))} chunks")
    
    result = await agent.extract(state["chunks"], state["schema"])
    
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])
    
    logger.info(f"[EXTRACTION DEBUG] Extracted {len(entities)} entities and {len(relationships)} relationships")
    
    if entities:
        # Log a sample of entities for debugging
        sample_names = [e.name if hasattr(e, 'name') else e.get('name', 'Unknown') for e in entities[:5]]
        logger.info(f"[EXTRACTION DEBUG] Sample entities: {sample_names}")
    else:
        logger.warning("[EXTRACTION DEBUG] No entities extracted! Check AI service and schema.")
    
    return {
        "entities": entities,
        "relationships": relationships
    }

async def deduplication_node(state: GraphState) -> Dict[str, Any]:
    """Phase 5: Neural Reconciliation"""
    logger.info("LangGraph: Entering Deduplication Phase")
    agent = DeduplicationAgent()
    
    if state.get("progress_callback"):
        await _report_progress(state, "deduplication", 5, "Performing neural reconciliation...")
        
    result = await agent.deduplicate(
        state["entities"], state["relationships"], state["folder_id"]
    )
    
    return {
        "entities": result.get("new_entities", []),
        "relationships": result.get("relationships", [])
    }

async def validation_node(state: GraphState) -> Dict[str, Any]:
    """Phase 6: Validation"""
    logger.info("LangGraph: Entering Validation Phase")
    agent = ValidationAgent()
    
    if state.get("progress_callback"):
        await _report_progress(state, "validation", 6, "Validating extractions...")
        
    result = await agent.validate(
        state["entities"], state["relationships"], state["chunks"]
    )
    
    # Determine if human review is needed
    # If auto_approve=True, NEVER require review (direct commit to Neo4j)
    # If auto_approve=False, ALWAYS require review so user can see data before commit
    if state.get("auto_approve", False):
        requires_review = False
        logger.info("[VALIDATION] Auto-approve enabled - proceeding directly to storage")
    else:
        requires_review = True  # Always require review when auto_approve=False
        logger.info(f"[VALIDATION] Review required - user will approve before Neo4j commit "
                   f"({len(result.validated_entities)} entities, {len(result.validated_relationships)} relationships)")
    
    return {
        "entities": result.validated_entities,
        "relationships": result.validated_relationships,
        "requires_review": requires_review,
        "validation_issues": result.issues
    }

async def human_review_node(state: GraphState) -> Dict[str, Any]:
    """Phase: Human Review (Persistence Gate)"""
    logger.info("LangGraph: Entering Human Review Node")
    storage = StorageAgent()
    
    entities = state.get("entities", [])
    relationships = state.get("relationships", [])
    
    logger.info(f"[HUMAN REVIEW DEBUG] Preparing to stage {len(entities)} entities and {len(relationships)} relationships")
    
    if not entities:
        logger.warning("[HUMAN REVIEW DEBUG] No entities to store in staging!")
    
    # Update file status to ready_for_review
    await storage.update_file_status(
        state["file_id"], 
        "ready_for_review",
        node_count=len(entities),
        relationship_count=len(relationships),
    )
    
    # Store in staging
    await storage.store_staging(
        file_id=state["file_id"],
        entities=entities,
        relationships=relationships,
    )
    
    logger.info(f"[HUMAN REVIEW DEBUG] Staging data stored for file {state['file_id']}")
    
    if state.get("progress_callback"):
        await _report_progress(state, "human_review", 7, "Ready for human review")
        
    return {"status": "paused"}

async def embedding_node(state: GraphState) -> Dict[str, Any]:
    """Phase 7: Embedding Generation"""
    logger.info("LangGraph: Entering Embedding Phase")
    agent = EmbeddingAgent()
    
    if state.get("progress_callback"):
        await _report_progress(state, "embedding", 7, "Generating embeddings...")
        
    embedded_entities = await agent.embed_entities(state["entities"])
    embedded_chunks = await agent.embed_chunks(state["chunks"])
    
    return {
        "entities": embedded_entities,
        "chunks": embedded_chunks
    }

async def storage_node(state: GraphState) -> Dict[str, Any]:
    """Phase 8: Storage & Finalization"""
    logger.info("LangGraph: Entering Storage Phase")
    storage = StorageAgent()
    graph_service = get_graph_service()
    
    entities = state.get("entities", [])
    relationships = state.get("relationships", [])
    
    logger.info(f"[STORAGE DEBUG] Storing {len(entities)} entities and {len(relationships)} relationships to Neo4j")
    
    if state.get("progress_callback"):
        await _report_progress(state, "storage", 8, "Committing to database...")
    
    if not entities:
        logger.warning("[STORAGE DEBUG] No entities to store! The extraction phase may have failed.")
        # Still update file status to completed with 0 counts
        await storage.update_file_status(
            state["file_id"],
            "completed",
            node_count=0,
            relationship_count=0,
        )
        return {"entity_id_map": {}, "status": "completed"}
        
    # Store entities
    entity_id_map = await storage.store_entities(
        entities, state["file_id"], state["folder_id"], state["user_id"]
    )
    logger.info(f"[STORAGE DEBUG] Stored {len(entity_id_map)} entities to Neo4j")
    
    # Store relationships
    rel_count = await storage.store_relationships(
        relationships, entity_id_map, state["file_id"], state["folder_id"]
    )
    logger.info(f"[STORAGE DEBUG] Stored {rel_count} relationships to Neo4j")
    
    # Store chunks
    chunk_count = await storage.store_chunks(
        state["chunks"], state["file_id"], state["folder_id"]
    )
    
    # Update status
    await storage.update_file_status(
        state["file_id"],
        "completed",
        node_count=len(entity_id_map),
        relationship_count=rel_count,
    )
    
    # Audit log
    await storage.create_audit_log(
        user_id=state["user_id"],
        action="ingest",
        target_type="file",
        target_id=state["file_id"],
        details={
            "entities_created": len(entity_id_map),
            "relationships_created": rel_count,
            "chunks_stored": chunk_count,
        }
    )
    
    # Run FastRP (optional, may fail if GDS not installed)
    try:
        await graph_service.run_fastrp_node_embeddings(state["folder_id"])
    except Exception as e:
        logger.warning(f"FastRP failed (optional): {e}")
    
    if state.get("progress_callback"):
        await _report_progress(state, "completed", 8, "Ingestion complete")
        
    return {"entity_id_map": entity_id_map, "status": "completed"}

# Helper stuff
def _check_review_required(state: GraphState) -> str:
    """Conditional edge logic."""
    if state["requires_review"]:
        return "human_review"
    return "embedding"

async def _report_progress(state: GraphState, phase_val: str, num: int, msg: str):
    """Helper to report progress through callback."""
    # This matches the PipelineProgress structure expected by SSE
    from app.agents.pipeline import PipelineProgress, PipelinePhase
    
    progress = PipelineProgress(
        phase=PipelinePhase(phase_val),
        phase_number=num,
        total_phases=8,
        progress_percent=int((num / 8) * 100),
        message=msg
    )
    
    if asyncio.iscoroutinefunction(state["progress_callback"]):
        await state["progress_callback"](progress)
    else:
        state["progress_callback"](progress)

# Building the Graph
def build_pipeline_graph():
    """Constructs the LangGraph for the ingestion pipeline."""
    workflow = StateGraph(GraphState)
    
    # Add Nodes
    workflow.add_node("layout", layout_node)
    workflow.add_node("chunking", chunking_node)
    workflow.add_node("ontology", ontology_node)
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("deduplication", deduplication_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("embedding", embedding_node)
    workflow.add_node("storage", storage_node)
    
    # Set Entry Point
    workflow.set_entry_point("layout")
    
    # Add Edges
    workflow.add_edge("layout", "chunking")
    workflow.add_edge("chunking", "ontology")
    workflow.add_edge("ontology", "extraction")
    workflow.add_edge("extraction", "deduplication")
    workflow.add_edge("deduplication", "validation")
    
    # Conditional Edge from validation
    workflow.add_conditional_edges(
        "validation",
        _check_review_required,
        {
            "human_review": "human_review",
            "embedding": "embedding"
        }
    )
    
    workflow.add_edge("human_review", END) # Process pauses here
    workflow.add_edge("embedding", "storage")
    workflow.add_edge("storage", END)
    
    return workflow.compile()

# Entry function to match run_pipeline signature
async def run_langgraph_pipeline(
    content: str,
    file_id: str,
    folder_id: str,
    user_id: str,
    file_type: str,
    auto_approve: bool = False,
    progress_callback: Optional[Callable] = None,
):
    """Execution entry point for the LangGraph pipeline."""
    graph = build_pipeline_graph()
    
    initial_state = {
        "content": content,
        "file_id": file_id,
        "folder_id": folder_id,
        "user_id": user_id,
        "file_type": file_type,
        "auto_approve": auto_approve,
        "progress_callback": progress_callback,
        "start_time": datetime.utcnow(),
        "requires_review": False,
        "validation_issues": [],
        "entities": [],
        "relationships": [],
        "chunks": [],
        "entity_id_map": {}, 
        "status": "starting"
    }
    
    try:
        # Run the graph
        final_state = await graph.ainvoke(initial_state)
        
        # Transform final state back to PipelineResult-like object if needed
        # For now, we return the final state for metadata access
        return final_state
        
    except Exception as e:
        logger.error(f"LangGraph Pipeline Error: {e}", exc_info=True)
        # Update storage one final time on failure
        storage = StorageAgent()
        await storage.update_file_status(file_id, "failed", error_message=str(e))
        raise
