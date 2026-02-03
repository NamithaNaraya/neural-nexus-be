"""
Extraction Agent - Phase 4 of the Ingestion Pipeline

Identifies and extracts entities/relationships from text chunks.
CONSTRAINT: System prompts strictly forbid hallucinations; 
every claim must have ground-truth text evidence.
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ExtractedEntity:
    """Represents an entity extracted from text."""
    id: str
    name: str
    type: str  # Person, Place, Organization, etc.
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunk_id: str = None
    source_text: str = None  # Ground-truth evidence
    confidence: float = 1.0


@dataclass
class ExtractedRelationship:
    """Represents a relationship extracted from text."""
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    description: str = None
    strength: float = 1.0
    source_chunk_id: str = None
    source_text: str = None  # Ground-truth evidence
    confidence: float = 1.0


class ExtractionAgent:
    """
    Phase 4: Entity & Relationship Extraction Agent
    
    Responsibilities:
    - Extract entities from text chunks
    - Identify relationships between entities
    - Provide ground-truth evidence for all extractions
    - Avoid hallucinations - only extract what's in the text
    """
    
    # Anti-hallucination system prompt
    SYSTEM_PROMPT = """You are a precise knowledge extraction agent.

STRICT RULES:
1. ONLY extract entities and relationships that are EXPLICITLY mentioned in the text.
2. NEVER infer or assume information not directly stated.
3. For EVERY entity and relationship, provide the exact quote from the text as evidence.
4. If you're uncertain about a fact, DO NOT include it.
5. Prefer specific, verifiable information over vague mentions.

Your output must be grounded in the source text. Hallucinations are strictly forbidden."""
    
    def __init__(self, model_name: str = "gemma2:latest"):
        self.model_name = model_name
        
    async def extract(self, 
                     chunks: List[Any],
                     schema: Dict[str, Any]) -> Dict[str, List]:
        """
        Extract entities and relationships from text chunks.
        
        Args:
            chunks: Text chunks from ChunkingAgent
            schema: Ontology schema from OntologyAgent
            
        Returns:
            Dictionary with 'entities' and 'relationships' lists
        """
        all_entities = []
        all_relationships = []
        
        for chunk in chunks:
            result = await self._extract_from_chunk(chunk, schema)
            all_entities.extend(result.get("entities", []))
            all_relationships.extend(result.get("relationships", []))
        
        return {
            "entities": all_entities,
            "relationships": all_relationships,
        }
    
    async def _extract_from_chunk(self, 
                                  chunk: Any, 
                                  schema: Dict[str, Any]) -> Dict[str, List]:
        """Extract entities and relationships from a single chunk."""
        # TODO: Implement AI-based extraction with anti-hallucination prompt
        return {
            "entities": [],
            "relationships": [],
        }
    
    async def _validate_evidence(self, 
                                 extraction: Any, 
                                 source_text: str) -> bool:
        """Validate that extraction has valid ground-truth evidence."""
        # TODO: Verify the source_text quote exists in the chunk
        return True
    
    async def _assign_confidence(self, extraction: Any) -> float:
        """Assign confidence score based on extraction quality."""
        # TODO: Implement confidence scoring
        return 1.0
