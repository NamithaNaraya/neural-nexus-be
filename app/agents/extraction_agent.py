"""
Extraction Agent - Phase 4 of the Ingestion Pipeline

Identifies and extracts entities/relationships from text chunks.
CONSTRAINT: System prompts strictly forbid hallucinations;
every claim must have ground-truth text evidence.
"""
import logging
import asyncio
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from app.services.ai_service import get_ollama_service

logger = logging.getLogger(__name__)


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
    
    SYSTEM_PROMPT = """You are a precise knowledge extraction agent for building knowledge graphs.

STRICT RULES:
1. ONLY extract entities and relationships that are EXPLICITLY mentioned in the text.
2. NEVER infer or assume information not directly stated.
3. For EVERY entity and relationship, provide the exact quote from the text as evidence.
4. If you're uncertain about a fact, DO NOT include it.
5. Use the provided entity types and relationship types from the schema.

OUTPUT FORMAT (JSON):
{
    "entities": [
        {
            "name": "John Smith",
            "type": "Person",
            "description": "A software engineer at TechCorp",
            "properties": {"occupation": "software engineer"},
            "evidence": "John Smith, a software engineer at TechCorp, presented..."
        }
    ],
    "relationships": [
        {
            "source": "John Smith",
            "target": "TechCorp",
            "type": "WORKS_AT",
            "description": "Employment relationship",
            "evidence": "John Smith, a software engineer at TechCorp"
        }
    ]
}

Hallucinations are strictly forbidden. Only extract what you can directly quote from the text."""
    
    def __init__(self, model_name: str = None):
        self.ollama = get_ollama_service()
        self.model_name = model_name
        
    async def extract(
        self,
        chunks: List[Any],
        schema: Dict[str, Any],
    ) -> Dict[str, List]:
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
        entity_name_to_id = {}  # Track entity names to IDs for relationship linking
        
        entity_types = [et["name"] for et in schema.get("entity_types", [])]
        rel_types = [rt["name"] for rt in schema.get("relationship_types", [])]
        
        logger.info(f"Extracting from {len(chunks)} chunks with schema: "
                   f"{len(entity_types)} entity types, {len(rel_types)} relationship types")
        
        # Limit concurrency to avoid hitting API rate limits
        sem = asyncio.Semaphore(5)

        async def _process_chunk_safe(i, chunk):
            async with sem:
                chunk_content = chunk.content if hasattr(chunk, 'content') else chunk.get('content', '')
                chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else chunk.get('chunk_id', str(i))
                
                if not chunk_content.strip():
                    return None
                
                try:
                    return await self._extract_from_chunk(
                        chunk_content=chunk_content,
                        chunk_id=chunk_id,
                        entity_types=entity_types,
                        rel_types=rel_types,
                    )
                except Exception as e:
                    logger.warning(f"Extraction failed for chunk {i}: {e}")
                    return None

        # Execute extractions in parallel
        tasks = [_process_chunk_safe(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            if not result:
                continue
            
            chunk_id = chunks[i].chunk_id if hasattr(chunks[i], 'chunk_id') else chunks[i].get('chunk_id', str(i))

            # Process entities
            for entity_data in result.get("entities", []):
                entity = ExtractedEntity(
                    id=str(uuid.uuid4()),
                    name=entity_data["name"],
                    type=entity_data.get("type", "Concept"),
                    description=entity_data.get("description", ""),
                    properties=entity_data.get("properties", {}),
                    source_chunk_id=chunk_id,
                    source_text=entity_data.get("evidence", ""),
                    confidence=0.9,
                )
                all_entities.append(entity)
                entity_name_to_id[entity.name.lower()] = entity.id
            
            # Process relationships (will link after all entities are extracted)
            for rel_data in result.get("relationships", []):
                all_relationships.append({
                    **rel_data,
                    "source_chunk_id": chunk_id,
                })
        
        # Link relationships to entity IDs
        linked_relationships = []
        for rel in all_relationships:
            source_name = rel.get("source", "").lower()
            target_name = rel.get("target", "").lower()
            
            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)
            
            if source_id and target_id:
                linked_relationships.append(ExtractedRelationship(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=rel.get("type", "RELATED_TO"),
                    description=rel.get("description", ""),
                    source_chunk_id=rel.get("source_chunk_id"),
                    source_text=rel.get("evidence", ""),
                    confidence=0.85,
                ))
            else:
                logger.debug(f"Could not link relationship: {source_name} -> {target_name}")
        
        logger.info(f"Extracted {len(all_entities)} entities, {len(linked_relationships)} relationships")
        
        return {
            "entities": all_entities,
            "relationships": linked_relationships,
        }
    
    async def _extract_from_chunk(
        self,
        chunk_content: str,
        chunk_id: str,
        entity_types: List[str],
        rel_types: List[str],
    ) -> Dict[str, List]:
        """Extract entities and relationships from a single chunk."""
        
        # Limit chunk size to prevent timeout
        content_to_process = chunk_content[:3000] if len(chunk_content) > 3000 else chunk_content
        
        prompt = f"""Extract entities and relationships from this text.

ALLOWED ENTITY TYPES: {', '.join(entity_types) if entity_types else 'Person, Place, Organization, Event, Concept, Object'}
ALLOWED RELATIONSHIP TYPES: {', '.join(rel_types) if rel_types else 'RELATED_TO, PART_OF, LOCATED_IN, WORKS_FOR, KNOWS'}

TEXT:
{content_to_process}

Remember: Only extract what is EXPLICITLY stated. Include exact quotes as evidence.
Respond with valid JSON containing "entities" and "relationships" arrays."""
        
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            logger.debug(f"Sending chunk {chunk_id} ({len(content_to_process)} chars) to AI for extraction")
            
            result = await self.ollama.chat_json(messages)
            
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])
            
            logger.debug(f"Chunk {chunk_id}: extracted {len(entities)} entities, {len(relationships)} relationships")
            
            return {
                "entities": entities,
                "relationships": relationships,
            }
            
        except Exception as e:
            logger.error(f"Chunk {chunk_id} extraction failed: {type(e).__name__}: {e}")
            # Return empty instead of raising to allow other chunks to process
            return {"entities": [], "relationships": []}
    
    def _validate_evidence(
        self,
        extraction: Any,
        source_text: str,
    ) -> bool:
        """Validate that extraction has valid ground-truth evidence."""
        evidence = extraction.source_text if hasattr(extraction, 'source_text') else ""
        
        if not evidence:
            return False
        
        # Check if evidence appears in the source
        evidence_lower = evidence.lower()[:50]  # First 50 chars
        source_lower = source_text.lower()
        
        return evidence_lower in source_lower
