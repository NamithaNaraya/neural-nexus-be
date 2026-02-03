"""
Ontology Agent - Phase 3 of the Ingestion Pipeline

Defines graph schema and ensures consistency with existing folder schemas.
RULE: Must check existing folder schema first to reuse labels 
(prevent "WORKS_AT" vs "EMPLOYED_BY" conflicts).
"""
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class EntityType:
    """Represents an entity type in the ontology."""
    name: str  # Person, Place, Organization, Concept, Event, Document
    description: str
    properties: List[str]
    examples: List[str] = None


@dataclass
class RelationshipType:
    """Represents a relationship type in the ontology."""
    name: str  # WORKS_FOR, LIVES_IN, RELATED_TO, etc.
    source_types: List[str]  # Allowed source entity types
    target_types: List[str]  # Allowed target entity types
    description: str
    is_directional: bool = True


class OntologyAgent:
    """
    Phase 3: Ontology Definition Agent
    
    Responsibilities:
    - Define graph schema for new documents
    - Check existing folder schema to reuse labels
    - Prevent label conflicts (WORKS_AT vs EMPLOYED_BY)
    - Suggest entity and relationship types based on content
    """
    
    def __init__(self, model_name: str = "gemma2:latest"):
        self.model_name = model_name
        self._default_entity_types = [
            "Person", "Place", "Organization", "Concept", "Event", "Document"
        ]
        
    async def define_schema(self, 
                           chunks: List[Any],
                           folder_id: str,
                           existing_schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Define the ontology schema for document extraction.
        
        Args:
            chunks: Text chunks from ChunkingAgent
            folder_id: ID of the folder (for schema consistency)
            existing_schema: Existing schema in the folder to maintain consistency
            
        Returns:
            Dictionary containing entity types and relationship types
        """
        # First, load existing schema from folder
        if existing_schema is None:
            existing_schema = await self._load_folder_schema(folder_id)
        
        # Analyze chunks to suggest new types
        suggested_types = await self._suggest_types(chunks)
        
        # Merge with existing schema (prefer existing labels)
        merged_schema = await self._merge_schemas(existing_schema, suggested_types)
        
        return merged_schema
    
    async def _load_folder_schema(self, folder_id: str) -> Dict[str, Any]:
        """Load existing ontology schema from folder."""
        # TODO: Implement schema loading from Neo4j
        return {
            "entity_types": [],
            "relationship_types": [],
        }
    
    async def _suggest_types(self, chunks: List[Any]) -> Dict[str, Any]:
        """Use AI to suggest entity and relationship types from content."""
        # TODO: Implement AI-based type suggestion
        return {
            "entity_types": self._default_entity_types,
            "relationship_types": [],
        }
    
    async def _merge_schemas(self, 
                            existing: Dict[str, Any], 
                            suggested: Dict[str, Any]) -> Dict[str, Any]:
        """Merge schemas, preferring existing labels to prevent conflicts."""
        # TODO: Implement smart schema merging
        # Rule: If "EMPLOYED_BY" exists, don't create "WORKS_AT"
        return {
            "entity_types": list(set(existing.get("entity_types", []) + 
                                    suggested.get("entity_types", []))),
            "relationship_types": existing.get("relationship_types", []) +
                                 suggested.get("relationship_types", []),
        }
    
    async def normalize_relationship(self, rel_type: str, existing_types: Set[str]) -> str:
        """Normalize relationship type to existing schema if similar."""
        # TODO: Use AI to map similar relationship types
        # e.g., "WORKS_AT" -> "EMPLOYED_BY" if that exists
        return rel_type
