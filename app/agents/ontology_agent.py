"""
Ontology Agent - Phase 3 of the Ingestion Pipeline

Defines graph schema and entity types for extraction.
RULE: Must check existing folder schema first to reuse labels
(prevent "WORKS_AT" vs "EMPLOYED_BY" conflicts).
"""
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from app.services.ai_service import get_ollama_service
from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)


@dataclass
class EntityType:
    """Represents an entity type in the schema."""
    name: str  # e.g., "Person", "Organization"
    description: str
    properties: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class RelationType:
    """Represents a relationship type in the schema."""
    name: str  # e.g., "WORKS_AT", "LIVES_IN"
    description: str
    source_types: List[str] = field(default_factory=list)
    target_types: List[str] = field(default_factory=list)


class OntologyAgent:
    """
    Phase 3: Ontology Definition Agent
    
    Responsibilities:
    - Define entity types for the document
    - Define relationship types
    - Reuse existing schema from folder when possible
    - Prevent semantic duplicates (WORKS_AT vs EMPLOYED_BY)
    """
    
    SYSTEM_PROMPT = """You are an ontology designer for knowledge graphs.

Given a document sample, identify the types of entities and relationships that should be extracted.

OUTPUT FORMAT (JSON):
{
    "entity_types": [
        {
            "name": "Person",
            "description": "A human individual mentioned in the text",
            "properties": ["age", "occupation", "nationality"],
            "examples": ["John Smith", "Dr. Jane Doe"]
        }
    ],
    "relationship_types": [
        {
            "name": "WORKS_AT",
            "description": "Employment relationship",
            "source_types": ["Person"],
            "target_types": ["Organization"]
        }
    ]
}

RULES:
1. Use UPPERCASE_SNAKE_CASE for relationship names
2. Use PascalCase for entity type names
3. Keep types generic enough to be reusable
4. Include common properties for each entity type
5. If given existing schema, REUSE those types instead of creating new ones"""
    
    def __init__(self, model_name: str = None):
        self.ollama = get_ollama_service()
        self.model_name = model_name
        
    async def define_schema(
        self,
        chunks: List[Any],
        folder_id: str,
        existing_schema: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Define ontology schema for extraction.
        
        Args:
            chunks: Text chunks from ChunkingAgent
            folder_id: Folder ID to check for existing schema
            existing_schema: Optional cached schema from folder
            
        Returns:
            Schema definition with entity and relationship types
        """
        logger.info(f"Defining ontology for folder {folder_id}")
        
        # Get existing schema from folder if not provided
        if existing_schema is None:
            existing_schema = await self._get_folder_schema(folder_id)
        
        # Sample text for analysis
        sample_text = self._get_sample_text(chunks)
        
        if existing_schema and existing_schema.get("entity_types"):
            # Merge with existing schema
            new_schema = await self._extend_schema(sample_text, existing_schema)
        else:
            # Create new schema
            new_schema = await self._create_schema(sample_text)
        
        logger.info(f"Schema defined with {len(new_schema['entity_types'])} entity types, "
                   f"{len(new_schema['relationship_types'])} relationship types")
        
        return new_schema
    
    async def _get_folder_schema(self, folder_id: str) -> Dict[str, Any]:
        """Get existing schema from folder's entities."""
        try:
            driver = get_neo4j_driver()
            
            async with driver.session() as session:
                # Get distinct entity types
                entity_result = await session.run("""
                    MATCH (e:Entity {folder_id: $folder_id})
                    RETURN DISTINCT e.type as type, count(e) as count
                    ORDER BY count DESC
                    LIMIT 20
                """, folder_id=folder_id)
                entity_records = await entity_result.data()
                
                # Get distinct relationship types
                rel_result = await session.run("""
                    MATCH (e1:Entity {folder_id: $folder_id})-[r:RELATIONSHIP]->(e2:Entity)
                    RETURN DISTINCT r.type as type, count(r) as count
                    ORDER BY count DESC
                    LIMIT 20
                """, folder_id=folder_id)
                rel_records = await rel_result.data()
                
                if not entity_records and not rel_records:
                    return {}
                
                return {
                    "entity_types": [
                        EntityType(
                            name=r["type"],
                            description=f"Existing entity type with {r['count']} instances",
                            properties=[],
                            examples=[],
                        ).__dict__
                        for r in entity_records if r["type"]
                    ],
                    "relationship_types": [
                        RelationType(
                            name=r["type"],
                            description=f"Existing relationship type with {r['count']} instances",
                            source_types=[],
                            target_types=[],
                        ).__dict__
                        for r in rel_records if r["type"]
                    ],
                }
                
        except Exception as e:
            logger.warning(f"Could not fetch folder schema: {e}")
            return {}
    
    def _get_sample_text(self, chunks: List[Any], max_chars: int = 3000) -> str:
        """Get sample text from chunks for analysis."""
        sample_parts = []
        total_chars = 0
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else chunk.get('content', '')
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                sample_parts.append(content[:remaining])
                break
            sample_parts.append(content)
            total_chars += len(content)
        
        return '\n\n'.join(sample_parts)
    
    async def _create_schema(self, sample_text: str) -> Dict[str, Any]:
        """Create new schema from scratch using AI."""
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text and define the ontology:\n\n{sample_text}"}
            ]
            
            result = await self.ollama.chat_json(messages, self.model_name)
            
            return {
                "entity_types": result.get("entity_types", []),
                "relationship_types": result.get("relationship_types", []),
            }
            
        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            # Return default schema
            return self._get_default_schema()
    
    async def _extend_schema(
        self,
        sample_text: str,
        existing_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extend existing schema with new types if needed."""
        existing_types = [t["name"] for t in existing_schema.get("entity_types", [])]
        existing_rels = [t["name"] for t in existing_schema.get("relationship_types", [])]
        
        prompt = f"""Given the existing schema:
Entity Types: {', '.join(existing_types)}
Relationship Types: {', '.join(existing_rels)}

Analyze the following text. If new entity or relationship types are needed, add them.
Otherwise, reuse the existing types.

Text:
{sample_text}"""
        
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            result = await self.ollama.chat_json(messages, self.model_name)
            
            # Merge with existing
            merged_entities = existing_schema.get("entity_types", [])
            merged_rels = existing_schema.get("relationship_types", [])
            
            # Add new types (avoid duplicates)
            for et in result.get("entity_types", []):
                if et["name"] not in existing_types:
                    merged_entities.append(et)
            
            for rt in result.get("relationship_types", []):
                if rt["name"] not in existing_rels:
                    merged_rels.append(rt)
            
            return {
                "entity_types": merged_entities,
                "relationship_types": merged_rels,
            }
            
        except Exception as e:
            logger.error(f"Schema extension failed: {e}")
            return existing_schema
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Return a sensible default schema."""
        return {
            "entity_types": [
                {
                    "name": "Person",
                    "description": "A human individual",
                    "properties": ["occupation", "age", "location"],
                    "examples": [],
                },
                {
                    "name": "Organization",
                    "description": "A company, institution, or group",
                    "properties": ["industry", "location", "size"],
                    "examples": [],
                },
                {
                    "name": "Location",
                    "description": "A geographical place",
                    "properties": ["type", "country"],
                    "examples": [],
                },
                {
                    "name": "Concept",
                    "description": "An abstract idea or topic",
                    "properties": ["category"],
                    "examples": [],
                },
            ],
            "relationship_types": [
                {
                    "name": "WORKS_AT",
                    "description": "Employment relationship",
                    "source_types": ["Person"],
                    "target_types": ["Organization"],
                },
                {
                    "name": "LOCATED_IN",
                    "description": "Location relationship",
                    "source_types": ["Person", "Organization"],
                    "target_types": ["Location"],
                },
                {
                    "name": "RELATED_TO",
                    "description": "General relationship",
                    "source_types": [],
                    "target_types": [],
                },
            ],
        }
