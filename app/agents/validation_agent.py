"""
Validation Agent - Phase 6 of the Ingestion Pipeline

Cross-verifies extracted facts against source text.
CONSTRAINT: Prompted to prioritize finding errors/hallucinations 
over simple confirmation.
"""
from typing import Any, Dict, List
from dataclasses import dataclass
from enum import Enum


class ValidationStatus(Enum):
    VERIFIED = "verified"
    SUSPICIOUS = "suspicious"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


@dataclass
class ValidationResult:
    """Result of validating an entity or relationship."""
    item_id: str
    item_type: str  # 'entity' or 'relationship'
    status: ValidationStatus
    confidence: float
    issues: List[str]
    evidence: str = None


class ValidationAgent:
    """
    Phase 6: Fact Validation Agent
    
    Responsibilities:
    - Cross-verify extracted entities against source text
    - Validate relationship accuracy
    - Flag suspicious or potentially hallucinated content
    - Prioritize finding errors over simple confirmation
    """
    
    # Validation-focused system prompt
    SYSTEM_PROMPT = """You are a critical fact-checking agent.

YOUR PRIMARY GOAL IS TO FIND ERRORS, not to confirm extractions.

For each extraction, you must:
1. Search for the EXACT evidence in the source text
2. Identify ANY inconsistency between extraction and source
3. Flag ANYTHING that seems inferred rather than stated
4. Be SKEPTICAL - assume extractions may be wrong until proven correct

Rate each extraction as:
- VERIFIED: Exact match with source text evidence
- SUSPICIOUS: Partially supported but has inconsistencies  
- REJECTED: Not supported by source text (hallucination)
- NEEDS_REVIEW: Ambiguous, requires human judgment

BE STRICT. It's better to flag a valid extraction for review than to approve a hallucination."""
    
    def __init__(self, model_name: str = "gemma2:latest"):
        self.model_name = model_name
        
    async def validate(self,
                      entities: List[Any],
                      relationships: List[Any],
                      chunks: List[Any]) -> Dict[str, Any]:
        """
        Validate all extracted entities and relationships.
        
        Args:
            entities: Deduplicated entities from Phase 5
            relationships: Updated relationships from Phase 5
            chunks: Original text chunks for evidence checking
            
        Returns:
            Dictionary with validation results and filtered data
        """
        entity_results = []
        relationship_results = []
        
        # Build chunk lookup for evidence checking
        chunk_map = {c.chunk_id: c for c in chunks if hasattr(c, 'chunk_id')}
        
        # Validate entities
        for entity in entities:
            result = await self._validate_entity(entity, chunk_map)
            entity_results.append(result)
        
        # Validate relationships
        for rel in relationships:
            result = await self._validate_relationship(rel, chunk_map)
            relationship_results.append(result)
        
        # Filter out rejected items
        verified_entities = [
            e for e, r in zip(entities, entity_results)
            if r.status != ValidationStatus.REJECTED
        ]
        verified_relationships = [
            r for r, v in zip(relationships, relationship_results)
            if v.status != ValidationStatus.REJECTED
        ]
        
        return {
            "entities": verified_entities,
            "relationships": verified_relationships,
            "entity_validations": entity_results,
            "relationship_validations": relationship_results,
            "rejection_count": sum(
                1 for r in entity_results + relationship_results
                if r.status == ValidationStatus.REJECTED
            ),
            "review_count": sum(
                1 for r in entity_results + relationship_results
                if r.status == ValidationStatus.NEEDS_REVIEW
            ),
        }
    
    async def _validate_entity(self, 
                              entity: Any, 
                              chunk_map: Dict) -> ValidationResult:
        """Validate a single entity against source chunks."""
        # TODO: Implement AI-based validation
        # - Check if entity name appears in source
        # - Verify entity type is appropriate
        # - Confirm description matches source context
        return ValidationResult(
            item_id=getattr(entity, 'id', 'unknown'),
            item_type='entity',
            status=ValidationStatus.NEEDS_REVIEW,
            confidence=0.0,
            issues=[],
        )
    
    async def _validate_relationship(self,
                                    relationship: Any,
                                    chunk_map: Dict) -> ValidationResult:
        """Validate a single relationship against source chunks."""
        # TODO: Implement AI-based validation
        # - Check if both entities are mentioned together
        # - Verify relationship type is supported by text
        # - Confirm directionality is correct
        return ValidationResult(
            item_id=f"{getattr(relationship, 'source_entity_id', '?')}->{getattr(relationship, 'target_entity_id', '?')}",
            item_type='relationship',
            status=ValidationStatus.NEEDS_REVIEW,
            confidence=0.0,
            issues=[],
        )
    
    async def _find_evidence(self, 
                            text: str, 
                            chunks: List[Any]) -> str:
        """Find the source text evidence for a claim."""
        # TODO: Search chunks for matching text
        return ""
