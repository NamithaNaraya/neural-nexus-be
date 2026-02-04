"""
Validation Agent - Phase 6 of the Ingestion Pipeline

Cross-verifies extracted facts against source text.
CONSTRAINT: Prompted to prioritize finding errors/hallucinations
over simple confirmation.
"""
import logging
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from app.services.ai_service import get_ollama_service

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found."""
    entity_id: str
    issue_type: str  # 'hallucination', 'missing_evidence', 'inconsistency', 'low_confidence'
    description: str
    severity: str  # 'critical', 'warning', 'info'
    suggested_action: str  # 'remove', 'flag_for_review', 'accept'


@dataclass
class ValidationResult:
    """Result of validation process."""
    is_valid: bool
    confidence: float
    issues: List[ValidationIssue]
    validated_entities: List[Any]
    validated_relationships: List[Any]
    removed_count: int
    flagged_count: int


class ValidationAgent:
    """
    Phase 6: Fact Validation Agent
    
    Responsibilities:
    - Verify entities have valid source evidence
    - Check for hallucinations (facts not in source)
    - Flag inconsistencies
    - Score extraction confidence
    """
    
    SYSTEM_PROMPT = """You are a critical fact-checker for knowledge graph extractions.

Your job is to FIND ERRORS and hallucinations. Be skeptical!

Given an extracted entity/relationship and the source text, determine:
1. Is this extraction supported by the source text?
2. Is the evidence quote accurate?
3. Are there any factual errors or assumptions?

OUTPUT FORMAT (JSON):
{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "issues": [
        {
            "type": "hallucination|missing_evidence|inconsistency",
            "description": "explanation",
            "severity": "critical|warning|info"
        }
    ],
    "corrected_entity": null or { corrected fields }
}

PRIORITY: Find errors over confirming validity. Be strict!"""
    
    CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(self, model_name: str = None):
        self.ollama = get_ollama_service()
        self.model_name = model_name
        
    async def validate(
        self,
        entities: List[Any],
        relationships: List[Any],
        chunks: List[Any],
    ) -> ValidationResult:
        """
        Validate extracted entities and relationships.
        
        Args:
            entities: Extracted entities
            relationships: Extracted relationships  
            chunks: Source text chunks (for evidence verification)
            
        Returns:
            ValidationResult with validated data and issues
        """
        logger.info(f"Validating {len(entities)} entities, {len(relationships)} relationships")
        
        # Build chunk lookup
        chunk_map = {}
        for chunk in chunks:
            chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else chunk.get('chunk_id')
            content = chunk.content if hasattr(chunk, 'content') else chunk.get('content', '')
            chunk_map[chunk_id] = content
        
        all_issues = []
        validated_entities = []
        validated_relationships = []
        removed_count = 0
        flagged_count = 0
        
        # Validate entities
        for entity in entities:
            result = await self._validate_entity(entity, chunk_map)
            
            if result["action"] == "remove":
                removed_count += 1
                all_issues.extend(result["issues"])
            elif result["action"] == "flag":
                flagged_count += 1
                
                # Update confidence
                if hasattr(entity, 'confidence'):
                    entity.confidence = result.get("confidence", 0.5)
                elif isinstance(entity, dict):
                    entity['confidence'] = result.get("confidence", 0.5)

                # Update properties
                if hasattr(entity, 'properties'):
                    if entity.properties is None:
                        entity.properties = {}
                    entity.properties["needs_review"] = True
                elif isinstance(entity, dict):
                    if entity.get("properties") is None:
                        entity["properties"] = {}
                    entity["properties"]["needs_review"] = True
                    
                validated_entities.append(entity)
                all_issues.extend(result["issues"])
            else:
                if hasattr(entity, 'confidence'):
                    entity.confidence = result.get("confidence", 0.9)
                elif isinstance(entity, dict):
                    entity['confidence'] = result.get("confidence", 0.9)
                validated_entities.append(entity)
        
        # Validate relationships
        valid_entity_ids = {
            e.id if hasattr(e, 'id') else e.get('id')
            for e in validated_entities
        }
        
        for rel in relationships:
            source_id = rel.source_entity_id if hasattr(rel, 'source_entity_id') else rel.get('source_entity_id')
            target_id = rel.target_entity_id if hasattr(rel, 'target_entity_id') else rel.get('target_entity_id')
            
            # Skip if either entity was removed
            if source_id not in valid_entity_ids or target_id not in valid_entity_ids:
                removed_count += 1
                continue
            
            result = await self._validate_relationship(rel, chunk_map)
            
            if result["action"] == "remove":
                removed_count += 1
                all_issues.extend(result["issues"])
            else:
                validated_relationships.append(rel)
        
        overall_valid = len(all_issues) == 0 or all(
            issue.severity != "critical" for issue in all_issues
        )
        
        avg_confidence = 0.0
        if validated_entities:
            avg_confidence = sum(
                e.confidence if hasattr(e, 'confidence') else e.get('confidence', 0.9)
                for e in validated_entities
            ) / len(validated_entities)
        
        logger.info(f"Validation complete: {len(validated_entities)} entities, "
                   f"{len(validated_relationships)} relationships "
                   f"({removed_count} removed, {flagged_count} flagged)")
        
        return ValidationResult(
            is_valid=overall_valid,
            confidence=avg_confidence,
            issues=all_issues,
            validated_entities=validated_entities,
            validated_relationships=validated_relationships,
            removed_count=removed_count,
            flagged_count=flagged_count,
        )
    
    async def _validate_entity(
        self,
        entity: Any,
        chunk_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """Validate a single entity."""
        name = entity.name if hasattr(entity, 'name') else entity.get('name', '')
        entity_type = entity.type if hasattr(entity, 'type') else entity.get('type', '')
        description = entity.description if hasattr(entity, 'description') else entity.get('description', '')
        evidence = entity.source_text if hasattr(entity, 'source_text') else entity.get('source_text', '')
        chunk_id = entity.source_chunk_id if hasattr(entity, 'source_chunk_id') else entity.get('source_chunk_id')
        entity_id = entity.id if hasattr(entity, 'id') else entity.get('id')
        
        # Get source text
        source_text = chunk_map.get(chunk_id, "")
        
        # Quick rule-based checks
        issues = []
        
        # Check if name is empty
        if not name.strip():
            issues.append(ValidationIssue(
                entity_id=entity_id,
                issue_type="missing_evidence",
                description="Entity has no name",
                severity="critical",
                suggested_action="remove",
            ))
            return {"action": "remove", "issues": issues}
        
        # Check if evidence exists in source
        if evidence and source_text:
            evidence_lower = evidence.lower()[:50]
            if evidence_lower not in source_text.lower():
                issues.append(ValidationIssue(
                    entity_id=entity_id,
                    issue_type="missing_evidence",
                    description=f"Evidence quote not found in source text",
                    severity="warning",
                    suggested_action="flag_for_review",
                ))
        
        # Check entity name is in source
        if source_text and name.lower() not in source_text.lower():
            issues.append(ValidationIssue(
                entity_id=entity_id,
                issue_type="hallucination",
                description=f"Entity name '{name}' not found in source chunk",
                severity="warning",
                suggested_action="flag_for_review",
            ))
        
        # Determine action
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            return {"action": "remove", "issues": issues, "confidence": 0.0}
        
        warning_issues = [i for i in issues if i.severity == "warning"]
        if warning_issues:
            return {"action": "flag", "issues": issues, "confidence": 0.6}
        
        return {"action": "accept", "issues": [], "confidence": 0.95}
    
    async def _validate_relationship(
        self,
        rel: Any,
        chunk_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """Validate a single relationship."""
        rel_type = rel.relationship_type if hasattr(rel, 'relationship_type') else rel.get('relationship_type', '')
        evidence = rel.source_text if hasattr(rel, 'source_text') else rel.get('source_text', '')
        chunk_id = rel.source_chunk_id if hasattr(rel, 'source_chunk_id') else rel.get('source_chunk_id')
        
        # Get source text
        source_text = chunk_map.get(chunk_id, "")
        
        issues = []
        
        # Check if relationship type is valid
        if not rel_type or rel_type.strip() == "":
            issues.append(ValidationIssue(
                entity_id="relationship",
                issue_type="missing_evidence",
                description="Relationship has no type",
                severity="critical",
                suggested_action="remove",
            ))
            return {"action": "remove", "issues": issues}
        
        # Check evidence exists in source
        if evidence and source_text:
            evidence_lower = evidence.lower()[:50]
            if evidence_lower not in source_text.lower():
                issues.append(ValidationIssue(
                    entity_id="relationship",
                    issue_type="missing_evidence",
                    description="Relationship evidence not found in source",
                    severity="warning",
                    suggested_action="flag_for_review",
                ))
        
        if any(i.severity == "critical" for i in issues):
            return {"action": "remove", "issues": issues}
        
        return {"action": "accept", "issues": issues, "confidence": 0.9}
