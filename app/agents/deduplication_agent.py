"""
Deduplication Agent - Phase 5 of the Ingestion Pipeline

Performs "Neural Reconciliation" to merge duplicate entities.
RULE: If descriptions differ significantly (e.g., "Prince" vs "Son"), 
keep as separate nodes. Users can manually merge them from the UI later.
"""
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class MergeCandidate:
    """Represents a potential merge of two entities."""
    entity_a_id: str
    entity_b_id: str
    similarity_score: float
    reason: str
    should_merge: bool


class DeduplicationAgent:
    """
    Phase 5: Neural Reconciliation Agent
    
    Responsibilities:
    - Identify duplicate entities across chunks
    - Merge entities with matching types and similar descriptions
    - Preserve separate nodes when descriptions differ significantly
    - Track entity aliases for merged entities
    """
    
    # Threshold for automatic merging (0.0 to 1.0)
    MERGE_THRESHOLD = 0.85
    
    def __init__(self, model_name: str = "gemma2:latest"):
        self.model_name = model_name
        
    async def deduplicate(self, 
                         entities: List[Any],
                         relationships: List[Any]) -> Dict[str, Any]:
        """
        Deduplicate entities and update relationships accordingly.
        
        Args:
            entities: List of extracted entities
            relationships: List of extracted relationships
            
        Returns:
            Dictionary with deduplicated entities, relationships, and merge log
        """
        # Find merge candidates
        candidates = await self._find_merge_candidates(entities)
        
        # Filter to only high-confidence merges
        approved_merges = [c for c in candidates if c.should_merge]
        
        # Perform merges
        merged_entities, id_mapping = await self._perform_merges(
            entities, approved_merges
        )
        
        # Update relationship references
        updated_relationships = await self._update_relationships(
            relationships, id_mapping
        )
        
        return {
            "entities": merged_entities,
            "relationships": updated_relationships,
            "merge_log": approved_merges,
            "id_mapping": id_mapping,
        }
    
    async def _find_merge_candidates(self, 
                                     entities: List[Any]) -> List[MergeCandidate]:
        """Find potential duplicate entities."""
        candidates = []
        
        for i, entity_a in enumerate(entities):
            for entity_b in entities[i+1:]:
                similarity = await self._calculate_similarity(entity_a, entity_b)
                
                if similarity > 0.5:  # Only consider potential matches
                    should_merge = await self._should_merge(
                        entity_a, entity_b, similarity
                    )
                    
                    candidates.append(MergeCandidate(
                        entity_a_id=entity_a.id,
                        entity_b_id=entity_b.id,
                        similarity_score=similarity,
                        reason=f"Name similarity: {similarity:.2f}",
                        should_merge=should_merge,
                    ))
        
        return candidates
    
    async def _calculate_similarity(self, entity_a: Any, entity_b: Any) -> float:
        """Calculate similarity score between two entities."""
        # TODO: Implement semantic similarity using embeddings
        # Consider: name, type, description
        return 0.0
    
    async def _should_merge(self, 
                           entity_a: Any, 
                           entity_b: Any, 
                           similarity: float) -> bool:
        """
        Determine if entities should be merged.
        
        Rule: If descriptions differ significantly, keep separate.
        Example: Same name "Rama" but one says "Prince" and other says "King"
        might indicate different contexts or time periods.
        """
        if similarity < self.MERGE_THRESHOLD:
            return False
        
        # Check if types match
        if hasattr(entity_a, 'type') and hasattr(entity_b, 'type'):
            if entity_a.type != entity_b.type:
                return False
        
        # TODO: Use AI to check if descriptions are compatible
        return True
    
    async def _perform_merges(self, 
                             entities: List[Any],
                             merges: List[MergeCandidate]) -> Tuple[List[Any], Dict]:
        """Perform the actual merges and return updated entity list."""
        # TODO: Implement merge logic
        # - Combine properties from both entities
        # - Keep the more complete description
        # - Track aliases
        id_mapping = {}
        return entities, id_mapping
    
    async def _update_relationships(self,
                                   relationships: List[Any],
                                   id_mapping: Dict) -> List[Any]:
        """Update relationship references after merges."""
        # TODO: Update source/target IDs based on mapping
        return relationships
