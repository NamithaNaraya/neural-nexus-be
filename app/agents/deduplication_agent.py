"""
Deduplication Agent - Phase 5 of the Ingestion Pipeline

Performs "Neural Reconciliation" to merge duplicate entities.
RULE: If descriptions differ significantly (e.g., "Prince" vs "Son"),
keep as separate nodes. Users can manually merge them from the UI later.
"""
import logging
from typing import Any, Dict, List, Set, Tuple
from dataclasses import dataclass

from app.services.ai_service import get_ollama_service
from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    merged_entities: List[Any]  # Entities after merging
    merge_map: Dict[str, str]  # Maps old IDs to canonical IDs
    new_entities: List[Any]  # Brand new entities (not merged)
    existing_matches: List[Tuple[str, str]]  # (new_id, existing_id) pairs


class DeduplicationAgent:
    """
    Phase 5: Neural Reconciliation Agent
    
    Responsibilities:
    - Find potential duplicate entities
    - Compare entity descriptions and properties
    - Merge duplicates while preserving source references
    - Handle edge cases (different roles, same name)
    """
    
    SYSTEM_PROMPT = """You are a deduplication specialist for knowledge graphs.

Given two entity descriptions, determine if they refer to the SAME real-world entity.

RULES:
1. Same name + same type = likely same entity (but verify roles)
2. Different descriptions might indicate different roles/contexts
3. "John Smith (CEO)" vs "John Smith (Engineer)" = DIFFERENT entities
4. "NYC" vs "New York City" = SAME entity

Output JSON:
{
    "is_same_entity": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}"""
    
    SIMILARITY_THRESHOLD = 0.85
    
    def __init__(self, model_name: str = None):
        self.ollama = get_ollama_service()
        self.model_name = model_name
        
    async def deduplicate(
        self,
        entities: List[Any],
        relationships: List[Any],
        folder_id: str,
    ) -> Dict[str, Any]:
        """
        Deduplicate entities within batch and against existing folder entities.
        
        Args:
            entities: Extracted entities from current batch
            relationships: Extracted relationships
            folder_id: Folder to check for existing entities
            
        Returns:
            Deduplication result with merged entities and relationship updates
        """
        logger.info(f"Deduplicating {len(entities)} entities for folder {folder_id}")
        
        # Step 1: Deduplicate within batch
        batch_dedup = self._deduplicate_batch(entities)
        
        # Step 2: Check against existing folder entities
        existing_matches = await self._match_existing_entities(
            batch_dedup["unique_entities"],
            folder_id,
        )
        
        # Step 3: Update relationship references
        updated_relationships = self._update_relationship_refs(
            relationships,
            batch_dedup["merge_map"],
            existing_matches,
        )
        
        # Separate new vs existing
        new_entities = []
        matched_entity_ids = {match[0] for match in existing_matches}
        
        def gv(obj, key, default=None):
            """Safe get for dict or object."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        for entity in batch_dedup["unique_entities"]:
            entity_id = gv(entity, 'id')
            if entity_id not in matched_entity_ids:
                new_entities.append(entity)
        
        logger.info(f"After deduplication: {len(new_entities)} new entities, "
                   f"{len(existing_matches)} matched to existing")
        
        return {
            "new_entities": new_entities,
            "existing_matches": existing_matches,
            "relationships": updated_relationships,
            "merge_map": batch_dedup["merge_map"],
            "stats": {
                "original_count": len(entities),
                "after_batch_dedup": len(batch_dedup["unique_entities"]),
                "new_entities": len(new_entities),
                "matched_existing": len(existing_matches),
            },
        }
    
    def _deduplicate_batch(self, entities: List[Any]) -> Dict[str, Any]:
        """Deduplicate entities within the current batch."""
        unique_entities = []
        merge_map = {}  # old_id -> canonical_id
        seen = {}  # (name_lower, type) -> canonical_entity
        
        def gv(obj, key, default=None):
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)

        for entity in entities:
            name = gv(entity, 'name', '')
            entity_type = gv(entity, 'type', '')
            entity_id = gv(entity, 'id')
            
            key = (name.lower().strip(), entity_type.lower())
            
            if key in seen:
                # Duplicate found - merge into existing
                canonical = seen[key]
                canonical_id = gv(canonical, 'id')
                merge_map[entity_id] = canonical_id
                
                # Merge properties if entity has additional info
                props = gv(entity, 'properties')
                if props:
                    canonical_props = gv(canonical, 'properties')
                    if canonical_props is not None:
                        canonical_props.update(props)
                    
            else:
                # New unique entity
                seen[key] = entity
                unique_entities.append(entity)
                merge_map[entity_id] = entity_id
        
        logger.debug(f"Batch dedup: {len(entities)} -> {len(unique_entities)} entities")
        
        return {
            "unique_entities": unique_entities,
            "merge_map": merge_map,
        }
    
    async def _match_existing_entities(
        self,
        entities: List[Any],
        folder_id: str,
    ) -> List[Tuple[str, str]]:
        """Match new entities against existing ones in the folder."""
        matches = []
        
        try:
            driver = get_neo4j_driver()
            
            async with driver.session() as session:
                for entity in entities:
                    def gv(obj, key, default=None):
                        if isinstance(obj, dict): return obj.get(key, default)
                        return getattr(obj, key, default)

                    name = gv(entity, 'name', '')
                    entity_type = gv(entity, 'type', '')
                    entity_id = gv(entity, 'id')
                    
                    # Query for potential matches
                    result = await session.run("""
                        MATCH (e:Entity {folder_id: $folder_id})
                        WHERE toLower(e.name) = toLower($name)
                          AND toLower(e.type) = toLower($type)
                        RETURN e.id as id, e.name as name, e.description as description
                        LIMIT 1
                    """, folder_id=folder_id, name=name, type=entity_type)
                    
                    record = await result.single()
                    
                    if record:
                        # Found existing match
                        existing_id = record["id"]
                        matches.append((entity_id, existing_id))
                        logger.debug(f"Matched '{name}' to existing entity {existing_id}")
                        
        except Exception as e:
            logger.warning(f"Error matching existing entities: {e}")
        
        return matches
    
    async def _should_merge(
        self,
        entity1: Any,
        entity2: Any,
    ) -> Tuple[bool, float]:
        """Use AI to determine if two entities should be merged."""
        def gv(obj, key, default=None):
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)

        name1 = gv(entity1, 'name', '')
        desc1 = gv(entity1, 'description', '')
        
        name2 = gv(entity2, 'name', '')
        desc2 = gv(entity2, 'description', '')
        
        prompt = f"""Compare these two entities:

Entity 1: {name1}
Description: {desc1}

Entity 2: {name2}
Description: {desc2}

Are they the same real-world entity?"""
        
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            result = await self.ollama.chat_json(messages)
            
            is_same = result.get("is_same_entity", False)
            confidence = result.get("confidence", 0.5)
            
            return is_same and confidence >= self.SIMILARITY_THRESHOLD, confidence
            
        except Exception as e:
            logger.warning(f"AI merge check failed: {e}")
            return False, 0.0
    
    def _update_relationship_refs(
        self,
        relationships: List[Any],
        batch_merge_map: Dict[str, str],
        existing_matches: List[Tuple[str, str]],
    ) -> List[Any]:
        """Update relationship references to point to canonical entities."""
        # Build complete merge map
        full_merge_map = dict(batch_merge_map)
        for new_id, existing_id in existing_matches:
            full_merge_map[new_id] = existing_id
        
        updated = []
        for rel in relationships:
            def gv(obj, key, default=None):
                if isinstance(obj, dict): return obj.get(key, default)
                return getattr(obj, key, default)

            def sv(obj, key, val):
                if isinstance(obj, dict): obj[key] = val
                else: setattr(obj, key, val)

            # Get current IDs (support multiple field names)
            source_id = gv(rel, 'source_entity_id') or gv(rel, 'source_id')
            target_id = gv(rel, 'target_entity_id') or gv(rel, 'target_id')
            
            # Map to canonical IDs
            canonical_source = full_merge_map.get(source_id, source_id)
            canonical_target = full_merge_map.get(target_id, target_id)
            
            # Update relationship (support multiple field names)
            if gv(rel, 'source_entity_id') is not None:
                sv(rel, 'source_entity_id', canonical_source)
                sv(rel, 'target_entity_id', canonical_target)
            else:
                sv(rel, 'source_id', canonical_source)
                sv(rel, 'target_id', canonical_target)
            
            updated.append(rel)
        
        return updated
