"""
Managed Cypher Service
Handles normalization and enrichment of nodes created via direct Cypher queries.
Ensures architectural compliance (e.g., :Entity label, file_id sync).
Syncs created data to PostgreSQL entity_staging for consistency with AI pipeline.
"""
import logging
import re
import uuid
from typing import List, Dict, Any, Optional
from app.db.connections import get_neo4j_driver
from app.agents.storage_agent import StorageAgent
from app.agents.embedding_agent import EmbeddingAgent

logger = logging.getLogger(__name__)

# Labels that are part of the system architecture and should NOT be removed
STANDARD_LABELS = {'Entity', 'Chunk', 'File', 'Folder'}


class ManagedCypherService:
    def __init__(self):
        self.driver = get_neo4j_driver()
        self.storage = StorageAgent()
        self.embedding_agent = EmbeddingAgent()

    from typing import List

    def _rewrite_query_for_folder(self, query: str, folder_id: str) -> List[str]:
        """
        Injects a folder-specific label into all node patterns in a Cypher query for isolation.
        Example: '(n:Herb)' becomes '(n:Herb:F_e59c8189)'
        Ensures MERGE operations are scoped to the folder without manual query edits.
        """
        folder_label = f"F_{folder_id.replace('-', '_')}"
        
        # Regex to find node patterns: ( [var] [ :Labels] [ {props} ] )
        node_pattern = r"(\(\s*)([a-zA-Z0-9_]*)(\s*:[a-zA-Z0-9_:]*)?(\s*\{.*?\})?(\s*\))"

        def inject(match):
            prefix = match.group(1) # '('
            var = match.group(2)    # 'n' or empty
            labels = match.group(3) or "" # ':Label' or empty
            props = match.group(4) or ""  # ' {..}' or empty
            suffix = match.group(5) # ')'
            
            if var and not labels and not props:
                return f"{prefix}{var}{suffix}"

            if not var and not labels and not props:
                return f"{prefix}:{folder_label}{suffix}"

            return f"{prefix}{var}{labels}:{folder_label}{props}{suffix}"

        # Split query by semicolon to separate schema (CONSTRAINT/INDEX) from data (MERGE/CREATE)
        # Process each statement individually to ensure isolation label is only applied to data.
        parts = query.split(';')
        rewritten_parts = []
        
        for part in parts:
            # Clean up comments for the check (avoid greedy DOTALL on single-line comments)
            check_part = re.sub(r'//.*', '', part)
            check_part = re.sub(r'/\*.*?\*/', '', check_part, flags=re.DOTALL)
            
            if not check_part.strip():
                rewritten_parts.append(part)
                continue
            
            # Check for schema commands (CONSTRAINT or INDEX)
            check_part = check_part.strip().upper()
            is_schema = any(check_part.startswith(kw) for kw in [
                "CREATE CONSTRAINT", "DROP CONSTRAINT", 
                "CREATE INDEX", "DROP INDEX", "SHOW", "ASSERT"
            ])
            
            if is_schema:
                rewritten_parts.append(part)
            else:
                rewritten_parts.append(re.sub(node_pattern, inject, part))
                
        return rewritten_parts

    async def execute_managed_query(
        self, 
        query: str, 
        file_id: str, 
        folder_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Executes a Cypher query and then normalizes all nodes touched by the operation.
        Steps:
          1. Execute user's Cypher statements
          2. Adopt orphan nodes (add :Entity label, file_id metadata)
          3. Generate UUIDs for nodes missing them
          4. Normalize type/name properties from original labels
          5. Remove non-standard labels (e.g., :Herb, :Karma → stored as type property)
          6. Generate embeddings for nodes missing them
          7. Sync entities & relationships to PostgreSQL entity_staging
          8. Return final counts
        """
        # Execute the query as a single block to preserve variables (MERGE h, etc.)
        # We only split if strictly necessary for certain system commands, 
        # but for user data, one block is better.
        # 0. Rewrite query for folder isolation
        folder_label = f"F_{folder_id.replace('-', '_')}"
        rewritten_statements = self._rewrite_query_for_folder(query, folder_id)
        logger.info(f"Rewrote Cypher for folder isolation: {folder_label}")

        async with self.driver.session() as session:
            # 1. Execute the user's query statements
            try:
                logger.info(f"Executing managed Cypher query for file {file_id}")
                for stmt in rewritten_statements:
                    await session.run(stmt, {"file_id": file_id, "folder_id": folder_id, "user_id": user_id})
            except Exception as e:
                logger.error(f"User Cypher execution failed: {e}")
                raise

            # 2. Targeted Adoption: Find nodes that should belong to this file.
            # We ONLY adopt nodes that have the folder-specific label (guaranteed isolation).
            # We also ensure :Entity label and system IDs are present.
            adoption_result = await session.run(f"""
                MATCH (n:{folder_label})
                WHERE (n.file_id IS NULL OR n.file_id = $file_id OR NOT $file_id IN n.file_ids)
                SET n.file_id = CASE WHEN n.file_id IS NULL THEN $file_id ELSE n.file_id END,
                    n.folder_id = CASE WHEN n.folder_id IS NULL THEN $folder_id ELSE n.folder_id END,
                    n.file_ids = CASE 
                        WHEN n.file_ids IS NULL THEN [$file_id]
                        WHEN NOT $file_id IN n.file_ids THEN n.file_ids + $file_id
                        ELSE n.file_ids
                    END,
                    n:Entity
                RETURN count(n) as adopted_count
            """, {"file_id": file_id, "folder_id": folder_id})
            adoption_record = await adoption_result.single()
            logger.info(f"Adopted {adoption_record['adopted_count']} nodes for file {file_id}")

            # 2.5 Adopt Relationships: Tag relationships between adopted entities
            # We use DISTINCT to avoid issues with multiple paths
            await session.run("""
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE ($file_id IN a.file_ids) 
                  AND ($file_id IN b.file_ids)
                  AND (r.file_ids IS NULL OR NOT $file_id IN r.file_ids)
                SET r.file_ids = CASE 
                    WHEN r.file_ids IS NULL THEN [$file_id]
                    ELSE r.file_ids + $file_id 
                END
            """, {"file_id": file_id})

            # 3. Robust ID and Name Generation
            try:
                await session.run("""
                    MATCH (n:Entity)
                    WHERE ($file_id IN n.file_ids OR n.file_id = $file_id) AND n.id IS NULL
                    SET n.id = apoc.create.uuid()
                """, {"file_id": file_id})
            except Exception as e:
                logger.warning(f"APOC UUID generation failed, will fallback to Python: {e}")

            # 4. File-to-Entity association is handled via the file_ids array on entities
            # (No :File nodes exist in Neo4j — ownership is tracked via e.file_ids property)
            logger.info(f"Entities for file {file_id} are tracked via file_ids array")

            # 5. Normalize names and types from original labels, catch missing IDs
            result = await session.run("""
                MATCH (n:Entity)
                WHERE $file_id IN n.file_ids OR n.file_id = $file_id
                SET n.type = CASE 
                        WHEN n.type IS NULL THEN [l IN labels(n) WHERE l <> 'Entity'][0]
                        ELSE n.type
                    END,
                    n.name = CASE 
                        WHEN n.name IS NULL THEN coalesce(n.label, n.title, n.id, 'Unknown')
                        ELSE n.name
                    END
                RETURN DISTINCT id(n) as internal_id, n.id as uuid
            """, {"file_id": file_id})
            
            import uuid
            async for record in result:
                if not record["uuid"]:
                    new_uuid = str(uuid.uuid4())
                    await session.run("MATCH (n) WHERE id(n) = $int_id SET n.id = $uuid", 
                                    {"int_id": record["internal_id"], "uuid": new_uuid})

            # 5. Remove non-standard labels (Herb, Karma, Quality, etc.)
            try:
                await session.run(f"""
                    MATCH (n:Entity:{folder_label})
                    WHERE $file_id IN n.file_ids OR n.file_id = $file_id
                    WITH n, [l IN labels(n) WHERE NOT l IN ['Entity', 'Chunk', 'File', 'Folder', $folder_label]] AS extra
                    WHERE size(extra) > 0
                    CALL apoc.create.removeLabels(n, extra) YIELD node
                    RETURN count(node)
                """, {"file_id": file_id, "folder_label": folder_label})

                # Explicitly remove the isolation label after normalization
                await session.run(f"MATCH (n:{folder_label}) REMOVE n:{folder_label}")
                logger.debug(f"Labels normalized for file {file_id}")
            except Exception as e:
                # Fallback implementation...
                extras_result = await session.run("""
                    MATCH (n:Entity)
                    WHERE $file_id IN n.file_ids OR n.file_id = $file_id
                    UNWIND labels(n) AS lbl
                    WITH DISTINCT lbl
                    WHERE NOT lbl IN ['Entity', 'Chunk', 'File', 'Folder']
                    RETURN collect(lbl) AS extras
                """, {"file_id": file_id})
                extras_record = await extras_result.single()
                if extras_record and extras_record["extras"]:
                    for label in extras_record["extras"]:
                        safe_label = label.replace('`', '``')
                        await session.run(
                            f"MATCH (n:`{safe_label}`:Entity) WHERE $file_id IN n.file_ids REMOVE n:`{safe_label}`",
                            {"file_id": file_id}
                        )

            # 6. Generate missing embeddings
            result = await session.run("""
                MATCH (n:Entity)
                WHERE ($file_id IN n.file_ids OR n.file_id = $file_id) AND n.embedding IS NULL
                RETURN DISTINCT n.id as id, n.name as name, n.type as type, coalesce(n.description, '') as description
            """, {"file_id": file_id})
            
            new_nodes = []
            async for record in result:
                if record["id"] and record["name"]:
                    new_nodes.append({
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"] or "Entity",
                        "description": record["description"]
                    })
            
            if new_nodes:
                try:
                    embedded_nodes = await self.embedding_agent.embed_entities(new_nodes)
                    for node in embedded_nodes:
                        if 'embedding' in node:
                            await session.run("""
                                MATCH (n:Entity {id: $id})
                                SET n.embedding = $embedding
                            """, {"id": node["id"], "embedding": node["embedding"]})
                except Exception as e:
                    logger.error(f"Embedding generation failed: {e}")

            # 7. Sync to PostgreSQL entity_staging
            try:
                # Collect all entities for this file
                # Use DISTINCT to avoid inflated counts from messy graphs
                entities_result = await session.run("""
                    MATCH (n:Entity)
                    WHERE $file_id IN n.file_ids OR n.file_id = $file_id
                    RETURN DISTINCT n.id as id, n.name as name, n.type as type, 
                           coalesce(n.description, '') as description,
                           coalesce(n.confidence, 1.0) as confidence
                """, {"file_id": file_id})
                
                entities_for_staging = []
                async for record in entities_result:
                    entities_for_staging.append({
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"] or "Entity",
                        "description": record["description"],
                        "confidence": record["confidence"],
                    })
                
                # Collect all relationships for this file
                # Use DISTINCT to avoid relationship duplication in the list
                rels_result = await session.run("""
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE ($file_id IN a.file_ids OR a.file_id = $file_id)
                      AND ($file_id IN b.file_ids OR b.file_id = $file_id)
                      AND ($file_id IN r.file_ids OR r.file_id = $file_id)
                    RETURN DISTINCT a.id as source, b.id as target, type(r) as rel_type,
                           coalesce(r.description, '') as description
                """, {"file_id": file_id})
                
                relationships_for_staging = []
                async for record in rels_result:
                    relationships_for_staging.append({
                        "source_entity_id": record["source"],
                        "target_entity_id": record["target"],
                        "type": record["rel_type"],
                        "description": record["description"],
                    })
                
                await self.storage.store_staging(file_id, entities_for_staging, relationships_for_staging)
                logger.info(f"Staged {len(entities_for_staging)} entities and {len(relationships_for_staging)} relationships for file {file_id}")
            except Exception as e:
                logger.error(f"PostgreSQL staging sync failed: {e}")

            # 8. Get final counts
            count_result = await session.run("""
                MATCH (n:Entity)
                WHERE $file_id IN n.file_ids OR n.file_id = $file_id
                RETURN count(n) as count
            """, {"file_id": file_id})
            node_count_record = await count_result.single()
            node_count = node_count_record["count"] if node_count_record else 0

            rel_count_result = await session.run("""
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE ($file_id IN a.file_ids OR a.file_id = $file_id)
                  AND ($file_id IN b.file_ids OR b.file_id = $file_id)
                  AND ($file_id IN r.file_ids OR r.file_id = $file_id)
                RETURN count(r) as count
            """, {"file_id": file_id})
            rel_count_record = await rel_count_result.single()
            rel_count = rel_count_record["count"] if rel_count_record else 0

            # 9. Invalidate Cache
            try:
                from app.services.cache_service import get_cache_service
                cache = get_cache_service()
                await cache.invalidate_graph(f"file_{file_id}")
                await cache.invalidate_graph(f"folder_{folder_id}_None_0_1000") # Common folder cache key
                # Also invalidate general folder cache
                await cache.invalidate_all() 
                logger.info(f"Invalidated graph cache for file {file_id}")
            except Exception as e:
                logger.warning(f"Cache invalidation failed: {e}")

            return {
                "node_count": node_count,
                "relationship_count": rel_count,
                "message": f"Successfully ingested and normalized {node_count} nodes and {rel_count} relationships."
            }


_managed_cypher_service = None

def get_managed_cypher_service() -> ManagedCypherService:
    global _managed_cypher_service
    if _managed_cypher_service is None:
        _managed_cypher_service = ManagedCypherService()
    return _managed_cypher_service
