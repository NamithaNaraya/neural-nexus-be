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
        folder_suffix = f"_F_{folder_id.replace('-', '_')}"
        folder_label = f"F_{folder_id.replace('-', '_')}"

        def prefix_labels(label_str: str) -> str:
            if not label_str:
                return ""
            # label_str looks like ":L1:L2"
            lbl_parts = label_str.split(':')
            prefixed = []
            for p in lbl_parts:
                if not p:
                    continue
                # Skip system labels and already prefixed labels
                if p in STANDARD_LABELS or p.startswith('F_') or p.endswith(folder_suffix):
                    prefixed.append(p)
                else:
                    prefixed.append(f"{p}{folder_suffix}")
            return ":" + ":".join(prefixed)

        # Regex to find node patterns: ( [var] [ :Labels] [ {props} ] )
        node_pattern = r"(\(\s*)([a-zA-Z0-9_]*)(\s*:[a-zA-Z0-9_:]*)?(\s*\{.*?\})?(\s*\))"

        parts = query.split(';')
        rewritten_parts = []

        for part in parts:
            seen_vars = set()

            def inject(match):
                prefix = match.group(1)  # '('
                var = match.group(2)     # 'n' or empty
                labels = match.group(3) or ""  # ':Label' or empty
                props = match.group(4) or ""   # ' {..}' or empty
                suffix = match.group(5)  # ')'

                # If it's a named variable we've already seen in this statement,
                # don't inject the folder label again.
                if var and var in seen_vars:
                    return f"{prefix}{var}{labels}{props}{suffix}"

                if var:
                    seen_vars.add(var)

                # Prefix all user labels in the pattern
                new_labels = prefix_labels(labels)

                if var and not new_labels and not props:
                    return f"{prefix}{var}:{folder_label}{suffix}"

                if not var and not new_labels and not props:
                    return f"{prefix}:{folder_label}{suffix}"

                # Append the technical isolation label as well
                return f"{prefix}{var}{new_labels}:{folder_label}{props}{suffix}"

            # Clean up comments
            check_part = re.sub(r'//.*', '', part)
            check_part = re.sub(r'/\*.*?\*/', '', check_part, flags=re.DOTALL)

            if not check_part.strip():
                rewritten_parts.append(part)
                continue

            # Check for schema commands (CONSTRAINT or INDEX)
            upper_part = check_part.strip().upper()
            is_schema = any(upper_part.startswith(kw) for kw in [
                "CREATE CONSTRAINT", "DROP CONSTRAINT",
                "CREATE INDEX", "DROP INDEX", "SHOW", "ASSERT"
            ])

            if is_schema:
                schema_lbl_pattern = r":([a-zA-Z0-9_]+)"

                def schema_inject(m):
                    lbl = m.group(1)
                    if lbl in STANDARD_LABELS or lbl.startswith('F_'):
                        return f":{lbl}"
                    return f":{lbl}{folder_suffix}"

                rewritten_part = re.sub(schema_lbl_pattern, schema_inject, part)
                rewritten_part = re.sub(
                    r"(CONSTRAINT|INDEX)\s+([a-zA-Z0-9_]+)\s+FOR",
                    r"\1 \2" + folder_suffix + " FOR",
                    rewritten_part,
                    flags=re.IGNORECASE
                )
                rewritten_parts.append(rewritten_part)
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
        folder_label = f"F_{folder_id.replace('-', '_')}"
        rewritten_statements = self._rewrite_query_for_folder(query, folder_id)
        logger.info(f"Rewrote Cypher for folder isolation: {folder_label}")

        async with self.driver.session() as session:
            # 0.5 Pre-Adoption: Restore isolation labels to orphaned or unassigned nodes
            try:
                pre_res = await session.run(f"""
                    MATCH (n)
                    WHERE (n.folder_id = $folder_id OR (n.folder_id IS NULL AND labels(n) <> []))
                    AND NOT n:{folder_label}
                    SET n:{folder_label},
                        n.folder_id = $folder_id
                    RETURN count(n) as repaired_count
                """, {"folder_id": folder_id})
                pre_rec = await pre_res.single()
                if pre_rec and pre_rec["repaired_count"] > 0:
                    logger.info(f"Pre-adopted {pre_rec['repaired_count']} orphaned nodes in folder {folder_id}")
            except Exception as e:
                logger.warning(f"Pre-adoption repair failed: {e}")

            # 1. Execute the user's query statements
            try:
                logger.info(f"Executing managed Cypher query for file {file_id}")
                for stmt in rewritten_statements:
                    await session.run(stmt, {"file_id": file_id, "folder_id": folder_id, "user_id": user_id})
            except Exception as e:
                logger.error(f"User Cypher execution failed: {e}")
                raise

            # 2. Targeted Adoption: Find nodes that should belong to this file.
            adoption_result = await session.run(f"""
                MATCH (n)
                WHERE (n:{folder_label} OR n.folder_id = $folder_id)
                AND (n.file_id IS NULL OR n.file_id = $file_id OR NOT $file_id IN n.file_ids)
                SET n.file_id = CASE WHEN n.file_id IS NULL THEN $file_id ELSE n.file_id END,
                    n.folder_id = CASE WHEN n.folder_id IS NULL THEN $folder_id ELSE n.folder_id END,
                    n.file_ids = CASE
                        WHEN n.file_ids IS NULL THEN [$file_id]
                        WHEN NOT $file_id IN n.file_ids THEN n.file_ids + $file_id
                        ELSE n.file_ids
                    END,
                    n:Entity,
                    n:{folder_label}
                RETURN count(n) as adopted_count
            """, {"file_id": file_id, "folder_id": folder_id})
            adoption_record = await adoption_result.single()
            logger.info(f"Adopted {adoption_record['adopted_count']} nodes for file {file_id}")

            # 2.5 Adopt Relationships: Tag relationships between adopted entities
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

            # 3. UUID Generation — use randomUUID() (built-in Neo4j 4.4+), fallback to Python
            try:
                await session.run("""
                    MATCH (n:Entity)
                    WHERE ($file_id IN n.file_ids OR n.file_id = $file_id) AND n.id IS NULL
                    SET n.id = randomUUID()
                """, {"file_id": file_id})
            except Exception as e:
                logger.warning(f"randomUUID() generation failed, will fallback to Python: {e}")

            # 4. File-to-Entity association is handled via the file_ids array on entities
            logger.info(f"Entities for file {file_id} are tracked via file_ids array")

            # 5. Normalize names and types from original labels, catch missing IDs
            # NOTE: Commented out to allow RAW node storage as nodes, not attributes
            """
            result = await session.run(\"\"\"
                MATCH (n:Entity)
                WHERE $file_id IN n.file_ids OR n.file_id = $file_id
                SET n.type = CASE
                        WHEN n.type IS NULL THEN
                            [l IN labels(n) WHERE NOT l IN ['Entity', 'Chunk', 'File', 'Folder'] AND NOT l STARTS WITH 'F_'][0]
                        ELSE n.type
                    END,
                    // Strip folder suffix from type if present (e.g. 'Herb_F_3c15...' -> 'Herb')
                    n.type = CASE
                        WHEN n.type CONTAINS '_F_' THEN split(n.type, '_F_')[0]
                        ELSE n.type
                    END,
                    // Improve naming heuristics to avoid UUIDs in the UI
                    n.name = CASE
                        WHEN n.name IS NULL OR n.name = n.id THEN
                            coalesce(
                                n.label, n.title, n.herb, n.quality, n.property,
                                n.value, n.text, n.display_name,
                                [lbl IN labels(n) WHERE NOT lbl IN ['Entity', 'Chunk', 'File', 'Folder'] AND NOT lbl STARTS WITH 'F_'][0],
                                n.id, 'Unknown'
                            )
                        ELSE n.name
                    END
                RETURN DISTINCT id(n) as internal_id, n.id as uuid
            \"\"\", {"file_id": file_id})
            """
            # Placeholder to ensure UUIDs are still generated if missing
            result = await session.run("""
                MATCH (n)
                WHERE ($file_id IN n.file_ids OR n.file_id = $file_id) AND n.id IS NULL
                RETURN DISTINCT id(n) as internal_id, n.id as uuid
            """, {"file_id": file_id})

            async for record in result:
                if not record["uuid"]:
                    new_uuid = str(uuid.uuid4())
                    await session.run(
                        "MATCH (n) WHERE id(n) = $int_id SET n.id = $uuid",
                        {"int_id": record["internal_id"], "uuid": new_uuid}
                    )

            # Step 5b: Retain all labels for idempotent Cypher compatibility.
            # The folder_label (:F_uuid) MUST persist for folder-based query scoping.
            logger.info("Retaining all labels for idempotent Cypher compatibility.")

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
                await cache.invalidate_graph(f"folder_{folder_id}_None_0_1000")
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