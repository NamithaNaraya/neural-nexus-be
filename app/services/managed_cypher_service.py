"""
Managed Cypher Service
Handles normalization and enrichment of nodes created via direct Cypher queries.
Ensures architectural compliance (e.g., :Entity label, file_id sync).
Syncs created data to PostgreSQL entity_staging for consistency with AI pipeline.

Robust Cypher rewriter that handles:
- CREATE CONSTRAINT ... IF NOT EXISTS FOR ...
- CREATE INDEX ... IF NOT EXISTS FOR ...
- DROP CONSTRAINT / DROP INDEX
- MERGE / CREATE / MATCH node patterns with folder isolation
- Multi-statement queries (semicolon-delimited)
- Comment lines (// and /* ... */)
- Relationship patterns (not misidentified as nodes)
"""
import logging
import re
import uuid
from typing import List, Dict, Any, Optional
from app.db.connections import get_neo4j_driver
from app.agents.storage_agent import StorageAgent
from app.agents.embedding_agent import EmbeddingAgent

logger = logging.getLogger(__name__)

# Labels that are part of the system architecture and should NOT be prefixed
STANDARD_LABELS = {'Entity', 'Chunk', 'File', 'Folder'}


class ManagedCypherService:
    def __init__(self):
        self.driver = get_neo4j_driver()
        self.storage = StorageAgent()
        self.embedding_agent = EmbeddingAgent()

    # ──────────────────────────────────────────────────────────────────────
    #  Cypher Rewriter — folder-level isolation via label injection
    # ──────────────────────────────────────────────────────────────────────

    def _rewrite_query_for_folder(self, query: str, folder_id: str) -> List[str]:
        """
        Rewrites a Cypher query for folder-level isolation by injecting a
        folder-specific label (e.g. :F_abc123) into all node patterns.

        Handles:
        - Schema commands (CREATE/DROP CONSTRAINT/INDEX with IF NOT EXISTS)
        - Data commands  (MERGE/CREATE/MATCH with node patterns)
        - Comments       (// line comments and /* block comments */)
        - Multi-statement queries (semicolon-delimited)

        Returns a list of individual Cypher statements ready for execution.
        """
        folder_label = f"F_{folder_id.replace('-', '_')}"
        folder_suffix = f"_{folder_label}"

        # ── Helper: prefix user labels with folder suffix ──
        def _prefix_labels(label_str: str) -> str:
            """
            Takes a label string like ':Herb:Spice' and returns ':Herb_F_xxx:Spice_F_xxx:F_xxx'.
            System labels (Entity, Chunk, etc.) and already-prefixed labels are left alone.
            """
            if not label_str:
                return ""
            parts = [p for p in label_str.split(':') if p]
            prefixed = []
            for p in parts:
                if p in STANDARD_LABELS or p.startswith('F_') or p.endswith(folder_suffix):
                    prefixed.append(p)
                else:
                    prefixed.append(f"{p}{folder_suffix}")
            return ":" + ":".join(prefixed) if prefixed else ""

        # ── Split into individual statements ──
        raw_parts = query.split(';')
        rewritten_statements: List[str] = []

        for part in raw_parts:
            # Strip the part for analysis but preserve original whitespace for output
            stripped = part.strip()
            if not stripped:
                continue

            # ── Remove comments for classification ──
            no_comments = re.sub(r'//.*', '', stripped)
            no_comments = re.sub(r'/\*.*?\*/', '', no_comments, flags=re.DOTALL)
            no_comments = no_comments.strip()

            if not no_comments:
                # Entire statement is comments — skip it entirely
                continue

            upper = no_comments.upper()

            # ── Classify: Schema command or Data command ──
            is_schema = any(upper.startswith(kw) for kw in [
                'CREATE CONSTRAINT', 'DROP CONSTRAINT',
                'CREATE INDEX', 'DROP INDEX',
                'SHOW CONSTRAINT', 'SHOW INDEX',
            ])

            if is_schema:
                rewritten_statements.append(
                    self._rewrite_schema_statement(stripped, folder_label, folder_suffix)
                )
            else:
                rewritten_statements.append(
                    self._rewrite_data_statement(stripped, folder_label, folder_suffix, _prefix_labels)
                )

        return rewritten_statements

    def _rewrite_schema_statement(
        self, stmt: str, folder_label: str, folder_suffix: str
    ) -> str:
        """
        Rewrites CREATE CONSTRAINT / CREATE INDEX statements.

        Examples:
          CREATE CONSTRAINT encounter_id IF NOT EXISTS FOR (e:Encounter) REQUIRE e.id IS UNIQUE
          →  CREATE CONSTRAINT encounter_id_F_xxx IF NOT EXISTS FOR (e:Encounter_F_xxx) REQUIRE e.id IS UNIQUE

          CREATE INDEX encounter_session IF NOT EXISTS FOR (e:Encounter) ON (e.session_id)
          →  CREATE INDEX encounter_session_F_xxx IF NOT EXISTS FOR (e:Encounter_F_xxx) ON (e.session_id)
        """
        result = stmt

        # 1) Rename the constraint/index name to include folder suffix
        #    Matches: CONSTRAINT  name  (IF NOT EXISTS)?  FOR
        #    Or:      INDEX       name  (IF NOT EXISTS)?  FOR
        name_pattern = re.compile(
            r'((?:CREATE|DROP)\s+(?:CONSTRAINT|INDEX)\s+)'   # prefix: CREATE CONSTRAINT
            r'([a-zA-Z_][a-zA-Z0-9_]*)'                     # constraint/index name
            r'(\s+(?:IF\s+NOT\s+EXISTS\s+)?FOR)',            # optional IF NOT EXISTS + FOR
            re.IGNORECASE
        )
        result = name_pattern.sub(
            lambda m: f"{m.group(1)}{m.group(2)}{folder_suffix}{m.group(3)}",
            result
        )

        # 2) Rename the label inside the FOR (...) pattern
        #    Matches: FOR (var:Label)
        for_label_pattern = re.compile(
            r'(FOR\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*):([a-zA-Z_][a-zA-Z0-9_]*)(\s*\))',
            re.IGNORECASE
        )

        def _replace_for_label(m):
            lbl = m.group(2)
            if lbl in STANDARD_LABELS or lbl.startswith('F_'):
                return m.group(0)
            return f"{m.group(1)}:{lbl}{folder_suffix}{m.group(3)}"

        result = for_label_pattern.sub(_replace_for_label, result)

        return result

    def _rewrite_data_statement(
        self, stmt: str, folder_label: str, folder_suffix: str, prefix_fn
    ) -> str:
        """
        Rewrites a data statement (MERGE/CREATE/MATCH/etc.) by injecting
        the folder label into all node patterns.

        Uses a careful regex that avoids matching relationship patterns like
        -[:REL_TYPE]-> or -[r:REL_TYPE {props}]->.
        """
        seen_vars: set = set()

        # ── Node pattern regex ──
        # Matches: (var:Label {props}) but NOT [...] (relationship patterns)
        # Key: We match opening '(' (not '[') to distinguish nodes from rels.
        node_pattern = re.compile(
            r'(\(\s*)'                          # Group 1: opening paren + optional whitespace
            r'([a-zA-Z_][a-zA-Z0-9_]*)?'        # Group 2: optional variable name
            r'(\s*(?::[a-zA-Z_][a-zA-Z0-9_:]*))?' # Group 3: optional labels like :Herb:Plant
            r'(\s*\{[^}]*\})?'                  # Group 4: optional properties block { ... }
            r'(\s*\))'                          # Group 5: closing paren
        )

        def _inject_folder(match: re.Match) -> str:
            prefix = match.group(1)     # '('
            var = match.group(2) or ""  # variable name or empty
            labels = match.group(3) or ""  # ':Label1:Label2' or empty
            props = match.group(4) or ""   # ' {name: "x"}' or empty
            suffix = match.group(5)     # ')'

            # If the variable was already seen in this statement, don't re-inject
            if var and var in seen_vars:
                return match.group(0)
            if var:
                seen_vars.add(var)

            # Prefix user labels
            new_labels = prefix_fn(labels)

            # Always ensure the folder label is present
            if folder_label not in (new_labels or ""):
                new_labels = f"{new_labels}:{folder_label}" if new_labels else f":{folder_label}"

            return f"{prefix}{var}{new_labels}{props}{suffix}"

        # Before applying the regex, strip comment lines so they don't get modified.
        # We process line-by-line: comment lines are passed through unchanged,
        # data lines get the node_pattern replacement.
        lines = stmt.split('\n')
        processed_lines = []

        for line in lines:
            stripped_line = line.strip()
            # Skip full-line comments
            if stripped_line.startswith('//') or stripped_line.startswith('/*'):
                processed_lines.append(line)
                continue
            # Apply node pattern injection on data lines
            processed_lines.append(node_pattern.sub(_inject_folder, line))

        return '\n'.join(processed_lines)

    # ──────────────────────────────────────────────────────────────────────
    #  Main Execution Entry Point
    # ──────────────────────────────────────────────────────────────────────

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
          1. Rewrite & execute user's Cypher statements (with folder isolation)
          2. Adopt orphan nodes (add :Entity label, file_id metadata)
          3. Generate UUIDs for nodes missing them
          4. Generate embeddings for nodes missing them
          5. Sync entities & relationships to PostgreSQL entity_staging
          6. Invalidate caches
          7. Return final counts
        """
        folder_label = f"F_{folder_id.replace('-', '_')}"

        # ── Rewrite query for folder isolation ──
        try:
            rewritten_statements = self._rewrite_query_for_folder(query, folder_id)
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            raise ValueError(f"Failed to rewrite Cypher for folder isolation: {e}")

        logger.info(
            f"Rewrote Cypher into {len(rewritten_statements)} statement(s) "
            f"for folder isolation: {folder_label}"
        )
        for i, stmt in enumerate(rewritten_statements):
            logger.debug(f"  Statement [{i}]: {stmt[:200]}...")

        async with self.driver.session() as session:
            # ── Step 1: Execute the rewritten statements ──
            try:
                logger.info(f"Executing managed Cypher query for file {file_id}")
                for idx, stmt in enumerate(rewritten_statements):
                    clean_stmt = stmt.strip()
                    if not clean_stmt:
                        continue
                    logger.debug(f"Running statement [{idx}]: {clean_stmt[:150]}")
                    try:
                        await session.run(
                            clean_stmt,
                            {
                                "file_id": file_id,
                                "folder_id": folder_id,
                                "user_id": user_id,
                            }
                        )
                    except Exception as stmt_err:
                        logger.error(
                            f"Statement [{idx}] failed: {stmt_err}\n"
                            f"Query was: {clean_stmt}"
                        )
                        raise
            except Exception as e:
                logger.error(f"User Cypher execution failed: {e}")
                raise

            # ── Step 2: Adopt nodes — tag with :Entity, file_id, folder_id ──
            try:
                adoption_result = await session.run(f"""
                    MATCH (n:{folder_label})
                    WHERE n.file_id IS NULL OR n.file_id = $file_id
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
                adopted = adoption_record["adopted_count"] if adoption_record else 0
                logger.info(f"Adopted {adopted} nodes for file {file_id}")
            except Exception as e:
                logger.error(f"Node adoption failed: {e}")
                # Non-fatal — continue with the rest

            # ── Step 2.5: Adopt Relationships ──
            try:
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
            except Exception as e:
                logger.warning(f"Relationship adoption failed: {e}")

            # ── Step 3: UUID Generation ──
            try:
                await session.run("""
                    MATCH (n:Entity)
                    WHERE ($file_id IN n.file_ids OR n.file_id = $file_id) AND n.id IS NULL
                    SET n.id = randomUUID()
                """, {"file_id": file_id})
            except Exception as e:
                logger.warning(f"randomUUID() failed, falling back to Python UUIDs: {e}")
                # Fallback: generate UUIDs via Python
                try:
                    result = await session.run("""
                        MATCH (n)
                        WHERE ($file_id IN n.file_ids OR n.file_id = $file_id) AND n.id IS NULL
                        RETURN DISTINCT id(n) as internal_id
                    """, {"file_id": file_id})
                    async for record in result:
                        new_uuid = str(uuid.uuid4())
                        await session.run(
                            "MATCH (n) WHERE id(n) = $int_id SET n.id = $uuid",
                            {"int_id": record["internal_id"], "uuid": new_uuid}
                        )
                except Exception as fallback_err:
                    logger.error(f"Python UUID fallback also failed: {fallback_err}")

            # ── Step 4: Generate missing embeddings ──
            try:
                result = await session.run("""
                    MATCH (n:Entity)
                    WHERE ($file_id IN n.file_ids OR n.file_id = $file_id) AND n.embedding IS NULL
                    RETURN DISTINCT n.id as id, n.name as name, n.type as type,
                           coalesce(n.description, '') as description
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
                        logger.info(f"Generated embeddings for {len(embedded_nodes)} nodes")
                    except Exception as e:
                        logger.error(f"Embedding generation failed: {e}")
            except Exception as e:
                logger.error(f"Embedding query failed: {e}")

            # ── Step 5: Sync to PostgreSQL entity_staging ──
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
                logger.info(
                    f"Staged {len(entities_for_staging)} entities and "
                    f"{len(relationships_for_staging)} relationships for file {file_id}"
                )
            except Exception as e:
                logger.error(f"PostgreSQL staging sync failed: {e}")

            # ── Step 6: Get final counts ──
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

            # ── Step 7: Invalidate Cache ──
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
                "message": (
                    f"Successfully ingested and normalized "
                    f"{node_count} nodes and {rel_count} relationships."
                )
            }


_managed_cypher_service = None


def get_managed_cypher_service() -> ManagedCypherService:
    global _managed_cypher_service
    if _managed_cypher_service is None:
        _managed_cypher_service = ManagedCypherService()
    return _managed_cypher_service