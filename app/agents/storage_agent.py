"""
Storage Agent - Persistence Layer for the Ingestion Pipeline

Handles all database writes:
- Neo4j (graph structure)
- PostgreSQL (audit logs, file metadata)
"""
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

from sqlalchemy import text

from app.db.connections import get_neo4j_driver, get_postgres_session
from app.db.neo4j_utils import sanitize_relationship_type

logger = logging.getLogger(__name__)


class StorageAgent:
    """
    Storage Agent for the Knowledge Graph Pipeline.
    
    Handles:
    - Neo4j entity and relationship creation
    - PostgreSQL audit logging
    - Reference counting for shared entities
    """
    
    def __init__(self):
        pass
        
    async def store_entities(
        self,
        entities: List[Any],
        file_id: str,
        folder_id: str,
        user_id: str,
    ) -> Dict[str, str]:
        """
        Store entities in Neo4j.
        
        Args:
            entities: List of validated entities
            file_id: Source file ID
            folder_id: Folder ID for scoping
            user_id: User ID for ownership
            
        Returns:
            Mapping of entity IDs to Neo4j node IDs
        """
        logger.info(f"Storing {len(entities)} entities to Neo4j")
        
        driver = get_neo4j_driver()
        entity_id_map = {}
        
        def get_value(obj, key, default=None):
            """Safely get value from dict or dataclass."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            elif hasattr(obj, key):
                return getattr(obj, key, default)
            return default
        
        async with driver.session() as session:
            for entity in entities:
                try:
                    # Extract entity data using helper
                    entity_id = get_value(entity, 'id', str(uuid.uuid4()))
                    name = get_value(entity, 'name', '')
                    entity_type = get_value(entity, 'type', 'Concept')
                    description = get_value(entity, 'description', '')
                    properties = get_value(entity, 'properties', {})
                    embedding = get_value(entity, 'embedding', None)
                    source_text = get_value(entity, 'source_text', '')
                    confidence = get_value(entity, 'confidence', 1.0)
                    
                    # Convert properties dict to JSON string (no APOC needed)
                    properties_json = json.dumps(properties) if properties else '{}'
                    
                    # Create entity node - simplified without APOC
                    result = await session.run("""
                        MERGE (e:Entity {
                            name: $name,
                            type: $type,
                            folder_id: $folder_id
                        })
                        ON CREATE SET
                            e.id = $entity_id,
                            e.description = $description,
                            e.properties = $properties_json,
                            e.embedding = $embedding,
                            e.source_text = $source_text,
                            e.confidence = $confidence,
                            e.user_id = $user_id,
                            e.file_ids = [$file_id],
                            e.created_at = datetime(),
                            e.source_count = 1
                        ON MATCH SET
                            e.file_ids = CASE 
                                WHEN NOT $file_id IN e.file_ids 
                                THEN e.file_ids + $file_id 
                                ELSE e.file_ids 
                            END,
                            e.source_count = size(e.file_ids),
                            e.updated_at = datetime()
                        RETURN e.id as node_id
                    """,
                        entity_id=entity_id,
                        name=name,
                        type=entity_type,
                        description=description,
                        properties_json=properties_json,
                        embedding=embedding,
                        source_text=source_text,
                        confidence=confidence,
                        folder_id=folder_id,
                        user_id=user_id,
                        file_id=file_id,
                    )
                    
                    record = await result.single()
                    if record:
                        entity_id_map[entity_id] = record["node_id"]
                        
                except Exception as e:
                    logger.error(f"Failed to store entity {name}: {e}")
                    continue
        
        logger.info(f"Stored {len(entity_id_map)} entities successfully")
        return entity_id_map
    
    async def store_relationships(
        self,
        relationships: List[Any],
        entity_id_map: Dict[str, str],
        file_id: str,
        folder_id: str,
    ) -> int:
        """
        Store relationships in Neo4j.
        
        Args:
            relationships: List of validated relationships
            entity_id_map: Mapping of entity IDs to Neo4j node IDs
            file_id: Source file ID
            folder_id: Folder ID for scoping
            
        Returns:
            Number of relationships created
        """
        logger.info(f"Storing {len(relationships)} relationships to Neo4j")
        
        driver = get_neo4j_driver()
        created_count = 0
        
        def get_value(obj, key, default=None):
            """Safely get value from dict or dataclass."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            elif hasattr(obj, key):
                return getattr(obj, key, default)
            return default
        
        async with driver.session() as session:
            for rel in relationships:
                try:
                    # Extract relationship data using helper
                    source_id = get_value(rel, 'source_entity_id')
                    target_id = get_value(rel, 'target_entity_id')
                    rel_type = get_value(rel, 'relationship_type', 'RELATED_TO')
                    description = get_value(rel, 'description', '')
                    strength = get_value(rel, 'strength', 1.0)
                    source_text = get_value(rel, 'source_text', '')
                    confidence = get_value(rel, 'confidence', 1.0)
                    
                    # Map to Neo4j IDs
                    neo4j_source = entity_id_map.get(source_id, source_id)
                    neo4j_target = entity_id_map.get(target_id, target_id)
                    
                    # Sanitize relationship type to valid Neo4j format
                    sanitized_type = sanitize_relationship_type(rel_type)
                    
                    # Try APOC-based dynamic relationship creation first
                    apoc_query = """
                        MATCH (source:Entity {id: $source_id, folder_id: $folder_id})
                        MATCH (target:Entity {id: $target_id, folder_id: $folder_id})
                        CALL apoc.merge.relationship(
                            source,
                            $rel_type,
                            {},
                            {
                                description: $description,
                                strength: $strength,
                                source_text: $source_text,
                                confidence: $confidence,
                                file_ids: [$file_id],
                                created_at: datetime()
                            },
                            target,
                            {}
                        ) YIELD rel
                        RETURN rel
                    """
                    
                    try:
                        result = await session.run(
                            apoc_query,
                            source_id=neo4j_source,
                            target_id=neo4j_target,
                            rel_type=sanitized_type,
                            description=description,
                            strength=strength,
                            source_text=source_text,
                            confidence=confidence,
                            folder_id=folder_id,
                            file_id=file_id,
                        )
                        record = await result.single()
                        if record:
                            created_count += 1
                    except Exception as apoc_error:
                        # APOC not available, use fallback with dynamic type in query
                        logger.debug(f"APOC not available, using fallback: {apoc_error}")
                        
                        fallback_query = f"""
                            MATCH (source:Entity {{id: $source_id, folder_id: $folder_id}})
                            MATCH (target:Entity {{id: $target_id, folder_id: $folder_id}})
                            MERGE (source)-[r:{sanitized_type}]->(target)
                            ON CREATE SET
                                r.description = $description,
                                r.strength = $strength,
                                r.source_text = $source_text,
                                r.confidence = $confidence,
                                r.file_ids = [$file_id],
                                r.created_at = datetime()
                            ON MATCH SET
                                r.file_ids = CASE 
                                    WHEN NOT $file_id IN r.file_ids 
                                    THEN r.file_ids + $file_id 
                                    ELSE r.file_ids 
                                END,
                                r.updated_at = datetime()
                            RETURN r
                        """
                        
                        result = await session.run(
                            fallback_query,
                            source_id=neo4j_source,
                            target_id=neo4j_target,
                            description=description,
                            strength=strength,
                            source_text=source_text,
                            confidence=confidence,
                            folder_id=folder_id,
                            file_id=file_id,
                        )
                        record = await result.single()
                        if record:
                            created_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to store relationship: {e}")
                    continue
        
        logger.info(f"Stored {created_count} relationships successfully")
        return created_count
    
    async def store_chunks(
        self,
        chunks: List[Any],
        file_id: str,
        folder_id: str,
    ) -> int:
        """
        Store text chunks in Neo4j for retrieval.
        
        Args:
            chunks: List of text chunks with embeddings
            file_id: Source file ID
            folder_id: Folder ID
            
        Returns:
            Number of chunks stored
        """
        logger.info(f"Storing {len(chunks)} chunks to Neo4j")
        
        driver = get_neo4j_driver()
        stored_count = 0
        
        def get_value(obj, key, default=None):
            """Safely get value from dict or dataclass."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            elif hasattr(obj, key):
                return getattr(obj, key, default)
            return default
        
        async with driver.session() as session:
            for chunk in chunks:
                try:
                    chunk_id = get_value(chunk, 'chunk_id', str(uuid.uuid4()))
                    content = get_value(chunk, 'content', '')
                    embedding = get_value(chunk, 'embedding', None)
                    section_type = get_value(chunk, 'section_type', 'paragraph')
                    section_title = get_value(chunk, 'section_title', None)
                    
                    await session.run("""
                        CREATE (c:Chunk {
                            id: $chunk_id,
                            content: $content,
                            embedding: $embedding,
                            section_type: $section_type,
                            section_title: $section_title,
                            file_id: $file_id,
                            folder_id: $folder_id,
                            created_at: datetime()
                        })
                    """,
                        chunk_id=chunk_id,
                        content=content,
                        embedding=embedding,
                        section_type=section_type,
                        section_title=section_title,
                        file_id=file_id,
                        folder_id=folder_id,
                    )
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to store chunk: {e}")
                    continue
        
        logger.info(f"Stored {stored_count} chunks successfully")
        return stored_count
    
    async def update_file_status(
        self,
        file_id: str,
        status: str,
        node_count: int = 0,
        relationship_count: int = 0,
        error_message: str = None,
    ) -> None:
        """
        Update file processing status in PostgreSQL.
        
        Args:
            file_id: File ID to update
            status: New status ('processing', 'ready_for_review', 'completed', 'failed')
            node_count: Number of nodes created
            relationship_count: Number of relationships created
            error_message: Error message if failed
        """
        async with get_postgres_session() as session:
            await session.execute(
                text("""
                    UPDATE neural_nexus.files 
                    SET status = :status,
                        node_count = :node_count,
                        relationship_count = :relationship_count,
                        error_message = :error_message,
                        processed_at = :processed_at
                    WHERE id = :file_id
                """),
                {
                    "file_id": file_id,
                    "status": status,
                    "node_count": node_count,
                    "relationship_count": relationship_count,
                    "error_message": error_message,
                    "processed_at": datetime.utcnow() if status in ['completed', 'failed'] else None,
                }
            )
            await session.commit()
        
        logger.info(f"Updated file {file_id} status to {status}")
    
    async def create_audit_log(
        self,
        user_id: str,
        action: str,
        target_type: str,
        target_id: str,
        details: Dict[str, Any] = None,
    ) -> None:
        """
        Create an audit log entry in PostgreSQL.
        
        Args:
            user_id: User who performed the action
            action: Action type (e.g., 'create', 'update', 'delete')
            target_type: Type of entity affected
            target_id: ID of affected entity
            details: Additional context
        """
        async with get_postgres_session() as session:
            await session.execute(
                text("""
                    INSERT INTO neural_nexus.audit_logs 
                    (user_id, action, target_type, target_id, new_value, timestamp)
                    VALUES (:user_id, :action, :target_type, :target_id, :details, :timestamp)
                """),
                {
                    "user_id": user_id,
                    "action": action,
                    "target_type": target_type,
                    "target_id": target_id,
                    "details": json.dumps(details) if details else None,
                    "timestamp": datetime.utcnow(),
                }
            )
            await session.commit()

    async def store_staging(
        self,
        file_id: str,
        entities: List[Any],
        relationships: List[Any],
    ) -> None:
        """
        Store extracted data in staging table.
        """
        # Convert entities/relationships to serializable format
        entity_dicts = []
        for e in entities:
            if hasattr(e, "model_dump"):
                entity_dicts.append(e.model_dump())
            elif hasattr(e, "__dict__"):
                entity_dicts.append({k: v for k, v in e.__dict__.items() if not k.startswith("_")})
            else:
                entity_dicts.append(e)

        rel_dicts = []
        for r in relationships:
            if hasattr(r, "model_dump"):
                rel_dicts.append(r.model_dump())
            elif hasattr(r, "__dict__"):
                rel_dicts.append({k: v for k, v in r.__dict__.items() if not k.startswith("_")})
            else:
                rel_dicts.append(r)

        async with get_postgres_session() as session:
            # Check if exists (upsert)
            result = await session.execute(
                text("SELECT id FROM neural_nexus.entity_staging WHERE file_id = :file_id"),
                {"file_id": file_id}
            )
            exists = result.fetchone()

            if exists:
                await session.execute(
                    text("""
                        UPDATE neural_nexus.entity_staging 
                        SET entity_data = :entity_data, 
                            relationship_data = :relationship_data,
                            created_at = :created_at
                        WHERE file_id = :file_id
                    """),
                    {
                        "file_id": file_id,
                        "entity_data": json.dumps(entity_dicts),
                        "relationship_data": json.dumps(rel_dicts),
                        "created_at": datetime.utcnow(),
                    }
                )
            else:
                await session.execute(
                    text("""
                        INSERT INTO neural_nexus.entity_staging 
                        (id, file_id, entity_data, relationship_data, created_at)
                        VALUES (:id, :file_id, :entity_data, :relationship_data, :created_at)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "file_id": file_id,
                        "entity_data": json.dumps(entity_dicts),
                        "relationship_data": json.dumps(rel_dicts),
                        "created_at": datetime.utcnow(),
                    }
                )
            await session.commit()
        
        logger.info(f"Stored {len(entities)} entities and {len(relationships)} relationships in staging for file {file_id}")
        # Debug: log first entity to verify format
        if entity_dicts:
            logger.info(f"[STAGING DEBUG] Sample entity: {entity_dicts[0].get('name', 'N/A')} ({entity_dicts[0].get('type', 'N/A')})")

    async def get_staging(self, file_id: str) -> Dict[str, List]:
        """Fetch staging data for a file."""
        async with get_postgres_session() as session:
            result = await session.execute(
                text("SELECT entity_data, relationship_data FROM neural_nexus.entity_staging WHERE file_id = :file_id"),
                {"file_id": file_id}
            )
            row = result.fetchone()
            if not row:
                logger.warning(f"[GET_STAGING DEBUG] No staging data found for file {file_id}")
                return {"entities": [], "relationships": []}
            
            entities = json.loads(row.entity_data) if isinstance(row.entity_data, str) else row.entity_data
            relationships = json.loads(row.relationship_data) if isinstance(row.relationship_data, str) else row.relationship_data
            
            logger.info(f"[GET_STAGING DEBUG] Retrieved {len(entities)} entities and {len(relationships)} relationships for file {file_id}")
            
            return {
                "entities": entities,
                "relationships": relationships,
            }

