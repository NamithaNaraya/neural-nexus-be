"""
Storage Agent - Persistence Layer for the Ingestion Pipeline

Handles all database writes:
- Neo4j (graph structure)
- PostgreSQL (audit logs, file metadata)
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

from sqlalchemy import text

from app.db.connections import get_neo4j_driver, get_postgres_session

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
        
        async with driver.session() as session:
            for entity in entities:
                try:
                    # Extract entity data
                    entity_id = entity.id if hasattr(entity, 'id') else entity.get('id', str(uuid.uuid4()))
                    name = entity.name if hasattr(entity, 'name') else entity.get('name', '')
                    entity_type = entity.type if hasattr(entity, 'type') else entity.get('type', 'Concept')
                    description = entity.description if hasattr(entity, 'description') else entity.get('description', '')
                    properties = entity.properties if hasattr(entity, 'properties') else entity.get('properties', {})
                    embedding = entity.embedding if hasattr(entity, 'embedding') else entity.get('embedding')
                    source_text = entity.source_text if hasattr(entity, 'source_text') else entity.get('source_text', '')
                    confidence = entity.confidence if hasattr(entity, 'confidence') else entity.get('confidence', 1.0)
                    
                    # Create entity node
                    result = await session.run("""
                        MERGE (e:Entity {
                            name: $name,
                            type: $type,
                            folder_id: $folder_id
                        })
                        ON CREATE SET
                            e.id = $entity_id,
                            e.description = $description,
                            e.properties = $properties,
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
                        properties=properties,
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
        
        async with driver.session() as session:
            for rel in relationships:
                try:
                    # Extract relationship data
                    source_id = rel.source_entity_id if hasattr(rel, 'source_entity_id') else rel.get('source_entity_id')
                    target_id = rel.target_entity_id if hasattr(rel, 'target_entity_id') else rel.get('target_entity_id')
                    rel_type = rel.relationship_type if hasattr(rel, 'relationship_type') else rel.get('relationship_type', 'RELATED_TO')
                    description = rel.description if hasattr(rel, 'description') else rel.get('description', '')
                    strength = rel.strength if hasattr(rel, 'strength') else rel.get('strength', 1.0)
                    source_text = rel.source_text if hasattr(rel, 'source_text') else rel.get('source_text', '')
                    confidence = rel.confidence if hasattr(rel, 'confidence') else rel.get('confidence', 1.0)
                    
                    # Map to Neo4j IDs
                    neo4j_source = entity_id_map.get(source_id, source_id)
                    neo4j_target = entity_id_map.get(target_id, target_id)
                    
                    # Create relationship
                    result = await session.run("""
                        MATCH (source:Entity {id: $source_id, folder_id: $folder_id})
                        MATCH (target:Entity {id: $target_id, folder_id: $folder_id})
                        MERGE (source)-[r:RELATIONSHIP {type: $rel_type}]->(target)
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
                    """,
                        source_id=neo4j_source,
                        target_id=neo4j_target,
                        rel_type=rel_type,
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
        
        async with driver.session() as session:
            for chunk in chunks:
                try:
                    chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else chunk.get('chunk_id', str(uuid.uuid4()))
                    content = chunk.content if hasattr(chunk, 'content') else chunk.get('content', '')
                    embedding = chunk.embedding if hasattr(chunk, 'embedding') else chunk.get('embedding')
                    section_type = chunk.section_type if hasattr(chunk, 'section_type') else chunk.get('section_type', 'paragraph')
                    section_title = chunk.section_title if hasattr(chunk, 'section_title') else chunk.get('section_title')
                    
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
                    "details": str(details) if details else None,
                    "timestamp": datetime.utcnow(),
                }
            )
            await session.commit()
