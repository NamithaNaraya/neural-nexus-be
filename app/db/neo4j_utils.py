"""
Neo4j Utility Functions

Provides base Cypher utilities and common graph operations:
- Index creation
- CRUD operations for entities
- Query builders
- Graph projection helpers
"""
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)


# === Index Management ===

async def create_indexes() -> None:
    """
    Create all required Neo4j indexes.
    
    Should be called during application startup.
    """
    driver = get_neo4j_driver()
    
    indexes = [
        # Primary lookups
        "CREATE INDEX entity_id_idx IF NOT EXISTS FOR (n:Entity) ON (n.id)",
        "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (n:Entity) ON (n.type)",
        
        # Ownership scoping
        "CREATE INDEX entity_folder_idx IF NOT EXISTS FOR (n:Entity) ON (n.folder_id)",
        "CREATE INDEX entity_user_idx IF NOT EXISTS FOR (n:Entity) ON (n.user_id)",
        "CREATE INDEX entity_file_idx IF NOT EXISTS FOR (n:Entity) ON (n.file_id)",
        
        # Chunk and file lookups
        "CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (n:Chunk) ON (n.id)",
        "CREATE INDEX file_id_idx IF NOT EXISTS FOR (n:File) ON (n.id)",
        "CREATE INDEX folder_id_idx IF NOT EXISTS FOR (n:Folder) ON (n.id)",
    ]
    
    async with driver.session() as session:
        for index_query in indexes:
            try:
                await session.run(index_query)
                logger.debug(f"Created index: {index_query}")
            except Exception as e:
                # Index might already exist
                logger.debug(f"Index creation skipped: {e}")
    
    logger.info("Neo4j indexes verified/created")


async def create_fulltext_indexes() -> None:
    """Create full-text search indexes."""
    driver = get_neo4j_driver()
    
    async with driver.session() as session:
        # Entity search index
        try:
            await session.run("""
                CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
                FOR (n:Entity) ON EACH [n.name, n.description]
            """)
            logger.info("Created entity full-text search index")
        except Exception as e:
            logger.debug(f"Full-text index exists: {e}")
        
        # Relationship search index (for Deep Search)
        try:
            await session.run("""
                CREATE FULLTEXT INDEX rel_search IF NOT EXISTS
                FOR ()-[r:RELATIONSHIP]-() ON EACH [r.description, r.type, r.category]
            """)
            logger.info("Created relationship full-text search index")
        except Exception as e:
            logger.debug(f"Relationship full-text index exists: {e}")
        
        # Relationship type index (for query optimization)
        try:
            await session.run("""
                CREATE INDEX rel_type_idx IF NOT EXISTS 
                FOR ()-[r:RELATIONSHIP]-() ON (r.type)
            """)
            logger.info("Created relationship type index")
        except Exception as e:
            logger.debug(f"Relationship type index exists: {e}")


async def create_vector_index(dimension: int = 768) -> None:
    """
    Create vector embedding index for semantic search.
    
    Args:
        dimension: Embedding vector dimension (default 768 for mxbai-embed-large)
    """
    driver = get_neo4j_driver()
    
    async with driver.session() as session:
        try:
            await session.run(f"""
                CREATE VECTOR INDEX embedding_idx IF NOT EXISTS
                FOR (n:Entity) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
            logger.info(f"Created vector index with dimension {dimension}")
        except Exception as e:
            logger.debug(f"Vector index exists: {e}")


# === Entity CRUD Operations ===

async def create_entity(
    entity_id: str,
    name: str,
    entity_type: str,
    folder_id: str,
    user_id: str,
    file_id: str,
    description: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None,
    embedding: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Create a new entity node in Neo4j.
    
    Returns the created entity data.
    """
    driver = get_neo4j_driver()
    
    query = """
    CREATE (e:Entity {
        id: $id,
        name: $name,
        type: $type,
        description: $description,
        properties: $properties,
        embedding: $embedding,
        folder_id: $folder_id,
        user_id: $user_id,
        file_id: $file_id,
        created_at: datetime()
    })
    RETURN e
    """
    
    async with driver.session() as session:
        result = await session.run(
            query,
            id=entity_id,
            name=name,
            type=entity_type,
            description=description or "",
            properties=properties or {},
            embedding=embedding,
            folder_id=folder_id,
            user_id=user_id,
            file_id=file_id,
        )
        record = await result.single()
        return dict(record["e"]) if record else {}


async def get_entity_by_id(entity_id: str) -> Optional[Dict[str, Any]]:
    """Get a single entity by ID."""
    driver = get_neo4j_driver()
    
    query = """
    MATCH (e:Entity {id: $id})
    RETURN e
    """
    
    async with driver.session() as session:
        result = await session.run(query, id=entity_id)
        record = await result.single()
        return dict(record["e"]) if record else None


async def update_entity(
    entity_id: str,
    updates: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Update an entity's properties.
    
    Returns the updated entity or None if not found.
    """
    driver = get_neo4j_driver()
    
    # Build dynamic SET clause
    set_clauses = [f"e.{key} = ${key}" for key in updates.keys()]
    set_clauses.append("e.updated_at = datetime()")
    set_clause = ", ".join(set_clauses)
    
    query = f"""
    MATCH (e:Entity {{id: $id}})
    SET {set_clause}
    RETURN e
    """
    
    params = {"id": entity_id, **updates}
    
    async with driver.session() as session:
        result = await session.run(query, **params)
        record = await result.single()
        return dict(record["e"]) if record else None


async def delete_entity(entity_id: str) -> bool:
    """
    Delete an entity and all its relationships.
    
    Returns True if entity was deleted.
    """
    driver = get_neo4j_driver()
    
    query = """
    MATCH (e:Entity {id: $id})
    DETACH DELETE e
    RETURN count(e) as deleted
    """
    
    async with driver.session() as session:
        result = await session.run(query, id=entity_id)
        record = await result.single()
        return record["deleted"] > 0 if record else False


# === Relationship Operations ===

async def create_relationship(
    source_id: str,
    target_id: str,
    rel_type: str,
    properties: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Create a relationship between two entities.
    
    Returns True if relationship was created.
    """
    driver = get_neo4j_driver()
    
    query = """
    MATCH (source:Entity {id: $source_id})
    MATCH (target:Entity {id: $target_id})
    CREATE (source)-[r:RELATIONSHIP {
        type: $rel_type,
        strength: $strength,
        description: $description,
        created_at: datetime()
    }]->(target)
    RETURN r
    """
    
    props = properties or {}
    
    async with driver.session() as session:
        result = await session.run(
            query,
            source_id=source_id,
            target_id=target_id,
            rel_type=rel_type,
            strength=props.get("strength", 1.0),
            description=props.get("description", ""),
        )
        record = await result.single()
        return record is not None


# === Graph Queries ===

async def get_folder_graph(
    folder_id: str,
    limit: int = 1000,
) -> Dict[str, List]:
    """
    Get all nodes and relationships for a folder.
    
    Returns dict with 'nodes' and 'links' arrays.
    """
    driver = get_neo4j_driver()
    
    query = """
    MATCH (e:Entity {folder_id: $folder_id})
    OPTIONAL MATCH (e)-[r:RELATIONSHIP]->(target:Entity {folder_id: $folder_id})
    WITH collect(DISTINCT e) as entities, collect(DISTINCT {source: e.id, target: target.id, type: r.type, strength: r.strength}) as rels
    RETURN entities, [r IN rels WHERE r.target IS NOT NULL] as relationships
    LIMIT $limit
    """
    
    async with driver.session() as session:
        result = await session.run(query, folder_id=folder_id, limit=limit)
        record = await result.single()
        
        if not record:
            return {"nodes": [], "links": []}
        
        nodes = [dict(e) for e in record["entities"]]
        links = record["relationships"]
        
        return {"nodes": nodes, "links": links}


async def get_file_graph(file_id: str) -> Dict[str, List]:
    """Get nodes and relationships for a specific file."""
    driver = get_neo4j_driver()
    
    query = """
    MATCH (e:Entity {file_id: $file_id})
    OPTIONAL MATCH (e)-[r:RELATIONSHIP]->(target:Entity {file_id: $file_id})
    RETURN collect(DISTINCT e) as entities, 
           collect(DISTINCT {source: e.id, target: target.id, type: r.type}) as relationships
    """
    
    async with driver.session() as session:
        result = await session.run(query, file_id=file_id)
        record = await result.single()
        
        if not record:
            return {"nodes": [], "links": []}
        
        nodes = [dict(e) for e in record["entities"]]
        links = [r for r in record["relationships"] if r["target"] is not None]
        
        return {"nodes": nodes, "links": links}


async def search_entities(
    query_text: str,
    folder_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Full-text search across entities.
    
    Optionally scope to a specific folder.
    """
    driver = get_neo4j_driver()
    
    if folder_id:
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_search', $query) YIELD node, score
        WHERE node.folder_id = $folder_id
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        params = {"query": query_text, "folder_id": folder_id, "limit": limit}
    else:
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_search', $query) YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        params = {"query": query_text, "limit": limit}
    
    async with driver.session() as session:
        result = await session.run(cypher, **params)
        records = await result.data()
        return [{"entity": dict(r["node"]), "score": r["score"]} for r in records]


async def vector_search(
    embedding: List[float],
    folder_id: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Vector similarity search using embeddings.
    
    Returns entities ordered by cosine similarity.
    """
    driver = get_neo4j_driver()
    
    if folder_id:
        cypher = """
        CALL db.index.vector.queryNodes('embedding_idx', $top_k, $embedding) YIELD node, score
        WHERE node.folder_id = $folder_id
        RETURN node, score
        ORDER BY score DESC
        """
        params = {"embedding": embedding, "folder_id": folder_id, "top_k": top_k}
    else:
        cypher = """
        CALL db.index.vector.queryNodes('embedding_idx', $top_k, $embedding) YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """
        params = {"embedding": embedding, "top_k": top_k}
    
    async with driver.session() as session:
        result = await session.run(cypher, **params)
        records = await result.data()
        return [{"entity": dict(r["node"]), "score": r["score"]} for r in records]


# === Batch Operations ===

async def delete_file_entities(file_id: str) -> int:
    """
    Delete all entities belonging to a file.
    
    Uses batch deletion for large files.
    Returns count of deleted entities.
    """
    driver = get_neo4j_driver()
    
    query = """
    MATCH (e:Entity {file_id: $file_id})
    WITH e LIMIT 5000
    DETACH DELETE e
    RETURN count(*) as deleted
    """
    
    total_deleted = 0
    
    async with driver.session() as session:
        while True:
            result = await session.run(query, file_id=file_id)
            record = await result.single()
            deleted = record["deleted"] if record else 0
            total_deleted += deleted
            
            if deleted < 5000:
                break
    
    logger.info(f"Deleted {total_deleted} entities for file {file_id}")
    return total_deleted
