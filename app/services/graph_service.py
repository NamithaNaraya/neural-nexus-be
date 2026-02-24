"""
Graph Service - Neo4j GDS Integration

Handles advanced graph algorithms like FastRP for node embeddings.
Fully utilizes GDS and APOC plugins for maximum performance.
"""
import logging
from typing import Dict, Any, List, Optional
from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)


class GraphService:
    """Service for advanced Neo4j Graph Data Science (GDS) and APOC operations."""
    
    def __init__(self):
        self.driver = get_neo4j_driver()

    async def check_gds_available(self) -> bool:
        """Check if GDS plugin is available."""
        async with self.driver.session() as session:
            try:
                result = await session.run("CALL gds.version() YIELD version RETURN version")
                record = await result.single()
                if record:
                    logger.info(f"GDS version: {record['version']}")
                    return True
            except Exception:
                pass
        return False

    async def check_apoc_available(self) -> bool:
        """Check if APOC plugin is available."""
        async with self.driver.session() as session:
            try:
                result = await session.run("CALL apoc.help('apoc') YIELD name RETURN count(name) as count")
                record = await result.single()
                return record and record["count"] > 0
            except Exception:
                pass
        return False

    # === GDS Embeddings ===
    async def run_fastrp_node_embeddings(
        self, 
        folder_id: str, 
        entity_types: Optional[List[str]] = None
    ) -> bool:
        """
        Run FastRP algorithm to generate structural node embeddings.
        
        This generates embeddings based on the graph topology within a folder.
        Optionally filter by entity types for type-specific embeddings.
        
        Args:
            folder_id: The folder to run FastRP on
            entity_types: Optional list of entity types to include (e.g., ["Person", "Organization"])
        """
        logger.info(f"Running FastRP for folder {folder_id}" + (f" with types {entity_types}" if entity_types else ""))
        
        projection_name = f"graph_{folder_id.replace('-', '_')}"
        
        # Build type filter clause
        type_filter = ""
        if entity_types:
            type_filter = " AND n.type IN $entity_types"
        
        async with self.driver.session() as session:
            try:
                # 1. Clean up old projection if it exists
                await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                
                # 2. Create projection with optional type filtering
                await session.run(f"""
                    CALL gds.graph.project.cypher(
                        $projection_name,
                        'MATCH (n:Entity) WHERE n.folder_id = $folder_id{type_filter} RETURN id(n) AS id',
                        'MATCH (a:Entity)-[r]->(b:Entity) 
                         WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id{type_filter.replace("n.", "a.") + type_filter.replace("n.", "b.") if entity_types else ""}
                         RETURN id(a) AS source, id(b) AS target, type(r) AS type',
                        {{parameters: {{folder_id: $folder_id, entity_types: $entity_types}}}}
                    )
                """, projection_name=projection_name, folder_id=folder_id, entity_types=entity_types or [])
                
                # 3. Run FastRP
                await session.run("""
                    CALL gds.fastRP.write(
                        $projection_name,
                        {
                            writeProperty: 'fastrp_embedding',
                            embeddingDimension: 256,
                            randomSeed: 42
                        }
                    )
                """, projection_name=projection_name)
                
                logger.info(f"FastRP completed for folder {folder_id}")
                return True
                
            except Exception as e:
                error_str = str(e)
                if "ProcedureNotFound" in error_str:
                    logger.warning("GDS plugin not available for FastRP")
                else:
                    logger.error(f"Failed to run FastRP: {e}")
                return False
            finally:
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except:
                    pass


    async def run_node2vec_embeddings(
        self, 
        folder_id: str,
        entity_types: Optional[List[str]] = None
    ) -> bool:
        """
        Run Node2Vec for random walk-based embeddings (alternative to FastRP).
        
        Args:
            folder_id: The folder to run Node2Vec on
            entity_types: Optional list of entity types to include
        """
        logger.info(f"Running Node2Vec for folder {folder_id}" + (f" with types {entity_types}" if entity_types else ""))
        projection_name = f"n2v_{folder_id.replace('-', '_')}"
        
        # Build type filter clause
        type_filter = ""
        if entity_types:
            type_filter = " AND n.type IN $entity_types"
        
        async with self.driver.session() as session:
            try:
                await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                
                await session.run(f"""
                    CALL gds.graph.project.cypher(
                        $projection_name,
                        'MATCH (n:Entity) WHERE n.folder_id = $folder_id{type_filter} RETURN id(n) AS id',
                        'MATCH (a:Entity)-[r]->(b:Entity) 
                         WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id{type_filter.replace("n.", "a.") + type_filter.replace("n.", "b.") if entity_types else ""}
                         RETURN id(a) AS source, id(b) AS target',
                        {{parameters: {{folder_id: $folder_id, entity_types: $entity_types}}}}
                    )
                """, projection_name=projection_name, folder_id=folder_id, entity_types=entity_types or [])
                
                await session.run("""
                    CALL gds.node2vec.write(
                        $projection_name,
                        {
                            writeProperty: 'node2vec_embedding',
                            embeddingDimension: 128,
                            walkLength: 80,
                            walksPerNode: 10,
                            inOutFactor: 1.0,
                            returnFactor: 1.0
                        }
                    )
                """, projection_name=projection_name)
                
                logger.info(f"Node2Vec completed for folder {folder_id}")
                return True
            except Exception as e:
                logger.error(f"Node2Vec failed: {e}")
                return False
            finally:
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except:
                    pass


    # === APOC Batch Operations ===
    async def batch_update_property(
        self, 
        folder_id: str, 
        property_name: str, 
        property_value: Any,
        batch_size: int = 1000
    ) -> int:
        """
        Batch update a property on all nodes using APOC periodic iterate.
        Returns count of updated nodes.
        """
        async with self.driver.session() as session:
            result = await session.run("""
                CALL apoc.periodic.iterate(
                    'MATCH (n:Entity {folder_id: $folder_id}) RETURN n',
                    'SET n[$prop] = $value',
                    {batchSize: $batch, params: {folder_id: $folder_id, prop: $prop, value: $value}}
                )
                YIELD batches, total, errorMessages
                RETURN total, errorMessages
            """, folder_id=folder_id, prop=property_name, value=property_value, batch=batch_size)
            
            record = await result.single()
            if record:
                if record["errorMessages"]:
                    logger.warning(f"Batch update errors: {record['errorMessages']}")
                return record["total"]
            return 0

    async def batch_create_relationships(
        self,
        relationships: List[Dict[str, str]],
        batch_size: int = 500
    ) -> int:
        """
        Batch create relationships using APOC.
        Each dict should have: source_id, target_id, rel_type, properties (optional)
        """
        if not relationships:
            return 0
            
        async with self.driver.session() as session:
            result = await session.run("""
                UNWIND $rels AS rel
                MATCH (a:Entity {id: rel.source_id}), (b:Entity {id: rel.target_id})
                CALL apoc.merge.relationship(a, rel.rel_type, {}, rel.properties, b, {})
                YIELD rel AS created
                RETURN count(created) AS count
            """, rels=relationships)
            
            record = await result.single()
            return record["count"] if record else 0

    # === APOC Text Processing ===
    async def clean_node_names(self, folder_id: str) -> int:
        """
        Clean and normalize all node names using APOC text functions.
        Returns count of cleaned nodes.
        """
        async with self.driver.session() as session:
            result = await session.run("""
                CALL apoc.periodic.iterate(
                    'MATCH (n:Entity {folder_id: $folder_id}) WHERE n.name IS NOT NULL RETURN n',
                    'SET n.name_clean = apoc.text.clean(n.name),
                         n.name_slug = apoc.text.slug(n.name, "-")',
                    {batchSize: 500, params: {folder_id: $folder_id}}
                )
                YIELD total
                RETURN total
            """, folder_id=folder_id)
            
            record = await result.single()
            return record["total"] if record else 0

    # === APOC UUID Generation ===
    async def ensure_node_uuids(self, folder_id: str) -> int:
        """
        Ensure all nodes have UUIDs using APOC.
        Returns count of nodes given new UUIDs.
        """
        async with self.driver.session() as session:
            result = await session.run("""
                CALL apoc.periodic.iterate(
                    'MATCH (n:Entity {folder_id: $folder_id}) WHERE n.uuid IS NULL RETURN n',
                    'SET n.uuid = apoc.create.uuid()',
                    {batchSize: 1000, params: {folder_id: $folder_id}}
                )
                YIELD total
                RETURN total
            """, folder_id=folder_id)
            
            record = await result.single()
            return record["total"] if record else 0

    # === APOC Graph Refactoring ===
    async def merge_duplicate_nodes(self, folder_id: str) -> Dict[str, Any]:
        """
        Merge nodes with the same name and type using APOC refactor.
        Returns merge statistics.
        """
        async with self.driver.session() as session:
            # Find and merge duplicates
            result = await session.run("""
                MATCH (n:Entity {folder_id: $folder_id})
                WITH n.name AS name, n.type AS type, collect(n) AS nodes
                WHERE size(nodes) > 1
                CALL apoc.refactor.mergeNodes(nodes, {properties: 'combine', mergeRels: true})
                YIELD node
                RETURN count(node) AS merged_count, 
                       sum(size(nodes) - 1) AS duplicates_removed
            """, folder_id=folder_id)
            
            record = await result.single()
            return {
                "merged_count": record["merged_count"] if record else 0,
                "duplicates_removed": record["duplicates_removed"] if record else 0,
            }

    async def normalize_relationship_types(self, folder_id: str) -> Dict[str, Any]:
        """
        Normalize relationship types to UPPERCASE using APOC.
        """
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (a:Entity {folder_id: $folder_id})-[r]->(b:Entity {folder_id: $folder_id})
                WHERE type(r) <> toUpper(type(r))
                WITH r, toUpper(type(r)) AS new_type
                CALL apoc.refactor.setType(r, new_type) YIELD output
                RETURN count(*) AS normalized_count
            """, folder_id=folder_id)
            
            record = await result.single()
            return {"normalized_count": record["normalized_count"] if record else 0}

    async def rename_relationship_type(self, old_type: str, new_type: str, folder_id: Optional[str] = None, file_id: Optional[str] = None) -> int:
        """
        Rename all relationships of a certain type, optionally scoped to a folder or file.
        Uses apoc.refactor.setType for efficiency.
        """
        # Sanitize new_type
        new_type = new_type.upper().replace(" ", "_").replace("-", "_")
        
        # Build dynamic query parts
        where_clauses = []
        params = {"new_type": new_type, "folder_id": folder_id, "file_id": file_id}
        
        if folder_id:
            where_clauses.append("a.folder_id = $folder_id")
        if file_id:
            # Check if relationship has this file_id in its metadata
            where_clauses.append("$file_id IN r.file_ids")
            
        scope_filter = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (a)-[r:{old_type}]->(b)
                {scope_filter}
                WITH r, $new_type AS nt
                CALL apoc.refactor.setType(r, nt) YIELD output
                RETURN count(*) AS count
            """, params)
            
            record = await result.single()
            return record["count"] if record else 0

    async def update_relationship(
        self, 
        relationship_id: str, 
        new_type: Optional[str] = None, 
        properties: Optional[Dict[str, Any]] = None,
        strength: Optional[float] = None
    ) -> bool:
        """
        Update a specific relationship by its ID.
        Handles type changes via apoc.refactor.setType and property updates via SET.
        """
        async with self.driver.session() as session:
            # 1. Update type if provided
            if new_type:
                new_type = new_type.upper().replace(" ", "_").replace("-", "_")
                await session.run("""
                    MATCH ()-[r]->()
                    WHERE r.id = $rel_id
                    CALL apoc.refactor.setType(r, $new_type) YIELD output
                    RETURN count(output)
                """, rel_id=relationship_id, new_type=new_type)

            # 2. Update properties if provided
            set_clauses = []
            params = {"rel_id": relationship_id}
            
            if strength is not None:
                set_clauses.append("r.strength = $strength")
                params["strength"] = strength
                
            if properties:
                for key, value in properties.items():
                    safe_key = key.replace(" ", "_").replace("-", "_")
                    set_clauses.append(f"r.{safe_key} = ${safe_key}")
                    params[safe_key] = value
            
            if set_clauses:
                query = f"""
                MATCH ()-[r]->()
                WHERE r.id = $rel_id
                SET {', '.join(set_clauses)}
                RETURN count(r)
                """
                await session.run(query, params)
                
            # 3. Add update timestamp
            from datetime import datetime
            await session.run("""
                MATCH ()-[r]->()
                WHERE r.id = $rel_id
                SET r.updated_at = $now
            """, rel_id=relationship_id, now=datetime.utcnow().isoformat())
                
            return True

    # === GDS Graph Statistics ===
    async def get_graph_stats(self, folder_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics using GDS.
        """
        projection_name = f"stats_{folder_id or 'all'}"
        
        async with self.driver.session() as session:
            try:
                # Create projection
                if folder_id:
                    await session.run("""
                        CALL gds.graph.project.cypher(
                            $name,
                            'MATCH (n:Entity) WHERE n.folder_id = $folder_id RETURN id(n) AS id',
                            'MATCH (a:Entity)-[r]->(b:Entity) 
                             WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id 
                             RETURN id(a) AS source, id(b) AS target',
                            {parameters: {folder_id: $folder_id}}
                        )
                    """, name=projection_name, folder_id=folder_id)
                else:
                    await session.run("""
                        CALL gds.graph.project($name, 'Entity', '*')
                    """, name=projection_name)
                
                # Get stats
                result = await session.run("""
                    CALL gds.graph.list($name)
                    YIELD graphName, nodeCount, relationshipCount, density
                    RETURN nodeCount, relationshipCount, density
                """, name=projection_name)
                
                record = await result.single()
                
                return {
                    "node_count": record["nodeCount"] if record else 0,
                    "relationship_count": record["relationshipCount"] if record else 0,
                    "density": record["density"] if record else 0,
                }
            finally:
                try:
                    await session.run("CALL gds.graph.drop($name, false)", name=projection_name)
                except:
                    pass


_graph_service = None

def get_graph_service() -> GraphService:
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service
