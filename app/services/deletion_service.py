"""
Deletion Service

Reliable batch deletion with reference counting.
Features:
- Background batch deletion using APOC
- Reference counting for shared nodes
- Shared nodes survive until last file is deleted
- Transaction-safe cascading deletes
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DeletionStatus(str, Enum):
    """Status of a deletion job."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeletionJob:
    """Represents a deletion job with progress tracking."""
    
    def __init__(
        self,
        job_id: str,
        target_type: str,  # "file", "folder", "entity"
        target_id: str,
        user_id: str,
    ):
        self.job_id = job_id
        self.target_type = target_type
        self.target_id = target_id
        self.user_id = user_id
        self.status = DeletionStatus.PENDING
        self.progress = 0.0
        self.message = "Queued for deletion"
        self.created_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.stats: Dict[str, int] = {
            "nodes_deleted": 0,
            "relationships_deleted": 0,
            "nodes_preserved": 0,  # Shared nodes kept
        }


class ReferenceCounter:
    """
    Tracks reference counts for shared entities.
    Entities are only deleted when their reference count reaches zero.
    """
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    
    async def get_reference_count(self, entity_id: str) -> int:
        """Get the number of files referencing an entity."""
        query = """
        MATCH (e:Entity {id: $entity_id})<-[:CONTAINS|MENTIONS]-(f:File)
        RETURN count(DISTINCT f) as ref_count
        """
        async with self.driver.session() as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()
            return record["ref_count"] if record else 0
    
    async def get_entities_with_refs(
        self,
        file_id: str,
    ) -> Dict[str, int]:
        """Get all entities in a file with their reference counts."""
        query = """
        MATCH (f:File {id: $file_id})-[:CONTAINS|MENTIONS]->(e:Entity)
        WITH e
        MATCH (e)<-[:CONTAINS|MENTIONS]-(other:File)
        RETURN e.id as entity_id, count(DISTINCT other) as ref_count
        """
        async with self.driver.session() as session:
            result = await session.run(query, file_id=file_id)
            records = await result.data()
            return {r["entity_id"]: r["ref_count"] for r in records}
    
    async def get_deletable_entities(
        self,
        file_id: str,
    ) -> Set[str]:
        """
        Get entities that can be safely deleted when file is removed.
        Only returns entities with reference count of 1 (only this file).
        """
        refs = await self.get_entities_with_refs(file_id)
        return {eid for eid, count in refs.items() if count <= 1}
    
    async def get_shared_entities(
        self,
        file_id: str,
    ) -> Set[str]:
        """Get entities shared with other files (will be preserved)."""
        refs = await self.get_entities_with_refs(file_id)
        return {eid for eid, count in refs.items() if count > 1}


class DeletionService:
    """
    Service for reliable background deletion with reference counting.
    Uses APOC for batch operations when available.
    """
    
    def __init__(self, neo4j_driver, db_session, cache_service=None, gds_service=None):
        self.driver = neo4j_driver
        self.db = db_session
        self.ref_counter = ReferenceCounter(neo4j_driver)
        self.active_jobs: Dict[str, DeletionJob] = {}
        self._has_apoc = None
        self.cache = cache_service
        self.gds = gds_service
    
    async def _check_apoc_available(self) -> bool:
        """Check if APOC procedures are available."""
        if self._has_apoc is not None:
            return self._has_apoc
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    "CALL dbms.procedures() YIELD name "
                    "WHERE name STARTS WITH 'apoc' RETURN count(*) as count"
                )
                record = await result.single()
                self._has_apoc = record["count"] > 0
        except Exception:
            self._has_apoc = False
        
        return self._has_apoc
    
    async def delete_file(
        self,
        file_id: str,
        user_id: str,
        background: bool = True,
    ) -> DeletionJob:
        """
        Delete a file and its non-shared entities.
        
        Args:
            file_id: File to delete
            user_id: User requesting deletion
            background: If True, run deletion in background
        
        Returns:
            DeletionJob tracking the deletion progress
        """
        import uuid
        
        job = DeletionJob(
            job_id=str(uuid.uuid4()),
            target_type="file",
            target_id=file_id,
            user_id=user_id,
        )
        self.active_jobs[job.job_id] = job
        
        if background:
            asyncio.create_task(self._execute_file_deletion(job))
        else:
            await self._execute_file_deletion(job)
        
        return job
    
    async def _execute_file_deletion(self, job: DeletionJob):
        """Execute file deletion with reference counting via file_ids array."""
        try:
            job.status = DeletionStatus.IN_PROGRESS
            job.message = "Cleaning graph entities..."
            job.progress = 0.1
            
            file_id = job.target_id
            
            # Remove file_id from entity file_ids arrays and delete entities with no remaining references
            async with self.driver.session() as neo_session:
                result = await neo_session.run("""
                    MATCH (e:Entity)
                    WHERE $file_id IN e.file_ids
                    SET e.file_ids = [x IN e.file_ids WHERE x <> $file_id]
                    WITH e, size(e.file_ids) as remaining
                    WHERE remaining = 0
                    DETACH DELETE e
                    RETURN count(e) as deleted
                """, file_id=file_id)
                record = await result.single()
                deleted_count = record["deleted"] if record else 0
                job.stats["nodes_deleted"] = deleted_count
                
                # Delete chunks for this file
                await neo_session.run(
                    "MATCH (c:Chunk {file_id: $file_id}) DELETE c",
                    file_id=file_id
                )
            
            job.message = f"Deleted {deleted_count} entities"
            job.progress = 0.8
            
            # Delete from PostgreSQL
            await self._delete_file_from_sql(file_id)
            
            job.status = DeletionStatus.COMPLETED
            job.message = "Deletion complete"
            job.progress = 1.0
            job.completed_at = datetime.utcnow()
            
            # Invalidate all caches after completion
            if self.cache:
                await self.cache.invalidate_all()
            if self.gds:
                await self.gds.invalidate_all()
            
            logger.info(
                f"File deletion complete: {file_id}, "
                f"deleted {job.stats['nodes_deleted']} nodes"
            )
            
        except Exception as e:
            job.status = DeletionStatus.FAILED
            job.error = str(e)
            job.message = f"Deletion failed: {str(e)}"
            logger.error(f"File deletion failed: {file_id}", exc_info=True)
    
    async def _detach_shared_entities(
        self,
        file_id: str,
        entity_ids: List[str],
    ):
        """Remove file-to-entity relationships without deleting entities."""
        query = """
        MATCH (f:File {id: $file_id})-[r:CONTAINS|MENTIONS]->(e:Entity)
        WHERE e.id IN $entity_ids
        DELETE r
        RETURN count(r) as deleted
        """
        async with self.driver.session() as session:
            result = await session.run(
                query,
                file_id=file_id,
                entity_ids=entity_ids,
            )
            return await result.single()
    
    async def _batch_delete_entities_apoc(
        self,
        entity_ids: List[str],
    ) -> Dict[str, int]:
        """Delete entities in batches using APOC for performance."""
        query = """
        CALL apoc.periodic.iterate(
            'UNWIND $entity_ids AS eid RETURN eid',
            'MATCH (e:Entity {id: eid})
             OPTIONAL MATCH (e)-[r]-()
             WITH e, count(r) as rel_count
             DETACH DELETE e
             RETURN rel_count',
            {batchSize: 500, parallel: true, params: {entity_ids: $entity_ids}}
        )
        YIELD batches, total, errorMessages
        RETURN total as nodes, 0 as relationships
        """
        async with self.driver.session() as session:
            result = await session.run(query, entity_ids=entity_ids)
            record = await result.single()
            return {
                "nodes": record["nodes"] if record else 0,
                "relationships": 0,  # APOC doesn't return this
            }
    
    async def _batch_delete_entities_native(
        self,
        entity_ids: List[str],
        batch_size: int = 500,
    ) -> Dict[str, int]:
        """Delete entities in batches using native Cypher."""
        total_nodes = 0
        total_rels = 0
        
        # Process in batches
        for i in range(0, len(entity_ids), batch_size):
            batch = entity_ids[i:i + batch_size]
            
            query = """
            MATCH (e:Entity)
            WHERE e.id IN $entity_ids
            OPTIONAL MATCH (e)-[r]-()
            WITH e, count(r) as rel_count
            DETACH DELETE e
            RETURN count(e) as nodes, sum(rel_count) as rels
            """
            async with self.driver.session() as session:
                result = await session.run(query, entity_ids=batch)
                record = await result.single()
                if record:
                    total_nodes += record["nodes"]
                    total_rels += record["rels"]
        
        return {"nodes": total_nodes, "relationships": total_rels}
    
    async def _delete_file_node(self, file_id: str):
        """Delete the file node from Neo4j."""
        query = """
        MATCH (f:File {id: $file_id})
        DETACH DELETE f
        """
        async with self.driver.session() as session:
            await session.run(query, file_id=file_id)
    
    async def _delete_file_from_sql(self, file_id: str):
        """Delete file record from PostgreSQL."""
        from sqlalchemy import delete
        from app.db.models import File
        
        await self.db.execute(
            delete(File).where(File.id == file_id)
        )
        await self.db.commit()
    
    async def delete_folder(
        self,
        folder_id: str,
        user_id: str,
        background: bool = True,
    ) -> DeletionJob:
        """
        Delete a folder and all its contents.
        Uses cascading deletion for contained files.
        """
        import uuid
        
        job = DeletionJob(
            job_id=str(uuid.uuid4()),
            target_type="folder",
            target_id=folder_id,
            user_id=user_id,
        )
        self.active_jobs[job.job_id] = job
        
        if background:
            asyncio.create_task(self._execute_folder_deletion(job))
        else:
            await self._execute_folder_deletion(job)
        
        return job
    
    async def _execute_folder_deletion(self, job: DeletionJob):
        """Execute folder deletion with cascading file deletions."""
        try:
            job.status = DeletionStatus.IN_PROGRESS
            job.message = "Finding folder contents..."
            job.progress = 0.1
            
            folder_id = job.target_id
            
            # Query PostgreSQL for file IDs (not Neo4j — :File nodes don't exist there)
            from sqlalchemy import text
            result = await self.db.execute(
                text("SELECT id FROM neural_nexus.files WHERE folder_id = :fid"),
                {"fid": folder_id}
            )
            file_ids = [str(r[0]) for r in result.fetchall()]
            
            job.message = f"Deleting {len(file_ids)} files..."
            
            # Delete each file's Neo4j entities via reference counting
            for i, file_id in enumerate(file_ids):
                try:
                    # Directly clean Neo4j entities for this file
                    async with self.driver.session() as neo_session:
                        # Remove file_id from entity file_ids array
                        # Delete entities that no longer have any file references
                        await neo_session.run("""
                            MATCH (e:Entity)
                            WHERE $file_id IN e.file_ids
                            SET e.file_ids = [x IN e.file_ids WHERE x <> $file_id]
                            WITH e
                            WHERE size(e.file_ids) = 0
                            DETACH DELETE e
                        """, file_id=file_id)
                        
                        # Delete chunks for this file
                        await neo_session.run(
                            "MATCH (c:Chunk {file_id: $file_id}) DELETE c",
                            file_id=file_id
                        )
                    job.stats["nodes_deleted"] += 1  # Approximate
                except Exception as e:
                    logger.warning(f"Failed to clean Neo4j for file {file_id}: {e}")
                
                job.progress = 0.1 + 0.6 * (i + 1) / max(len(file_ids), 1)
            
            # Safety net: delete ANY remaining entities tagged with this folder_id
            try:
                async with self.driver.session() as neo_session:
                    result = await neo_session.run(
                        "MATCH (e:Entity {folder_id: $fid}) DETACH DELETE e RETURN count(e) as cnt",
                        fid=folder_id
                    )
                    record = await result.single()
                    orphan_count = record["cnt"] if record else 0
                    if orphan_count > 0:
                        logger.info(f"Safety net cleaned {orphan_count} orphaned entities for folder {folder_id}")
                        job.stats["nodes_deleted"] += orphan_count
                    
                    # Also clean any chunks tagged with this folder
                    await neo_session.run(
                        "MATCH (c:Chunk {folder_id: $fid}) DELETE c",
                        fid=folder_id
                    )
            except Exception as e:
                logger.warning(f"Safety net Neo4j cleanup failed for folder {folder_id}: {e}")
            
            job.progress = 0.85
            
            # Delete from PostgreSQL (CASCADE will handle files + entity_staging)
            from sqlalchemy import delete as sql_delete
            from app.db.models import Folder
            
            await self.db.execute(
                sql_delete(Folder).where(Folder.id == folder_id)
            )
            await self.db.commit()
            
            # Invalidate all caches
            if self.cache:
                await self.cache.invalidate_all()
            
            job.status = DeletionStatus.COMPLETED
            job.message = "Folder deletion complete"
            job.progress = 1.0
            job.completed_at = datetime.utcnow()
            
            logger.info(f"Folder deletion complete: {folder_id}, deleted {job.stats['nodes_deleted']} nodes")
            
        except Exception as e:
            job.status = DeletionStatus.FAILED
            job.error = str(e)
            job.message = f"Deletion failed: {str(e)}"
            logger.error(f"Folder deletion failed: {folder_id}", exc_info=True)
    
    def get_job_status(self, job_id: str) -> Optional[DeletionJob]:
        """Get status of a deletion job."""
        return self.active_jobs.get(job_id)
    
    def get_active_jobs(self, user_id: Optional[str] = None) -> List[DeletionJob]:
        """Get all active deletion jobs, optionally filtered by user."""
        jobs = list(self.active_jobs.values())
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]
        return jobs


# Singleton instance
_deletion_service: Optional[DeletionService] = None


def get_deletion_service(neo4j_driver, db_session, cache_service=None, gds_service=None) -> DeletionService:
    """Get or create deletion service instance."""
    global _deletion_service
    if _deletion_service is None:
        _deletion_service = DeletionService(neo4j_driver, db_session, cache_service, gds_service)
    return _deletion_service
