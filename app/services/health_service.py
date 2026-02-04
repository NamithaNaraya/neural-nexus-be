"""
System Health Service

Comprehensive system health monitoring and Core-Sync checks.
Features:
- Database connectivity monitoring
- Index status tracking
- Worker health checks
- Resource usage monitoring
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceStatus:
    """Status of an individual service."""
    
    def __init__(
        self,
        name: str,
        status: HealthStatus = HealthStatus.UNKNOWN,
        latency_ms: Optional[float] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.status = status
        self.latency_ms = latency_ms
        self.message = message
        self.details = details or {}
        self.checked_at = datetime.utcnow()


class IndexStatus:
    """Status of database indexes."""
    
    def __init__(
        self,
        name: str,
        database: str,
        status: str,
        population_percent: float = 100.0,
        label_or_table: Optional[str] = None,
        properties: Optional[List[str]] = None,
    ):
        self.name = name
        self.database = database
        self.status = status  # "ONLINE", "POPULATING", "FAILED"
        self.population_percent = population_percent
        self.label_or_table = label_or_table
        self.properties = properties or []


class SystemHealthService:
    """
    Service for monitoring system health and performing Core-Sync checks.
    """
    
    def __init__(self, neo4j_driver, db_session, redis_client=None):
        self.neo4j = neo4j_driver
        self.db = db_session
        self.redis = redis_client
        self._last_check: Optional[datetime] = None
        self._cached_status: Optional[Dict[str, Any]] = None
        self._cache_ttl = timedelta(seconds=30)
    
    async def get_full_health(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Get comprehensive health status of all system components.
        
        Returns cached result if within TTL unless force_refresh is True.
        """
        now = datetime.utcnow()
        
        # Return cached if still valid
        if (
            not force_refresh
            and self._cached_status
            and self._last_check
            and now - self._last_check < self._cache_ttl
        ):
            return self._cached_status
        
        # Run all health checks in parallel
        checks = await asyncio.gather(
            self._check_neo4j(),
            self._check_postgresql(),
            self._check_redis(),
            self._check_ollama(),
            return_exceptions=True,
        )
        
        services = {}
        for check in checks:
            if isinstance(check, ServiceStatus):
                services[check.name] = {
                    "status": check.status.value,
                    "latency_ms": check.latency_ms,
                    "message": check.message,
                    "details": check.details,
                    "checked_at": check.checked_at.isoformat(),
                }
            elif isinstance(check, Exception):
                logger.error(f"Health check failed: {check}")
        
        # Determine overall status
        statuses = [s.get("status") for s in services.values()]
        if all(s == HealthStatus.HEALTHY.value for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY.value for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED
        
        result = {
            "status": overall.value,
            "services": services,
            "checked_at": now.isoformat(),
            "version": "2.1.0",
        }
        
        self._cached_status = result
        self._last_check = now
        
        return result
    
    async def _check_neo4j(self) -> ServiceStatus:
        """Check Neo4j connectivity and status."""
        start = datetime.utcnow()
        try:
            async with self.neo4j.session() as session:
                result = await session.run(
                    "CALL dbms.components() YIELD name, versions, edition "
                    "RETURN name, versions, edition"
                )
                record = await result.single()
                
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            return ServiceStatus(
                name="neo4j",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Connected",
                details={
                    "name": record["name"] if record else "Unknown",
                    "version": record["versions"][0] if record and record["versions"] else "Unknown",
                    "edition": record["edition"] if record else "Unknown",
                },
            )
        except Exception as e:
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return ServiceStatus(
                name="neo4j",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e),
            )
    
    async def _check_postgresql(self) -> ServiceStatus:
        """Check PostgreSQL connectivity."""
        start = datetime.utcnow()
        try:
            from sqlalchemy import text
            result = await self.db.execute(text("SELECT version()"))
            version = result.scalar()
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            return ServiceStatus(
                name="postgresql",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Connected",
                details={"version": version},
            )
        except Exception as e:
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return ServiceStatus(
                name="postgresql",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e),
            )
    
    async def _check_redis(self) -> ServiceStatus:
        """Check Redis connectivity."""
        if not self.redis:
            return ServiceStatus(
                name="redis",
                status=HealthStatus.UNKNOWN,
                message="Redis client not configured",
            )
        
        start = datetime.utcnow()
        try:
            await self.redis.ping()
            info = await self.redis.info()
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            return ServiceStatus(
                name="redis",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Connected",
                details={
                    "version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                },
            )
        except Exception as e:
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return ServiceStatus(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e),
            )
    
    async def _check_ollama(self) -> ServiceStatus:
        """Check Ollama API availability."""
        start = datetime.utcnow()
        try:
            from app.core.config import settings
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{settings.OLLAMA_BASE_URL}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m.get("name") for m in data.get("models", [])]
                        
                        latency = (datetime.utcnow() - start).total_seconds() * 1000
                        
                        return ServiceStatus(
                            name="ollama",
                            status=HealthStatus.HEALTHY,
                            latency_ms=latency,
                            message="Connected",
                            details={
                                "models": models[:5],  # First 5 models
                                "model_count": len(models),
                            },
                        )
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return ServiceStatus(
                name="ollama",
                status=HealthStatus.DEGRADED,  # Degraded, not unhealthy (optional service)
                latency_ms=latency,
                message=str(e),
            )
    
    async def get_index_status(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get status of all database indexes."""
        result = {
            "neo4j": [],
            "postgresql": [],
        }
        
        # Neo4j indexes
        try:
            async with self.neo4j.session() as session:
                query = """
                SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state, populationPercent
                RETURN name, type, labelsOrTypes, properties, state, populationPercent
                """
                neo4j_result = await session.run(query)
                records = await neo4j_result.data()
                
                for r in records:
                    result["neo4j"].append({
                        "name": r["name"],
                        "type": r["type"],
                        "labels": r["labelsOrTypes"],
                        "properties": r["properties"],
                        "status": r["state"],
                        "population_percent": r.get("populationPercent", 100),
                    })
        except Exception as e:
            logger.error(f"Failed to get Neo4j indexes: {e}")
        
        # PostgreSQL indexes
        try:
            from sqlalchemy import text
            pg_query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """)
            pg_result = await self.db.execute(pg_query)
            
            for row in pg_result.fetchall():
                result["postgresql"].append({
                    "name": row.indexname,
                    "table": row.tablename,
                    "status": "ONLINE",  # PG indexes are always online once created
                    "definition": row.indexdef[:100] + "..." if len(row.indexdef) > 100 else row.indexdef,
                })
        except Exception as e:
            logger.error(f"Failed to get PostgreSQL indexes: {e}")
        
        return result
    
    async def get_index_population_summary(self) -> Dict[str, Any]:
        """Get summary of index population status."""
        indexes = await self.get_index_status()
        
        neo4j_indexes = indexes.get("neo4j", [])
        pg_indexes = indexes.get("postgresql", [])
        
        neo4j_online = sum(1 for i in neo4j_indexes if i.get("status") == "ONLINE")
        neo4j_populating = sum(1 for i in neo4j_indexes if i.get("status") == "POPULATING")
        
        avg_population = (
            sum(i.get("population_percent", 100) for i in neo4j_indexes) / len(neo4j_indexes)
            if neo4j_indexes else 100
        )
        
        return {
            "neo4j": {
                "total": len(neo4j_indexes),
                "online": neo4j_online,
                "populating": neo4j_populating,
                "avg_population_percent": round(avg_population, 1),
                "all_ready": neo4j_populating == 0,
            },
            "postgresql": {
                "total": len(pg_indexes),
                "online": len(pg_indexes),  # All PG indexes are online
                "all_ready": True,
            },
            "overall_ready": neo4j_populating == 0,
        }
    
    async def run_core_sync_check(self) -> Dict[str, Any]:
        """
        Run comprehensive Core-Sync check.
        Verifies all essential services and data integrity.
        """
        results = {
            "passed": True,
            "checks": [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # 1. Service connectivity
        health = await self.get_full_health(force_refresh=True)
        services_ok = health["status"] == HealthStatus.HEALTHY.value
        results["checks"].append({
            "name": "Service Connectivity",
            "passed": services_ok,
            "message": f"Overall status: {health['status']}",
        })
        results["passed"] = results["passed"] and services_ok
        
        # 2. Index readiness
        index_summary = await self.get_index_population_summary()
        indexes_ok = index_summary["overall_ready"]
        results["checks"].append({
            "name": "Index Population",
            "passed": indexes_ok,
            "message": f"Neo4j: {index_summary['neo4j']['online']}/{index_summary['neo4j']['total']} online",
        })
        results["passed"] = results["passed"] and indexes_ok
        
        # 3. Data consistency check
        consistency = await self._check_data_consistency()
        results["checks"].append({
            "name": "Data Consistency",
            "passed": consistency["consistent"],
            "message": consistency["message"],
        })
        results["passed"] = results["passed"] and consistency["consistent"]
        
        # 4. Worker availability
        worker_check = await self._check_worker_availability()
        results["checks"].append({
            "name": "Worker Availability",
            "passed": worker_check["available"],
            "message": worker_check["message"],
        })
        # Workers are optional - don't fail core sync for missing workers
        
        return results
    
    async def _check_data_consistency(self) -> Dict[str, Any]:
        """Check data consistency between Neo4j and PostgreSQL."""
        try:
            # Count files in PostgreSQL
            from sqlalchemy import text
            pg_result = await self.db.execute(text("SELECT COUNT(*) FROM files"))
            pg_file_count = pg_result.scalar() or 0
            
            # Count files in Neo4j
            async with self.neo4j.session() as session:
                result = await session.run("MATCH (f:File) RETURN count(f) as count")
                record = await result.single()
                neo4j_file_count = record["count"] if record else 0
            
            # Allow some tolerance for in-flight operations
            diff = abs(pg_file_count - neo4j_file_count)
            tolerance = max(5, int(pg_file_count * 0.01))  # 1% or 5 files
            
            consistent = diff <= tolerance
            
            return {
                "consistent": consistent,
                "message": f"PostgreSQL: {pg_file_count} files, Neo4j: {neo4j_file_count} files",
                "details": {
                    "postgresql_files": pg_file_count,
                    "neo4j_files": neo4j_file_count,
                    "difference": diff,
                },
            }
        except Exception as e:
            return {
                "consistent": False,
                "message": f"Consistency check failed: {str(e)}",
            }
    
    async def _check_worker_availability(self) -> Dict[str, Any]:
        """Check Celery worker availability."""
        if not self.redis:
            return {
                "available": False,
                "message": "Redis not configured for worker check",
            }
        
        try:
            # Check for active Celery workers via Redis
            # This is a simplified check - production would use Celery inspect
            keys = await self.redis.keys("celery-task-meta-*")
            
            return {
                "available": True,
                "message": f"Celery broker connected, {len(keys)} task results cached",
                "details": {
                    "cached_results": len(keys),
                },
            }
        except Exception as e:
            return {
                "available": False,
                "message": f"Worker check failed: {str(e)}",
            }


# Singleton instance
_health_service: Optional[SystemHealthService] = None


def get_health_service(neo4j_driver, db_session, redis_client=None) -> SystemHealthService:
    """Get or create health service instance."""
    global _health_service
    if _health_service is None:
        _health_service = SystemHealthService(neo4j_driver, db_session, redis_client)
    return _health_service
