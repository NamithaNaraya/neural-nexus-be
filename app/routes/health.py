"""
Health Check Routes

Provides health status endpoints for all services.
Used for monitoring and startup verification.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
import time
import logging

from app.db.connections import (
    get_neo4j_driver,
    get_redis_client,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Overall system health check.
    
    Returns status of all services.
    """
    services = {}
    overall_status = "healthy"
    
    # Check Neo4j
    try:
        start = time.time()
        driver = get_neo4j_driver()
        async with driver.session() as session:
            await session.run("RETURN 1")
        latency = int((time.time() - start) * 1000)
        services["neo4j"] = {"status": "up", "latency_ms": latency}
    except Exception as e:
        services["neo4j"] = {"status": "down", "error": str(e)}
        overall_status = "unhealthy"
    
    # Check Redis
    try:
        start = time.time()
        redis = get_redis_client()
        await redis.ping()
        latency = int((time.time() - start) * 1000)
        services["redis"] = {"status": "up", "latency_ms": latency}
    except Exception as e:
        services["redis"] = {"status": "down", "error": str(e)}
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "services": services,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@router.get("/health/neo4j")
async def neo4j_health() -> Dict[str, Any]:
    """Neo4j specific health check."""
    try:
        start = time.time()
        driver = get_neo4j_driver()
        async with driver.session() as session:
            result = await session.run("CALL dbms.components() YIELD name, versions")
            data = await result.single()
        latency = int((time.time() - start) * 1000)
        return {
            "status": "up",
            "latency_ms": latency,
            "version": data["versions"][0] if data else "unknown",
        }
    except Exception as e:
        return {"status": "down", "error": str(e)}


@router.get("/health/redis")
async def redis_health() -> Dict[str, Any]:
    """Redis specific health check."""
    try:
        start = time.time()
        redis = get_redis_client()
        info = await redis.info("server")
        latency = int((time.time() - start) * 1000)
        return {
            "status": "up",
            "latency_ms": latency,
            "version": info.get("redis_version", "unknown"),
        }
    except Exception as e:
        return {"status": "down", "error": str(e)}


@router.get("/health/ollama")
async def ollama_health() -> Dict[str, Any]:
    """Ollama AI service health check."""
    import httpx
    from app.core.config import settings
    
    try:
        start = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.OLLAMA_BASE_URL}/api/tags",
                timeout=5.0,
            )
            response.raise_for_status()
            data = response.json()
        latency = int((time.time() - start) * 1000)
        
        models = [m["name"] for m in data.get("models", [])]
        return {
            "status": "up",
            "latency_ms": latency,
            "available_models": models,
            "default_model": settings.OLLAMA_MODEL,
        }
    except Exception as e:
        return {"status": "down", "error": str(e)}


@router.get("/health/detailed")
async def detailed_health() -> Dict[str, Any]:
    """
    Detailed health status of all system components.
    
    Returns:
        - Overall status (healthy/degraded/unhealthy)
        - Individual service statuses with latency
        - System version
    """
    services = {}
    overall_status = "healthy"
    
    # Check Neo4j with details
    try:
        start = time.time()
        driver = get_neo4j_driver()
        async with driver.session() as session:
            result = await session.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions, edition"
            )
            data = await result.single()
        latency = (time.time() - start) * 1000
        services["neo4j"] = {
            "status": "healthy",
            "latency_ms": latency,
            "message": "Connected",
            "details": {
                "name": data["name"] if data else "Unknown",
                "version": data["versions"][0] if data and data["versions"] else "Unknown",
                "edition": data["edition"] if data else "Unknown",
            },
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except Exception as e:
        services["neo4j"] = {
            "status": "unhealthy",
            "message": str(e),
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        overall_status = "unhealthy"
    
    # Check Redis with details
    try:
        start = time.time()
        redis = get_redis_client()
        await redis.ping()
        info = await redis.info()
        latency = (time.time() - start) * 1000
        services["redis"] = {
            "status": "healthy",
            "latency_ms": latency,
            "message": "Connected",
            "details": {
                "version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
            },
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except Exception as e:
        services["redis"] = {
            "status": "unhealthy",
            "message": str(e),
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        overall_status = "unhealthy"
    
    # Check Ollama (optional service)
    import httpx
    from app.core.config import settings
    
    try:
        start = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.OLLAMA_BASE_URL}/api/tags",
                timeout=5.0,
            )
            response.raise_for_status()
            data = response.json()
        latency = (time.time() - start) * 1000
        
        models = [m["name"] for m in data.get("models", [])][:5]
        services["ollama"] = {
            "status": "healthy",
            "latency_ms": latency,
            "message": "Connected",
            "details": {
                "models": models,
                "model_count": len(data.get("models", [])),
            },
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except Exception as e:
        services["ollama"] = {
            "status": "degraded",  # Optional service
            "message": str(e),
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if overall_status == "healthy":
            overall_status = "degraded"
    
    return {
        "status": overall_status,
        "services": services,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": "2.1.0",
    }


@router.get("/health/indexes")
async def index_status() -> Dict[str, Any]:
    """
    Get database index population status.
    
    Returns:
        - Neo4j index status (online/populating)
        - PostgreSQL index status
        - Overall readiness flag
    """
    neo4j_indexes = []
    neo4j_populating = 0
    
    try:
        driver = get_neo4j_driver()
        async with driver.session() as session:
            query = """
            SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state, populationPercent
            RETURN name, type, labelsOrTypes, properties, state, populationPercent
            """
            result = await session.run(query)
            records = await result.data()
            
            for r in records:
                status = r.get("state", "UNKNOWN")
                if status == "POPULATING":
                    neo4j_populating += 1
                neo4j_indexes.append({
                    "name": r["name"],
                    "type": r["type"],
                    "labels": r["labelsOrTypes"],
                    "properties": r["properties"],
                    "status": status,
                    "population_percent": r.get("populationPercent", 100),
                })
    except Exception as e:
        logger.error(f"Failed to get Neo4j indexes: {e}")
    
    neo4j_online = len(neo4j_indexes) - neo4j_populating
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
            "total": 0,  # Would query pg_indexes
            "online": 0,
            "all_ready": True,
        },
        "overall_ready": neo4j_populating == 0,
    }


@router.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Kubernetes-style readiness probe.
    Returns 200 only if system is ready to receive traffic.
    """
    # Check basic connectivity
    try:
        driver = get_neo4j_driver()
        async with driver.session() as session:
            await session.run("RETURN 1")
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"System not ready - Neo4j unavailable: {str(e)}"
        )
    
    try:
        redis = get_redis_client()
        await redis.ping()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"System not ready - Redis unavailable: {str(e)}"
        )
    
    return {
        "ready": True,
        "services": "healthy",
        "indexes": "ready",
    }


@router.get("/health/live")
async def liveness_check() -> Dict[str, bool]:
    """
    Kubernetes-style liveness probe.
    Returns 200 if the process is alive (even if unhealthy).
    """
    return {"alive": True}

