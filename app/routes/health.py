"""
Health Check Routes

Provides health status endpoints for all services.
Used for monitoring and startup verification.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends
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
