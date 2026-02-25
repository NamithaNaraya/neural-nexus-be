"""
Neural Nexus Backend - Main Application Entry Point

Knowledge Graph Platform - FastAPI Backend
Version: 2.1.0
"""
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import health, auth, folders, files, upload, graph, query, analytics, sse, websocket, deletion, dashboard, reasoning, browse, analytics_chat
from app.routes.ml import ml_routes
from app.db.connections import (
    init_neo4j, 
    close_neo4j, 
    init_postgres, 
    close_postgres,
    init_redis,
    close_redis,
)
from app.db.neo4j_utils import create_indexes, create_fulltext_indexes, create_vector_index
from app.core.config import settings
from app.core.middleware import RequestLoggingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Default to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Silence noisy libraries
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)  # Silence SQL queries
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
if not settings.DEBUG:
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Initializes all database connections on startup and
    closes them gracefully on shutdown.
    """
    logger.info("🚀 Starting Neural Nexus Backend...")
    
    # Initialize connections
    try:
        await init_neo4j()
        logger.info("✅ Neo4j connection established")
        
        # Create Neo4j indexes
        await create_indexes()
        await create_fulltext_indexes()
        await create_vector_index()
        logger.info("✅ Neo4j indexes created/verified")
        
        # Clear any stale GDS projections from previous runs
        try:
            from app.services.gds_service import get_gds_service
            gds = get_gds_service()
            await gds.invalidate_all()
            logger.info("✅ GDS projections cleared")
        except Exception as e:
            logger.warning(f"⚠️ Could not clear GDS projections: {e}")
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
    
    try:
        await init_postgres()
        logger.info("✅ PostgreSQL connection established")
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
    
    try:
        await init_redis()
        logger.info("✅ Redis connection established")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
    
    logger.info("🧠 Neural Nexus Backend is ready!")
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down Neural Nexus Backend...")
    await close_neo4j()
    await close_postgres()
    await close_redis()
    logger.info("👋 Neural Nexus Backend shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Neural Nexus API",
    description="Knowledge Graph Platform - Enterprise-grade knowledge extraction and visualization",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
if settings.DEBUG:
    app.add_middleware(RequestLoggingMiddleware)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
        }
    )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(folders.router, prefix="/api/v1/folders", tags=["Folders"])
app.include_router(files.router, prefix="/api/v1/files", tags=["Files"])
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(graph.router, prefix="/api/v1/graph", tags=["Graph"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(sse.router, prefix="/api/v1/sse", tags=["SSE"])
app.include_router(websocket.router, prefix="/api/v1/ws", tags=["WebSocket"])
app.include_router(deletion.router, prefix="/api/v1", tags=["Deletion"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(reasoning.router, prefix="/api/v1", tags=["Reasoning"])
app.include_router(browse.router, prefix="/api/v1/browse", tags=["Browse"])
app.include_router(ml_routes.router, prefix="/api/v1/ml", tags=["Machine Learning"])
app.include_router(analytics_chat.router, prefix="/api/v1/analytics-chat", tags=["Analytic Chat"])


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint - API information."""
    return {
        "name": "Neural Nexus API",
        "version": "2.1.0",
        "status": "running",
        "docs": "/api/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
