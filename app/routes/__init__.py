# API Routes Package
# All API endpoints are centralized here for easy discovery

from app.routes import health
from app.routes import auth
from app.routes import upload
from app.routes import graph
from app.routes import query
from app.routes import analytics
from app.routes import sse
from app.routes import websocket

__all__ = [
    "health",
    "auth",
    "upload", 
    "graph",
    "query",
    "analytics",
    "sse",
    "websocket",
]
