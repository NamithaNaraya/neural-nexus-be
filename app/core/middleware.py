"""
Middleware Module

Provides FastAPI middleware for:
- Global error handling
- Request logging
- CORS configuration
"""
import time
import logging
import traceback
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.
    
    Catches all unhandled exceptions and returns
    a consistent JSON error response.
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # Generate error ID for tracking
            error_id = str(uuid4())[:8]
            
            # Log the full exception
            logger.error(
                f"Unhandled exception [{error_id}]: {str(exc)}\n"
                f"Path: {request.url.path}\n"
                f"Method: {request.method}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            
            # Return sanitized error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "error_id": error_id,
                    "message": "An unexpected error occurred. Please try again.",
                },
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware.
    
    Logs all incoming requests with timing information.
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Start timer
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"- {response.status_code} ({duration_ms}ms)"
        )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms}ms"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.
    
    Limits requests per IP per minute.
    Uses in-memory storage (use Redis for production).
    """
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: dict = {}
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get current minute
        current_minute = int(time.time() / 60)
        key = f"{client_ip}:{current_minute}"
        
        # Check rate limit
        if key in self.request_counts:
            if self.request_counts[key] >= self.requests_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too many requests",
                        "message": f"Rate limit of {self.requests_per_minute} requests per minute exceeded.",
                        "retry_after": 60 - (int(time.time()) % 60),
                    },
                )
            self.request_counts[key] += 1
        else:
            # Clean old entries
            self.request_counts = {
                k: v for k, v in self.request_counts.items()
                if k.endswith(f":{current_minute}")
            }
            self.request_counts[key] = 1
        
        return await call_next(request)
