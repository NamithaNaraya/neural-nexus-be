"""
Server-Sent Events (SSE) Routes

Real-time unidirectional streaming for:
- Ingestion progress updates
- AI response streaming
- Task status notifications
"""
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging

from app.core.security import get_current_user
from app.db.connections import get_redis_client

router = APIRouter()
logger = logging.getLogger(__name__)


async def event_generator(user_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    Generate SSE events for a user.
    
    Listens to Redis pub/sub for user-specific events.
    """
    try:
        redis = get_redis_client()
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"user:{user_id}:events")
        
        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connected', 'user_id': user_id})}\n\n"
        
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break
            
            # Get message from Redis
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            
            if message and message["type"] == "message":
                data = message["data"]
                yield f"data: {data}\n\n"
            
            # Send heartbeat every 30 seconds
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"SSE error for user {user_id}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    finally:
        await pubsub.unsubscribe(f"user:{user_id}:events")


@router.get("/tasks/{user_id}")
async def stream_task_events(
    user_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream real-time task events to the client.
    
    Events include:
    - Ingestion progress (phase, percentage)
    - AI response chunks (word-by-word)
    - Graph updates (new nodes, relationships)
    """
    # Verify user can only subscribe to their own events
    if current_user.user_id != user_id and current_user.role != "admin":
        user_id = current_user.user_id
    
    return StreamingResponse(
        event_generator(user_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def publish_event(user_id: str, event_type: str, data: dict) -> None:
    """
    Publish an event to a user's SSE stream.
    
    Call this from other parts of the application to send real-time updates.
    """
    try:
        redis = get_redis_client()
        event = json.dumps({
            "type": event_type,
            **data,
        })
        await redis.publish(f"user:{user_id}:events", event)
    except Exception as e:
        logger.error(f"Failed to publish event: {e}")


# Helper functions for common events
async def publish_ingestion_progress(
    user_id: str, 
    file_id: str, 
    phase: str, 
    progress: int,
    message: str = "",
) -> None:
    """Publish ingestion progress update."""
    await publish_event(user_id, "ingestion_progress", {
        "file_id": file_id,
        "phase": phase,
        "progress": progress,
        "message": message,
    })


async def publish_ai_chunk(
    user_id: str,
    session_id: str,
    chunk: str,
    is_final: bool = False,
) -> None:
    """Publish AI response chunk for streaming."""
    await publish_event(user_id, "ai_chunk", {
        "session_id": session_id,
        "chunk": chunk,
        "is_final": is_final,
    })
