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
from app.core.pubsub import manager as local_pubsub

router = APIRouter()
logger = logging.getLogger(__name__)

async def event_generator(user_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    Generate SSE events for a user.
    Uses LocalPubSub for immediate delivery and attempts Redis subscription as secondary.
    """
    channel = f"user:{user_id}:events"
    queue = await local_pubsub.subscribe(channel)
    
    try:
        # Attempt Redis subscription if possible, but don't crash if it fails
        redis_pubsub = None
        try:
            redis = get_redis_client()
            redis_pubsub = redis.pubsub()
            await redis_pubsub.subscribe(channel)
            logger.info(f"✅ Subscribed to Redis channel {channel}")
        except Exception as e:
            logger.warning(f"⚠️ Redis sub failed (using Local fallback): {e}")

        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connected', 'user_id': user_id})}\n\n"
        
        while True:
            if await request.is_disconnected():
                break
            
            try:
                # 1. Check Local Queue (Instant)
                try:
                    local_msg = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield f"data: {local_msg}\n\n"
                except asyncio.TimeoutError:
                    pass

                # 2. Check Redis (if available)
                if redis_pubsub:
                    redis_msg = await redis_pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                    if redis_msg and redis_msg["type"] == "message":
                        data = redis_msg["data"]
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        yield f"data: {data}\n\n"
                
                # 3. Heartbeat
                yield ": keepalive\n\n"
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                
            except Exception as e:
                logger.warning(f"Transient loop error: {e}")
                await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"SSE error for user {user_id}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    finally:
        await local_pubsub.unsubscribe(channel, queue)
        if redis_pubsub:
            try:
                await redis_pubsub.unsubscribe(channel)
            except:
                pass


@router.get("/tasks/{user_id}")
async def stream_task_events(
    user_id: str,
    request: Request,
    token: str = None,  # Accept token as query param for EventSource
) -> StreamingResponse:
    """
    Stream real-time task events to the client.
    """
    # Validate token if provided
    if token:
        try:
            from app.core.security import decode_token
            payload = decode_token(token)
            if payload and payload.get("sub") == user_id:
                pass # Valid
            elif payload:
                user_id = payload.get("sub", user_id)
        except Exception as e:
            logger.warning(f"Invalid SSE token: {e}")
    
    return StreamingResponse(
        event_generator(user_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


async def publish_event(user_id: str, event_type: str, data: dict) -> None:
    """
    Publish an event to a user's SSE stream.
    Publishes to BOTH LocalPubSub and Redis to ensure delivery.
    """
    channel = f"user:{user_id}:events"
    event = json.dumps({
        "type": event_type,
        **data,
    })
    
    # 1. Publish Locally (Instant, no network required)
    await local_pubsub.publish(channel, event)
    
    # 2. Publish to Redis (if available)
    try:
        redis = get_redis_client()
        await redis.publish(channel, event)
    except Exception as e:
        # Silently fail Redis publish, we already sent it locally
        logger.debug(f"Redis publish skipped: {e}")


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
