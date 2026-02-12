"""
WebSocket Routes

Bi-directional real-time communication for:
- Collaborative editing
- Live node selection sync
- Camera position sync
- Instant notifications
"""
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import json
import logging
import asyncio

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time features.
    
    Supports:
    - User-specific connections
    - Room-based broadcasting (by folder_id)
    - Presence tracking
    """
    
    def __init__(self):
        # user_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # folder_id -> set of user_ids
        self.room_members: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        """Accept and register a new connection."""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        logger.info(f"WebSocket connected: user {user_id}")
    
    def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        """Remove a connection."""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        logger.info(f"WebSocket disconnected: user {user_id}")
    
    async def send_personal(self, message: dict, user_id: str) -> None:
        """Send message to a specific user."""
        if user_id in self.active_connections:
            data = json.dumps(message)
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_text(data)
                except Exception:
                    pass
    
    async def broadcast_to_room(self, message: dict, folder_id: str, exclude_user: str = None) -> None:
        """Broadcast message to all users in a folder room."""
        if folder_id in self.room_members:
            for user_id in self.room_members[folder_id]:
                if user_id != exclude_user:
                    await self.send_personal(message, user_id)
    
    def join_room(self, user_id: str, folder_id: str) -> None:
        """Add user to a folder room."""
        if folder_id not in self.room_members:
            self.room_members[folder_id] = set()
        self.room_members[folder_id].add(user_id)
    
    def leave_room(self, user_id: str, folder_id: str) -> None:
        """Remove user from a folder room."""
        if folder_id in self.room_members:
            self.room_members[folder_id].discard(user_id)


# Global connection manager
manager = ConnectionManager()


@router.websocket("")
@router.websocket("/")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = Query(None),
    token: str = Query(None),
):
    """
    WebSocket endpoint for real-time collaboration.
    """
    # Extract user_id from token if not provided
    if not user_id and token:
        try:
            from app.core.security import decode_token
            payload = decode_token(token)
            if payload:
                user_id = payload.get("sub")
            else:
                logger.warning("WebSocket token decoding returned None")
                await websocket.close(code=4003)
                return
        except Exception as e:
            logger.warning(f"WebSocket token validation failed: {e}")
            await websocket.close(code=4003) # Forbidden
            return

    if not user_id:
        logger.warning("WebSocket connection attempt without user_id")
        await websocket.close(code=4003)
        return

    await manager.connect(websocket, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            event_type = message.get("type")
            
            if event_type == "ping":
                await manager.send_personal({"type": "pong"}, user_id)
            
            elif event_type == "join_room":
                folder_id = message.get("folder_id")
                manager.join_room(user_id, folder_id)
                await manager.broadcast_to_room(
                    {"type": "collaborator_joined", "user_id": user_id},
                    folder_id,
                    exclude_user=user_id,
                )
            
            elif event_type == "leave_room":
                folder_id = message.get("folder_id")
                manager.leave_room(user_id, folder_id)
                await manager.broadcast_to_room(
                    {"type": "collaborator_left", "user_id": user_id},
                    folder_id,
                )
            
            elif event_type == "node_select":
                folder_id = message.get("folder_id")
                node_id = message.get("node_id")
                await manager.broadcast_to_room(
                    {"type": "node_selected", "node_id": node_id, "by_user": user_id},
                    folder_id,
                    exclude_user=user_id,
                )
            
            elif event_type == "camera_sync":
                folder_id = message.get("folder_id")
                position = message.get("position")
                await manager.broadcast_to_room(
                    {"type": "camera_update", "position": position, "by_user": user_id},
                    folder_id,
                    exclude_user=user_id,
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        # Notify rooms about disconnect
        for folder_id, members in manager.room_members.items():
            if user_id in members:
                members.discard(user_id)
                await manager.broadcast_to_room(
                    {"type": "collaborator_left", "user_id": user_id},
                    folder_id,
                )
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, user_id)
