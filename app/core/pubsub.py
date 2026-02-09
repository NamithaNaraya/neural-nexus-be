"""
Local Pub/Sub fallback for environments without Redis.
Ensures SSE and WebSockets work even if Redis is unreachable.
"""
import asyncio
from typing import Dict, Set, Any
import logging

logger = logging.getLogger(__name__)

class LocalPubSub:
    """Simple in-memory pub-sub to replace Redis for local development."""
    def __init__(self):
        self.subscribers: Dict[str, Set[asyncio.Queue]] = {}

    async def subscribe(self, channel: str):
        if channel not in self.subscribers:
            self.subscribers[channel] = set()
        queue = asyncio.Queue()
        self.subscribers[channel].add(queue)
        return queue

    async def unsubscribe(self, channel: str, queue: asyncio.Queue):
        if channel in self.subscribers:
            self.subscribers[channel].discard(queue)
            if not self.subscribers[channel]:
                del self.subscribers[channel]

    async def publish(self, channel: str, message: Any):
        if channel in self.subscribers:
            for queue in self.subscribers[channel]:
                await queue.put(message)

# Global instance
manager = LocalPubSub()
