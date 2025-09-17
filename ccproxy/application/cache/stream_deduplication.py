"""Stream deduplication for handling concurrent identical streaming requests."""

import asyncio
from typing import Dict, List, AsyncIterator, Optional
from asyncio import Queue, QueueFull

from ...logging import debug, info, warning, LogRecord, LogEvent


class StreamDeduplicator:
    """
    Manages stream deduplication to handle concurrent identical requests.

    When multiple identical streaming requests arrive simultaneously,
    this class ensures only one upstream request is made while all
    clients receive the same streamed response.
    """

    def __init__(self, max_queue_size: int = 100):
        """Initialize stream deduplicator."""
        self.subscribers: Dict[str, List[Queue]] = {}
        self.lock = asyncio.Lock()
        self.max_queue_size = max_queue_size
        self.active_streams: Dict[str, bool] = {}

    async def subscribe(
        self, key: str, request_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Subscribe to a stream for the given cache key.

        Args:
            key: Cache key identifying the request
            request_id: Optional request ID for logging

        Yields:
            Lines from the deduplicated stream
        """
        queue: Queue[Optional[str]] = Queue(maxsize=self.max_queue_size)

        async with self.lock:
            if key not in self.subscribers:
                self.subscribers[key] = []
                self.active_streams[key] = True

            self.subscribers[key].append(queue)
            subscriber_count = len(self.subscribers[key])

            debug(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Stream subscription added for key {key[:8]}...",
                    request_id=request_id,
                    data={
                        "cache_key": key[:8] + "...",
                        "subscriber_count": subscriber_count,
                    },
                )
            )

        try:
            while True:
                line = await queue.get()
                if line is None:  # Stream ended
                    break
                yield line
        finally:
            async with self.lock:
                if key in self.subscribers and queue in self.subscribers[key]:
                    self.subscribers[key].remove(queue)
                    if not self.subscribers[key]:
                        del self.subscribers[key]
                        self.active_streams.pop(key, None)

    async def publish(self, key: str, line: str) -> None:
        """
        Publish a line to all subscribers of a stream.

        Args:
            key: Cache key identifying the stream
            line: Line to publish to all subscribers
        """
        async with self.lock:
            if key not in self.subscribers:
                return

            dead_queues = []
            for queue in self.subscribers[key]:
                try:
                    queue.put_nowait(line)
                except QueueFull:
                    dead_queues.append(queue)
                    warning(
                        LogRecord(
                            event=LogEvent.CACHE_EVENT.value,
                            message=f"Stream subscriber queue full for key {key[:8]}...",
                            request_id=None,
                            data={"cache_key": key[:8] + "..."},
                        )
                    )

            # Remove dead queues
            for queue in dead_queues:
                self.subscribers[key].remove(queue)

    async def finalize(self, key: str) -> None:
        """
        Finalize a stream, notifying all subscribers that it has ended.

        Args:
            key: Cache key identifying the stream
        """
        async with self.lock:
            if key not in self.subscribers:
                return

            # Send None to signal end of stream
            for queue in self.subscribers[key]:
                try:
                    queue.put_nowait(None)
                except QueueFull:
                    pass  # Queue is full, subscriber will timeout

            info(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Stream finalized for key {key[:8]}...",
                    request_id=None,
                    data={
                        "cache_key": key[:8] + "...",
                        "subscriber_count": len(self.subscribers.get(key, [])),
                    },
                )
            )

            # Mark stream as inactive
            self.active_streams[key] = False

    def is_active(self, key: str) -> bool:
        """Check if a stream is currently active."""
        return self.active_streams.get(key, False)

    def has_subscribers(self, key: str) -> bool:
        """Check if a stream has any subscribers."""
        return key in self.subscribers and len(self.subscribers[key]) > 0

    def get_subscriber_count(self, key: str) -> int:
        """Get the number of subscribers for a stream."""
        return len(self.subscribers.get(key, []))

    async def clear(self):
        """Clear all active streams and subscribers."""
        async with self.lock:
            # Finalize all active streams
            for key in list(self.subscribers.keys()):
                for queue in self.subscribers[key]:
                    try:
                        queue.put_nowait(None)
                    except QueueFull:
                        pass

            self.subscribers.clear()
            self.active_streams.clear()