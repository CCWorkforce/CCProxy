"""Stream deduplication for handling concurrent identical streaming requests."""

import anyio
from anyio import WouldBlock, create_memory_object_stream
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
from typing import Dict, List, Optional, Tuple

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
        self.subscribers: Dict[
            str, List[Tuple[MemoryObjectSendStream, MemoryObjectReceiveStream]]
        ] = {}
        self.lock = anyio.Lock()
        self.max_queue_size = max_queue_size
        self.active_streams: Dict[str, bool] = {}

    async def register(
        self, key: str, request_id: Optional[str] = None
    ) -> Tuple[bool, MemoryObjectReceiveStream]:
        """Register a subscriber channel for the given stream key."""

        # Create a channel with the specified buffer size
        send_stream, receive_stream = create_memory_object_stream(
            max_buffer_size=self.max_queue_size
        )

        async with self.lock:
            is_primary = key not in self.subscribers
            if is_primary:
                self.subscribers[key] = []
                self.active_streams[key] = True

            self.subscribers[key].append((send_stream, receive_stream))
            subscriber_count = len(self.subscribers[key])

            debug(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Stream subscription added for key {key[:8]}...",
                    request_id=request_id,
                    data={
                        "cache_key": key[:8] + "...",
                        "subscriber_count": subscriber_count,
                        "is_primary": is_primary,
                    },
                )
            )

        return is_primary, receive_stream

    async def unregister(
        self, key: str, receive_stream: MemoryObjectReceiveStream
    ) -> None:
        """Remove a subscriber channel from the stream."""

        async with self.lock:
            if key in self.subscribers:
                # Find and remove the matching receive stream
                for i, (send, recv) in enumerate(self.subscribers[key]):
                    if recv == receive_stream:
                        self.subscribers[key].pop(i)
                        # Close the channels
                        await send.aclose()
                        await recv.aclose()
                        break

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

            dead_streams = []
            for send_stream, receive_stream in self.subscribers[key]:
                try:
                    send_stream.send_nowait(line)
                except anyio.WouldBlock:
                    dead_streams.append((send_stream, receive_stream))
                    warning(
                        LogRecord(
                            event=LogEvent.CACHE_EVENT.value,
                            message=f"Stream subscriber channel full for key {key[:8]}...",
                            request_id=None,
                            data={"cache_key": key[:8] + "..."},
                        )
                    )
                except anyio.ClosedResourceError:
                    # Channel was closed
                    dead_streams.append((send_stream, receive_stream))

            # Remove dead streams
            for dead_pair in dead_streams:
                self.subscribers[key].remove(dead_pair)

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
            for send_stream, receive_stream in self.subscribers[key]:
                try:
                    send_stream.send_nowait(None)
                except (anyio.WouldBlock, anyio.ClosedResourceError):
                    pass  # Channel is full or closed, subscriber will timeout

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
                for send_stream, _ in self.subscribers[key]:
                    try:
                        send_stream.send_nowait(None)
                    except WouldBlock:
                        pass

            self.subscribers.clear()
            self.active_streams.clear()
