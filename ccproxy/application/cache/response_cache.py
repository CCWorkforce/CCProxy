"""Modular response cache implementation using specialized components."""

import asyncio
import hashlib
import time
from typing import Any, Dict, Optional, List, AsyncIterator, Tuple

from .models import CachedResponse
from .statistics import CacheStatistics
from .circuit_breaker import CacheCircuitBreaker
from .stream_deduplication import StreamDeduplicator
from .memory_manager import CacheMemoryManager
from ...domain.models import (
    MessagesRequest,
    MessagesResponse,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockRedactedThinking,
)
from ...logging import debug, info, warning, LogRecord, LogEvent
from ...constants import (
    DEFAULT_CACHE_MAX_SIZE,
    DEFAULT_CACHE_MAX_MEMORY_MB,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS,
    DEFAULT_CACHE_VALIDATION_FAILURE_THRESHOLD,
)


class ResponseCache:
    """
    Advanced response cache with memory management and TTL support.

    This modular implementation delegates specific responsibilities to:
    - CacheMemoryManager: Memory management and LRU eviction
    - CacheStatistics: Performance metrics tracking
    - CacheCircuitBreaker: Validation failure handling
    - StreamDeduplicator: Concurrent stream deduplication
    """

    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_MAX_SIZE,
        max_memory_mb: int = DEFAULT_CACHE_MAX_MEMORY_MB,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        cleanup_interval_seconds: int = DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS,
        validation_failure_threshold: int = DEFAULT_CACHE_VALIDATION_FAILURE_THRESHOLD,
        redact_fields: Optional[List[str]] = None,
    ):
        """Initialize response cache with modular components."""
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._redact_fields = (
            [f.lower() for f in redact_fields] if redact_fields else []
        )

        # Initialize components
        self._memory_manager = CacheMemoryManager(
            max_memory_mb=max_memory_mb,
            max_size=max_size,
        )
        self._statistics = CacheStatistics()
        self._circuit_breaker = CacheCircuitBreaker(
            failure_threshold=validation_failure_threshold,
        )
        self._stream_deduplicator = StreamDeduplicator()

        # Pending requests tracking
        self._pending_requests: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

        # Cleanup task
        self._cleanup_task = None

    async def start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self):
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self):
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                warning(
                    LogRecord(
                        LogEvent.CACHE_EVENT.value,
                        f"Error in cache cleanup: {str(e)}",
                        None,
                        {"error": str(e)},
                    )
                )

    async def _cleanup_expired(self):
        """Remove expired entries from cache."""
        expired_keys = await self._memory_manager.evict_expired(self._ttl_seconds)
        if expired_keys:
            for _ in expired_keys:
                self._statistics.record_eviction()

    def _generate_cache_key(self, request: MessagesRequest) -> str:
        """Generate a unique cache key for the request."""
        request_json = request.model_dump_json(exclude_none=True)
        return hashlib.sha256(request_json.encode()).hexdigest()

    def _is_caching_disabled(self) -> bool:
        """Check if caching is currently disabled."""
        return self._circuit_breaker.is_open()

    def _validate_response_for_caching(self, response: MessagesResponse) -> bool:
        """
        Validate response structure for caching.

        Returns:
            True if response is valid for caching, False otherwise.
        """
        try:
            # Basic validation
            if not response or not response.content:
                return False

            # Validate content structure
            content_items = response.content
            if not isinstance(content_items, list):
                return False

            # Ensure all content blocks are valid
            for item in content_items:
                if isinstance(item, (ContentBlockText, ContentBlockThinking)):
                    if not hasattr(item, "text") or item.text is None:
                        return False
                elif isinstance(item, ContentBlockRedactedThinking):
                    pass  # Redacted thinking blocks are valid without text
                # Add other content block validations as needed

            # Validate JSON serialization
            _ = response.model_dump_json()
            return True

        except Exception as e:
            debug(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Response validation failed: {str(e)}",
                    request_id=None,
                    data={"error": str(e)},
                )
            )
            return False

    def _redact_sensitive_data(self, data: Any) -> Any:
        """Redact sensitive data from cache entries."""
        if isinstance(data, dict):
            return {
                k: "***REDACTED***"
                if k.lower() in self._redact_fields
                else self._redact_sensitive_data(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._redact_sensitive_data(item) for item in data]
        return data

    async def get_cached_response(
        self,
        request: MessagesRequest,
        request_id: Optional[str] = None,
        wait_for_pending: bool = False,
        timeout_seconds: float = 30.0,
    ) -> Optional[MessagesResponse]:
        """
        Get cached response for a request.

        Args:
            request: The request to look up
            request_id: Optional request ID for logging
            wait_for_pending: Whether to wait for pending requests
            timeout_seconds: Maximum time to wait for pending requests

        Returns:
            Cached response if found and valid, None otherwise
        """
        if self._is_caching_disabled():
            self._statistics.record_miss()
            return None

        cache_key = self._generate_cache_key(request)

        # Check if request is pending and wait if requested
        if wait_for_pending:
            async with self._lock:
                if cache_key in self._pending_requests:
                    event = self._pending_requests[cache_key]

            if "event" in locals():
                debug(
                    LogRecord(
                        event=LogEvent.CACHE_EVENT.value,
                        message="Request pending, waiting for completion",
                        request_id=request_id,
                        data={
                            "cache_key": cache_key[:8] + "...",
                            "timeout": timeout_seconds,
                        },
                    )
                )
                try:
                    await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    debug(
                        LogRecord(
                            event=LogEvent.CACHE_EVENT.value,
                            message="Timeout waiting for pending request",
                            request_id=request_id,
                            data={
                                "cache_key": cache_key[:8] + "...",
                                "timeout": timeout_seconds,
                            },
                        )
                    )
                    # Continue to check cache even after timeout

        # Get from cache
        cached = await self._memory_manager.get(cache_key)

        if cached:
            self._statistics.record_hit()

            info(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message="Cache hit",
                    request_id=request_id,
                    data={
                        "cache_key": cache_key[:8] + "...",
                        "waited_for_pending": wait_for_pending,
                    },
                )
            )
            return cached.response

        self._statistics.record_miss()
        return None

    async def mark_request_pending(self, request: MessagesRequest) -> str:
        """Mark a request as pending to prevent duplicate upstream calls."""
        cache_key = self._generate_cache_key(request)

        async with self._lock:
            if cache_key not in self._pending_requests:
                self._pending_requests[cache_key] = asyncio.Event()

        return cache_key

    async def cache_response(
        self,
        request: MessagesRequest,
        response: MessagesResponse,
        request_id: Optional[str] = None,
    ) -> bool:
        """
        Cache a response with validation.

        Args:
            request: The request that generated the response
            response: The response to cache
            request_id: Optional request ID for logging

        Returns:
            True if cached successfully, False otherwise
        """
        if self._is_caching_disabled():
            return False

        self._statistics.record_cache_attempt()

        # Validate response
        if not self._validate_response_for_caching(response):
            self._statistics.record_validation_failure()
            self._circuit_breaker.record_failure()

            warning(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message="Response validation failed",
                    request_id=request_id,
                    data={"validation_failures": self._statistics.validation_failures},
                )
            )
            return False

        self._circuit_breaker.record_success()

        # Create cached response
        cache_key = self._generate_cache_key(request)
        cached_response = CachedResponse(
            response=response,
            request_hash=cache_key,
            timestamp=time.time(),
        )

        # Add to cache
        success = await self._memory_manager.add(cache_key, cached_response, request_id)

        if success:
            self._statistics.set_memory_usage(self._memory_manager.memory_usage_bytes)

            info(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message="Response cached successfully",
                    request_id=request_id,
                    data={
                        "cache_key": cache_key[:8] + "...",
                        "size_bytes": cached_response.size_bytes,
                    },
                )
            )

        return success

    async def clear_pending_request(self, request: MessagesRequest):
        """Clear pending status for a request."""
        cache_key = self._generate_cache_key(request)

        async with self._lock:
            if cache_key in self._pending_requests:
                event = self._pending_requests.pop(cache_key)
                event.set()

    async def subscribe_stream(
        self, request: MessagesRequest, request_id: Optional[str] = None
    ) -> Tuple[bool, AsyncIterator[str], str]:
        """Subscribe to a deduplicated stream, returning iterator metadata."""
        cache_key = self._generate_cache_key(request)
        is_primary, queue = await self._stream_deduplicator.register(
            cache_key, request_id
        )

        async def iterator():
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                await self._stream_deduplicator.unregister(cache_key, queue)

        return is_primary, iterator(), cache_key

    async def publish_stream_line(self, key: str, line: str) -> None:
        """Publish a line to stream subscribers."""
        await self._stream_deduplicator.publish(key, line)

    async def finalize_stream(self, key: str) -> None:
        """Finalize a stream."""
        await self._stream_deduplicator.finalize(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self._statistics.get_stats()
        stats.update(self._memory_manager.get_stats())
        stats["circuit_breaker"] = self._circuit_breaker.get_status()
        return stats

    async def clear(self):
        """Clear all cache entries and reset statistics."""
        await self._memory_manager.clear()
        await self._stream_deduplicator.clear()
        self._statistics.reset()
        self._circuit_breaker.reset()

        async with self._lock:
            for event in self._pending_requests.values():
                event.set()
            self._pending_requests.clear()

        info(
            LogRecord(
                event=LogEvent.CACHE_EVENT.value,
                message="Cache cleared",
                request_id=None,
                data={},
            )
        )
