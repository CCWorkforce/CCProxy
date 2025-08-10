"""Response caching system with memory management for handling timeouts and repeated requests."""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import sys

from ..domain.models import MessagesRequest, MessagesResponse
from ..logging import debug, info, warning, LogRecord, LogEvent


@dataclass
class CachedResponse:
    """Represents a cached response with metadata."""
    response: MessagesResponse
    request_hash: str
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0

    def __post_init__(self):
        """Calculate size after initialization."""
        if self.size_bytes == 0:
            # Estimate size of the cached response
            self.size_bytes = sys.getsizeof(self.response.model_dump_json())


class ResponseCache:
    """
    Advanced response cache with memory management and TTL support.
    Handles timeout scenarios by returning cached responses for identical requests.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 500,
        ttl_seconds: int = 3600,
        cleanup_interval_seconds: int = 300
    ):
        """
        Initialize response cache.

        Args:
            max_size: Maximum number of cached responses
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for cached responses in seconds
            cleanup_interval_seconds: Interval for cleanup task
        """
        self._max_size = max_size
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds

        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CachedResponse] = OrderedDict()
        self._pending_requests: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._memory_usage_bytes = 0
        self._evictions = 0
        self._timeouts_prevented = 0
        self._validation_failures = 0
        self._total_cache_attempts = 0

        # Circuit breaker for validation failures
        self._validation_failure_threshold = 10  # Disable caching after 10 consecutive failures
        self._validation_failure_reset_time = 300  # Reset after 5 minutes
        self._consecutive_validation_failures = 0
        self._caching_disabled_until = 0

        # Start cleanup task
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
                warning(LogRecord(
                    LogEvent.PARAMETER_UNSUPPORTED.value,
                    f"Error in cache cleanup: {str(e)}",
                    None,
                    {"error": str(e)}
                ))

    async def _cleanup_expired(self):
        """Remove expired entries from cache."""
        async with self._lock:
            current_time = time.time()
            expired_keys = []

            for key, cached in self._cache.items():
                if current_time - cached.timestamp > self._ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                cached = self._cache.pop(key)
                self._memory_usage_bytes -= cached.size_bytes
                debug(LogRecord(
                    LogEvent.PARAMETER_UNSUPPORTED.value,
                    "Expired cache entry removed",
                    None,
                    {"key": key[:8], "age_seconds": current_time - cached.timestamp}
                ))

    def _generate_cache_key(self, request: MessagesRequest) -> str:
        """Generate a unique cache key for the request."""
        # Create a deterministic string representation
        request_dict = request.model_dump(exclude_unset=True)
        request_str = json.dumps(request_dict, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()

    def _is_caching_disabled(self) -> bool:
        """
        Check if caching is temporarily disabled due to validation failures.

        Returns:
            True if caching is disabled, False otherwise
        """
        current_time = time.time()

        # Check if we're still in the disabled period
        if current_time < self._caching_disabled_until:
            return True

        # Reset circuit breaker if we were previously disabled but enough time has passed
        if self._caching_disabled_until > 0 and current_time >= self._caching_disabled_until:
            self._consecutive_validation_failures = 0
            self._caching_disabled_until = 0

        return False

    def _validate_response_for_caching(self, response: MessagesResponse) -> bool:
        """
        Validate that a response is safe to cache.

        Checks for malformed responses that could cause issues when retrieved from cache.

        Args:
            response: The response to validate

        Returns:
            True if response is valid for caching, False otherwise
        """
        try:
            # Test JSON serialization - this will catch malformed responses
            response_json = response.model_dump_json()

            # Try to parse it back to ensure it's valid JSON
            parsed = json.loads(response_json)

            # Basic structure validation
            if not isinstance(parsed, dict):
                warning(LogRecord(
                    LogEvent.PARAMETER_UNSUPPORTED.value,
                    "Response validation failed: not a dictionary",
                    None,
                    {"response_type": type(parsed).__name__}
                ))
                return False

            # Check for essential fields
            required_fields = ["id", "content", "model", "stop_reason", "usage"]
            missing_fields = [field for field in required_fields if field not in parsed]
            if missing_fields:
                warning(LogRecord(
                    LogEvent.PARAMETER_UNSUPPORTED.value,
                    "Response validation failed: missing required fields",
                    None,
                    {"missing_fields": missing_fields}
                ))
                return False

            # Validate text content can be properly encoded/decoded
            if parsed.get("content") and isinstance(parsed["content"], list):
                for content_block in parsed["content"]:
                    if isinstance(content_block, dict) and "text" in content_block:
                        text_content = content_block["text"]
                        if isinstance(text_content, str):
                            # Test UTF-8 encoding/decoding
                            try:
                                text_content.encode('utf-8').decode('utf-8')
                            except UnicodeDecodeError:
                                warning(LogRecord(
                                    LogEvent.PARAMETER_UNSUPPORTED.value,
                                    "Response validation failed: invalid UTF-8 content",
                                    None,
                                    {"content_type": "text"}
                                ))
                                return False

            return True

        except (TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
            warning(LogRecord(
                LogEvent.PARAMETER_UNSUPPORTED.value,
                "Response validation failed: JSON/encoding error",
                None,
                {"error": str(e), "error_type": type(e).__name__}
            ))
            return False
        except Exception as e:
            warning(LogRecord(
                LogEvent.PARAMETER_UNSUPPORTED.value,
                "Response validation failed: unexpected error",
                None,
                {"error": str(e), "error_type": type(e).__name__}
            ))
            return False

    async def get_cached_response(
        self,
        request: MessagesRequest,
        request_id: Optional[str] = None,
        wait_for_pending: bool = True,
        timeout_seconds: float = 30.0
    ) -> Optional[MessagesResponse]:
        """
        Get cached response for the request.

        Args:
            request: The request to look up
            request_id: Request ID for logging
            wait_for_pending: Whether to wait for pending requests
            timeout_seconds: Timeout for waiting on pending requests

        Returns:
            Cached response if available, None otherwise
        """
        cache_key = self._generate_cache_key(request)

        event: Optional[asyncio.Event] = None

        async with self._lock:
            # Check if response is already cached
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                current_time = time.time()

                # Check if cache entry is still valid
                if current_time - cached.timestamp <= self._ttl_seconds:
                    self._cache_hits += 1
                    cached.access_count += 1
                    cached.last_accessed = current_time

                    # Move to end for LRU
                    self._cache.move_to_end(cache_key)

                    info(LogRecord(
                        LogEvent.ANTHROPIC_REQUEST.value,
                        "Returning cached response",
                        request_id,
                        {
                            "cache_hit": True,
                            "cache_key": cache_key[:8],
                            "age_seconds": current_time - cached.timestamp,
                            "access_count": cached.access_count,
                            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses)
                        }
                    ))
                    return cached.response
                else:
                    # Remove expired entry
                    self._cache.pop(cache_key)
                    self._memory_usage_bytes -= cached.size_bytes

            # Check if request is pending
            if wait_for_pending and cache_key in self._pending_requests:
                event = self._pending_requests[cache_key]

        # Wait for pending request outside the lock
        if wait_for_pending and event:
            try:
                debug(LogRecord(
                    LogEvent.ANTHROPIC_REQUEST.value,
                    "Waiting for pending identical request",
                    request_id,
                    {"cache_key": cache_key[:8], "timeout": timeout_seconds}
                ))

                await asyncio.wait_for(event.wait(), timeout=timeout_seconds)

                # Try to get the response again after waiting
                async with self._lock:
                    if cache_key in self._cache:
                        cached = self._cache[cache_key]
                        self._timeouts_prevented += 1
                        info(LogRecord(
                            LogEvent.ANTHROPIC_REQUEST.value,
                            "Prevented timeout by using pending request's response",
                            request_id,
                            {"cache_key": cache_key[:8]}
                        ))
                        return cached.response
            except asyncio.TimeoutError:
                warning(LogRecord(
                    LogEvent.ANTHROPIC_REQUEST.value,
                    "Timeout waiting for pending request",
                    request_id,
                    {"cache_key": cache_key[:8]}
                ))

        self._cache_misses += 1
        return None

    async def mark_request_pending(self, request: MessagesRequest) -> str:
        """Mark a request as pending to prevent duplicate processing."""
        cache_key = self._generate_cache_key(request)

        async with self._lock:
            if cache_key not in self._pending_requests:
                self._pending_requests[cache_key] = asyncio.Event()

        return cache_key

    async def cache_response(
        self,
        request: MessagesRequest,
        response: MessagesResponse,
        request_id: Optional[str] = None
    ) -> bool:
        """
        Cache a response with memory management and validation.

        Args:
            request: The original request
            response: The response to cache
            request_id: Request ID for logging

        Returns:
            True if cached successfully, False otherwise
        """
        cache_key = self._generate_cache_key(request)
        self._total_cache_attempts += 1

        # Check if caching is temporarily disabled due to validation failures
        if self._is_caching_disabled():
            warning(LogRecord(
                LogEvent.PARAMETER_UNSUPPORTED.value,
                "Caching temporarily disabled due to validation failures",
                request_id,
                {
                    "cache_key": cache_key[:8],
                    "consecutive_failures": self._consecutive_validation_failures,
                    "disabled_until": self._caching_disabled_until
                }
            ))
            # Still clear pending requests to unblock waiting clients
            async with self._lock:
                if cache_key in self._pending_requests:
                    event = self._pending_requests.pop(cache_key)
                    event.set()
            return False

        # Validate response before caching to prevent malformed responses from being cached
        if not self._validate_response_for_caching(response):
            self._validation_failures += 1
            self._consecutive_validation_failures += 1

            # Trigger circuit breaker if too many consecutive failures
            if self._consecutive_validation_failures >= self._validation_failure_threshold:
                self._caching_disabled_until = time.time() + self._validation_failure_reset_time
                warning(LogRecord(
                    LogEvent.PARAMETER_UNSUPPORTED.value,
                    "Disabling caching due to consecutive validation failures",
                    request_id,
                    {
                        "consecutive_failures": self._consecutive_validation_failures,
                        "disabled_until": self._caching_disabled_until,
                        "threshold": self._validation_failure_threshold
                    }
                ))

            warning(LogRecord(
                LogEvent.PARAMETER_UNSUPPORTED.value,
                "Refusing to cache invalid response",
                request_id,
                {
                    "cache_key": cache_key[:8],
                    "reason": "response_validation_failed",
                    "consecutive_failures": self._consecutive_validation_failures
                }
            ))
            # Still clear pending requests to unblock waiting clients
            async with self._lock:
                if cache_key in self._pending_requests:
                    event = self._pending_requests.pop(cache_key)
                    event.set()
            return False

        # Reset consecutive failures on successful validation
        self._consecutive_validation_failures = 0

        async with self._lock:
            # Create cached response
            cached = CachedResponse(
                response=response,
                request_hash=cache_key,
                timestamp=time.time()
            )

            # Check memory constraints
            if self._memory_usage_bytes + cached.size_bytes > self._max_memory_bytes:
                await self._evict_entries(cached.size_bytes)

            # Check size constraints
            while len(self._cache) >= self._max_size:
                # Remove least recently used
                lru_key, lru_cached = self._cache.popitem(last=False)
                self._memory_usage_bytes -= lru_cached.size_bytes
                self._evictions += 1
                debug(LogRecord(
                    LogEvent.PARAMETER_UNSUPPORTED.value,
                    "Evicted LRU cache entry",
                    request_id,
                    {"evicted_key": lru_key[:8], "size_bytes": lru_cached.size_bytes}
                ))

            # Add to cache
            self._cache[cache_key] = cached
            self._memory_usage_bytes += cached.size_bytes

            # Signal any waiting requests
            if cache_key in self._pending_requests:
                event = self._pending_requests.pop(cache_key)
                event.set()

            info(LogRecord(
                LogEvent.ANTHROPIC_RESPONSE.value,
                "Response cached successfully",
                request_id,
                {
                    "cache_key": cache_key[:8],
                    "size_bytes": cached.size_bytes,
                    "cache_size": len(self._cache),
                    "memory_usage_mb": self._memory_usage_bytes / (1024 * 1024)
                }
            ))

            return True

    async def _evict_entries(self, required_bytes: int):
        """Evict entries to free up memory."""
        freed_bytes = 0
        current_time = time.time()

        # First, remove expired entries
        expired_keys = []
        for key, cached in self._cache.items():
            if current_time - cached.timestamp > self._ttl_seconds:
                expired_keys.append(key)
                freed_bytes += cached.size_bytes
                if freed_bytes >= required_bytes:
                    break

        for key in expired_keys:
            cached = self._cache.pop(key)
            self._memory_usage_bytes -= cached.size_bytes
            self._evictions += 1

        # If still need more space, remove least recently accessed
        while freed_bytes < required_bytes and self._cache:
            # Find least recently accessed entry
            lra_key = min(self._cache.keys(),
                         key=lambda k: self._cache[k].last_accessed)
            lra_cached = self._cache.pop(lra_key)
            self._memory_usage_bytes -= lra_cached.size_bytes
            freed_bytes += lra_cached.size_bytes
            self._evictions += 1

    async def clear_pending_request(self, request: MessagesRequest):
        """Clear a pending request without caching (e.g., on error)."""
        cache_key = self._generate_cache_key(request)

        async with self._lock:
            if cache_key in self._pending_requests:
                event = self._pending_requests.pop(cache_key)
                event.set()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including validation failure metrics."""
        total_requests = self._cache_hits + self._cache_misses
        return {
            "cache_size": len(self._cache),
            "max_size": self._max_size,
            "memory_usage_mb": round(self._memory_usage_bytes / (1024 * 1024), 2),
            "max_memory_mb": self._max_memory_bytes / (1024 * 1024),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, total_requests),
            "evictions": self._evictions,
            "timeouts_prevented": self._timeouts_prevented,
            "pending_requests": len(self._pending_requests),
            "total_requests": total_requests,
            "validation_failures": self._validation_failures,
            "total_cache_attempts": self._total_cache_attempts,
            "validation_failure_rate": self._validation_failures / max(1, self._total_cache_attempts),
            "consecutive_validation_failures": self._consecutive_validation_failures,
            "caching_disabled": self._is_caching_disabled(),
            "caching_disabled_until": self._caching_disabled_until if self._caching_disabled_until > time.time() else None
        }

    async def clear(self):
        """Clear all cached responses."""
        async with self._lock:
            self._cache.clear()
            self._pending_requests.clear()
            self._memory_usage_bytes = 0
            info(LogRecord(
                LogEvent.PARAMETER_UNSUPPORTED.value,
                "Cache cleared",
                None,
                {"previous_size": len(self._cache)}
            ))


# Global response cache instance
response_cache = ResponseCache(
    max_size=1000,
    max_memory_mb=500,
    ttl_seconds=3600,
    cleanup_interval_seconds=300
)
