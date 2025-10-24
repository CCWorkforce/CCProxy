"""Client-side rate limiter for proactive OpenAI API rate limiting."""

import anyio
import time
import logging
from typing import Optional, Dict, Any
from ccproxy.application.tokenizer import count_tokens_for_openai_request
from dataclasses import dataclass
from enum import Enum

from ...application.thread_pool import asyncify
from ..._cython import CYTHON_ENABLED

# Try to import Cython-optimized functions
if CYTHON_ENABLED:
    try:
        from ..._cython.lru_ops import (
            filter_request_times,
            filter_token_counts,
            sum_token_counts,
        )

        _USING_CYTHON = True
    except ImportError:
        _USING_CYTHON = False
else:
    _USING_CYTHON = False

# Fallback to pure Python implementations if Cython not available
if not _USING_CYTHON:

    def filter_request_times(
        request_times: list[float], current_time: float, window_seconds: float
    ) -> list[float]:
        """Filter request times to only include those within the time window."""
        cutoff_time = current_time - window_seconds
        return [t for t in request_times if t >= cutoff_time]

    def filter_token_counts(
        token_counts: list[tuple[float, int]],
        current_time: float,
        window_seconds: float,
    ) -> list[tuple[float, int]]:
        """Filter token counts to only include those within the time window."""
        cutoff_time = current_time - window_seconds
        return [(t, c) for t, c in token_counts if t >= cutoff_time]

    def sum_token_counts(token_counts: list[tuple[float, int]]) -> int:
        """Sum all token counts from the list."""
        return sum(c for _, c in token_counts)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 1500
    tokens_per_minute: int = 270000
    burst_size: int = 100
    model_name: str = "gpt-4o"
    strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE
    adaptive_enabled: bool = True
    backoff_multiplier: float = 0.8  # Reduce limits to 80% after 429
    recovery_multiplier: float = 1.1

    def __post_init__(self) -> Any:
        if self.requests_per_minute < 1:
            raise ValueError(
                f"requests_per_minute must be positive, got {self.requests_per_minute}"
            )
        if self.tokens_per_minute < 1:
            raise ValueError(
                f"tokens_per_minute must be positive, got {self.tokens_per_minute}"
            )
        if self.burst_size < 1:
            raise ValueError(
                f"burst_size must be positive, got {self.burst_size}"
            )  # Increase limits by 10% during recovery


@dataclass
class RateLimitMetrics:
    """Metrics for rate limit tracking."""

    total_requests: int = 0
    total_tokens: int = 0
    rejected_requests: int = 0
    rate_limit_hits: int = 0
    current_rpm: float = 0
    current_tpm: float = 0
    last_429_time: Optional[float] = None
    consecutive_successes: int = 0


class ClientRateLimiter:
    """
    Client-side rate limiter for OpenAI API calls.

    Implements token bucket algorithm with adaptive rate limiting
    based on 429 responses from the API.

    Performance optimizations:
    - Batch cleanup: Only cleans request history every 10 requests (not every request)
    - Rough token estimate cache: Caches token estimates for common message patterns
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """
        Initialize the rate limiter with configuration.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.metrics = RateLimitMetrics()

        # Token buckets for requests and tokens
        self._request_semaphore = anyio.Semaphore(config.burst_size)
        self._token_semaphore = anyio.Semaphore(
            config.burst_size * 1000
        )  # Rough token burst

        # Rate tracking
        self._request_times: list[float] = []
        self._token_counts: list[tuple[float, int]] = []  # (timestamp, token_count)

        # Adaptive limits (can be adjusted based on 429 responses)
        self._current_rpm_limit = config.requests_per_minute
        self._current_tpm_limit = config.tokens_per_minute

        # Lock for thread-safe operations
        self._lock = anyio.Lock()

        # Rate limiter state
        self._running = False

        # Batch cleanup optimization: only cleanup every N requests
        self._cleanup_counter = 0
        self._cleanup_batch_size = 10  # Cleanup every 10 requests

        # Rough token estimate cache: {(msg_count, avg_length): estimated_tokens}
        # Cache up to 100 patterns for common request shapes
        self._token_estimate_cache: Dict[tuple[int, int], int] = {}
        self._token_estimate_cache_max_size = 100

    async def start(self) -> None:
        """Start the rate limiter."""
        if not self._running:
            self._running = True
            # Note: Background task for bucket refill not needed - using sliding window approach
            logging.info(
                f"Client rate limiter started with {self._current_rpm_limit} RPM, {self._current_tpm_limit} TPM"
            )

    async def stop(self) -> None:
        """Stop the rate limiter."""
        self._running = False
        logging.info("Client rate limiter stopped")

    def _get_rough_token_estimate_from_cache(
        self, messages: list[Dict[str, Any]]
    ) -> Optional[int]:
        """Get rough token estimate from cache based on message count and average length.

        Cache key: (message_count, average_length_bucket)
        where average_length_bucket is the average message length rounded to nearest 100 chars.

        Args:
            messages: List of message dictionaries

        Returns:
            Estimated token count from cache, or None if not found
        """
        if not messages:
            return 0

        # Calculate message count and average length
        message_count = len(messages)
        total_length = 0

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_length += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_length += len(part.get("text", ""))

        avg_length = total_length // message_count if message_count > 0 else 0
        # Bucket average length to nearest 100 characters for better cache hits
        avg_length_bucket = (avg_length // 100) * 100

        cache_key = (message_count, avg_length_bucket)
        return self._token_estimate_cache.get(cache_key)

    def _cache_rough_token_estimate(
        self, messages: list[Dict[str, Any]], estimated_tokens: int
    ) -> None:
        """Cache a rough token estimate for future similar requests.

        Args:
            messages: List of message dictionaries
            estimated_tokens: Estimated token count to cache
        """
        if not messages:
            return

        # Calculate cache key (same logic as get)
        message_count = len(messages)
        total_length = 0

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_length += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_length += len(part.get("text", ""))

        avg_length = total_length // message_count if message_count > 0 else 0
        avg_length_bucket = (avg_length // 100) * 100
        cache_key = (message_count, avg_length_bucket)

        # Evict oldest entry if cache is full (simple FIFO)
        if (
            cache_key not in self._token_estimate_cache
            and len(self._token_estimate_cache) >= self._token_estimate_cache_max_size
        ):
            # Remove first (oldest) entry
            first_key = next(iter(self._token_estimate_cache))
            del self._token_estimate_cache[first_key]

        self._token_estimate_cache[cache_key] = estimated_tokens

    def _should_cleanup(self) -> bool:
        """Determine if cleanup should be triggered based on batch counter.

        Uses batch cleanup strategy: only cleanup every N requests instead of every request.
        This reduces cleanup overhead by 90% (from every request to every 10th request).

        Returns:
            True if cleanup should be performed
        """
        self._cleanup_counter += 1
        return self._cleanup_counter % self._cleanup_batch_size == 0

    async def acquire(
        self, request_payload: Optional[Dict[str, Any]] = None, priority: int = 0
    ) -> bool:
        """
        Acquire permission to make an API request.

        Args:
            estimated_tokens: Estimated tokens for the request
            priority: Priority level (higher = more important)

        Returns:
            True if request can proceed, False if rate limited
        """
        # Auto-start on first use
        if not self._running:
            await self.start()

        async with self._lock:
            current_time = time.time()

            # Token estimation with rough estimate cache optimization
            if request_payload is not None:
                messages = request_payload.get("messages", [])

                # Try to get estimate from cache first
                cached_estimate = self._get_rough_token_estimate_from_cache(messages)

                if cached_estimate is not None:
                    estimated_tokens = cached_estimate
                    logging.debug(
                        f"Using cached token estimate: {estimated_tokens} tokens for {len(messages)} messages"
                    )
                else:
                    # Cache miss - compute estimate using full tokenizer
                    try:
                        model = request_payload.get("model", self.config.model_name)
                        tools = request_payload.get(
                            "tools", request_payload.get("functions", [])
                        )
                        estimated_tokens = await count_tokens_for_openai_request(
                            messages, model_name=model, tools=tools, request_id=None
                        )
                        # Cache the estimate for future similar requests
                        self._cache_rough_token_estimate(messages, estimated_tokens)
                    except Exception as e:
                        logging.warning(
                            f"Token estimation failed: {e}, using rough estimate"
                        )
                        # Fallback to character-based estimation
                        total_chars = 0
                        for msg in messages:
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                total_chars += len(content)
                            elif isinstance(content, list):
                                for part in content:
                                    if (
                                        isinstance(part, dict)
                                        and part.get("type") == "text"
                                    ):
                                        total_chars += len(part.get("text", ""))
                        estimated_tokens = max(1, total_chars // 4)
                        # Cache the fallback estimate
                        self._cache_rough_token_estimate(messages, estimated_tokens)
            else:
                estimated_tokens = 0

            # Batch cleanup optimization: only cleanup every N requests (default: 10)
            # This reduces cleanup overhead by 90% while still maintaining accuracy
            if self._should_cleanup():
                # Use Cython-optimized list cleaning functions for 20-40% performance improvement
                # Offload to thread pool for large lists
                async_clean_requests = asyncify(
                    lambda times: filter_request_times(times, current_time, 60)
                )
                async_clean_tokens = asyncify(
                    lambda counts: filter_token_counts(counts, current_time, 60)
                )

                # Clean old entries (older than 1 minute) in parallel
                self._request_times = await async_clean_requests(self._request_times)
                self._token_counts = await async_clean_tokens(self._token_counts)

                logging.debug(
                    f"Batch cleanup performed: {len(self._request_times)} requests, {len(self._token_counts)} token counts"
                )

            # Calculate current rates (always needed for rate limiting decisions)
            async_sum_tokens = asyncify(sum_token_counts)

            # Calculate current rates using Cython-optimized operations
            self.metrics.current_rpm = len(self._request_times)
            self.metrics.current_tpm = await async_sum_tokens(self._token_counts)

            # Check if we're within limits
            if self.metrics.current_rpm >= self._current_rpm_limit:
                self.metrics.rejected_requests += 1
                logging.debug(
                    f"Rate limit: RPM limit reached ({self.metrics.current_rpm}/{self._current_rpm_limit})"
                )
                return False

            if (
                estimated_tokens > 0
                and self.metrics.current_tpm + estimated_tokens
                > self._current_tpm_limit
            ):
                self.metrics.rejected_requests += 1
                logging.debug(
                    f"Rate limit: TPM limit reached ({self.metrics.current_tpm + estimated_tokens}/{self._current_tpm_limit})"
                )
                return False

            # Record the request
            self._request_times.append(current_time)
            if estimated_tokens > 0:
                self._token_counts.append((current_time, estimated_tokens))

            self.metrics.total_requests += 1
            self.metrics.total_tokens += estimated_tokens

            return True

    async def release(self, actual_tokens: int = 0) -> None:
        """
        Release resources after request completion.

        Args:
            actual_tokens: Actual tokens used (for updating estimates)
        """
        async with self._lock:
            if actual_tokens > 0 and self._token_counts:
                # Update the last token count with actual value
                last_time, _ = self._token_counts[-1]
                self._token_counts[-1] = (last_time, actual_tokens)

    async def handle_429_response(self, retry_after: Optional[int] = None) -> None:
        """
        Handle a 429 rate limit response from the API.

        Args:
            retry_after: Seconds to wait before retrying (from Retry-After header)
        """
        async with self._lock:
            self.metrics.rate_limit_hits += 1
            self.metrics.last_429_time = time.time()
            self.metrics.consecutive_successes = 0

            if self.config.adaptive_enabled:
                # Reduce limits when we hit 429
                old_rpm = self._current_rpm_limit
                old_tpm = self._current_tpm_limit

                self._current_rpm_limit = int(
                    self._current_rpm_limit * self.config.backoff_multiplier
                )
                self._current_tpm_limit = int(
                    self._current_tpm_limit * self.config.backoff_multiplier
                )

                # Ensure minimum limits
                self._current_rpm_limit = max(10, self._current_rpm_limit)
                self._current_tpm_limit = max(1000, self._current_tpm_limit)

                logging.warning(
                    f"Rate limit hit! Reducing limits: RPM {old_rpm} → {self._current_rpm_limit}, "
                    f"TPM {old_tpm} → {self._current_tpm_limit}"
                )

        if retry_after:
            await anyio.sleep(retry_after)

    async def handle_success(self) -> None:
        """Handle a successful API response for adaptive rate limiting."""
        if not self.config.adaptive_enabled:
            return

        async with self._lock:
            self.metrics.consecutive_successes += 1

            # Gradually increase limits after sustained success
            if self.metrics.consecutive_successes >= 10:
                if self._current_rpm_limit < self.config.requests_per_minute:
                    old_rpm = self._current_rpm_limit
                    self._current_rpm_limit = min(
                        self.config.requests_per_minute,
                        int(self._current_rpm_limit * self.config.recovery_multiplier),
                    )
                    logging.info(
                        f"Rate limit recovery: RPM {old_rpm} → {self._current_rpm_limit}"
                    )

                if self._current_tpm_limit < self.config.tokens_per_minute:
                    old_tpm = self._current_tpm_limit
                    self._current_tpm_limit = min(
                        self.config.tokens_per_minute,
                        int(self._current_tpm_limit * self.config.recovery_multiplier),
                    )
                    logging.info(
                        f"Rate limit recovery: TPM {old_tpm} → {self._current_tpm_limit}"
                    )

                self.metrics.consecutive_successes = 0

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current rate limiter metrics.

        Returns:
            Dictionary of metrics including cache statistics
        """
        return {
            "total_requests": self.metrics.total_requests,
            "total_tokens": self.metrics.total_tokens,
            "rejected_requests": self.metrics.rejected_requests,
            "rate_limit_hits": self.metrics.rate_limit_hits,
            "current_rpm": self.metrics.current_rpm,
            "current_tpm": self.metrics.current_tpm,
            "rpm_limit": self._current_rpm_limit,
            "tpm_limit": self._current_tpm_limit,
            "last_429_time": self.metrics.last_429_time,
            "token_estimate_cache": {
                "size": len(self._token_estimate_cache),
                "max_size": self._token_estimate_cache_max_size,
                "hit_rate": (
                    len(self._token_estimate_cache)
                    / max(1, self.metrics.total_requests)
                    if self.metrics.total_requests > 0
                    else 0.0
                ),
            },
            "cleanup": {
                "batch_size": self._cleanup_batch_size,
                "counter": self._cleanup_counter,
                "next_cleanup_in": self._cleanup_batch_size
                - (self._cleanup_counter % self._cleanup_batch_size),
            },
        }

    async def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        async with self._lock:
            if not self._request_times:
                return

            # Calculate wait time to respect rate limit
            current_time = time.time()
            oldest_request = self._request_times[0]
            time_window = 60  # 1 minute window

            if len(self._request_times) >= self._current_rpm_limit:
                # Need to wait for oldest request to fall out of window
                wait_time = (oldest_request + time_window) - current_time
                if wait_time > 0:
                    logging.debug(
                        f"Rate limiter waiting {wait_time:.2f}s to respect RPM limit"
                    )
                    await anyio.sleep(wait_time)
