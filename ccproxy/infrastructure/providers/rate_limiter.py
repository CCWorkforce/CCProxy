"""Client-side rate limiter for proactive OpenAI API rate limiting."""

import anyio
import time
import logging
from typing import Optional, Dict, Any
from ccproxy.application.tokenizer import count_tokens_for_openai_request
from dataclasses import dataclass
from enum import Enum

from ...application.thread_pool import asyncify


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
    recovery_multiplier: float = 1.1  # Increase limits by 10% during recovery


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

    async def acquire(self, request_payload: Optional[Dict[str, Any]] = None, priority: int = 0) -> bool:
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

            if request_payload is not None:
                try:
                    model = request_payload.get("model", self.config.model_name)
                    messages = request_payload.get("messages", [])
                    tools = request_payload.get("tools", request_payload.get("functions", []))
                    estimated_tokens = await count_tokens_for_openai_request(
                        messages, model_name=model, tools=tools, request_id=None
                    )
                except Exception as e:
                    logging.warning(f"Token estimation failed: {e}, using rough estimate")
                    messages = request_payload.get("messages", [])
                    total_chars = 0
                    for msg in messages:
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            total_chars += len(content)
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    total_chars += len(part.get("text", ""))
                    estimated_tokens = max(1, total_chars // 4)
            else:
                estimated_tokens = 0

            # Offload list cleaning to thread pool for large lists
            async_clean_requests = asyncify(
                lambda times, cutoff: [t for t in times if current_time - t < cutoff]
            )
            async_clean_tokens = asyncify(
                lambda counts, cutoff: [
                    (t, c) for t, c in counts if current_time - t < cutoff
                ]
            )

            # Clean old entries (older than 1 minute) in parallel
            self._request_times = await async_clean_requests(self._request_times, 60)
            self._token_counts = await async_clean_tokens(self._token_counts, 60)

            # Calculate current rates using async operations
            async_sum_tokens = asyncify(lambda counts: sum(c for _, c in counts))

            # Calculate current rates
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
            Dictionary of metrics
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
