"""Client-side rate limiter for proactive OpenAI API rate limiting."""

import asyncio
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 500
    tokens_per_minute: int = 90000
    burst_size: int = 100
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
        self._request_semaphore = asyncio.Semaphore(config.burst_size)
        self._token_semaphore = asyncio.Semaphore(config.burst_size * 1000)  # Rough token burst

        # Rate tracking
        self._request_times: list[float] = []
        self._token_counts: list[tuple[float, int]] = []  # (timestamp, token_count)

        # Adaptive limits (can be adjusted based on 429 responses)
        self._current_rpm_limit = config.requests_per_minute
        self._current_tpm_limit = config.tokens_per_minute

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Background task for refilling buckets
        self._refill_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the rate limiter background tasks."""
        if not self._running:
            self._running = True
            self._refill_task = asyncio.create_task(self._refill_buckets())
            logging.info(f"Client rate limiter started with {self._current_rpm_limit} RPM, {self._current_tpm_limit} TPM")

    async def stop(self) -> None:
        """Stop the rate limiter background tasks."""
        self._running = False
        if self._refill_task:
            self._refill_task.cancel()
            try:
                await self._refill_task
            except asyncio.CancelledError:
                pass
            self._refill_task = None
            logging.info("Client rate limiter stopped")

    async def acquire(self, estimated_tokens: int = 0, priority: int = 0) -> bool:
        """
        Acquire permission to make an API request.

        Args:
            estimated_tokens: Estimated tokens for the request
            priority: Priority level (higher = more important)

        Returns:
            True if request can proceed, False if rate limited
        """
        async with self._lock:
            current_time = time.time()

            # Clean old entries (older than 1 minute)
            self._request_times = [t for t in self._request_times if current_time - t < 60]
            self._token_counts = [(t, c) for t, c in self._token_counts if current_time - t < 60]

            # Calculate current rates
            self.metrics.current_rpm = len(self._request_times)
            self.metrics.current_tpm = sum(c for _, c in self._token_counts)

            # Check if we're within limits
            if self.metrics.current_rpm >= self._current_rpm_limit:
                self.metrics.rejected_requests += 1
                logging.debug(f"Rate limit: RPM limit reached ({self.metrics.current_rpm}/{self._current_rpm_limit})")
                return False

            if estimated_tokens > 0 and self.metrics.current_tpm + estimated_tokens > self._current_tpm_limit:
                self.metrics.rejected_requests += 1
                logging.debug(f"Rate limit: TPM limit reached ({self.metrics.current_tpm + estimated_tokens}/{self._current_tpm_limit})")
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

                self._current_rpm_limit = int(self._current_rpm_limit * self.config.backoff_multiplier)
                self._current_tpm_limit = int(self._current_tpm_limit * self.config.backoff_multiplier)

                # Ensure minimum limits
                self._current_rpm_limit = max(10, self._current_rpm_limit)
                self._current_tpm_limit = max(1000, self._current_tpm_limit)

                logging.warning(
                    f"Rate limit hit! Reducing limits: RPM {old_rpm} → {self._current_rpm_limit}, "
                    f"TPM {old_tpm} → {self._current_tpm_limit}"
                )

        if retry_after:
            await asyncio.sleep(retry_after)

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
                        int(self._current_rpm_limit * self.config.recovery_multiplier)
                    )
                    logging.info(f"Rate limit recovery: RPM {old_rpm} → {self._current_rpm_limit}")

                if self._current_tpm_limit < self.config.tokens_per_minute:
                    old_tpm = self._current_tpm_limit
                    self._current_tpm_limit = min(
                        self.config.tokens_per_minute,
                        int(self._current_tpm_limit * self.config.recovery_multiplier)
                    )
                    logging.info(f"Rate limit recovery: TPM {old_tpm} → {self._current_tpm_limit}")

                self.metrics.consecutive_successes = 0

    async def _refill_buckets(self) -> None:
        """Background task to refill token buckets."""
        while self._running:
            try:
                await asyncio.sleep(1)  # Refill every second

                # Calculate refill amounts (per second)
                # TODO: Implement actual bucket refill logic
                # requests_per_second = self._current_rpm_limit / 60
                # tokens_per_second = self._current_tpm_limit / 60

                # Note: In a real implementation, we'd need more sophisticated
                # bucket management here. This is simplified for demonstration.

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in rate limiter refill task: {e}")

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
                    logging.debug(f"Rate limiter waiting {wait_time:.2f}s to respect RPM limit")
                    await asyncio.sleep(wait_time)