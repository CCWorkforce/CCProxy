"""Circuit breaker for cache validation failures."""

import time

from ...constants import (
    DEFAULT_CACHE_VALIDATION_FAILURE_THRESHOLD,
    DEFAULT_CACHE_VALIDATION_FAILURE_RESET_TIME,
)
from ...logging import warning, LogRecord, LogEvent
from typing import Any


class CacheCircuitBreaker:
    """
    Circuit breaker pattern for handling cache validation failures.

    Temporarily disables caching when validation failures exceed threshold
    to prevent cascading failures and protect system stability.
    """

    def __init__(
        self,
        failure_threshold: int = DEFAULT_CACHE_VALIDATION_FAILURE_THRESHOLD,
        reset_time: int = DEFAULT_CACHE_VALIDATION_FAILURE_RESET_TIME,
    ):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.reset_time = reset_time
        self.consecutive_failures = 0
        self.disabled_until = 0
        self.max_consecutive_failures = failure_threshold * 10  # Cap at 10x threshold

    def is_open(self) -> bool:
        """Check if circuit breaker is open (caching disabled)."""
        current_time = time.time()

        # Check if we're in a disabled period
        if current_time < self.disabled_until:
            return True

        # Check if we've exceeded the failure threshold
        if self.consecutive_failures >= self.failure_threshold:
            # Cap the consecutive failures
            if self.consecutive_failures > self.max_consecutive_failures:
                self.consecutive_failures = self.max_consecutive_failures

            # Calculate disabled period with exponential backoff
            backoff_multiplier = min(
                self.consecutive_failures / self.failure_threshold, 10
            )
            disabled_duration = self.reset_time * backoff_multiplier
            self.disabled_until = current_time + disabled_duration  # type: ignore[assignment]

            warning(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Cache circuit breaker opened due to {self.consecutive_failures} validation failures",
                    request_id=None,
                    data={
                        "consecutive_failures": self.consecutive_failures,
                        "disabled_duration": disabled_duration,
                        "disabled_until": self.disabled_until,
                    },
                )
            )
            return True

        return False

    def record_success(self) -> Any:
        """Record a successful validation."""
        if self.consecutive_failures > 0:
            self.consecutive_failures = max(0, self.consecutive_failures - 1)
            if self.consecutive_failures == 0:
                self.disabled_until = 0

    def record_failure(self) -> Any:
        """Record a validation failure."""
        self.consecutive_failures += 1

        # Cap at maximum to prevent overflow
        if self.consecutive_failures > self.max_consecutive_failures:
            self.consecutive_failures = self.max_consecutive_failures

    def reset(self) -> Any:
        """Reset the circuit breaker."""
        self.consecutive_failures = 0
        self.disabled_until = 0

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        current_time = time.time()
        is_open = self.is_open()

        return {
            "is_open": is_open,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "disabled_until": self.disabled_until if is_open else None,
            "time_until_reset": max(0, self.disabled_until - current_time)
            if is_open
            else 0,
        }
