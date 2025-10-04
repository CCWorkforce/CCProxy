"""
Resilience patterns for provider implementations.
Includes circuit breaker and retry logic for handling failures gracefully.
"""

import anyio
import logging
import random
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

import httpx
import openai

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for API calls with automatic recovery."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_requests: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_requests: Successful requests needed to close circuit
        """
        self.failure_threshold = failure_threshold
        if self.failure_threshold < 1:
            raise ValueError(
                f"failure_threshold must be at least 1, got {self.failure_threshold}"
            )

        self.recovery_timeout = recovery_timeout
        if self.recovery_timeout < 1:
            raise ValueError(
                f"recovery_timeout must be at least 1, got {self.recovery_timeout}"
            )
        # seconds
        self.half_open_requests = half_open_requests
        if self.half_open_requests < 1:
            raise ValueError(
                f"half_open_requests must be at least 1, got {self.half_open_requests}"
            )

        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_successes = 0
        self._lock = anyio.Lock()

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from the function

        Raises:
            Exception: If circuit is open or function fails
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_successes = 0
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self.consecutive_failures = 0
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_requests:
                    self.state = CircuitState.CLOSED
                    logging.info("Circuit breaker CLOSED - service recovered")

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self.consecutive_failures += 1
            self.last_failure_time = datetime.now()

            if self.consecutive_failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logging.warning(
                    f"Circuit breaker OPEN - {self.consecutive_failures} consecutive failures"
                )
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logging.warning("Circuit breaker OPEN - failed during recovery test")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time:
            time_since_failure = (
                datetime.now() - self.last_failure_time
            ).total_seconds()
            return time_since_failure >= self.recovery_timeout
        return False

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN


class RetryHandler:
    """Handles retry logic with exponential backoff for transient failures."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        jitter: float = 0.1,
    ):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            jitter: Random jitter to add to delays (0-jitter seconds)
        """
        self.max_retries = max_retries
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )

        self.base_delay = base_delay
        self.jitter = jitter

    async def execute_with_retry(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute a function with exponential backoff retry logic.

        Args:
            func: The async function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result from the function

        Raises:
            The last exception if all retries fail
        """
        attempt = 0
        while True:
            try:
                return await func(*args, **kwargs)
            except openai.RateLimitError as e:
                if attempt >= self.max_retries:
                    raise e
                delay = self._calculate_delay(attempt)
                logging.debug(
                    f"Rate limit hit, retrying in {delay:.2f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                await anyio.sleep(delay)
                attempt += 1
            except (
                openai.APIConnectionError,
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as e:
                if attempt >= self.max_retries:
                    raise e
                delay = self._calculate_delay(attempt)
                logging.debug(
                    f"Network error, retrying in {delay:.2f}s "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                await anyio.sleep(delay)
                attempt += 1

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        return self.base_delay * (2**attempt) + random.uniform(0, self.jitter)


class ResilientExecutor:
    """
    Combines circuit breaker and retry logic for resilient execution.
    Provides a unified interface for executing functions with both patterns.
    """

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_handler: Optional[RetryHandler] = None,
    ):
        """
        Initialize resilient executor.

        Args:
            circuit_breaker: Optional circuit breaker instance
            retry_handler: Optional retry handler instance
        """
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.retry_handler = retry_handler or RetryHandler()

    async def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function with both circuit breaker and retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from the function

        Raises:
            Exception: If circuit is open or all retries fail
        """
        # Check circuit breaker first
        if self.circuit_breaker.is_open:
            logging.warning("Request blocked by open circuit breaker")
            raise Exception("Service temporarily unavailable - circuit breaker is open")

        # Execute with circuit breaker and retry
        return await self.circuit_breaker.call(
            self.retry_handler.execute_with_retry,
            func,
            *args,
            **kwargs,
        )
