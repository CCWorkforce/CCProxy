"""
Optimized OpenAI provider with high-performance HTTP client configuration.
Supports multiple HTTP client backends for maximum performance.
"""

from openai import AsyncOpenAI
from openai import DefaultAioHttpClient
from openai import DefaultAsyncHttpxClient
from typing import Any, Optional, Dict, List
import httpx
import os
import logging
import asyncio
import random
import openai
import time
import uuid
import statistics
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ...config import Settings
from ...monitoring import PerformanceMonitor
from ...application.error_tracker import ErrorTracker, ErrorType


def _safe_decode_response(response_bytes: bytes, context: str = "API response") -> str:
    """
    Safely decode response bytes to UTF-8 string with error handling.

    Args:
        response_bytes: Raw bytes from HTTP response
        context: Description of the response context for logging

    Returns:
        Decoded UTF-8 string

    Raises:
        UnicodeDecodeError: If decoding fails even with error handling
    """
    try:
        # First try strict UTF-8 decoding
        return response_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        # Log the issue
        logging.warning(
            f"Malformed UTF-8 bytes detected in {context}. "
            f"Attempting recovery with byte replacement. Error: {str(e)}"
        )

        try:
            # Try with error replacement - replaces malformed bytes with replacement character
            decoded = response_bytes.decode("utf-8", errors="replace")

            # Log successful recovery
            logging.info(
                f"Successfully recovered {context} by replacing malformed UTF-8 bytes"
            )
            return decoded
        except Exception as recovery_error:
            # If even replacement fails, raise the original error
            logging.error(
                f"Failed to recover {context} even with byte replacement. "
                f"Recovery error: {str(recovery_error)}"
            )
            raise e


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ProviderMetrics:
    """Detailed provider metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    avg_latency_ms: float = 0
    p95_latency_ms: float = 0
    p99_latency_ms: float = 0
    tokens_processed: int = 0
    circuit_state: str = CircuitState.CLOSED.value
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    health_score: float = 100.0  # 0-100 scale


class CircuitBreaker:
    """Circuit breaker for OpenAI API calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, half_open_requests: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout  # seconds
        self.half_open_requests = half_open_requests
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_successes = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
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

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self.consecutive_failures = 0
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_requests:
                    self.state = CircuitState.CLOSED
                    logging.info("Circuit breaker CLOSED - service recovered")

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.consecutive_failures += 1
            self.last_failure_time = datetime.now()

            if self.consecutive_failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logging.warning(f"Circuit breaker OPEN - {self.consecutive_failures} consecutive failures")
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logging.warning("Circuit breaker OPEN - failed during recovery test")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time:
            time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
            return time_since_failure >= self.recovery_timeout
        return False

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN


class OpenAIProvider:
    """
    High-performance OpenAI provider with monitoring and resilience.

    Key features:
    - HTTP/2 support for multiplexing
    - Circuit breaker for failure protection
    - Comprehensive metrics collection
    - Request/response logging with correlation IDs
    - Adaptive timeout based on performance
    - Health monitoring and scoring
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._openAIClient: Optional[AsyncOpenAI] = None
        self._max_retries = settings.provider_max_retries
        self._base_delay = settings.provider_retry_base_delay
        self._jitter = settings.provider_retry_jitter
        self._http_client = None  # Will be managed by OpenAI SDK

        # Monitoring and resilience
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            half_open_requests=3
        )
        self._performance_monitor = PerformanceMonitor()
        self._error_tracker = ErrorTracker()  # Singleton, no parameters
        self._metrics = ProviderMetrics()
        self._request_log: Dict[str, Dict[str, Any]] = {}  # Correlation ID -> request details
        self._latency_history: List[float] = []  # For adaptive timeout
        self._metrics_lock = asyncio.Lock()

        try:
            # High-performance configuration
            _read_timeout = float(self.settings.max_stream_seconds)

            # Check deployment environment
            is_local = os.getenv("IS_LOCAL_DEPLOYMENT", "False").lower() == "true"

            if is_local:
                # Use httpx for local development (better debugging)
                http_client_kwargs = {
                    "limits": httpx.Limits(
                        max_keepalive_connections=50,
                        max_connections=500,
                        keepalive_expiry=120,
                    ),
                    "timeout": httpx.Timeout(
                        connect=10.0,
                        read=_read_timeout,
                        write=30.0,
                        pool=10.0,
                    ),
                    "verify": os.getenv("SSL_CERT_FILE", True),
                    "follow_redirects": True,
                }

                try:
                    # Try with HTTP/2
                    self._http_client = DefaultAsyncHttpxClient(
                        **http_client_kwargs,
                        http2=True
                    )
                    logging.info("Using DefaultAsyncHttpxClient with HTTP/2 for local deployment")
                except ImportError:
                    # Fallback without HTTP/2
                    self._http_client = DefaultAsyncHttpxClient(**http_client_kwargs)
                    logging.info("Using DefaultAsyncHttpxClient (HTTP/1.1) for local deployment")
            else:
                # Use aiohttp for production (better performance under high concurrency)
                # Note: DefaultAioHttpClient requires the openai SDK to be installed with aiohttp extra
                # Try to use aiohttp client if available, fallback to httpx with optimized settings
                try:
                    # Try to create aiohttp client - will fail if aiohttp extra not installed
                    self._http_client = DefaultAioHttpClient()
                    logging.info("Using DefaultAioHttpClient for production deployment")
                except (ImportError, RuntimeError):
                    # Fallback to optimized httpx client for production
                    # RuntimeError is raised when aiohttp extra is not installed
                    # Try with HTTP/2 first, fallback without if h2 not available
                    http_client_kwargs = {
                        "limits": httpx.Limits(
                            max_keepalive_connections=100,  # Higher for production
                            max_connections=300,  # Reasonable limit to avoid resource exhaustion
                            keepalive_expiry=120,
                        ),
                        "timeout": httpx.Timeout(
                            connect=10.0,
                            read=_read_timeout,
                            write=30.0,
                            pool=10.0,
                        ),
                        "verify": os.getenv("SSL_CERT_FILE", True),
                        "follow_redirects": True,
                    }

                    try:
                        # Try with HTTP/2
                        self._http_client = DefaultAsyncHttpxClient(
                            **http_client_kwargs,
                            http2=True
                        )
                        logging.info(
                            "Using optimized DefaultAsyncHttpxClient with HTTP/2 for production. "
                            "For better performance, install aiohttp: pip install 'openai[aiohttp]'"
                        )
                    except ImportError:
                        # Fallback without HTTP/2
                        self._http_client = DefaultAsyncHttpxClient(**http_client_kwargs)
                        logging.info(
                            "Using optimized DefaultAsyncHttpxClient (HTTP/1.1) for production. "
                            "Install h2 for HTTP/2: pip install 'httpx[http2]'. "
                            "For better performance, install aiohttp: pip install 'openai[aiohttp]'"
                        )

            # Initialize OpenAI client with the appropriate HTTP client
            self._openAIClient = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.base_url,
                default_headers={
                    "HTTP-Referer": self.settings.referer_url,
                    "X-Title": self.settings.app_name,
                    "Accept-Charset": "utf-8",
                },
                timeout=_read_timeout,
                http_client=self._http_client,
                max_retries=self._max_retries,
            )
        except Exception:
            # Clean up if initialization fails
            # The OpenAI SDK clients handle their own cleanup
            self._http_client = None
            raise

    async def _execute_with_retry(self, func: Any, *args: Any, **kwargs: Any) -> Any:
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
                if attempt >= self._max_retries:
                    raise e
                delay = self._base_delay * (2**attempt) + random.uniform(0, self._jitter)
                logging.debug(f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt + 1}/{self._max_retries})")
                await asyncio.sleep(delay)
                attempt += 1
            except (
                openai.APIConnectionError,
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as e:
                if attempt >= self._max_retries:
                    raise e
                delay = self._base_delay * (2**attempt) + random.uniform(0, self._jitter)
                logging.debug(f"Network error, retrying in {delay:.2f}s (attempt {attempt + 1}/{self._max_retries}): {e}")
                await asyncio.sleep(delay)
                attempt += 1

    async def create_chat_completion(self, **params: Any) -> Any:
        """Create a chat completion with monitoring and resilience.

        Args:
            **params: Parameters to pass to the OpenAI chat completion API

        Returns:
            The API response from OpenAI (regular object for non-streaming,
            async generator for streaming)

        Raises:
            openai.APIError: If there are issues with the API call after retries
            Exception: If circuit breaker is open
        """

        if not self._openAIClient:
            raise ValueError("OpenAI client not properly initialized")

        # Generate correlation ID for request tracking
        correlation_id = str(uuid.uuid4())
        request_start = time.monotonic()

        # Log request
        self._log_request(correlation_id, params)

        # Start performance monitoring
        await self._performance_monitor.start_request(correlation_id)

        try:
            # Update metrics
            async with self._metrics_lock:
                self._metrics.total_requests += 1

            # Check circuit breaker
            if self._circuit_breaker.is_open:
                logging.warning(f"Request {correlation_id} blocked by open circuit breaker")
                async with self._metrics_lock:
                    self._metrics.failed_requests += 1
                raise Exception("Service temporarily unavailable - circuit breaker is open")

            # Handle streaming case separately
            stream = params.get('stream', False)
            if stream:
                # For streaming, use circuit breaker with retry logic
                response = await self._circuit_breaker.call(
                    self._execute_with_retry,
                    self._openAIClient.chat.completions.create,
                    **params
                )
                await self._on_request_success(correlation_id, request_start)
                return response

            # Non-streaming case with additional UTF-8 handling
            response = await self._circuit_breaker.call(
                self._execute_with_retry,
                self._openAIClient.chat.completions.create,
                **params
            )

            # Decode bytes if needed
            if hasattr(response, 'choices') and response.choices:
                for choice in response.choices:
                    if (hasattr(choice, 'message') and
                        hasattr(choice.message, 'content') and
                        isinstance(choice.message.content, bytes)):
                        choice.message.content = _safe_decode_response(
                            choice.message.content,
                            "chat completion response"
                        )

            # Update metrics and log success
            await self._on_request_success(correlation_id, request_start, response)
            return response

        except UnicodeDecodeError as e:
            await self._on_request_failure(correlation_id, request_start, e, ErrorType.CONVERSION_ERROR)
            raise ValueError(
                f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}"
            ) from e
        except openai.RateLimitError as e:
            await self._on_request_failure(correlation_id, request_start, e, ErrorType.RATE_LIMIT_ERROR)
            raise
        except openai.AuthenticationError as e:
            await self._on_request_failure(correlation_id, request_start, e, ErrorType.AUTH_ERROR)
            raise
        except Exception as e:
            error_type = ErrorType.API_ERROR
            if "utf-8" in str(e).lower() or "codec can't decode" in str(e).lower():
                error_type = ErrorType.CONVERSION_ERROR
            elif "timeout" in str(e).lower():
                error_type = ErrorType.TIMEOUT_ERROR

            await self._on_request_failure(correlation_id, request_start, e, error_type)

            if error_type == ErrorType.CONVERSION_ERROR:
                raise ValueError(
                    f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}"
                ) from e
            raise

    def _log_request(self, correlation_id: str, params: Dict[str, Any]) -> None:
        """Log request with correlation ID."""
        self._request_log[correlation_id] = {
            "timestamp": datetime.now().isoformat(),
            "params": {k: v for k, v in params.items() if k not in ["api_key", "messages"]},
            "model": params.get("model", "unknown")
        }
        logging.debug(f"Request {correlation_id}: model={params.get('model')}, stream={params.get('stream', False)}")

    async def _on_request_success(self, correlation_id: str, request_start: float, response: Any = None) -> None:
        """Handle successful request - update metrics and monitoring."""
        latency_ms = (time.monotonic() - request_start) * 1000

        # Update performance monitor
        await self._performance_monitor.end_request(correlation_id, success=True)

        # Update metrics
        async with self._metrics_lock:
            self._metrics.successful_requests += 1
            self._metrics.total_latency_ms += latency_ms
            self._metrics.avg_latency_ms = self._metrics.total_latency_ms / max(self._metrics.successful_requests, 1)

            # Track latency for percentiles and adaptive timeout
            self._latency_history.append(latency_ms)
            if len(self._latency_history) > 1000:  # Keep last 1000 requests
                self._latency_history = self._latency_history[-1000:]

            # Calculate percentiles
            if len(self._latency_history) >= 10:
                sorted_latencies = sorted(self._latency_history)
                self._metrics.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
                self._metrics.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]

            # Update health score (100 = perfect, 0 = dead)
            self._update_health_score()

            # Count tokens if available
            if response and hasattr(response, 'usage'):
                self._metrics.tokens_processed += getattr(response.usage, 'total_tokens', 0)

        # Clean up request log
        if correlation_id in self._request_log:
            del self._request_log[correlation_id]

        logging.debug(f"Request {correlation_id} succeeded in {latency_ms:.2f}ms")

    async def _on_request_failure(self, correlation_id: str, request_start: float, error: Exception, error_type: ErrorType) -> None:
        """Handle failed request - update metrics and error tracking."""
        latency_ms = (time.monotonic() - request_start) * 1000

        # Update performance monitor
        await self._performance_monitor.end_request(correlation_id, success=False)

        # Track error (ErrorTracker.track_error doesn't have a context parameter)
        await self._error_tracker.track_error(
            error=error,
            error_type=error_type
        )

        # Update metrics
        async with self._metrics_lock:
            self._metrics.failed_requests += 1
            self._metrics.consecutive_failures = self._circuit_breaker.consecutive_failures
            self._metrics.last_failure_time = datetime.now()
            self._metrics.circuit_state = self._circuit_breaker.state.value

            # Update health score
            self._update_health_score()

        # Clean up request log
        if correlation_id in self._request_log:
            del self._request_log[correlation_id]

        logging.error(f"Request {correlation_id} failed after {latency_ms:.2f}ms: {error}")

    def _update_health_score(self) -> None:
        """Update health score based on metrics (call with lock held)."""
        if self._metrics.total_requests == 0:
            self._metrics.health_score = 100.0
            return

        # Calculate base score from success rate
        success_rate = self._metrics.successful_requests / self._metrics.total_requests
        base_score = success_rate * 100

        # Apply penalties
        penalties = 0

        # Circuit breaker state penalty
        if self._circuit_breaker.state == CircuitState.OPEN:
            penalties += 50
        elif self._circuit_breaker.state == CircuitState.HALF_OPEN:
            penalties += 25

        # High latency penalty (if p99 > 5 seconds)
        if self._metrics.p99_latency_ms > 5000:
            penalties += 20

        # Recent failures penalty
        if self._metrics.last_failure_time:
            minutes_since_failure = (datetime.now() - self._metrics.last_failure_time).total_seconds() / 60
            if minutes_since_failure < 1:
                penalties += 15
            elif minutes_since_failure < 5:
                penalties += 10

        self._metrics.health_score = max(0, min(100, base_score - penalties))

    async def get_metrics(self) -> ProviderMetrics:
        """Get current metrics snapshot."""
        async with self._metrics_lock:
            # Update circuit state
            self._metrics.circuit_state = self._circuit_breaker.state.value
            return ProviderMetrics(**self._metrics.__dict__)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        metrics = await self.get_metrics()

        # Determine health status
        if metrics.health_score >= 80:
            status = "healthy"
        elif metrics.health_score >= 50:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "health_score": metrics.health_score,
            "circuit_breaker": metrics.circuit_state,
            "success_rate": metrics.successful_requests / max(metrics.total_requests, 1),
            "avg_latency_ms": metrics.avg_latency_ms,
            "p99_latency_ms": metrics.p99_latency_ms,
            "active_requests": self._performance_monitor.metrics.active_requests,
            "tokens_processed": metrics.tokens_processed
        }

    def get_adaptive_timeout(self) -> float:
        """Calculate adaptive timeout based on recent performance."""
        if not self._latency_history:
            return float(self.settings.max_stream_seconds)

        # Use p95 latency * 2 as timeout, with min/max bounds
        p95_latency = statistics.quantiles(self._latency_history, n=20)[18]  # 95th percentile
        adaptive_timeout_s = min(
            float(self.settings.max_stream_seconds),
            max(10.0, p95_latency * 2 / 1000)  # Convert ms to seconds
        )

        logging.debug(f"Adaptive timeout: {adaptive_timeout_s:.2f}s (based on p95={p95_latency:.2f}ms)")
        return adaptive_timeout_s

    async def close(self) -> None:
        """Clean up the HTTP client when shutting down."""
        # Properly close HTTP clients to avoid resource leaks
        if self._http_client:
            try:
                if hasattr(self._http_client, 'aclose'):
                    # For httpx-based clients
                    await self._http_client.aclose()
                elif hasattr(self._http_client, '_session') and hasattr(self._http_client._session, 'close'):
                    # For aiohttp-based clients
                    await self._http_client._session.close()
            except Exception as e:
                logging.warning(f"Error closing HTTP client: {e}")
            finally:
                self._http_client = None

        self._openAIClient = None

    async def __aenter__(self) -> "OpenAIProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
