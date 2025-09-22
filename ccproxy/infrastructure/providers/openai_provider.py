"""
Optimized OpenAI provider with high-performance HTTP client configuration.
Supports multiple HTTP client backends for maximum performance.
"""

from openai import AsyncOpenAI
from typing import Any, Optional, Dict, List
import logging
import time
import openai

from ...config import Settings
from ...monitoring import PerformanceMonitor
from ...application.error_tracker import ErrorTracker, ErrorType
from .rate_limiter import ClientRateLimiter, RateLimitConfig

from .resilience import CircuitBreaker, RetryHandler, ResilientExecutor
from .metrics import (
    ProviderMetrics,
    MetricsCollector,
    HealthMonitor,
    AdaptiveTimeoutCalculator,
)
from .http_client_factory import HttpClientFactory, HttpClientConfig
from .response_handlers import ResponseProcessor, ErrorResponseHandler
from .request_logger import RequestLogger, PerformanceTracker


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
        self._http_client = None  # Will be managed by OpenAI SDK

        # Initialize resilience components
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            recovery_timeout=settings.circuit_breaker_recovery_timeout,
            half_open_requests=settings.circuit_breaker_half_open_requests,
        )
        self._retry_handler = RetryHandler(
            max_retries=settings.provider_max_retries,
            base_delay=settings.provider_retry_base_delay,
            jitter=settings.provider_retry_jitter,
        )
        self._resilient_executor = ResilientExecutor(
            circuit_breaker=self._circuit_breaker,
            retry_handler=self._retry_handler,
        )

        # Initialize monitoring components
        self._metrics_collector = MetricsCollector()
        self._health_monitor = HealthMonitor(self._metrics_collector)
        self._timeout_calculator = AdaptiveTimeoutCalculator(
            self._metrics_collector,
            max_timeout=float(settings.max_stream_seconds),
        )
        self._performance_monitor = PerformanceMonitor()
        self._error_tracker = ErrorTracker()  # Singleton, no parameters

        # Initialize logging components
        self._request_logger = RequestLogger()
        self._performance_tracker = PerformanceTracker()

        # Initialize response processors
        self._response_processor = ResponseProcessor()
        self._error_handler = ErrorResponseHandler()

        # Initialize client-side rate limiter
        if settings.client_rate_limit_enabled:
            rate_limit_config = RateLimitConfig(
                requests_per_minute=settings.client_rate_limit_rpm,
                tokens_per_minute=settings.client_rate_limit_tpm,
                burst_size=settings.client_rate_limit_burst,
                adaptive_enabled=settings.client_rate_limit_adaptive,
            )
            self._rate_limiter = ClientRateLimiter(rate_limit_config)
            logging.info(
                f"Client rate limiter initialized: {settings.client_rate_limit_rpm} RPM, {settings.client_rate_limit_tpm} TPM"
            )
        else:
            self._rate_limiter = None
            logging.info("Client rate limiting disabled")

        try:
            # Create HTTP client using factory
            self._http_client = HttpClientFactory.create_client(settings)
            HttpClientConfig.log_client_configuration(self._http_client, settings)

            # Initialize OpenAI client with the configured HTTP client
            self._openAIClient = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.base_url,
                default_headers=HttpClientConfig.get_default_headers(settings),
                timeout=float(settings.max_stream_seconds),
                http_client=self._http_client,
                max_retries=settings.provider_max_retries,
            )
        except Exception:
            # Clean up if initialization fails
            self._http_client = None
            raise

    # Retry logic is now handled by RetryHandler in resilience.py

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
        correlation_id = self._request_logger.generate_correlation_id()
        request_start = time.monotonic()

        # Extract trace context and prepare headers
        trace_headers = self._request_logger.prepare_trace_headers(correlation_id)
        trace_id = None
        if trace_headers:
            params["extra_headers"] = trace_headers
            trace_id = trace_headers.get("X-Trace-ID")

        # Log request with trace context
        self._request_logger.log_request(correlation_id, params, trace_id)

        # Start performance monitoring
        await self._performance_monitor.start_request(correlation_id)
        await self._performance_tracker.start_request(correlation_id)

        try:
            # Check circuit breaker
            if self._circuit_breaker.is_open:
                logging.warning(
                    f"Request {correlation_id} blocked by open circuit breaker"
                )
                await self._metrics_collector.record_failure(
                    0, self._circuit_breaker.consecutive_failures
                )
                raise Exception(
                    "Service temporarily unavailable - circuit breaker is open"
                )

            # Apply client-side rate limiting
            if self._rate_limiter:
                # Accurately count tokens using tiktoken
                estimated_tokens = await self._estimate_tokens(
                    params.get("messages", []), params.get("model")
                )

                # Wait if necessary to respect rate limits
                await self._rate_limiter.wait_if_needed()

                # Try to acquire rate limit permit
                if not await self._rate_limiter.acquire(estimated_tokens):
                    logging.warning(
                        f"Request {correlation_id} blocked by client rate limiter"
                    )
                    await self._metrics_collector.record_failure(
                        0, self._circuit_breaker.consecutive_failures
                    )
                    raise Exception(
                        "Client-side rate limit exceeded. Please retry after a short delay."
                    )

            # Handle streaming case separately
            stream = params.get("stream", False)
            if stream:
                # For streaming, use resilient executor
                response = await self._resilient_executor.execute(
                    self._openAIClient.chat.completions.create,
                    **params,
                )
                await self._on_request_success(correlation_id, request_start)
                return response

            # Non-streaming case with additional UTF-8 handling
            response = await self._resilient_executor.execute(
                self._openAIClient.chat.completions.create,
                **params,
            )

            # Process response for UTF-8 handling
            response = self._response_processor.process_chat_completion_response(
                response
            )

            # Update metrics and log success
            await self._on_request_success(correlation_id, request_start, response)

            # Update rate limiter on success
            if self._rate_limiter:
                await self._rate_limiter.handle_success()
                # Release with actual token count if available
                usage_info = self._response_processor.extract_usage_info(response)
                if usage_info:
                    await self._rate_limiter.release(usage_info["total_tokens"])

            return response

        except UnicodeDecodeError as e:
            await self._on_request_failure(
                correlation_id, request_start, e, ErrorType.CONVERSION_ERROR
            )
            raise ValueError(
                f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}"
            ) from e
        except openai.RateLimitError as e:
            await self._on_request_failure(
                correlation_id, request_start, e, ErrorType.RATE_LIMIT_ERROR
            )
            # Update rate limiter on 429
            if self._rate_limiter:
                retry_after = getattr(e.response, "headers", {}).get("retry-after")
                await self._rate_limiter.handle_429_response(
                    int(retry_after) if retry_after else None
                )
            raise
        except openai.AuthenticationError as e:
            await self._on_request_failure(
                correlation_id, request_start, e, ErrorType.AUTH_ERROR
            )
            raise
        except Exception as e:
            # Use error handler to classify error
            error_category = self._error_handler.classify_error(e)
            error_type_map = {
                "conversion_error": ErrorType.CONVERSION_ERROR,
                "timeout_error": ErrorType.TIMEOUT_ERROR,
                "rate_limit_error": ErrorType.RATE_LIMIT_ERROR,
                "auth_error": ErrorType.AUTH_ERROR,
                "network_error": ErrorType.API_ERROR,
                "api_error": ErrorType.API_ERROR,
            }
            error_type = error_type_map.get(error_category, ErrorType.API_ERROR)

            await self._on_request_failure(correlation_id, request_start, e, error_type)

            if error_type == ErrorType.CONVERSION_ERROR:
                raise ValueError(
                    f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}"
                ) from e
            raise

    # Request logging is now handled by RequestLogger in request_logger.py

    async def _on_request_success(
        self, correlation_id: str, request_start: float, response: Any = None
    ) -> None:
        """Handle successful request - update metrics and monitoring."""
        latency_ms = (time.monotonic() - request_start) * 1000

        # Update performance monitor
        await self._performance_monitor.end_request(correlation_id, success=True)

        # Update performance tracker
        await self._performance_tracker.end_request(correlation_id)

        # Extract token usage
        tokens = 0
        usage_info = self._response_processor.extract_usage_info(response)
        if usage_info:
            tokens = usage_info["total_tokens"]

        # Update metrics collector
        await self._metrics_collector.record_success(latency_ms, tokens)

        # Update circuit state in metrics
        await self._metrics_collector.update_circuit_state(self._circuit_breaker.state)

        # Log response
        self._request_logger.log_response(
            correlation_id, latency_ms, success=True, response=response
        )

    async def _on_request_failure(
        self,
        correlation_id: str,
        request_start: float,
        error: Exception,
        error_type: ErrorType,
    ) -> None:
        """Handle failed request - update metrics and error tracking."""
        latency_ms = (time.monotonic() - request_start) * 1000

        # Update performance monitor
        await self._performance_monitor.end_request(correlation_id, success=False)

        # Update performance tracker
        await self._performance_tracker.end_request(correlation_id)

        # Track error
        await self._error_tracker.track_error(error=error, error_type=error_type)

        # Update metrics collector
        await self._metrics_collector.record_failure(
            latency_ms, self._circuit_breaker.consecutive_failures
        )

        # Update circuit state in metrics
        await self._metrics_collector.update_circuit_state(self._circuit_breaker.state)

        # Log response
        self._request_logger.log_response(
            correlation_id, latency_ms, success=False, error=error
        )

    # Health score calculation is now handled by HealthMonitor in metrics.py

    async def get_metrics(self) -> ProviderMetrics:
        """Get current metrics snapshot."""
        metrics = await self._metrics_collector.get_snapshot()
        # Update circuit state
        await self._metrics_collector.update_circuit_state(self._circuit_breaker.state)
        return metrics

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        return await self._health_monitor.get_health_check(
            circuit_state=self._circuit_breaker.state,
            active_requests=self._performance_monitor.metrics.active_requests,
        )

    def get_adaptive_timeout(self) -> float:
        """Calculate adaptive timeout based on recent performance."""
        return self._timeout_calculator.calculate_timeout()

    async def _estimate_tokens(
        self, messages: List[Dict[str, Any]], model_name: str = None
    ) -> int:
        """
        Accurately count tokens for messages using tiktoken.

        Args:
            messages: List of message dictionaries
            model_name: Optional model name for accurate encoding

        Returns:
            Accurate token count
        """
        if not messages:
            return 0

        try:
            # Import tokenizer function
            from ...application.tokenizer import count_tokens_for_openai_request

            # Use the provided model or default from settings
            model = model_name or self.settings.big_model_name

            # Get accurate token count
            token_count = await count_tokens_for_openai_request(
                messages=messages,
                model_name=model,
                request_id=None,  # Could pass correlation_id if available
            )

            return token_count
        except Exception as e:
            logging.warning(
                f"Failed to count tokens accurately, falling back to rough estimate: {e}"
            )

            # Fallback to rough estimation: ~4 characters per token
            total_chars = 0
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    total_chars += len(content)
            # Handle tool calls, function calls, etc.
            if "tool_calls" in messages[0] if messages else False:
                total_chars += len(str(message["tool_calls"]))

        # Add overhead for message structure
        total_chars += len(messages) * 10

        return total_chars // 4

    async def close(self) -> None:
        """Clean up resources when shutting down."""
        # Stop rate limiter if running
        if self._rate_limiter:
            try:
                await self._rate_limiter.stop()
                logging.info("Rate limiter stopped successfully")
            except Exception as e:
                logging.error(f"Error stopping rate limiter: {e}")

        # Close HTTP client using factory method
        await HttpClientFactory.close_client(self._http_client)
        self._http_client = None
        self._openAIClient = None

    async def __aenter__(self) -> "OpenAIProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
