"""
Request lifecycle observer for coordinating updates to monitoring systems.
Centralizes the handling of request success and failure events.
"""

import time
from typing import Any, Optional

from ...monitoring import PerformanceMonitor
from ...application.error_tracker import ErrorTracker, ErrorType
from .resilience import CircuitBreaker
from .metrics import MetricsCollector
from .response_handlers import ResponseProcessor
from .request_logger import RequestLogger, PerformanceTracker


class RequestLifecycleObserver:
    """
    Observer for request lifecycle events.

    Coordinates updates to multiple monitoring and tracking systems
    when requests complete successfully or fail.
    """

    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        performance_tracker: PerformanceTracker,
        metrics_collector: MetricsCollector,
        error_tracker: ErrorTracker,
        circuit_breaker: CircuitBreaker,
        request_logger: RequestLogger,
        response_processor: ResponseProcessor,
    ):
        """
        Initialize the lifecycle observer.

        Args:
            performance_monitor: Application-level performance monitor
            performance_tracker: Request-level performance tracker
            metrics_collector: Provider metrics collector
            error_tracker: Centralized error tracking
            circuit_breaker: Circuit breaker for state updates
            request_logger: Request/response logger
            response_processor: Response processing utilities
        """
        self._performance_monitor = performance_monitor
        self._performance_tracker = performance_tracker
        self._metrics_collector = metrics_collector
        self._error_tracker = error_tracker
        self._circuit_breaker = circuit_breaker
        self._request_logger = request_logger
        self._response_processor = response_processor

    async def on_success(
        self,
        correlation_id: str,
        request_start: float,
        response: Optional[Any] = None,
    ) -> None:
        """
        Handle successful request completion.

        Updates all monitoring systems with success metrics.

        Args:
            correlation_id: Unique request identifier
            request_start: Request start time (from time.monotonic())
            response: Optional API response for token extraction
        """
        latency_ms = (time.monotonic() - request_start) * 1000

        # Update performance monitors
        await self._performance_monitor.end_request(correlation_id, success=True)
        await self._performance_tracker.end_request(correlation_id)

        # Extract token usage
        tokens = 0
        if response:
            usage_info = self._response_processor.extract_usage_info(response)
            if usage_info:
                tokens = usage_info.get("total_tokens", 0)

        # Update metrics
        await self._metrics_collector.record_success(latency_ms, tokens)
        await self._metrics_collector.update_circuit_state(self._circuit_breaker.state)

        # Log response
        self._request_logger.log_response(
            correlation_id=correlation_id,
            latency_ms=latency_ms,
            success=True,
            response=response,
        )

    async def on_failure(
        self,
        correlation_id: str,
        request_start: float,
        error: Exception,
        error_type: ErrorType,
    ) -> None:
        """
        Handle failed request.

        Updates all monitoring systems with failure metrics.

        Args:
            correlation_id: Unique request identifier
            request_start: Request start time (from time.monotonic())
            error: The exception that occurred
            error_type: Categorized error type
        """
        latency_ms = (time.monotonic() - request_start) * 1000

        # Update performance monitors
        await self._performance_monitor.end_request(correlation_id, success=False)
        await self._performance_tracker.end_request(correlation_id)

        # Track error
        await self._error_tracker.track_error(
            error=error,
            error_type=error_type,
            request_id=correlation_id,
        )

        # Update metrics
        await self._metrics_collector.record_failure(
            latency_ms, self._circuit_breaker.consecutive_failures
        )
        await self._metrics_collector.update_circuit_state(self._circuit_breaker.state)

        # Log response
        self._request_logger.log_response(
            correlation_id=correlation_id,
            latency_ms=latency_ms,
            success=False,
            error=error,
        )

    async def on_request_start(
        self,
        correlation_id: str,
        params: dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Handle request start event.

        Args:
            correlation_id: Unique request identifier
            params: Request parameters
            trace_id: Optional distributed trace ID
        """
        # Log request
        self._request_logger.log_request(correlation_id, params, trace_id)

        # Start performance monitoring
        await self._performance_monitor.start_request(correlation_id)
        await self._performance_tracker.start_request(correlation_id)

    def classify_error(self, error: Exception) -> ErrorType:
        """
        Classify an error into an ErrorType category.

        Args:
            error: The exception to classify

        Returns:
            Appropriate ErrorType for the error
        """
        # Use existing error handler classification
        from .response_handlers import ErrorResponseHandler

        error_handler = ErrorResponseHandler()
        error_category = error_handler.classify_error(error)

        error_type_map = {
            "conversion_error": ErrorType.CONVERSION_ERROR,
            "timeout_error": ErrorType.TIMEOUT_ERROR,
            "rate_limit_error": ErrorType.RATE_LIMIT_ERROR,
            "auth_error": ErrorType.AUTH_ERROR,
            "network_error": ErrorType.API_ERROR,
            "api_error": ErrorType.API_ERROR,
        }

        return error_type_map.get(error_category, ErrorType.API_ERROR)
