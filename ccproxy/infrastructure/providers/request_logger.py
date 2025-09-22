"""
Request logging and tracing for provider implementations.
Handles correlation IDs, trace context, and request/response logging.
"""

import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

# Context variable for trace context propagation
trace_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "trace_context", default=None
)


class RequestLogger:
    """Handles request logging with correlation and tracing."""

    def __init__(self):
        """Initialize request logger."""
        self._request_log: Dict[str, Dict[str, Any]] = {}
        self._tracing_manager = self._init_tracing_manager()

    def _init_tracing_manager(self) -> Optional[Any]:
        """
        Initialize tracing manager if available.

        Returns:
            Tracing manager instance or None
        """
        try:
            from ...tracing import get_tracing_manager

            return get_tracing_manager()
        except ImportError:
            logging.debug("Tracing not available")
            return None

    def generate_correlation_id(self) -> str:
        """
        Generate a unique correlation ID for request tracking.

        Returns:
            UUID string for correlation
        """
        return str(uuid.uuid4())

    def extract_trace_context(self) -> Optional[Dict[str, Any]]:
        """
        Extract trace context from context variables.

        Returns:
            Trace context dictionary or None
        """
        return trace_context.get()

    def prepare_trace_headers(self, correlation_id: str) -> Optional[Dict[str, str]]:
        """
        Prepare trace headers for propagation.

        Args:
            correlation_id: Request correlation ID

        Returns:
            Dictionary of trace headers or None
        """
        if not self._tracing_manager or not self._tracing_manager.enabled:
            return None

        trace_info = self.extract_trace_context()
        trace_id = None

        # Get trace ID from context or generate new one
        if trace_info:
            trace_id = trace_info.get("trace_id")
        else:
            trace_id = self._tracing_manager.get_current_trace_id()

        if not trace_id:
            return None

        # Prepare headers for injection
        headers = {}
        self._tracing_manager.inject_context(headers)

        # Add custom trace headers
        headers["X-Trace-ID"] = trace_id
        headers["X-Correlation-ID"] = correlation_id

        return headers

    def log_request(
        self,
        correlation_id: str,
        params: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Log request with correlation ID and optional trace ID.

        Args:
            correlation_id: Request correlation ID
            params: Request parameters
            trace_id: Optional trace ID for distributed tracing
        """
        # Build request info
        request_info = {
            "timestamp": datetime.now().isoformat(),
            "params": self._sanitize_params(params),
            "model": params.get("model", "unknown"),
        }

        if trace_id:
            request_info["trace_id"] = trace_id

        # Store in request log
        self._request_log[correlation_id] = request_info

        # Log the request
        log_msg = (
            f"Request {correlation_id}: "
            f"model={params.get('model')}, "
            f"stream={params.get('stream', False)}"
        )
        if trace_id:
            log_msg += f", trace_id={trace_id}"

        logging.debug(log_msg)

    def log_response(
        self,
        correlation_id: str,
        latency_ms: float,
        success: bool = True,
        error: Optional[Exception] = None,
        response: Optional[Any] = None,
    ) -> None:
        """
        Log response with metrics.

        Args:
            correlation_id: Request correlation ID
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            error: Optional error if failed
            response: Optional response object
        """
        if success:
            log_msg = f"Request {correlation_id} succeeded in {latency_ms:.2f}ms"

            # Add token usage if available
            if response and hasattr(response, "usage"):
                usage = response.usage
                if usage:
                    log_msg += f", tokens: {getattr(usage, 'total_tokens', 0)} total"

            logging.debug(log_msg)
        else:
            logging.error(
                f"Request {correlation_id} failed after {latency_ms:.2f}ms: {error}"
            )

        # Clean up request log entry
        self.cleanup_request(correlation_id)

    def cleanup_request(self, correlation_id: str) -> None:
        """
        Clean up request log entry.

        Args:
            correlation_id: Request correlation ID
        """
        if correlation_id in self._request_log:
            del self._request_log[correlation_id]

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize request parameters for logging.

        Args:
            params: Request parameters

        Returns:
            Sanitized parameters safe for logging
        """
        # List of sensitive keys to exclude
        sensitive_keys = {
            "api_key",
            "messages",
            "extra_headers",
            "authorization",
            "password",
            "secret",
        }

        # Filter out sensitive information
        sanitized = {}
        for key, value in params.items():
            if key.lower() not in sensitive_keys:
                sanitized[key] = value

        return sanitized

    def get_request_info(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get request information by correlation ID.

        Args:
            correlation_id: Request correlation ID

        Returns:
            Request information or None
        """
        return self._request_log.get(correlation_id)


class PerformanceTracker:
    """Tracks performance metrics for requests."""

    def __init__(self):
        """Initialize performance tracker."""
        self._active_requests: Dict[str, float] = {}

    async def start_request(self, correlation_id: str) -> None:
        """
        Start tracking a request.

        Args:
            correlation_id: Request correlation ID
        """
        self._active_requests[correlation_id] = time.monotonic()
        logging.debug(f"Started tracking request {correlation_id}")

    async def end_request(self, correlation_id: str) -> Optional[float]:
        """
        End tracking a request and return latency.

        Args:
            correlation_id: Request correlation ID

        Returns:
            Latency in milliseconds or None
        """
        if correlation_id not in self._active_requests:
            logging.warning(f"Request {correlation_id} not found in active requests")
            return None

        start_time = self._active_requests.pop(correlation_id)
        latency_ms = (time.monotonic() - start_time) * 1000

        logging.debug(f"Request {correlation_id} completed in {latency_ms:.2f}ms")
        return latency_ms

    def get_active_request_count(self) -> int:
        """
        Get count of currently active requests.

        Returns:
            Number of active requests
        """
        return len(self._active_requests)

    def get_active_request_ids(self) -> list:
        """
        Get list of active request correlation IDs.

        Returns:
            List of correlation IDs
        """
        return list(self._active_requests.keys())


class RequestMetadata:
    """Manages metadata for requests."""

    @staticmethod
    def create_metadata(
        correlation_id: str,
        model: str,
        stream: bool = False,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create metadata dictionary for a request.

        Args:
            correlation_id: Request correlation ID
            model: Model name
            stream: Whether request is streaming
            trace_id: Optional trace ID

        Returns:
            Metadata dictionary
        """
        metadata = {
            "correlation_id": correlation_id,
            "model": model,
            "stream": stream,
            "timestamp": datetime.now().isoformat(),
        }

        if trace_id:
            metadata["trace_id"] = trace_id

        return metadata

    @staticmethod
    def extract_from_params(
        params: Dict[str, Any], correlation_id: str
    ) -> Dict[str, Any]:
        """
        Extract metadata from request parameters.

        Args:
            params: Request parameters
            correlation_id: Request correlation ID

        Returns:
            Extracted metadata
        """
        return RequestMetadata.create_metadata(
            correlation_id=correlation_id,
            model=params.get("model", "unknown"),
            stream=params.get("stream", False),
        )
