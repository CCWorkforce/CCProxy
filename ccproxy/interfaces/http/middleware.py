"""Common FastAPI middleware utilities for CCProxy HTTP interface."""

import time
import uuid
from typing import Awaitable, Callable, Optional

from fastapi import Request
from fastapi.responses import Response

# Import tracing if available
try:
    from ...tracing import get_tracing_manager

    tracing_available = True
except ImportError:
    tracing_available = False
    get_tracing_manager = None


async def logging_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach request ID, timing headers, and handle distributed tracing.

    This middleware:
    1. Generates a per-request UUID (``X-Request-ID``)
    2. Extracts trace context from incoming headers
    3. Creates or continues a trace span
    4. Measures wall-clock latency
    5. Stores identifiers on ``request.state`` for downstream handlers
    """
    # Generate request ID
    if not hasattr(request.state, "request_id"):
        request.state.request_id = str(uuid.uuid4())
    if not hasattr(request.state, "start_time_monotonic"):
        request.state.start_time_monotonic = time.monotonic()

    # Handle distributed tracing if available
    trace_id: Optional[str] = None
    span_context = None

    if tracing_available and get_tracing_manager:
        tracing_manager = get_tracing_manager()
        if tracing_manager and tracing_manager.enabled:
            # Extract trace context from headers
            headers_dict = dict(request.headers)
            span_context = tracing_manager.extract_context(headers_dict)

            # Get or create trace ID
            trace_id = (
                headers_dict.get("x-trace-id")
                or headers_dict.get("x-b3-traceid")
                or headers_dict.get("uber-trace-id", "").split(":")[0]
                if "uber-trace-id" in headers_dict
                else None or tracing_manager.get_current_trace_id()
            )

            # Store trace context in request state for downstream use
            request.state.trace_context = span_context
            request.state.trace_id = trace_id

    # Process request with optional tracing
    if (
        tracing_available
        and get_tracing_manager
        and tracing_manager
        and tracing_manager.enabled
    ):
        # Start a server span for this request
        span_attributes = {
            "http.method": request.method,
            "http.url": str(request.url),
            "http.scheme": request.url.scheme,
            "http.host": request.url.hostname,
            "http.target": request.url.path,
            "request.id": request.state.request_id,
        }

        if trace_id:
            span_attributes["trace.id"] = trace_id

        with tracing_manager.start_span(
            f"{request.method} {request.url.path}",
            attributes=span_attributes,
            kind=1,  # SERVER span
            context=span_context,
        ) as span:
            response = await call_next(request)

            # Add response attributes
            if span:
                tracing_manager.add_span_attributes(
                    {
                        "http.status_code": response.status_code,
                        "response.size": len(response.body)
                        if hasattr(response, "body")
                        else 0,
                    }
                )
    else:
        # No tracing, just process the request
        response = await call_next(request)

    # Add standard response headers
    response.headers["X-Request-ID"] = request.state.request_id
    duration_ms = (time.monotonic() - request.state.start_time_monotonic) * 1000
    response.headers["X-Response-Time-ms"] = str(duration_ms)

    # Add trace ID to response if available
    if trace_id:
        response.headers["X-Trace-ID"] = trace_id

    return response
