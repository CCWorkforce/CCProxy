"""OpenTelemetry distributed tracing support for CCProxy."""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import extract, inject
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Optional exporters
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except ImportError:
    OTLPSpanExporter = None

try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
except ImportError:
    JaegerExporter = None


class TracingManager:
    """Manages OpenTelemetry tracing configuration and context propagation."""

    def __init__(self, settings: "Any") -> None:
        """Initialize tracing based on configuration settings.

        Args:
            settings: Application settings containing tracing configuration
        """
        self.enabled = settings.tracing_enabled
        self.tracer: Optional[trace.Tracer] = None
        self.propagator = TraceContextTextMapPropagator()

        if not self.enabled:
            logging.info("Distributed tracing is disabled")
            return

        # Create resource with service information
        resource = Resource.create(
            {
                "service.name": settings.tracing_service_name,
                "service.version": settings.version
                if hasattr(settings, "version")
                else "1.0.0",
                "deployment.environment": "production"
                if os.getenv("IS_LOCAL_DEPLOYMENT", "False").lower() != "true"
                else "development",
            }
        )

        # Initialize tracer provider
        provider = TracerProvider(resource=resource)

        # Configure exporter based on settings
        exporter = self._create_exporter(settings)
        if exporter:
            provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as global tracer provider
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)

        logging.info(
            f"Distributed tracing enabled with {settings.tracing_exporter} exporter"
        )

    def _create_exporter(self, settings: Any) -> Optional[Any]:
        """Create the appropriate span exporter based on configuration.

        Args:
            settings: Application settings

        Returns:
            Configured span exporter or None if unavailable
        """
        exporter_type = settings.tracing_exporter.lower()

        if exporter_type == "console":
            return ConsoleSpanExporter()

        elif exporter_type == "otlp" and OTLPSpanExporter:
            if not settings.tracing_endpoint:
                logging.warning(
                    "OTLP exporter configured but no endpoint specified, using default"
                )
                return OTLPSpanExporter()
            return OTLPSpanExporter(endpoint=settings.tracing_endpoint)

        elif exporter_type == "jaeger" and JaegerExporter:
            if not settings.tracing_endpoint:
                # Default Jaeger agent endpoint
                return JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=6831,
                )
            # Parse endpoint for Jaeger collector
            parts = settings.tracing_endpoint.split(":")
            if len(parts) == 2:
                return JaegerExporter(
                    agent_host_name=parts[0],
                    agent_port=int(parts[1]),
                )
            return JaegerExporter()

        else:
            logging.warning(
                f"Unsupported or unavailable exporter: {exporter_type}, using console"
            )
            return ConsoleSpanExporter()

    def extract_context(self, headers: Dict[str, str]) -> Optional[Any]:
        """Extract trace context from incoming request headers.

        Supports multiple trace context formats:
        - W3C Trace Context (traceparent)
        - B3 Single (X-B3-TraceId)
        - Jaeger (uber-trace-id)

        Args:
            headers: HTTP request headers

        Returns:
            Extracted context or None
        """
        if not self.enabled:
            return None

        # Extract using W3C Trace Context propagator
        return extract(headers)

    def inject_context(
        self, headers: Dict[str, str], context: Optional[Any] = None
    ) -> None:
        """Inject trace context into outgoing request headers.

        Args:
            headers: HTTP headers dictionary to inject into
            context: Optional context to inject, uses current if not provided
        """
        if not self.enabled:
            return

        inject(headers, context)

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
        context: Optional[Any] = None,
    ):
        """Start a new span with the given name and attributes.

        Args:
            name: Span name
            attributes: Optional span attributes
            kind: Span kind (CLIENT, SERVER, INTERNAL, PRODUCER, CONSUMER)
            context: Optional parent context

        Yields:
            The created span
        """
        if not self.enabled or not self.tracer:
            # Return a no-op context manager
            yield None
            return

        with self.tracer.start_as_current_span(
            name,
            context=context,
            kind=kind,
            attributes=attributes or {},
        ) as span:
            try:
                yield span
            except Exception as e:
                # Record exception in span
                if span and span.is_recording():
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            else:
                # Set success status
                if span and span.is_recording():
                    span.set_status(Status(StatusCode.OK))

    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID as a string.

        Returns:
            Trace ID string or None if not in a trace context
        """
        if not self.enabled:
            return None

        span = trace.get_current_span()
        if span and span.is_recording():
            context = span.get_span_context()
            if context and context.is_valid:
                return format(context.trace_id, "032x")
        return None

    def add_span_attributes(self, attributes: Dict[str, Any]) -> None:
        """Add attributes to the current active span.

        Args:
            attributes: Dictionary of attributes to add
        """
        if not self.enabled:
            return

        span = trace.get_current_span()
        if span and span.is_recording():
            for key, value in attributes.items():
                span.set_attribute(key, value)

    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the current span.

        Args:
            exception: The exception to record
        """
        if not self.enabled:
            return

        span = trace.get_current_span()
        if span and span.is_recording():
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def init_tracing(settings: Any) -> TracingManager:
    """Initialize the global tracing manager.

    Args:
        settings: Application settings

    Returns:
        The initialized tracing manager
    """
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager(settings)
    return _tracing_manager


def get_tracing_manager() -> Optional[TracingManager]:
    """Get the global tracing manager instance.

    Returns:
        The tracing manager or None if not initialized
    """
    return _tracing_manager
