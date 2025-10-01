"""Comprehensive error tracking system for debugging and monitoring."""

import asyncio
import json
import re
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from fastapi import Request, Response
from asyncer import create_task_group
from .thread_pool import asyncify

from ..config import Settings
from ..logging import debug, info, warning, LogRecord, LogEvent


class ErrorType(str, Enum):
    """Categories of errors for classification and analysis."""

    CONVERSION_ERROR = "conversion_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    CACHE_ERROR = "cache_error"
    STREAM_ERROR = "stream_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTH_ERROR = "auth_error"
    INTERNAL_ERROR = "internal_error"


@dataclass
class RequestSnapshot:
    """Captures request details for error context."""

    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, str]
    body: Optional[Any] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class ResponseSnapshot:
    """Captures response details for error context."""

    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    body: Optional[Any] = None
    elapsed_ms: Optional[float] = None


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    request_id: Optional[str] = None
    error_type: ErrorType = ErrorType.INTERNAL_ERROR
    error_message: str = ""
    stack_trace: Optional[str] = None
    request_snapshot: Optional[RequestSnapshot] = None
    response_snapshot: Optional[ResponseSnapshot] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    redacted_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert enum to string
        result["error_type"] = self.error_type.value
        return result


class ErrorTracker:
    """
    Singleton error tracking service for comprehensive error logging.

    Features:
    - Async, non-blocking writes to dedicated error log
    - Automatic sensitive data redaction
    - Request/response snapshot capture
    - Log rotation and retention management
    - Performance metrics tracking
    """

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._settings: Optional[Settings] = None
        self._file_handle = None
        self._write_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._writer_task = None
        self._redact_patterns: List[re.Pattern] = []
        self._setup_redaction_patterns()

    def _setup_redaction_patterns(self):
        """Setup regex patterns for sensitive data redaction."""
        # API keys and tokens
        self._redact_patterns.extend(
            [
                re.compile(
                    r'(api[_-]?key|token|secret|password)["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)',
                    re.IGNORECASE,
                ),
                re.compile(r"Bearer\s+([A-Za-z0-9\-._~+/]+=*)", re.IGNORECASE),
                re.compile(
                    r"(sk-[a-zA-Z0-9]{48}|sk-proj-[a-zA-Z0-9]{48})"
                ),  # OpenAI keys
            ]
        )

    async def initialize(self, settings: Settings):
        """Initialize the error tracker with settings."""
        self._settings = settings

        if not settings.error_tracking_enabled:
            debug(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message="Error tracking disabled",
                )
            )
            return

        # Create error log directory if needed
        error_log_path = Path(settings.error_tracking_file)
        error_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Start background writer
        if not self._writer_task:
            self._writer_task = asyncio.create_task(self._writer_loop())

        info(
            LogRecord(
                event=LogEvent.CACHE_EVENT.value,
                message=f"Error tracking initialized: {settings.error_tracking_file}",
            )
        )

    async def shutdown(self):
        """Shutdown the error tracker gracefully."""
        if self._writer_task:
            # Signal shutdown by adding None to queue
            await self._write_queue.put(None)
            await self._writer_task
            self._writer_task = None

        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    async def _writer_loop(self):
        """Background task to write errors to file."""
        while True:
            try:
                # Get error context from queue
                error_context = await self._write_queue.get()

                # Check for shutdown signal
                if error_context is None:
                    break

                # Write to file
                await self._write_error(error_context)

            except Exception as e:
                # Log but don't crash the writer
                warning(
                    LogRecord(
                        event=LogEvent.CACHE_EVENT.value,
                        message=f"Error writer exception: {str(e)}",
                    )
                )

    async def _write_error(self, error_context: ErrorContext):
        """Write error context to file."""
        if not self._settings or not self._settings.error_tracking_enabled:
            return

        try:
            # Check and rotate log if needed
            await self._rotate_log_if_needed()

            # Open file if not open
            if not self._file_handle:
                self._file_handle = open(self._settings.error_tracking_file, "a")

            # Serialize JSON asynchronously for large error contexts
            json_dumps_async = asyncify(lambda obj: json.dumps(obj, default=str))
            json_line = await json_dumps_async(error_context.to_dict())

            # Write to file (could use aiofiles but keeping sync for simplicity here)
            self._file_handle.write(json_line + "\n")
            self._file_handle.flush()

        except Exception as e:
            warning(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Failed to write error log: {str(e)}",
                )
            )

    async def _rotate_log_if_needed(self):
        """Rotate log file if it exceeds size limit."""
        if not self._settings:
            return

        try:
            log_path = Path(self._settings.error_tracking_file)
            if not log_path.exists():
                return

            # Check file size
            size_mb = log_path.stat().st_size / (1024 * 1024)
            if size_mb > self._settings.error_tracking_max_size_mb:
                # Close current file
                if self._file_handle:
                    self._file_handle.close()
                    self._file_handle = None

                # Rotate with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rotated_path = log_path.with_suffix(f".{timestamp}.jsonl")
                log_path.rename(rotated_path)

                info(
                    LogRecord(
                        event=LogEvent.CACHE_EVENT.value,
                        message=f"Rotated error log: {rotated_path}",
                    )
                )

                # Clean old logs
                await self._clean_old_logs()

        except Exception as e:
            warning(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Log rotation failed: {str(e)}",
                )
            )

    async def _clean_old_logs(self):
        """Remove old rotated logs based on retention policy."""
        if not self._settings:
            return

        try:
            log_dir = Path(self._settings.error_tracking_file).parent
            cutoff_date = datetime.now() - timedelta(
                days=self._settings.error_tracking_retention_days
            )

            for file_path in log_dir.glob("errors_detailed.*.jsonl"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    debug(
                        LogRecord(
                            event=LogEvent.CACHE_EVENT.value,
                            message=f"Deleted old error log: {file_path}",
                        )
                    )
        except Exception:
            pass  # Best effort cleanup

    async def _redact_sensitive_data_async(self, data: Any) -> Any:
        """Recursively redact sensitive data asynchronously."""
        if isinstance(data, dict):
            # Process dictionary values in parallel for better performance
            redacted = {}
            items_to_process = []

            for key, value in data.items():
                # Check if key contains sensitive keywords
                if any(
                    keyword in key.lower()
                    for keyword in ["password", "secret", "token", "key", "auth"]
                ):
                    redacted[key] = "[REDACTED]"
                else:
                    items_to_process.append((key, value))

            # Process remaining items asynchronously
            if items_to_process:
                async with create_task_group() as tg:
                    soon_values = []
                    for key, value in items_to_process:
                        soon_values.append(
                            (key, tg.soonify(self._redact_sensitive_data_async)(value))
                        )

                for key, sv in soon_values:
                    redacted[key] = sv.value

            return redacted
        elif isinstance(data, list):
            # Process list items in parallel
            if data:
                async with create_task_group() as tg:
                    soon_values = [
                        tg.soonify(self._redact_sensitive_data_async)(item)
                        for item in data
                    ]
                return [sv.value for sv in soon_values]
            return []
        elif isinstance(data, str):
            # Apply redaction patterns (CPU-intensive for large strings)
            redact_func = asyncify(self._apply_redaction_patterns)
            return await redact_func(data)
        return data

    def _apply_redaction_patterns(self, text: str) -> str:
        """Apply redaction patterns to text."""
        result = text
        for pattern in self._redact_patterns:
            result = pattern.sub("[REDACTED]", result)
        return result

    def _redact_sensitive_data(self, data: Any) -> Any:
        """Synchronous version for backward compatibility."""
        if isinstance(data, dict):
            redacted = {}
            for key, value in data.items():
                # Check if key contains sensitive keywords
                if any(
                    keyword in key.lower()
                    for keyword in ["password", "secret", "token", "key", "auth"]
                ):
                    redacted[key] = "[REDACTED]"
                else:
                    redacted[key] = self._redact_sensitive_data(value)
            return redacted
        elif isinstance(data, list):
            return [self._redact_sensitive_data(item) for item in data]
        elif isinstance(data, str):
            # Apply redaction patterns
            return self._apply_redaction_patterns(data)
        return data

    def _truncate_large_data(self, data: Any, max_size: int = None) -> Any:
        """Truncate large data structures to prevent log bloat."""
        max_size = max_size or (
            self._settings.error_tracking_max_body_size if self._settings else 10000
        )

        if isinstance(data, str) and len(data) > max_size:
            return data[:max_size] + f"...[truncated {len(data) - max_size} chars]"
        elif isinstance(data, dict):
            return {k: self._truncate_large_data(v, max_size) for k, v in data.items()}
        elif isinstance(data, list):
            if len(data) > 100:  # Limit array size
                return data[:100] + [f"...[truncated {len(data) - 100} items]"]
            return [self._truncate_large_data(item, max_size) for item in data]
        return data

    async def capture_request_snapshot(self, request: Request) -> RequestSnapshot:
        """Capture request details for error context."""
        try:
            # Get headers (redact sensitive ones)
            headers = dict(request.headers)
            headers = self._redact_sensitive_data(headers)

            # Get body if configured
            body = None
            if self._settings and self._settings.error_tracking_capture_request:
                try:
                    # Try to get cached body first
                    if hasattr(request, "_body"):
                        body = request._body
                    else:
                        # Read body (this consumes the stream)
                        body_bytes = await request.body()
                        request._body = body_bytes  # Cache for later use
                        body = body_bytes.decode("utf-8") if body_bytes else None

                    # Try to parse as JSON
                    if body:
                        try:
                            body = json.loads(body)
                        except json.JSONDecodeError:
                            pass  # Keep as string

                    # Redact and truncate
                    body = self._redact_sensitive_data(body)
                    body = self._truncate_large_data(body)
                except Exception:
                    body = None

            return RequestSnapshot(
                method=request.method,
                path=str(request.url.path),
                headers=headers,
                query_params=dict(request.query_params),
                body=body,
                client_ip=request.client.host if request.client else None,
                user_agent=headers.get("user-agent"),
            )
        except Exception as e:
            return RequestSnapshot(
                method="UNKNOWN",
                path="UNKNOWN",
                headers={},
                query_params={},
                body=f"Failed to capture: {str(e)}",
            )

    def capture_response_snapshot(
        self,
        response: Optional[Response] = None,
        status_code: Optional[int] = None,
        body: Optional[Any] = None,
        elapsed_ms: Optional[float] = None,
    ) -> ResponseSnapshot:
        """Capture response details for error context."""
        try:
            headers = {}
            if response and hasattr(response, "headers"):
                headers = dict(response.headers)
                headers = self._redact_sensitive_data(headers)

            # Process body
            if (
                body is not None
                and self._settings
                and self._settings.error_tracking_capture_response
            ):
                body = self._redact_sensitive_data(body)
                body = self._truncate_large_data(body)

            return ResponseSnapshot(
                status_code=status_code or (response.status_code if response else None),
                headers=headers,
                body=body,
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            return ResponseSnapshot(
                status_code=status_code,
                body=f"Failed to capture: {str(e)}",
            )

    async def track_error(
        self,
        error: Exception,
        error_type: ErrorType,
        request_id: Optional[str] = None,
        request_snapshot: Optional[RequestSnapshot] = None,
        response_snapshot: Optional[ResponseSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Track an error with full context."""
        if not self._settings or not self._settings.error_tracking_enabled:
            return

        try:
            # Create error context
            error_context = ErrorContext(
                request_id=request_id,
                error_type=error_type,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                request_snapshot=request_snapshot,
                response_snapshot=response_snapshot,
                metadata=self._redact_sensitive_data(metadata or {}),
            )

            # Queue for async write (non-blocking)
            if not self._write_queue.full():
                await self._write_queue.put(error_context)
            else:
                # Queue is full, log a warning
                warning(
                    LogRecord(
                        event=LogEvent.CACHE_EVENT.value,
                        message="Error tracking queue full, dropping error",
                        request_id=request_id,
                    )
                )

        except Exception as e:
            # Don't let error tracking errors break the application
            debug(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Error tracking failed: {str(e)}",
                    request_id=request_id,
                )
            )

    @asynccontextmanager
    async def track_context(
        self,
        error_type: ErrorType,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for error tracking."""
        try:
            yield
        except Exception as e:
            await self.track_error(
                error=e,
                error_type=error_type,
                request_id=request_id,
                metadata=metadata,
            )
            raise

    def track_errors_decorator(
        self,
        error_type: ErrorType = ErrorType.INTERNAL_ERROR,
        include_request: bool = True,
    ):
        """Decorator for automatic error tracking."""

        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Try to extract request from args
                    request_snapshot = None
                    request_id = None

                    if include_request:
                        for arg in args:
                            if isinstance(arg, Request):
                                request_snapshot = await self.capture_request_snapshot(
                                    arg
                                )
                                request_id = getattr(arg.state, "request_id", None)
                                break

                    await self.track_error(
                        error=e,
                        error_type=error_type,
                        request_id=request_id,
                        request_snapshot=request_snapshot,
                        metadata={"function": func.__name__},
                    )
                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # For sync functions, try to queue error tracking
                    asyncio.create_task(
                        self.track_error(
                            error=e,
                            error_type=error_type,
                            metadata={"function": func.__name__},
                        )
                    )
                    raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator


# Global singleton instance
error_tracker = ErrorTracker()


# Convenience functions
async def track_error(
    error: Exception,
    error_type: ErrorType = ErrorType.INTERNAL_ERROR,
    request_id: Optional[str] = None,
    request: Optional[Request] = None,
    response: Optional[Response] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Convenience function to track errors."""
    request_snapshot = None
    if request:
        request_snapshot = await error_tracker.capture_request_snapshot(request)

    response_snapshot = None
    if response:
        response_snapshot = error_tracker.capture_response_snapshot(response)

    await error_tracker.track_error(
        error=error,
        error_type=error_type,
        request_id=request_id,
        request_snapshot=request_snapshot,
        response_snapshot=response_snapshot,
        metadata=metadata,
    )


def get_error_type_from_exception(error: Exception) -> ErrorType:
    """Determine error type from exception type."""
    error_name = type(error).__name__.lower()

    if "timeout" in error_name:
        return ErrorType.TIMEOUT_ERROR
    elif "validation" in error_name or "invalid" in error_name:
        return ErrorType.VALIDATION_ERROR
    elif "auth" in error_name or "permission" in error_name:
        return ErrorType.AUTH_ERROR
    elif "rate" in error_name and "limit" in error_name:
        return ErrorType.RATE_LIMIT_ERROR
    elif "api" in error_name:
        return ErrorType.API_ERROR
    elif "cache" in error_name:
        return ErrorType.CACHE_ERROR
    elif "stream" in error_name:
        return ErrorType.STREAM_ERROR
    elif "convert" in error_name or "conversion" in error_name:
        return ErrorType.CONVERSION_ERROR
    else:
        return ErrorType.INTERNAL_ERROR
