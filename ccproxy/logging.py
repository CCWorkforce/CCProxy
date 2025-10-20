import dataclasses
import enum
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener
import queue
import sys
from typing import Any, Dict, Optional, Tuple, List
from logging import Handler

from .config import Settings
from ._cython import CYTHON_ENABLED

# Try to import Cython-optimized functions
if CYTHON_ENABLED:
    try:
        from ._cython.dict_ops import (
            recursive_filter_none,
        )
        from ._cython.json_ops import (
            json_dumps_compact,
            is_json_serializable,
        )

        _USING_CYTHON = True
    except ImportError:
        _USING_CYTHON = False
else:
    _USING_CYTHON = False

# Fallback to pure Python implementations if Cython not available
if not _USING_CYTHON:

    def recursive_filter_none(data: Any) -> Any:
        """Recursively remove None values."""
        if isinstance(data, dict):
            return {
                k: recursive_filter_none(v) for k, v in data.items() if v is not None
            }
        elif isinstance(data, list):
            return [recursive_filter_none(item) for item in data if item is not None]
        return data

    def json_dumps_compact(obj: Any) -> str:
        """Compact JSON serialization with minimal separators."""
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    def is_json_serializable(obj: Any) -> bool:
        """Check if object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False


_REDACT_KEYS: set[str] = set()

_logger: Optional[logging.Logger] = None
_log_listener: Optional[QueueListener] = None


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize an object for JSON serialization.

    Converts non-serializable types (bytes, dataclasses, etc.) into JSON-compatible structures while redacting sensitive fields. Handles:
    - Bytes: Decoded as UTF-8 (with 'replace' on error), falling back to Latin-1
    - Dataclasses: Converted to dictionaries
    - Dictionaries: Redacts keys listed in _REDACT_KEYS and removes null values
    - Lists/sets/tuples: Recursively sanitizes each element
    - Non-serializable objects: Converted via repr()

    Args:
        obj (Any): Input object to sanitize.

    Returns:
        Any: JSON-serializable structure with sensitive data redacted and null values removed.
    """
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", "replace")
        except Exception:
            try:
                return obj.decode("latin-1", "replace")
            except Exception:
                return repr(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _sanitize_for_json(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        redacted = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _REDACT_KEYS:
                redacted[k] = "***REDACTED***"
            else:
                sanitized_value = _sanitize_for_json(v)
                # Only include non-null values
                if sanitized_value is not None:
                    redacted[k] = sanitized_value
        return redacted
    if isinstance(obj, (list, tuple, set)):
        sanitized_list = [_sanitize_for_json(x) for x in obj]
        # Filter out None values from lists (Cython-optimized)
        return recursive_filter_none(sanitized_list)
    # Use Cython-optimized JSON serialization check
    if is_json_serializable(obj):
        return obj
    return repr(obj)


class LogEvent(enum.Enum):
    """Enumeration of structured log events emitted throughout CCProxy.

    Each value marks a distinct milestone or error category during request
    processing, model selection, streaming, tool handling, or health checks.
    These constants are used in ``LogRecord.event`` for consistent analytics
    and monitoring.
    """

    MODEL_SELECTION = "model_selection"
    REQUEST_START = "request_start"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILURE = "request_failure"
    ANTHROPIC_REQUEST = "anthropic_body"
    OPENAI_REQUEST = "openai_request"
    OPENAI_RESPONSE = "openai_response"
    ANTHROPIC_RESPONSE = "anthropic_response"
    STREAMING_REQUEST = "streaming_request"
    STREAM_INTERRUPTED = "stream_interrupted"
    TOKEN_COUNT = "token_count"
    TOKEN_ENCODER_LOAD_FAILED = "token_encoder_load_failed"
    SYSTEM_PROMPT_ADJUSTED = "system_prompt_adjusted"
    TOOL_INPUT_SERIALIZATION_FAILURE = "tool_input_serialization_failure"
    IMAGE_FORMAT_UNSUPPORTED = "image_format_unsupported"
    MESSAGE_FORMAT_NORMALIZED = "message_format_normalized"
    TOOL_RESULT_SERIALIZATION_FAILURE = "tool_result_serialization_failure"
    TOOL_RESULT_PROCESSING = "tool_result_processing"
    TOOL_CHOICE_UNSUPPORTED = "tool_choice_unsupported"
    TOOL_ARGS_TYPE_MISMATCH = "tool_args_type_mismatch"
    TOOL_ARGS_PARSE_FAILURE = "tool_args_parse_failure"
    TOOL_ARGS_UNEXPECTED = "tool_args_unexpected"
    TOOL_ID_PLACEHOLDER = "tool_id_placeholder"
    TOOL_ID_UPDATED = "tool_id_updated"
    PARAMETER_UNSUPPORTED = "parameter_unsupported"
    HEALTH_CHECK = "health_check"
    PROVIDER_ERROR_DETAILS = "provider_error_details"
    CACHE_EVENT = "cache_event"
    STREAM_EVENT = "stream_event"
    CONVERSION_EVENT = "conversion_event"


@dataclasses.dataclass
class LogError:
    """Structured representation of an exception attached to a log entry.

    Attributes:
        name: Exception class name.
        message: Human-readable description.
        stack_trace: Full traceback string (may be ``None`` when suppressed).
        args: JSON-safe serialization of ``Exception.args``.
    """

    name: str
    message: str
    stack_trace: Optional[str] = None
    args: Optional[Tuple[Any, ...]] = None


@dataclasses.dataclass
class LogRecord:
    """Primary payload transported via the logging system.

    Attributes:
        event: Identifier from :class:`LogEvent` or custom tag.
        message: Short human-readable summary.
        request_id: Correlator generated per HTTP request.
        data: Arbitrary contextual dictionary (sanitized/truncated).
        error: Optional :class:`LogError` with exception details.
    """

    event: str
    message: str
    request_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[LogError] = None


class JSONFormatter(logging.Formatter):
    """Formatter that outputs log records as compact JSON lines.

    Used for file logging or machine-ingestible stdout. It injects timestamp,
    level and logger name, serializes attached :class:`LogRecord`, truncates
    oversized strings, and redacts configured sensitive fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        header: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        log_payload = getattr(record, "log_record", None)
        if isinstance(log_payload, LogRecord):
            detail = _sanitize_for_json(dataclasses.asdict(log_payload))
            # Limit data field size for performance
            if (
                isinstance(detail, dict)
                and detail.get("data")
                and isinstance(detail["data"], dict)
            ):
                for key, value in detail["data"].items():
                    if isinstance(value, str) and len(value) > 5000:
                        detail["data"][key] = value[:5000] + "...[truncated]"
            header["detail"] = detail
        else:
            header["message"] = record.getMessage()
            if record.exc_info:
                exc_type, exc_value, exc_tb = record.exc_info
                header["error"] = _sanitize_for_json(
                    {
                        "name": exc_type.__name__ if exc_type else "UnknownError",
                        "message": str(exc_value),
                        "stack_trace": "".join(
                            traceback.format_exception(exc_type, exc_value, exc_tb)
                        ),
                        "args": exc_value.args
                        if exc_value and hasattr(exc_value, "args")
                        else [],
                    }
                )
        # Use Cython-optimized JSON serialization
        return json_dumps_compact(_sanitize_for_json(header))


class ConsoleJSONFormatter(JSONFormatter):
    """Variant of :class:`JSONFormatter` tuned for interactive consoles.

    Removes stack traces for brevity and supports pretty-print when
    ``settings.log_pretty_console`` is true while preserving JSON structure.
    """

    def format(self, record: logging.LogRecord) -> str:
        header: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        log_payload = getattr(record, "log_record", None)
        if isinstance(log_payload, LogRecord):
            detail = _sanitize_for_json(dataclasses.asdict(log_payload))
            if (
                isinstance(detail, dict)
                and detail.get("error")
                and detail["error"].get("stack_trace")
            ):
                detail["error"]["stack_trace"] = None
            header["detail"] = detail
        else:
            header["message"] = record.getMessage()
            if record.exc_info:
                exc_type, exc_value, _ = record.exc_info
                header["error"] = _sanitize_for_json(
                    {
                        "name": exc_type.__name__ if exc_type else "UnknownError",
                        "message": str(exc_value),
                        "args": exc_value.args
                        if exc_value and hasattr(exc_value, "args")
                        else [],
                    }
                )
        # Use Cython-optimized JSON serialization
        return json_dumps_compact(_sanitize_for_json(header))


def init_logging(settings: Settings) -> logging.Logger:
    global _logger
    global _log_listener
    _log_listener = None

    # Create logging queue and handler
    log_queue = queue.Queue(-1)
    queue_handler = QueueHandler(log_queue)

    # Setup main console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(
        ConsoleJSONFormatter() if settings.log_pretty_console else JSONFormatter()
    )

    handlers: List[Handler] = [console_handler]

    # Setup file handlers if configured
    if settings.log_file_path:
        try:
            log_dir = os.path.dirname(settings.log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                settings.log_file_path, mode="a", encoding="utf-8"
            )
            file_handler.setFormatter(JSONFormatter())
            handlers.append(file_handler)
        except Exception as e:
            if _logger:
                _logger.warning("Failed to configure file logging: %s", e)

    error_log_path = getattr(settings, "error_log_file_path", None)
    if error_log_path:
        try:
            err_dir = os.path.dirname(error_log_path)
            if err_dir:
                os.makedirs(err_dir, exist_ok=True)
            err_handler = logging.FileHandler(
                error_log_path, mode="a", encoding="utf-8"
            )
            err_handler.setLevel(logging.ERROR)
            err_handler.setFormatter(JSONFormatter())
            handlers.append(err_handler)
        except Exception as e:
            if _logger:
                _logger.warning("Failed to configure error file logging: %s", e)

    # Configure QueueListener with all handlers
    _log_listener = QueueListener(log_queue, *handlers)
    _log_listener.start()

    # Configure loggers to use queue handler
    for logger_name in [
        "",
        settings.app_name,
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ]:
        logger = logging.getLogger(logger_name)
        logger.handlers = [queue_handler]
        logger.propagate = False if logger_name != "" else True
        logger.setLevel(
            logging.WARNING
            if logger_name == ""
            else settings.log_level.upper()
            if logger_name == settings.app_name
            else "INFO"
        )
    _logger = logging.getLogger(settings.app_name)
    global _REDACT_KEYS
    _REDACT_KEYS = {k.lower() for k in settings.redact_log_fields}
    return _logger


def shutdown_logging() -> None:
    """Safely shutdown logging system, flushing all messages."""
    global _log_listener
    if _log_listener:
        _log_listener.stop()
        _log_listener = None


def _log(level: int, record: LogRecord, exc: Optional[Exception] = None) -> None:
    """Internal helper to log structured messages with exception handling.

    Processes the exception (if provided) into the LogRecord's error field
    and emits the log entry at the specified level.

    Args:
        level: The logging level (e.g., logging.DEBUG, logging.ERROR)
        record: The structured log record containing event details
        exc: Optional exception to include in error details
    """
    if exc:
        include_stack = False
        try:
            include_stack = any(
                isinstance(h, logging.FileHandler)
                for h in (_logger.handlers if _logger else [])
            )
        except Exception:
            include_stack = False
        stack_str = None
        if include_stack:
            stack_str = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        sanitized_args = None
        if hasattr(exc, "args"):
            sanitized = _sanitize_for_json(exc.args)
            sanitized_args = (
                tuple(sanitized)
                if isinstance(sanitized, (list, tuple))
                else (sanitized,)
            )

        record.error = LogError(
            name=type(exc).__name__,
            message=str(exc),
            stack_trace=stack_str,
            args=sanitized_args,
        )
        if not record.message and str(exc):
            record.message = str(exc)
        elif not record.message:
            record.message = "An unspecified error occurred"

    if _logger:
        _logger.log(level=level, msg=record.message, extra={"log_record": record})


def is_debug_enabled() -> bool:
    return _logger is not None and _logger.isEnabledFor(logging.DEBUG)


def debug(record: LogRecord) -> None:
    _log(logging.DEBUG, record)


def info(record: LogRecord) -> None:
    _log(logging.INFO, record)


def warning(record: LogRecord, exc: Optional[Exception] = None) -> None:
    _log(logging.WARNING, record, exc=exc)


def error(record: LogRecord, exc: Optional[Exception] = None) -> None:
    _log(logging.ERROR, record, exc=exc)


def critical(record: LogRecord, exc: Optional[Exception] = None) -> None:
    _log(logging.CRITICAL, record, exc=exc)
