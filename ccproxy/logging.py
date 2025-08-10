import dataclasses
import enum
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from logging.config import dictConfig
from typing import Any, Dict, Optional, Tuple

from .config import Settings

_REDACT_KEYS: list[str] = []

def _sanitize_for_json(obj):
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", "replace")
        except Exception:
            try:
                return obj.decode("latin-1", "replace")
            except Exception:
                return repr(obj)
    if dataclasses.is_dataclass(obj):
        return _sanitize_for_json(dataclasses.asdict(obj))
    if isinstance(obj, dict):
                redacted = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _REDACT_KEYS:
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = _sanitize_for_json(v)
        return redacted
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return repr(obj)


class LogEvent(enum.Enum):
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


@dataclasses.dataclass
class LogError:
    name: str
    message: str
    stack_trace: Optional[str] = None
    args: Optional[Tuple[Any, ...]] = None


@dataclasses.dataclass
class LogRecord:
    event: str
    message: str
    request_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[LogError] = None


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        header = {
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
            if detail.get("data") and isinstance(detail["data"], dict):
                for key, value in detail["data"].items():
                    if isinstance(value, str) and len(value) > 5000:
                        detail["data"][key] = value[:5000] + "...[truncated]"
            header["detail"] = detail
        else:
            header["message"] = record.getMessage()
            if record.exc_info:
                exc_type, exc_value, exc_tb = record.exc_info
                header["error"] = _sanitize_for_json({
                    "name": exc_type.__name__ if exc_type else "UnknownError",
                    "message": str(exc_value),
                    "stack_trace": "".join(
                        traceback.format_exception(exc_type, exc_value, exc_tb)
                    ),
                    "args": exc_value.args if hasattr(exc_value, "args") else [],
                })
        return json.dumps(_sanitize_for_json(header), ensure_ascii=False, separators=(",", ":"))


class ConsoleJSONFormatter(JSONFormatter):
    def format(self, record: logging.LogRecord) -> str:
        header = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        log_payload = getattr(record, "log_record", None)
        if isinstance(log_payload, LogRecord):
            detail = _sanitize_for_json(dataclasses.asdict(log_payload))
            if detail.get("error") and detail["error"].get("stack_trace"):
                detail["error"]["stack_trace"] = None
            header["detail"] = detail
        else:
            header["message"] = record.getMessage()
            if record.exc_info:
                exc_type, exc_value, _ = record.exc_info
                header["error"] = _sanitize_for_json({
                    "name": exc_type.__name__ if exc_type else "UnknownError",
                    "message": str(exc_value),
                    "args": exc_value.args if hasattr(exc_value, "args") else [],
                })
        return json.dumps(_sanitize_for_json(header), separators=(",", ":"))


_logger: Optional[logging.Logger] = None


def init_logging(settings: Settings) -> logging.Logger:
    global _logger
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {"()": JSONFormatter},
                "console_json": {"()": ConsoleJSONFormatter},
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "console_json",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "": {"handlers": ["default"], "level": "WARNING"},
                settings.app_name: {
                    "handlers": ["default"],
                    "level": settings.log_level.upper(),
                    "propagate": False,
                },
                "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "uvicorn.error": {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )
    _logger = logging.getLogger(settings.app_name)
    global _REDACT_KEYS
    _REDACT_KEYS = [k.lower() for k in settings.redact_log_fields]
    if settings.log_file_path:
        try:
            log_dir = os.path.dirname(settings.log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(settings.log_file_path, mode="a")
            file_handler.setFormatter(JSONFormatter())
            _logger.addHandler(file_handler)
        except Exception as e:
            _logger.warning("Failed to configure file logging: %s", e)
    return _logger


def _log(level: int, record: LogRecord, exc: Optional[Exception] = None) -> None:
    if exc:
        include_stack = False
        try:
            include_stack = any(isinstance(h, logging.FileHandler) for h in (_logger.handlers if _logger else []))
        except Exception:
            include_stack = False
        stack_str = None
        if include_stack:
            stack_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        record.error = LogError(
            name=type(exc).__name__,
            message=str(exc),
            stack_trace=stack_str,
            args=_sanitize_for_json(exc.args) if hasattr(exc, "args") else tuple(),
        )
        if not record.message and str(exc):
            record.message = str(exc)
        elif not record.message:
            record.message = "An unspecified error occurred"

    _logger.log(level=level, msg=record.message, extra={"log_record": record})


def is_debug_enabled() -> bool:
    return _logger is not None and _logger.isEnabledFor(logging.DEBUG)

def debug(record: LogRecord):
    _log(logging.DEBUG, record)


def info(record: LogRecord):
    _log(logging.INFO, record)


def warning(record: LogRecord, exc: Optional[Exception] = None):
    _log(logging.WARNING, record, exc=exc)


def error(record: LogRecord, exc: Optional[Exception] = None):
    _log(logging.ERROR, record, exc=exc)


def critical(record: LogRecord, exc: Optional[Exception] = None):
    _log(logging.CRITICAL, record, exc=exc)
