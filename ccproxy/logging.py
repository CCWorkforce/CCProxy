import dataclasses
import enum
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from logging.config import dictConfig
from typing import Any, Dict, Optional, Tuple

from .config import Settings


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
            header["detail"] = dataclasses.asdict(log_payload)
        else:
            header["message"] = record.getMessage()
            if record.exc_info:
                exc_type, exc_value, exc_tb = record.exc_info
                header["error"] = {
                    "name": exc_type.__name__ if exc_type else "UnknownError",
                    "message": str(exc_value),
                    "stack_trace": "".join(
                        traceback.format_exception(exc_type, exc_value, exc_tb)
                    ),
                    "args": exc_value.args if hasattr(exc_value, "args") else [],
                }
        return json.dumps(header, ensure_ascii=False)


class ConsoleJSONFormatter(JSONFormatter):
    def format(self, record: logging.LogRecord) -> str:
        log_dict = json.loads(super().format(record))
        if (
            "detail" in log_dict
            and "error" in log_dict["detail"]
            and log_dict["detail"]["error"]
        ):
            if "stack_trace" in log_dict["detail"]["error"]:
                del log_dict["detail"]["error"]["stack_trace"]
        elif "error" in log_dict and log_dict["error"]:
            if "stack_trace" in log_dict["error"]:
                del log_dict["error"]["stack_trace"]
        return json.dumps(log_dict)


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
        record.error = LogError(
            name=type(exc).__name__,
            message=str(exc),
            stack_trace="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            args=exc.args if hasattr(exc, "args") else tuple(),
        )
        if not record.message and str(exc):
            record.message = str(exc)
        elif not record.message:
            record.message = "An unspecified error occurred"

    _logger.log(level=level, msg=record.message, extra={"log_record": record})


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
