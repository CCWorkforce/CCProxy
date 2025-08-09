import json
import time
from typing import Any, Dict, Optional, Tuple

import openai
from fastapi import Request
from fastapi.responses import JSONResponse

from ...domain.models import AnthropicErrorType, AnthropicErrorDetail, AnthropicErrorResponse, ProviderErrorMetadata
from ...logging import error, warning, LogRecord, LogEvent


STATUS_CODE_ERROR_MAP: Dict[int, AnthropicErrorType] = {
    400: AnthropicErrorType.INVALID_REQUEST,
    401: AnthropicErrorType.AUTHENTICATION,
    403: AnthropicErrorType.PERMISSION,
    404: AnthropicErrorType.NOT_FOUND,
    413: AnthropicErrorType.REQUEST_TOO_LARGE,
    422: AnthropicErrorType.INVALID_REQUEST,
    429: AnthropicErrorType.RATE_LIMIT,
    500: AnthropicErrorType.API_ERROR,
    502: AnthropicErrorType.API_ERROR,
    503: AnthropicErrorType.OVERLOADED,
    504: AnthropicErrorType.API_ERROR,
}


def extract_provider_error_details(
    error_details_dict: Optional[Dict[str, Any]],
) -> Optional[ProviderErrorMetadata]:
    if not isinstance(error_details_dict, dict):
        return None
    metadata = error_details_dict.get("metadata")
    if not isinstance(metadata, dict):
        return None
    provider_name = metadata.get("provider_name")
    raw_error_str = metadata.get("raw")

    if not provider_name or not isinstance(provider_name, str):
        return None

    parsed_raw_error: Optional[Dict[str, Any]] = None
    if isinstance(raw_error_str, str):
        try:
            parsed_raw_error = json.loads(raw_error_str)
        except json.JSONDecodeError:
            warning(
                LogRecord(
                    event=LogEvent.PROVIDER_ERROR_DETAILS.value,
                    message=f"Failed to parse raw provider error string for {provider_name}.",
                )
            )
            parsed_raw_error = {"raw_string_parse_failed": raw_error_str}
    elif isinstance(raw_error_str, dict):
        parsed_raw_error = raw_error_str

    return ProviderErrorMetadata(
        provider_name=provider_name, raw_error=parsed_raw_error
    )


def _get_anthropic_error_details_from_exc(
    exc: Exception,
) -> Tuple[AnthropicErrorType, str, int, Optional[ProviderErrorMetadata]]:
    """Maps caught exceptions to Anthropic error type, message, status code, and provider details."""
    error_type = AnthropicErrorType.API_ERROR
    error_message = str(exc)
    status_code = 500
    provider_details: Optional[ProviderErrorMetadata] = None

    if isinstance(exc, openai.APIError):
        error_message = exc.message or str(exc)
        status_code = exc.status_code or 500
        error_type = STATUS_CODE_ERROR_MAP.get(
            status_code, AnthropicErrorType.API_ERROR
        )

        if hasattr(exc, "body") and isinstance(exc.body, dict):
            actual_error_details = exc.body.get("error", exc.body)
            provider_details = extract_provider_error_details(actual_error_details)

    if isinstance(exc, openai.AuthenticationError):
        error_type = AnthropicErrorType.AUTHENTICATION
    elif isinstance(exc, openai.RateLimitError):
        error_type = AnthropicErrorType.RATE_LIMIT
    elif isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        error_type = AnthropicErrorType.INVALID_REQUEST
    elif isinstance(exc, openai.PermissionDeniedError):
        error_type = AnthropicErrorType.PERMISSION
    elif isinstance(exc, openai.NotFoundError):
        error_type = AnthropicErrorType.NOT_FOUND

    return error_type, error_message, status_code, provider_details


def _format_anthropic_error_sse_event(
    error_type: AnthropicErrorType,
    message: str,
    provider_details: Optional[ProviderErrorMetadata] = None,
) -> str:
    """Formats an error into the Anthropic SSE 'error' event structure."""
    anthropic_err_detail = AnthropicErrorDetail(type=error_type, message=message)
    if provider_details:
        anthropic_err_detail.provider = provider_details.provider_name
        if provider_details.raw_error and isinstance(
            provider_details.raw_error.get("error"), dict
        ):
            prov_err_obj = provider_details.raw_error["error"]
            anthropic_err_detail.provider_message = prov_err_obj.get("message")
            anthropic_err_detail.provider_code = prov_err_obj.get("code")
        elif provider_details.raw_error and isinstance(
            provider_details.raw_error.get("message"), str
        ):
            anthropic_err_detail.provider_message = provider_details.raw_error.get(
                "message"
            )
            anthropic_err_detail.provider_code = provider_details.raw_error.get("code")

    error_response = AnthropicErrorResponse(error=anthropic_err_detail)
    return f"event: error\ndata: {error_response.model_dump_json()}\n\n"


def _build_anthropic_error_response(
    error_type: AnthropicErrorType,
    message: str,
    status_code: int,
    provider_details: Optional[ProviderErrorMetadata] = None,
) -> JSONResponse:
    """Creates a JSONResponse with Anthropic-formatted error."""
    err_detail = AnthropicErrorDetail(type=error_type, message=message)
    if provider_details:
        err_detail.provider = provider_details.provider_name
        if provider_details.raw_error:
            if isinstance(provider_details.raw_error, dict):
                prov_err_obj = provider_details.raw_error.get("error")
                if isinstance(prov_err_obj, dict):
                    err_detail.provider_message = prov_err_obj.get("message")
                    err_detail.provider_code = prov_err_obj.get("code")
                elif isinstance(provider_details.raw_error.get("message"), str):
                    err_detail.provider_message = provider_details.raw_error.get(
                        "message"
                    )
                    err_detail.provider_code = provider_details.raw_error.get("code")

    error_resp_model = AnthropicErrorResponse(error=err_detail)
    return JSONResponse(
        status_code=status_code, content=error_resp_model.model_dump(exclude_unset=True)
    )


async def _log_and_return_error_response(
    request: Request,
    status_code: int,
    anthropic_error_type: AnthropicErrorType,
    error_message: str,
    provider_details: Optional[ProviderErrorMetadata] = None,
    caught_exception: Optional[Exception] = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    start_time_mono = getattr(request.state, "start_time_monotonic", time.monotonic())
    duration_ms = (time.monotonic() - start_time_mono) * 1000

    log_data = {
        "status_code": status_code,
        "duration_ms": duration_ms,
        "error_type": anthropic_error_type.value,
        "client_ip": request.client.host if request.client else "unknown",
    }
    if provider_details:
        log_data["provider_name"] = provider_details.provider_name
        log_data["provider_raw_error"] = provider_details.raw_error

    error(
        LogRecord(
            event=LogEvent.REQUEST_FAILURE.value,
            message=f"Request failed: {error_message}",
            request_id=request_id,
            data=log_data,
        ),
        exc=caught_exception,
    )
    return _build_anthropic_error_response(
        anthropic_error_type, error_message, status_code, provider_details
    )
