from typing import Any, Dict, List, Optional, Union, cast
import json
import time
import uuid
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
import openai

from ....config import Settings
from ....logging import debug, info, warning, error, LogRecord, LogEvent
from ....domain.models import (
    MessagesRequest, TokenCountRequest, TokenCountResponse, AnthropicErrorType
)
from ....application.tokenizer import count_tokens_for_anthropic_request
from ....application.converters import (
    convert_anthropic_to_openai_messages,
    convert_anthropic_tools_to_openai,
    convert_anthropic_tool_choice_to_openai,
    convert_openai_to_anthropic_response,
)
from ....application.model_selection import select_target_model
from ...http.streaming import handle_anthropic_streaming_response_from_openai_stream
from ...http.errors import _log_and_return_error_response, _get_anthropic_error_details_from_exc

router = APIRouter()


@router.post("/v1/messages")
async def create_message_proxy(request: Request) -> Union[JSONResponse, StreamingResponse]:
    settings: Settings = request.app.state.settings
    provider = request.app.state.provider

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    request.state.request_id = request_id
    request.state.start_time_monotonic = getattr(request.state, "start_time_monotonic", time.monotonic())

    try:
        raw_body = await request.json()
        debug(LogRecord(LogEvent.ANTHROPIC_REQUEST.value, "Received Anthropic request body", request_id, {"body": raw_body}))
        anthropic_request = MessagesRequest.model_validate(raw_body)
    except json.JSONDecodeError as e:
        return await _log_and_return_error_response(request, 400, AnthropicErrorType.INVALID_REQUEST, "Invalid JSON body.", caught_exception=e)
    except Exception as e:
        # Pydantic v2 ValidationError or others
        return await _log_and_return_error_response(request, 422, AnthropicErrorType.INVALID_REQUEST, f"Invalid request body: {str(e)}", caught_exception=e)

    if getattr(anthropic_request, "top_k", None) is not None:
        warning(LogRecord(LogEvent.PARAMETER_UNSUPPORTED.value, "Parameter 'top_k' provided but not supported by OpenAI Chat Completions API; it will be ignored.", request_id, {"parameter": "top_k", "value": anthropic_request.top_k}))

    is_stream = bool(anthropic_request.stream)
    target_model_name = select_target_model(anthropic_request.model, request_id, settings.big_model_name, settings.small_model_name)

    estimated_input_tokens = count_tokens_for_anthropic_request(
        messages=anthropic_request.messages,
        system=anthropic_request.system,
        model_name=anthropic_request.model,
        tools=anthropic_request.tools,
        request_id=request_id,
    )

    info(LogRecord(
        event=LogEvent.REQUEST_START.value,
        message="Processing new message request",
        request_id=request_id,
        data={
            "client_model": anthropic_request.model,
            "target_model": target_model_name,
            "stream": is_stream,
            "estimated_input_tokens": estimated_input_tokens,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
        },
    ))

    try:
        openai_messages = convert_anthropic_to_openai_messages(anthropic_request.messages, anthropic_request.system, request_id=request_id)
        openai_tools = convert_anthropic_tools_to_openai(anthropic_request.tools)
        openai_tool_choice = convert_anthropic_tool_choice_to_openai(anthropic_request.tool_choice, request_id=request_id)
    except Exception as e:
        return await _log_and_return_error_response(request, 500, AnthropicErrorType.API_ERROR, "Error during request conversion.", caught_exception=e)

    openai_params: Dict[str, Any] = {
        "model": target_model_name,
        "messages": cast(List[Dict[str, Any]], openai_messages),
        "max_tokens": anthropic_request.max_tokens,
        "stream": is_stream,
    }
    if anthropic_request.temperature is not None:
        openai_params["temperature"] = anthropic_request.temperature
    if anthropic_request.top_p is not None:
        openai_params["top_p"] = anthropic_request.top_p
    if anthropic_request.stop_sequences:
        openai_params["stop"] = anthropic_request.stop_sequences
    if openai_tools:
        openai_params["tools"] = openai_tools
    if openai_tool_choice:
        openai_params["tool_choice"] = openai_tool_choice
    if anthropic_request.metadata and anthropic_request.metadata.get("user_id"):
        user_val = str(anthropic_request.metadata["user_id"])
        openai_params["user"] = user_val[:128] if len(user_val) > 128 else user_val

    debug(LogRecord(LogEvent.OPENAI_REQUEST.value, "Prepared OpenAI request parameters", request_id, {"params": openai_params}))

    try:
        if is_stream:
            debug(LogRecord(LogEvent.STREAMING_REQUEST.value, "Initiating streaming request to OpenAI-compatible API", request_id))
            openai_stream_response = await provider.create_chat_completion(**openai_params)
            return StreamingResponse(
                handle_anthropic_streaming_response_from_openai_stream(
                    openai_stream_response,
                    anthropic_request.model,
                    estimated_input_tokens,
                    request_id,
                    request.state.start_time_monotonic,
                ),
                media_type="text/event-stream",
            )
        else:
            debug(LogRecord(LogEvent.OPENAI_REQUEST.value, "Sending non-streaming request to OpenAI-compatible API", request_id))
            openai_response_obj = await provider.create_chat_completion(**openai_params)
            debug(LogRecord(LogEvent.OPENAI_RESPONSE.value, "Received OpenAI response", request_id, {"response": getattr(openai_response_obj, "model_dump", lambda: {} )()}))
            anthropic_response_obj = convert_openai_to_anthropic_response(openai_response_obj, anthropic_request.model, request_id=request_id)
            duration_ms = (time.monotonic() - request.state.start_time_monotonic) * 1000
            info(LogRecord(
                event=LogEvent.REQUEST_COMPLETED.value,
                message="Non-streaming request completed successfully",
                request_id=request_id,
                data={
                    "status_code": 200,
                    "duration_ms": duration_ms,
                    "input_tokens": anthropic_response_obj.usage.input_tokens,
                    "output_tokens": anthropic_response_obj.usage.output_tokens,
                    "stop_reason": anthropic_response_obj.stop_reason,
                },
            ))
            return JSONResponse(content=anthropic_response_obj.model_dump(exclude_unset=True))
    except openai.APIError as e:
        err_type, err_msg, err_status, prov_details = _get_anthropic_error_details_from_exc(e)
        return await _log_and_return_error_response(request, err_status, err_type, err_msg, prov_details, e)
    except Exception as e:
        return await _log_and_return_error_response(request, 500, AnthropicErrorType.API_ERROR, "An unexpected error occurred while processing the request.", caught_exception=e)


@router.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: Request) -> TokenCountResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    setattr(request.state, "request_id", request_id)
    start_time_mono = time.monotonic()

    body = await request.json()
    count_request = TokenCountRequest.model_validate(body)

    token_count = count_tokens_for_anthropic_request(
        messages=count_request.messages,
        system=count_request.system,
        model_name=count_request.model,
        tools=count_request.tools,
        request_id=request_id,
    )
    duration_ms = (time.monotonic() - start_time_mono) * 1000
    info(LogRecord(
        event=LogEvent.TOKEN_COUNT.value,
        message=f"Counted {token_count} tokens",
        request_id=request_id,
        data={"duration_ms": duration_ms, "token_count": token_count, "model": count_request.model},
    ))
    return TokenCountResponse(input_tokens=token_count)
