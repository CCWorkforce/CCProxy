from typing import Any, Dict, List, cast
import json
import time
import uuid
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from asyncio import create_task
import openai

from ....application.response_cache import ResponseCache

from ....config import (
    TOP_TIER_MODELS,
    ReasoningEfforts,
    Settings,
    NO_SUPPORT_TEMPERATURE_MODELS,
    SUPPORT_REASONING_EFFORT_MODELS,
    MODEL_INPUT_TOKEN_LIMIT_MAP,
)
from ....logging import debug, info, warning, LogRecord, LogEvent, is_debug_enabled
from ....domain.models import (
    MessagesRequest,
    TokenCountRequest,
    TokenCountResponse,
    AnthropicErrorType,
)
from ....application.tokenizer import (
    count_tokens_for_anthropic_request,
    truncate_request,
)
from ....application.converters import (
    convert_anthropic_to_openai_messages,
    convert_anthropic_tools_to_openai,
    convert_anthropic_tool_choice_to_openai,
    convert_openai_to_anthropic_response,
)
from ....application.model_selection import select_target_model
from ..streaming import handle_anthropic_streaming_response_from_openai_stream
from ..errors import (
    log_and_return_error_response,
    get_anthropic_error_details_from_execution,
    format_anthropic_error_sse_event,
)
from ..http_status import (
    BAD_REQUEST,
    UNPROCESSABLE_ENTITY,
    INTERNAL_SERVER_ERROR,
    PAYLOAD_TOO_LARGE,
)
from ....infrastructure.providers.openai_provider import OpenAIProvider

router = APIRouter()


@router.post("/v1/messages", response_model=None)
async def create_message_proxy(request: Request) -> Response:
    """Proxy Anthropic `/v1/messages` requests to configured OpenAI-compatible provider.

    Handles request validation, caching, model selection, parameter translation,
    streaming bridging, and maps provider errors back to Anthropic-style responses.

    Args:
        request (Request): Incoming HTTP request with Anthropic Messages payload

    Returns:
        Response: JSONResponse for non-streaming requests or StreamingResponse
            for streaming requests (when `stream=true`), formatted per Anthropic API.
    """
    settings: Settings = request.app.state.settings
    provider: OpenAIProvider = request.app.state.provider
    response_cache: ResponseCache = request.app.state.response_cache

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    request.state.request_id = request_id
    request.state.start_time_monotonic = getattr(
        request.state, "start_time_monotonic", time.monotonic()
    )

    try:
        raw_body = await request.json()
        debug(
            LogRecord(
                LogEvent.ANTHROPIC_REQUEST.value,
                "Received Anthropic request body",
                request_id,
                {"body": raw_body},
            )
        )
        anthropic_request = MessagesRequest.model_validate(raw_body)
    except json.JSONDecodeError as e:
        return await log_and_return_error_response(
            request,
            BAD_REQUEST,
            AnthropicErrorType.INVALID_REQUEST,
            "Invalid JSON body.",
            caught_exception=e,
        )
    except Exception as e:
        # Pydantic v2 ValidationError or others
        return await log_and_return_error_response(
            request,
            UNPROCESSABLE_ENTITY,
            AnthropicErrorType.INVALID_REQUEST,
            f"Invalid request body: {str(e)}",
            caught_exception=e,
        )

    if getattr(anthropic_request, "top_k", None) is not None:
        warning(
            LogRecord(
                LogEvent.STREAM_EVENT.value,
                "Parameter 'top_k' provided but not supported by OpenAI Chat Completions API; it will be omitted.",
                request_id,
                {"parameter": "top_k", "value": anthropic_request.top_k},
            )
        )

    is_stream = bool(anthropic_request.stream)

    target_model = select_target_model(
        anthropic_request.model,
        request_id,
        settings.big_model_name,
        settings.small_model_name,
    )

    # Check cache for non-streaming requests
    if not is_stream:
        cached_response = await response_cache.get_cached_response(
            anthropic_request,
            request_id=request_id,
            wait_for_pending=True,
            timeout_seconds=30.0,
        )
        if cached_response:
            duration_ms = (time.monotonic() - request.state.start_time_monotonic) * 1000
            info(
                LogRecord(
                    event=LogEvent.REQUEST_COMPLETED.value,
                    message=f"Returned cached response for {target_model}",
                    request_id=request_id,
                    data={
                        "status_code": 200,
                        "duration_ms": duration_ms,
                        "from_cache": True,
                        "client_model": anthropic_request.model,
                        "target_model": target_model,
                        "input_tokens": cached_response.usage.input_tokens,
                        "output_tokens": cached_response.usage.output_tokens,
                    },
                )
            )
            return JSONResponse(content=cached_response.model_dump(exclude_unset=True))

        # Mark request as pending to prevent duplicate processing
        await response_cache.mark_request_pending(anthropic_request)

    estimated_input_tokens = count_tokens_for_anthropic_request(
        messages=anthropic_request.messages,
        system=anthropic_request.system,
        model_name=anthropic_request.model,
        tools=anthropic_request.tools,
        request_id=request_id,
        settings=settings,
    )

    _limit = MODEL_INPUT_TOKEN_LIMIT_MAP.get(target_model, 200_000)
    if estimated_input_tokens > _limit:
        if settings.truncate_long_requests:
            anthropic_request.messages, anthropic_request.system = truncate_request(
                anthropic_request.messages,
                anthropic_request.system,
                anthropic_request.model,
                _limit,
                settings.truncation_config,
            )
            estimated_input_tokens = count_tokens_for_anthropic_request(
                messages=anthropic_request.messages,
                system=anthropic_request.system,
                model_name=anthropic_request.model,
                tools=anthropic_request.tools,
                request_id=request_id,
                settings=settings,
            )
        else:
            return await log_and_return_error_response(
                request,
                PAYLOAD_TOO_LARGE,
                AnthropicErrorType.REQUEST_TOO_LARGE,
                f"Input tokens {estimated_input_tokens} exceed limit {_limit} for model {target_model}.",
            )

    info(
        LogRecord(
            event=LogEvent.REQUEST_START.value,
            message=f"Processing request for {anthropic_request.model} (target: {target_model})",
            request_id=request_id,
            data={
                "client_model": anthropic_request.model,
                "target_model": target_model,
                "stream": is_stream,
                "estimated_input_tokens": estimated_input_tokens,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            },
        )
    )

    try:
        openai_messages = convert_anthropic_to_openai_messages(
            anthropic_request.messages,
            anthropic_request.system,
            request_id=request_id,
            target_model=target_model,
            settings=settings,
        )
        openai_tools = convert_anthropic_tools_to_openai(
            anthropic_request.tools, settings=settings
        )
        openai_tool_choice = convert_anthropic_tool_choice_to_openai(
            anthropic_request.tool_choice, request_id=request_id, settings=settings
        )
    except Exception as e:
        return await log_and_return_error_response(
            request,
            INTERNAL_SERVER_ERROR,
            AnthropicErrorType.API_ERROR,
            "Error during request conversion.",
            caught_exception=e,
        )

    openai_params: Dict[str, Any] = {
        "model": target_model,
        "messages": cast(List[Dict[str, Any]], openai_messages),
        "stream": is_stream,
    }
    if target_model not in SUPPORT_REASONING_EFFORT_MODELS:
        openai_params["max_tokens"] = anthropic_request.max_tokens
    else:
        warning(
            LogRecord(
                LogEvent.STREAM_EVENT.value,
                f"Model supports reasoning; 'max_tokens' will be omitted for model {target_model}.",
                request_id,
                {
                    "parameter": "max_tokens",
                    "value": anthropic_request.max_tokens,
                    "target_model": target_model,
                },
            )
        )
    if anthropic_request.temperature is not None:
        if target_model in NO_SUPPORT_TEMPERATURE_MODELS:
            warning(
                LogRecord(
                    LogEvent.STREAM_EVENT.value,
                    f"Model does not support 'temperature'; it will be omitted for model {target_model}.",
                    request_id,
                    {
                        "parameter": "temperature",
                        "value": anthropic_request.temperature,
                        "target_model": target_model,
                    },
                )
            )
        else:
            openai_params["temperature"] = anthropic_request.temperature or 0
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
    if (
        anthropic_request.thinking is not None
        and target_model in SUPPORT_REASONING_EFFORT_MODELS
    ):
        reasoning_effort = (
            ReasoningEfforts.High.value
            if is_stream
            else (
                ReasoningEfforts.Low.value
                if target_model in TOP_TIER_MODELS
                else ReasoningEfforts.Medium.value
            )
        )
        warning(
            LogRecord(
                LogEvent.STREAM_EVENT.value,
                f"Model supports reasoning; 'reasoning_effort' with value {reasoning_effort} will be added for model {target_model}.",
                request_id,
                {"parameter": "reasoning_effort", "value": f"{reasoning_effort}"},
            )
        )
        openai_params["reasoning_effort"] = reasoning_effort

    debug(
        LogRecord(
            LogEvent.OPENAI_REQUEST.value,
            f"Prepared OpenAI request parameters for {target_model}",
            request_id,
            {"params": openai_params},
        )
    )

    try:
        if is_stream:
            debug(
                LogRecord(
                    LogEvent.STREAMING_REQUEST.value,
                    f"Initiating streaming request to OpenAI-compatible API for {target_model}",
                    request_id,
                )
            )
            if settings.stream_dedupe_enabled:
                is_primary, q, key = await response_cache.subscribe_stream(
                    anthropic_request
                )
                if is_primary:
                    try:
                        openai_stream_response = await provider.create_chat_completion(
                            **openai_params
                        )
                    except openai.APIError as e:
                        err_type, err_msg, _, prov_details = (
                            get_anthropic_error_details_from_execution(e)
                        )
                        await response_cache.publish_stream_line(
                            key,
                            format_anthropic_error_sse_event(
                                err_type, err_msg, prov_details
                            ),
                        )
                        await response_cache.finalize_stream(key)
                        raise
                    except Exception:
                        await response_cache.publish_stream_line(
                            key,
                            format_anthropic_error_sse_event(
                                AnthropicErrorType.API_ERROR,
                                "Streaming request failed to start.",
                                None,
                            ),
                        )
                        await response_cache.finalize_stream(key)
                        raise

                    async def fanout():
                        async for (
                            line
                        ) in handle_anthropic_streaming_response_from_openai_stream(
                            openai_stream_response,
                            anthropic_request.model,
                            estimated_input_tokens,
                            request_id,
                            request.state.start_time_monotonic,
                            thinking_enabled=anthropic_request.thinking is not None,
                        ):
                            await response_cache.publish_stream_line(key, line)
                        await response_cache.finalize_stream(key)

                    create_task(fanout())

                async def gen():
                    try:
                        while True:
                            item = await q.get()
                            if item is None:
                                break
                            yield item
                    finally:
                        await response_cache.finalize_stream(key)

                return StreamingResponse(gen(), media_type="text/event-stream")
            else:
                openai_stream_response = await provider.create_chat_completion(
                    **openai_params
                )
                return StreamingResponse(
                    handle_anthropic_streaming_response_from_openai_stream(
                        openai_stream_response,
                        anthropic_request.model,
                        estimated_input_tokens,
                        request_id,
                        request.state.start_time_monotonic,
                        thinking_enabled=anthropic_request.thinking is not None,
                    ),
                    media_type="text/event-stream",
                )
        else:
            debug(
                LogRecord(
                    LogEvent.OPENAI_REQUEST.value,
                    f"Sending non-streaming request to OpenAI-compatible API for {target_model}",
                    request_id,
                )
            )
            openai_response_obj = await provider.create_chat_completion(**openai_params)
            if is_debug_enabled():
                response_data = openai_response_obj.model_dump()
                # Truncate large content for logging performance
                if "choices" in response_data:
                    for choice in response_data["choices"]:
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"]["content"]
                            if content and len(content) > 1000:
                                choice["message"]["content"] = (
                                    content[:1000] + "...[truncated]"
                                )
                debug(
                    LogRecord(
                        LogEvent.OPENAI_RESPONSE.value,
                        f"Received OpenAI-compatible response from {target_model}",
                        request_id,
                        {"response": response_data},
                    )
                )
            anthropic_response_obj = convert_openai_to_anthropic_response(
                openai_response_obj, anthropic_request.model, request_id=request_id
            )

            # Cache the successful response for future use
            await response_cache.cache_response(
                anthropic_request, anthropic_response_obj, request_id=request_id
            )

            duration_ms = (time.monotonic() - request.state.start_time_monotonic) * 1000
            info(
                LogRecord(
                    event=LogEvent.REQUEST_COMPLETED.value,
                    message=f"Non-streaming request completed successfully for {target_model}",
                    request_id=request_id,
                    data={
                        "status_code": 200,
                        "duration_ms": duration_ms,
                        "input_tokens": anthropic_response_obj.usage.input_tokens,
                        "output_tokens": anthropic_response_obj.usage.output_tokens,
                        "stop_reason": anthropic_response_obj.stop_reason,
                    },
                )
            )
            return JSONResponse(
                content=anthropic_response_obj.model_dump(exclude_unset=True)
            )
    except openai.APIError as e:
        # Clear pending request on error
        if not is_stream:
            await response_cache.clear_pending_request(anthropic_request)
        err_type, err_msg, err_status, prov_details = (
            get_anthropic_error_details_from_execution(e)
        )
        return await log_and_return_error_response(
            request, err_status, err_type, err_msg, prov_details, e
        )
    except Exception as e:
        # Clear pending request on error
        if not is_stream:
            await response_cache.clear_pending_request(anthropic_request)
        return await log_and_return_error_response(
            request,
            INTERNAL_SERVER_ERROR,
            AnthropicErrorType.API_ERROR,
            "An unexpected error occurred while processing the request.",
            caught_exception=e,
        )


@router.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: Request) -> TokenCountResponse:
    """Process token count requests per Anthropic's Messages API specification.

    Accepts Messages-like payloads and returns computed token count using application
    tokenization utilities. Mirrors Anthropic's `/v1/messages/count_tokens` endpoint.

    Args:
        request: The HTTP request containing TokenCountRequest data

    Returns:
        TokenCountResponse: Object containing the calculated input token count
    """
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
        settings=request.app.state.settings,
    )
    duration_ms = (time.monotonic() - start_time_mono) * 1000
    info(
        LogRecord(
            event=LogEvent.TOKEN_COUNT.value,
            message=f"Counted {token_count} tokens",
            request_id=request_id,
            data={
                "duration_ms": duration_ms,
                "token_count": token_count,
                "model": count_request.model,
            },
        )
    )
    return TokenCountResponse(input_tokens=token_count)
