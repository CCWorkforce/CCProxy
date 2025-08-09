import json
import time
import uuid
from typing import AsyncGenerator, Dict, Optional, Literal, Any

import openai

from ...application.tokenizer import get_token_encoder
from ...logging import warning, debug, error, info, LogRecord, LogEvent
from .errors import _get_anthropic_error_details_from_exc, format_anthropic_error_sse_event


StopReasonType = Optional[
    Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]
]


async def handle_anthropic_streaming_response_from_openai_stream(
    openai_stream: openai.AsyncStream[openai.types.chat.ChatCompletionChunk],
    original_anthropic_model_name: str,
    estimated_input_tokens: int,
    request_id: str,
    start_time_mono: float,
) -> AsyncGenerator[str, None]:
    """
    Consumes an OpenAI stream and yields Anthropic-compatible SSE events.
    BUGFIX: Correctly handles content block indexing for mixed text/tool_use.
    """

    anthropic_message_id = f"msg_stream_{request_id}_{uuid.uuid4().hex[:8]}"

    next_anthropic_block_idx = 0
    text_block_anthropic_idx: Optional[int] = None

    openai_tool_idx_to_anthropic_block_idx: Dict[int, int] = {}

    tool_states: Dict[int, Dict[str, Any]] = {}

    sent_tool_block_starts: set[int] = set()

    output_token_count = 0
    final_anthropic_stop_reason: StopReasonType = None

    enc = get_token_encoder(original_anthropic_model_name, request_id)

    openai_to_anthropic_stop_reason_map: Dict[Optional[str], StopReasonType] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "stop_sequence",
        None: None,
    }

    stream_status_code = 200
    stream_final_message = "Streaming request completed successfully."
    stream_log_event = LogEvent.REQUEST_COMPLETED.value

    try:
        message_start_event_data = {
            "type": "message_start",
            "message": {
                "id": anthropic_message_id,
                "type": "message",
                "role": "assistant",
                "model": original_anthropic_model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": estimated_input_tokens, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start_event_data)}\n\n"
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        async for chunk in openai_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            openai_finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                # Cache content for batch encoding (more efficient)
                content_batch = delta.content
                output_token_count += len(enc.encode(content_batch))
                if text_block_anthropic_idx is None:
                    text_block_anthropic_idx = next_anthropic_block_idx
                    next_anthropic_block_idx += 1
                    start_text_event = {
                        "type": "content_block_start",
                        "index": text_block_anthropic_idx,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(start_text_event)}\n\n"

                text_delta_event = {
                    "type": "content_block_delta",
                    "index": text_block_anthropic_idx,
                    "delta": {"type": "text_delta", "text": delta.content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(text_delta_event)}\n\n"

            if delta.tool_calls:
                for tool_delta in delta.tool_calls:
                    openai_tc_idx = tool_delta.index

                    if openai_tc_idx not in openai_tool_idx_to_anthropic_block_idx:
                        current_anthropic_tool_block_idx = next_anthropic_block_idx
                        next_anthropic_block_idx += 1
                        openai_tool_idx_to_anthropic_block_idx[openai_tc_idx] = (
                            current_anthropic_tool_block_idx
                        )

                        tool_states[current_anthropic_tool_block_idx] = {
                            "id": tool_delta.id
                            or f"tool_ph_{request_id}_{current_anthropic_tool_block_idx}",
                            "name": "",
                            "arguments_buffer": "",
                        }
                        if not tool_delta.id:
                            warning(
                                LogRecord(
                                    LogEvent.TOOL_ID_PLACEHOLDER.value,
                                    f"Generated placeholder Tool ID for OpenAI tool index {openai_tc_idx} -> Anthropic block {current_anthropic_tool_block_idx}",
                                    request_id,
                                )
                            )
                    else:
                        current_anthropic_tool_block_idx = (
                            openai_tool_idx_to_anthropic_block_idx[openai_tc_idx]
                        )

                    tool_state = tool_states[current_anthropic_tool_block_idx]

                    if tool_delta.id and tool_state["id"].startswith("tool_ph_"):
                        debug(
                            LogRecord(
                                LogEvent.TOOL_ID_UPDATED.value,
                                f"Updated placeholder Tool ID for Anthropic block {current_anthropic_tool_block_idx} to {tool_delta.id}",
                                request_id,
                            )
                        )
                        tool_state["id"] = tool_delta.id

                    if tool_delta.function:
                        if tool_delta.function.name:
                            tool_state["name"] = tool_delta.function.name
                        if tool_delta.function.arguments:
                            tool_state["arguments_buffer"] += (
                                tool_delta.function.arguments
                            )
                            output_token_count += len(
                                enc.encode(tool_delta.function.arguments)
                            )

                    if (
                        current_anthropic_tool_block_idx not in sent_tool_block_starts
                        and tool_state["id"]
                        and not tool_state["id"].startswith("tool_ph_")
                        and tool_state["name"]
                    ):
                        start_tool_event = {
                            "type": "content_block_start",
                            "index": current_anthropic_tool_block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_state["id"],
                                "name": tool_state["name"],
                                "input": {},
                            },
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(start_tool_event)}\n\n"
                        sent_tool_block_starts.add(current_anthropic_tool_block_idx)

                    if (
                        tool_delta.function
                        and tool_delta.function.arguments
                        and current_anthropic_tool_block_idx in sent_tool_block_starts
                    ):
                        args_delta_event = {
                            "type": "content_block_delta",
                            "index": current_anthropic_tool_block_idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": tool_delta.function.arguments,
                            },
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(args_delta_event)}\n\n"

            if openai_finish_reason:
                final_anthropic_stop_reason = openai_to_anthropic_stop_reason_map.get(
                    openai_finish_reason, "end_turn"
                )
                if openai_finish_reason == "tool_calls":
                    final_anthropic_stop_reason = "tool_use"
                break

        if text_block_anthropic_idx is not None:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_anthropic_idx})}\n\n"

        for anthropic_tool_idx in sent_tool_block_starts:
            tool_state_to_finalize = tool_states.get(anthropic_tool_idx)
            if tool_state_to_finalize:
                try:
                    json.loads(tool_state_to_finalize["arguments_buffer"])
                except json.JSONDecodeError:
                    warning(
                        LogRecord(
                            event=LogEvent.TOOL_ARGS_PARSE_FAILURE.value,
                            message=f"Buffered arguments for tool '{tool_state_to_finalize.get('name')}' (Anthropic block {anthropic_tool_idx}) did not form valid JSON.",
                            request_id=request_id,
                            data={
                                "buffered_args": tool_state_to_finalize[
                                    "arguments_buffer"
                                ][:100]
                            },
                        )
                    )
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': anthropic_tool_idx})}\n\n"

        if final_anthropic_stop_reason is None:
            final_anthropic_stop_reason = "end_turn"

        message_delta_event = {
            "type": "message_delta",
            "delta": {
                "stop_reason": final_anthropic_stop_reason,
                "stop_sequence": None,
            },
            "usage": {"output_tokens": output_token_count},
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta_event)}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    except Exception as e:
        stream_status_code = 500
        stream_log_event = LogEvent.REQUEST_FAILURE.value
        error_type, error_msg_str, _, provider_err_details = (
            _get_anthropic_error_details_from_exc(e)
        )
        stream_final_message = f"Error during OpenAI stream conversion: {error_msg_str}"
        final_anthropic_stop_reason = "error"

        error(
            LogRecord(
                event=LogEvent.STREAM_INTERRUPTED.value,
                message=stream_final_message,
                request_id=request_id,
                data={
                    "error_type": error_type.value,
                    "provider_details": provider_err_details.model_dump()
                    if provider_err_details
                    else None,
                },
            ),
            exc=e,
        )
        yield format_anthropic_error_sse_event(
            error_type, error_msg_str, provider_err_details
        )

    finally:
        duration_ms = (time.monotonic() - start_time_mono) * 1000
        log_data = {
            "status_code": stream_status_code,
            "duration_ms": duration_ms,
            "input_tokens": estimated_input_tokens,
            "output_tokens": output_token_count,
            "stop_reason": final_anthropic_stop_reason,
        }
        if stream_log_event == LogEvent.REQUEST_COMPLETED.value:
            info(
                LogRecord(
                    event=stream_log_event,
                    message=stream_final_message,
                    request_id=request_id,
                    data=log_data,
                )
            )
        else:
            error(
                LogRecord(
                    event=stream_log_event,
                    message=stream_final_message,
                    request_id=request_id,
                    data=log_data,
                )
            )
