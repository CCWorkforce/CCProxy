import json
import time
import uuid
from typing import AsyncGenerator, Dict, Optional, Literal

import openai
from openai.types.chat import ChatCompletionChunk

from ...application.tokenizer import get_token_encoder
from ...application.error_tracker import error_tracker, ErrorType
from ...logging import error, info, LogRecord, LogEvent
from .errors import (
    get_anthropic_error_details_from_execution,
    format_anthropic_error_sse_event,
)
from .http_status import OK, INTERNAL_SERVER_ERROR


# Content block type constants
CONTENT_TYPE_THINKING = "thinking"
CONTENT_TYPE_TEXT = "text"
CONTENT_TYPE_TOOL_USE = "tool_use"

# Event type constants
EVENT_TYPE_CONTENT_BLOCK_START = "content_block_start"
EVENT_TYPE_CONTENT_BLOCK_DELTA = "content_block_delta"
EVENT_TYPE_CONTENT_BLOCK_STOP = "content_block_stop"
EVENT_TYPE_MESSAGE_START = "message_start"
EVENT_TYPE_MESSAGE_DELTA = "message_delta"
EVENT_TYPE_MESSAGE_STOP = "message_stop"
EVENT_TYPE_PING = "ping"


class StreamProcessor:
    class ThinkingState:
        def __init__(self):
            self.idx: Optional[int] = None
            self.buffer = ""
            self.signature = ""
            self.started = False

    class TextState:
        def __init__(self):
            self.idx: Optional[int] = None
            self.content = ""

    def __init__(self, enc, request_id, thinking_enabled):
        self.enc = enc
        self.request_id = request_id
        self.thinking_enabled = thinking_enabled
        self.thinking = self.ThinkingState()
        self.text = self.TextState()
        self.tools = {}
        self.tool_starts = set()
        self.output_token_count = 0
        self.next_anthropic_block_idx = 0

    def snapshot_content(self) -> Dict[str, Optional[str]]:
        """Return a snapshot of accumulated content for diagnostics."""
        return {
            "thinking": self.thinking.buffer if self.thinking else None,
            "text": self.text.content if self.text else None,
        }

    def snapshot_tool_calls(self) -> list:
        """Return simplified view of tool calls for diagnostics."""
        return [
            {
                "id": tool_id,
                "name": tool.get("name"),
                "arguments": tool.get("arguments", ""),
            }
            for tool_id, tool in self.tools.items()
        ]

    async def process_thinking_content(self, content: str) -> list:
        # Process thinking content with state tracking
        events = []
        if not self.thinking.started:
            self.thinking.idx = self.next_anthropic_block_idx
            self.next_anthropic_block_idx += 1
            self.thinking.started = True
            event_data = {
                "type": EVENT_TYPE_CONTENT_BLOCK_START,
                "index": self.thinking.idx,
                "content": {"type": CONTENT_TYPE_THINKING, "text": content},
            }
            events.append(
                f"event: {EVENT_TYPE_CONTENT_BLOCK_START}\ndata: {json.dumps(event_data)}\n\n"
            )
            self.thinking.buffer = content
        else:
            self.thinking.buffer += content
            event_data = {
                "type": EVENT_TYPE_CONTENT_BLOCK_DELTA,
                "index": self.thinking.idx,
                "delta": {"text": content},
            }
            events.append(
                f"event: {EVENT_TYPE_CONTENT_BLOCK_DELTA}\ndata: {json.dumps(event_data)}\n\n"
            )
        tokens = self.enc.encode(content)
        self.output_token_count += len(tokens)
        return events

    async def process_text_content(self, content: str):
        # Process text content
        self.text.content += content
        tokens = self.enc.encode(content)
        self.output_token_count += len(tokens)
        events = []

        if not self.text.idx:
            self.text.idx = self.next_anthropic_block_idx
            event_data = {
                "type": EVENT_TYPE_CONTENT_BLOCK_START,
                "index": self.text.idx,
                "content": {"type": CONTENT_TYPE_TEXT, "text": self.text.content},
            }
            events.append(
                f"event: {EVENT_TYPE_CONTENT_BLOCK_START}\ndata: {json.dumps(event_data)}\n\n"
            )
            self.next_anthropic_block_idx += 1
        else:
            event_data = {
                "type": EVENT_TYPE_CONTENT_BLOCK_DELTA,
                "index": self.text.idx,
                "delta": {"text": content},
            }
            events.append(
                f"event: {EVENT_TYPE_CONTENT_BLOCK_DELTA}\ndata: {json.dumps(event_data)}\n\n"
            )
        return events

    async def process_tool_call(self, tool_delta):
        # Process tool calls
        tool_id = tool_delta.id
        events = []

        if tool_id not in self.tools:
            self.tools[tool_id] = {
                "name": tool_delta.function.name,
                "arguments": "",
                "index": self.next_anthropic_block_idx,
            }
            self.next_anthropic_block_idx += 1
            # Pre-allocate start event format
            start_event_data = {
                "type": EVENT_TYPE_CONTENT_BLOCK_START,
                "index": self.tools[tool_id]["index"],
                "content": {
                    "type": CONTENT_TYPE_TOOL_USE,
                    "id": tool_id,
                    "name": tool_delta.function.name,
                    "input": {},
                },
            }
            json_str = json.dumps(start_event_data)
            self.tools[tool_id]["start_event"] = (
                f"event: {EVENT_TYPE_CONTENT_BLOCK_START}\ndata: {json_str}\n\n"
            )
            events.append(self.tools[tool_id]["start_event"])

        if tool_delta.function.arguments:
            self.tools[tool_id]["arguments"] += tool_delta.function.arguments
            tokens = self.enc.encode(tool_delta.function.arguments)
            self.output_token_count += len(tokens)
            event_data = {
                "type": EVENT_TYPE_CONTENT_BLOCK_DELTA,
                "index": self.tools[tool_id]["index"],
                "delta": {"arguments": tool_delta.function.arguments},
            }
            events.append(
                f"event: {EVENT_TYPE_CONTENT_BLOCK_DELTA}\ndata: {json.dumps(event_data)}\n\n"
            )

        return events

    async def finalize_blocks(self, thinking_enabled):
        # Finalize all blocks
        events = []
        # Finalize thinking block if exists
        if self.thinking.buffer and thinking_enabled:
            thinking_event_data = {
                "type": EVENT_TYPE_CONTENT_BLOCK_STOP,
                "index": self.thinking.idx or self.next_anthropic_block_idx - 1,
            }
            events.append(
                f"event: {EVENT_TYPE_CONTENT_BLOCK_STOP}\ndata: {json.dumps(thinking_event_data)}\n\n"
            )
            self.thinking.buffer = ""

        # Finalize text block if exists
        if self.text.idx is not None and self.text.content:
            text_event_data = {
                "type": EVENT_TYPE_CONTENT_BLOCK_STOP,
                "index": self.text.idx,
            }
            events.append(
                f"event: {EVENT_TYPE_CONTENT_BLOCK_STOP}\ndata: {json.dumps(text_event_data)}\n\n"
            )
            self.text.idx = None

        # Finalize tool blocks
        for tool_id, tool in self.tools.items():
            if "arguments" in tool and tool["arguments"]:
                tool_event_data = {
                    "type": EVENT_TYPE_CONTENT_BLOCK_STOP,
                    "index": tool["index"],
                }
                events.append(
                    f"event: {EVENT_TYPE_CONTENT_BLOCK_STOP}\ndata: {json.dumps(tool_event_data)}\n\n"
                )
        return events


StopReasonType = Optional[
    Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]
]


async def handle_anthropic_streaming_response_from_openai_stream(
    openai_stream: openai.AsyncStream[ChatCompletionChunk],
    original_anthropic_model_name: str,
    estimated_input_tokens: int,
    request_id: str,
    start_time_mono: float,
    thinking_enabled: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Consumes an OpenAI stream and yields Anthropic-compatible SSE events.
    Supports thinking blocks for Claude 4 and beyond Anthropic models.
    """

    anthropic_message_id = f"msg_stream_{request_id}_{uuid.uuid4().hex[:8]}"

    # Consolidated state tracking
    processor = StreamProcessor(
        enc=get_token_encoder(original_anthropic_model_name, request_id),
        request_id=request_id,
        thinking_enabled=thinking_enabled,
    )
    final_anthropic_stop_reason: StopReasonType = None

    openai_to_anthropic_stop_reason_map: Dict[Optional[str], StopReasonType] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "stop_sequence",
        None: None,
    }

    stream_status_code = OK
    stream_final_message = "Streaming request completed successfully."
    stream_log_event = LogEvent.REQUEST_COMPLETED.value

    try:
        message_start_event_data = {
            "type": EVENT_TYPE_MESSAGE_START,
            "message": {
                "id": anthropic_message_id,
                "type": "message",
                "role": "assistant",
                "model": original_anthropic_model_name,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": estimated_input_tokens, "output_tokens": 0},
            },
        }
        yield f"event: {EVENT_TYPE_MESSAGE_START}\ndata: {json.dumps(message_start_event_data)}\n\n"
        yield f"event: {EVENT_TYPE_PING}\ndata: {json.dumps({'type': EVENT_TYPE_PING})}\n\n"

        async for chunk in openai_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            openai_finish_reason = chunk.choices[0].finish_reason

            # Handle thinking content with consolidated state
            if thinking_enabled and delta.content:
                events = await processor.process_thinking_content(delta.content)
                for event in events:
                    yield event
            elif delta.content:
                # Handle text content with state management
                events = await processor.process_text_content(delta.content)
                for event in events:
                    yield event

            # Process tool calls using state manager
            if delta.tool_calls:
                for tool_delta in delta.tool_calls:
                    events = await processor.process_tool_call(tool_delta)
                    for event in events:
                        yield event

            # Map finish reason with simplified logic
            if openai_finish_reason:
                final_anthropic_stop_reason = openai_to_anthropic_stop_reason_map.get(
                    openai_finish_reason, "end_turn"
                )
                break

        # Finalize all blocks with consolidated state
        events = await processor.finalize_blocks(thinking_enabled)
        for event in events:
            yield event

        if final_anthropic_stop_reason is None:
            final_anthropic_stop_reason = "end_turn"

        message_delta_event = {
            "type": EVENT_TYPE_MESSAGE_DELTA,
            "delta": {
                "stop_reason": final_anthropic_stop_reason,
                "stop_sequence": None,
            },
            "usage": {"output_tokens": processor.output_token_count},
        }
        yield f"event: {EVENT_TYPE_MESSAGE_DELTA}\ndata: {json.dumps(message_delta_event)}\n\n"
        yield f"event: {EVENT_TYPE_MESSAGE_STOP}\ndata: {json.dumps({'type': EVENT_TYPE_MESSAGE_STOP})}\n\n"

    except Exception as e:
        stream_status_code = INTERNAL_SERVER_ERROR
        stream_log_event = LogEvent.REQUEST_FAILURE.value
        error_type, error_msg_str, _, provider_err_details = (
            get_anthropic_error_details_from_execution(e)
        )
        stream_final_message = f"Error during OpenAI stream conversion: {error_msg_str}"
        final_anthropic_stop_reason = "error"

        # Comprehensive error tracking for streaming
        metadata = {
            "conversion_type": "openai_to_anthropic_stream",
            "original_model": original_anthropic_model_name,
            "error_message": error_msg_str,
            "error_type": error_type.value,
            "thinking_enabled": thinking_enabled,
            "stream_state": {
                "message_id": anthropic_message_id,
                "content": processor.snapshot_content() if processor else None,
                "tool_calls": processor.snapshot_tool_calls() if processor else None,
                "output_tokens": processor.output_token_count if processor else 0,
            },
        }

        if provider_err_details:
            metadata["provider_details"] = provider_err_details.model_dump()

        # Track the streaming error
        # We're in an async generator context, we can just await directly
        await error_tracker.track_error(
            error=e,
            error_type=ErrorType.STREAM_ERROR,
            request_id=request_id,
            metadata=metadata,
        )

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
            "output_tokens": processor.output_token_count,
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
