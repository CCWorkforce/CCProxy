import json
import time
import uuid
from typing import AsyncGenerator, Dict, Optional, Literal, Any

import openai

from ...application.tokenizer import get_token_encoder
from ...logging import warning, debug, error, info, LogRecord, LogEvent
from .errors import (
    get_anthropic_error_details_from_execution,
    format_anthropic_error_sse_event,
)
from .http_status import OK, INTERNAL_SERVER_ERROR


class StreamProcessor:
    class ThinkingState:
        def __init__(self):
            self.idx = None
            self.buffer = ''
            self.signature = ''
            self.started = False

    class TextState:
        def __init__(self):
            self.idx = None
            self.content = ''

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

    async def process_thinking_content(self, content: str, yield_fn):
        # Process thinking content
        self.thinking.buffer += content
        tokens = self.enc.encode(self.thinking.buffer)
        self.output_token_count += len(tokens)
        events = []
        if 'END_THINKING' in self.thinking.buffer:
            event_data = {
                'type': 'content_block_start',
                'index': self.next_anthropic_block_idx,
                'content': {'type': 'thinking', 'text': self.thinking.buffer}
            }
            yield_fn(f'event: content_block_start\ndata: {json.dumps(event_data)}\n\n')
            self.thinking.buffer = ''
            self.thinking.started = False
            self.next_anthropic_block_idx += 1

    async def process_text_content(self, content: str):
        # Process text content
        self.text.content += content
        tokens = self.enc.encode(content)
        self.output_token_count += len(tokens)
        events = []
        if not self.text.idx:
            self.text.idx = self.next_anthropic_block_idx
            self.next_anthropic_block_idx += 1
            events.append(f"event: content_block_start\ndata: {json.dumps({
                'type': 'content_block_start',
                'index': self.text.idx,
                'content': {'type': 'text', 'text': self.text.content}
            })}\n\n")
        else:
            events.append(f"event: content_block_delta\ndata: {json.dumps({
                'type': 'content_block_delta',
                'index': self.text.idx,
                'delta': {'text': content}
            })}\n\n")
        return events

    async def process_tool_call(self, tool_delta):
        # Process tool calls
        tool_id = tool_delta.id
        events = []

        if tool_id not in self.tools:
            self.tools[tool_id] = {
                'name': tool_delta.function.name,
                'arguments': '',
                'index': self.next_anthropic_block_idx
            }
            self.next_anthropic_block_idx += 1
            event_data = {
                'type': 'content_block_start',
                'index': self.tools[tool_id]['index'],
                'content': {
                    'type': 'tool_use',
                    'id': tool_id,
                    'name': tool_delta.function.name,
                    'input': {}
                }
            }
            events.append(f'event: content_block_start\ndata: {json.dumps(event_data)}\n\n')

        if tool_delta.function.arguments:
            self.tools[tool_id]['arguments'] += tool_delta.function.arguments
            tokens = self.enc.encode(tool_delta.function.arguments)
            self.output_token_count += len(tokens)
            event_data = {
                'type': 'content_block_delta',
                'index': self.tools[tool_id]['index'],
                'delta': {'arguments': tool_delta.function.arguments}
            }
            events.append(f'event: content_block_delta\ndata: {json.dumps(event_data)}\n\n')

        return events

    async def finalize_blocks(self, thinking_enabled):
        # Finalize all blocks
        events = []
        # Finalize thinking block if exists
        if self.thinking.buffer and thinking_enabled:
            event_data = {
                'type': 'content_block_stop',
                'index': self.thinking.idx or self.next_anthropic_block_idx - 1
            }
            events.append(f'event: content_block_stop\ndata: {json.dumps(event_data)}\n\n')
            self.thinking.buffer = ''

        # Finalize text block if exists
        if self.text.idx is not None and self.text.content:
            event_data = {
                'type': 'content_block_stop',
                'index': self.text.idx
            }
            events.append(f'event: content_block_stop\ndata: {json.dumps(event_data)}\n\n')
            self.text.idx = None

        # Finalize tool blocks
        for tool_id, tool in self.tools.items():
            if 'arguments' in tool and tool['arguments']:
                event_data = {
                    'type': 'content_block_stop',
                    'index': tool['index']
                }
            events.append(f'event: content_block_stop\ndata: {json.dumps(event_data)}\n\n')
        return events

StopReasonType = Optional[
    Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]
]


async def handle_anthropic_streaming_response_from_openai_stream(
    openai_stream: openai.AsyncStream[openai.types.chat.ChatCompletionChunk],
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

    next_anthropic_block_idx = 0
    # Consolidated state tracking
    processor = StreamProcessor(
        enc=get_token_encoder(original_anthropic_model_name, request_id),
        request_id=request_id,
        thinking_enabled=thinking_enabled
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
                    openai_finish_reason,
                    "end_turn"
                )
                break

        # Finalize all blocks with consolidated state
        events = await processor.finalize_blocks(thinking_enabled)
        for event in events:
            yield event

        if final_anthropic_stop_reason is None:
            final_anthropic_stop_reason = "end_turn"

        message_delta_event = {
            "type": "message_delta",
            "delta": {
                "stop_reason": final_anthropic_stop_reason,
                "stop_sequence": None,
            },
            "usage": {"output_tokens": processor.output_token_count},
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta_event)}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    except Exception as e:
        stream_status_code = INTERNAL_SERVER_ERROR
        stream_log_event = LogEvent.REQUEST_FAILURE.value
        error_type, error_msg_str, _, provider_err_details = (
            get_anthropic_error_details_from_execution(e)
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
