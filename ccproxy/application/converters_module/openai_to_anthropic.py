"""Converter for OpenAI to Anthropic response format."""

import json
from typing import Any, Optional

from openai.types.chat.chat_completion import ChatCompletion

from .base import ResponseConverter, ConversionContext
from .tool_converter import ToolConverter
from ...domain.models import (
    MessagesResponse,
    ContentBlockText,
    ContentBlockToolUse,
    Usage,
)
from ...logging import debug, warning, LogRecord, LogEvent


class OpenAIToAnthropicConverter(ResponseConverter):
    """Converts OpenAI responses to Anthropic format."""

    def __init__(self, context: Optional[ConversionContext] = None):
        super().__init__(context)
        self.tool_converter = ToolConverter()

    def convert_response(self, response: ChatCompletion) -> MessagesResponse:
        """
        Convert an OpenAI ChatCompletion to Anthropic MessagesResponse.

        Args:
            response: OpenAI ChatCompletion response

        Returns:
            Anthropic MessagesResponse
        """
        anthropic_content = []
        anthropic_stop_reason = None
        usage = None

        if response.choices:
            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            # Map stop reason
            anthropic_stop_reason = self.tool_converter.map_stop_reason(finish_reason)

            # Convert message content
            if message.content:
                anthropic_content.append(
                    ContentBlockText(type="text", text=message.content)
                )

            # Convert tool calls
            if message.tool_calls:
                for call in message.tool_calls:
                    tool_block = self._convert_tool_call(call)
                    if tool_block:
                        anthropic_content.append(tool_block)

        # Convert usage statistics
        if response.usage:
            usage = self._convert_usage(response.usage)

        # Filter out any None or invalid content blocks
        anthropic_content = [
            block
            for block in anthropic_content
            if block is not None and hasattr(block, "type")
        ]

        # Ensure we have at least some content
        if not anthropic_content:
            anthropic_content = [ContentBlockText(type="text", text="")]
            debug(
                LogRecord(
                    event=LogEvent.CONVERSION_EVENT.value,
                    message="No valid content in OpenAI response, adding empty text block",
                    request_id=self.context.request_id,
                )
            )

        return MessagesResponse(
            id=response.id or "msg_converted",
            type="message",
            role="assistant",
            content=anthropic_content,
            model=self.context.original_model or response.model,
            stop_reason=anthropic_stop_reason,
            stop_sequence=None,
            usage=usage,
        )

    def _convert_tool_call(self, call: Any) -> Optional[ContentBlockToolUse]:
        """
        Convert an OpenAI tool call to Anthropic ContentBlockToolUse.

        Args:
            call: OpenAI tool call object

        Returns:
            Anthropic ContentBlockToolUse or None if conversion fails
        """
        if call.type != "function":
            debug(
                LogRecord(
                    event=LogEvent.CONVERSION_EVENT.value,
                    message=f"Skipping non-function tool call type: {call.type}",
                    request_id=self.context.request_id,
                )
            )
            return None

        tool_input_dict = {}
        try:
            parsed_input = json.loads(call.function.arguments)
            if isinstance(parsed_input, dict):
                tool_input_dict = parsed_input
            else:
                warning(
                    LogRecord(
                        event=LogEvent.CONVERSION_EVENT.value,
                        message=f"Tool arguments not a dict: {type(parsed_input).__name__}",
                        request_id=self.context.request_id,
                        data={
                            "tool_name": call.function.name,
                            "arguments_type": type(parsed_input).__name__,
                        },
                    )
                )
                tool_input_dict = {"value": parsed_input}
        except json.JSONDecodeError as e:
            warning(
                LogRecord(
                    event=LogEvent.CONVERSION_EVENT.value,
                    message=f"Failed to parse tool arguments: {e}",
                    request_id=self.context.request_id,
                    data={
                        "tool_name": call.function.name,
                        "raw_arguments": call.function.arguments[:100],
                        "error": str(e),
                    },
                )
            )
            # Use raw string as fallback
            tool_input_dict = {"raw_arguments": call.function.arguments}

        return ContentBlockToolUse(
            type="tool_use",
            id=call.id,
            name=call.function.name,
            input=tool_input_dict,
        )

    def _convert_usage(self, openai_usage: Any) -> Usage:
        """
        Convert OpenAI usage statistics to Anthropic format.

        Args:
            openai_usage: OpenAI usage object

        Returns:
            Anthropic Usage object
        """
        # Handle different possible attribute names
        input_tokens = getattr(openai_usage, "prompt_tokens", 0)
        output_tokens = getattr(openai_usage, "completion_tokens", 0)

        # Some OpenAI responses might have these alternative names
        if hasattr(openai_usage, "input_tokens"):
            input_tokens = openai_usage.input_tokens
        if hasattr(openai_usage, "output_tokens"):
            output_tokens = openai_usage.output_tokens

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
