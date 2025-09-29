"""Async-optimized message converters for improved performance."""

import json
from typing import Any, Dict, List, Optional, Union

from asyncer import asyncify, create_task_group

from ...domain.models import (
    Message,
    MessagesResponse,
    SystemContent,
    ContentBlockText,
    ContentBlockToolUse,
    ContentBlockToolResult,
    Usage,
)
from .base import ConversionContext, BaseConverter
from .content_converter import ContentConverter
from .tool_converter import ToolConverter


class AsyncMessageConverter(BaseConverter):
    """Async-optimized message converter with parallel processing."""

    def __init__(self, context: Optional[ConversionContext] = None):
        """Initialize async converter with thread pool for CPU-bound operations."""
        super().__init__(context)
        self.content_converter = ContentConverter()
        self.tool_converter = ToolConverter()

    def convert(self, source: Any) -> Any:
        """Synchronous convert method (not used, async version preferred)."""
        # This is here to satisfy the abstract base class requirement
        # Use convert_messages_async instead
        from .anthropic_to_openai import AnthropicToOpenAIConverter

        sync_converter = AnthropicToOpenAIConverter(self.context)
        return sync_converter.convert(source)

    async def convert_messages_async(
        self,
        messages: List[Message],
        system: Optional[Union[str, List[SystemContent]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously convert Anthropic messages to OpenAI format.

        Optimizations:
        - Parallel processing of messages
        - Async JSON serialization for large payloads
        - Batch processing for multiple content blocks
        """
        openai_messages = []

        # Add system message if provided
        if system:
            system_content = (
                system
                if isinstance(system, str)
                else self._serialize_system_content(system)
            )
            openai_messages.append({"role": "system", "content": system_content})

        # Convert messages in parallel
        if len(messages) > 3:  # Only parallelize for multiple messages
            async with create_task_group() as tg:
                soon_values = []
                for msg in messages:
                    soon_values.append(
                        tg.soonify(self._convert_single_message_async)(msg)
                    )
            # After task group completes, extract values
            converted_messages = [sv.value for sv in soon_values]
            openai_messages.extend(converted_messages)
        else:
            # For small message counts, sequential is fine
            for msg in messages:
                converted = await self._convert_single_message_async(msg)
                openai_messages.append(converted)

        return openai_messages

    async def _convert_single_message_async(self, message: Message) -> Dict[str, Any]:
        """Convert a single message asynchronously."""
        role = self._map_role(message.role)

        # Handle empty content
        if not message.content:
            return {"role": role}

        # Handle string content directly
        if isinstance(message.content, str):
            return {"role": role, "content": message.content}

        # Process different content types
        if len(message.content) == 1:
            # Single content block - optimize for common case
            content_block = message.content[0]

            if content_block.type == "text":
                return {"role": role, "content": content_block.text}
            elif content_block.type == "tool_result":
                return self._convert_tool_result_message(content_block)
            elif content_block.type == "tool_use":
                return await self._convert_tool_use_async(role, [content_block])

        # Multiple content blocks or complex content
        return await self._convert_complex_message_async(message, role)

    async def _convert_complex_message_async(
        self, message: Message, role: str
    ) -> Dict[str, Any]:
        """Convert complex messages with multiple content blocks."""
        text_parts = []
        non_text_parts = []
        tool_calls = []

        # Categorize content blocks
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)
            else:
                non_text_parts.append(block)

        # Build OpenAI message
        openai_msg = {"role": role}

        # Handle tool calls
        if tool_calls:
            openai_msg["tool_calls"] = await self._convert_tool_calls_async(tool_calls)
            if text_parts:
                openai_msg["content"] = " ".join(text_parts)
        elif text_parts and non_text_parts:
            # Mixed content - needs array format
            content = []
            for text in text_parts:
                content.append({"type": "text", "text": text})
            for block in non_text_parts:
                if block.type == "image":
                    content.append(
                        self.content_converter.convert_image_block_to_openai(block)
                    )
                else:
                    # For other types, just skip or add as text
                    content.append({"type": "text", "text": str(block)})
            openai_msg["content"] = content
        elif text_parts:
            # Only text
            openai_msg["content"] = " ".join(text_parts)
        elif non_text_parts:
            # Only non-text
            content = []
            for block in non_text_parts:
                if block.type == "image":
                    content.append(
                        self.content_converter.convert_image_block_to_openai(block)
                    )
                elif block.type == "tool_use":
                    # Tool use blocks should be handled separately, not in content
                    pass
                else:
                    # For other types, add as text
                    content.append({"type": "text", "text": str(block)})

            if content:
                if len(content) == 1:
                    openai_msg["content"] = content[0]
                else:
                    openai_msg["content"] = content

        return openai_msg

    async def _convert_tool_calls_async(
        self, tool_calls: List[ContentBlockToolUse]
    ) -> List[Dict[str, Any]]:
        """Convert tool calls asynchronously with parallel JSON serialization."""

        async def serialize_tool_call(tool: ContentBlockToolUse) -> Dict[str, Any]:
            # Offload JSON serialization to thread pool for large inputs
            if isinstance(tool.input, dict) and len(str(tool.input)) > 1000:
                args_str = await asyncify(json.dumps)(tool.input)
            else:
                args_str = json.dumps(tool.input) if tool.input else "{}"

            return {
                "id": tool.id,
                "type": "function",
                "function": {
                    "name": tool.name,
                    "arguments": args_str,
                },
            }

        # Process tool calls in parallel
        async with create_task_group() as tg:
            soon_values = [tg.soonify(serialize_tool_call)(tool) for tool in tool_calls]
        return [sv.value for sv in soon_values]

    async def _convert_tool_use_async(
        self, role: str, tool_uses: List[ContentBlockToolUse]
    ) -> Dict[str, Any]:
        """Convert tool use blocks asynchronously."""
        tool_calls = await self._convert_tool_calls_async(tool_uses)
        return {"role": role, "tool_calls": tool_calls}

    def _convert_tool_result_message(
        self, tool_result: ContentBlockToolResult
    ) -> Dict[str, Any]:
        """Convert tool result message (synchronous as it's simple)."""
        content = self.content_converter.serialize_tool_result_content(tool_result)
        return {
            "role": "tool",
            "tool_call_id": tool_result.tool_use_id,
            "content": content,
        }

    def _map_role(self, anthropic_role: str) -> str:
        """Map Anthropic role to OpenAI role."""
        role_mapping = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
        }
        return role_mapping.get(anthropic_role, anthropic_role)

    def _serialize_system_content(self, system_content: List[SystemContent]) -> str:
        """Serialize system content blocks to string."""
        parts = []
        for content in system_content:
            if content.type == "text":
                parts.append(content.text)
        return "\n".join(parts)


class AsyncResponseConverter:
    """Async-optimized response converter."""

    def __init__(self, context: Optional[ConversionContext] = None):
        """Initialize async response converter."""
        self.context = context or ConversionContext()
        self.content_converter = ContentConverter()
        self.tool_converter = ToolConverter()

    async def convert_response_async(
        self, openai_response: Any, request_id: Optional[str] = None
    ) -> MessagesResponse:
        """
        Asynchronously convert OpenAI response to Anthropic format.

        Optimizations:
        - Async processing of tool calls
        - Parallel content block conversion
        """
        if not openai_response.choices:
            raise ValueError("OpenAI response has no choices")

        choice = openai_response.choices[0]
        message = choice.message

        # Build content blocks asynchronously
        content = await self._build_content_blocks_async(message)

        # Map finish reason
        stop_reason = self._map_finish_reason(choice.finish_reason)

        # Build usage
        usage = None
        if hasattr(openai_response, "usage") and openai_response.usage:
            usage = Usage(
                input_tokens=openai_response.usage.prompt_tokens,
                output_tokens=openai_response.usage.completion_tokens,
            )

        return MessagesResponse(
            id=openai_response.id,
            type="message",
            role="assistant",
            model=openai_response.model,
            content=content,
            stop_reason=stop_reason,
            usage=usage,
        )

    async def _build_content_blocks_async(self, message: Any) -> List[Any]:
        """Build content blocks asynchronously."""
        content = []

        # Handle text content
        if message.content:
            content.append(ContentBlockText(type="text", text=message.content))

        # Handle tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_blocks = await self._convert_tool_calls_to_blocks_async(
                message.tool_calls
            )
            content.extend(tool_blocks)

        # Handle refusal
        if hasattr(message, "refusal") and message.refusal:
            content.append(ContentBlockText(type="text", text=message.refusal))

        # Ensure at least one content block
        if not content:
            content.append(ContentBlockText(type="text", text=""))

        return content

    async def _convert_tool_calls_to_blocks_async(
        self, tool_calls: List[Any]
    ) -> List[ContentBlockToolUse]:
        """Convert OpenAI tool calls to Anthropic tool use blocks asynchronously."""

        async def parse_tool_call(tool_call: Any) -> ContentBlockToolUse:
            # Parse arguments asynchronously for large payloads
            args_str = tool_call.function.arguments
            if len(args_str) > 1000:
                tool_input = await asyncify(json.loads)(args_str)
            else:
                tool_input = json.loads(args_str) if args_str else {}

            return ContentBlockToolUse(
                type="tool_use",
                id=tool_call.id,
                name=tool_call.function.name,
                input=tool_input,
            )

        # Process tool calls in parallel
        async with create_task_group() as tg:
            soon_values = [tg.soonify(parse_tool_call)(tc) for tc in tool_calls]
        return [sv.value for sv in soon_values]

    def _map_finish_reason(self, openai_reason: Optional[str]) -> str:
        """Map OpenAI finish reason to Anthropic stop reason."""
        if not openai_reason:
            return "stop_sequence"

        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
        }
        return mapping.get(openai_reason, "stop_sequence")


# Async wrapper functions for backward compatibility
async def convert_messages_async(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]] = None,
    context: Optional[ConversionContext] = None,
) -> List[Dict[str, Any]]:
    """Async wrapper for message conversion."""
    converter = AsyncMessageConverter(context)
    return await converter.convert_messages_async(messages, system)


async def convert_response_async(
    openai_response: Any,
    request_id: Optional[str] = None,
    context: Optional[ConversionContext] = None,
) -> MessagesResponse:
    """Async wrapper for response conversion."""
    converter = AsyncResponseConverter(context)
    return await converter.convert_response_async(openai_response, request_id)
