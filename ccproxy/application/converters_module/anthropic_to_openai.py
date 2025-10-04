"""Converter for Anthropic to OpenAI message format."""

from typing import Any, Dict, List, Optional, Union

from .base import MessageConverter, ConversionContext
from .content_converter import ContentConverter
from .tool_converter import ToolConverter
from ...domain.models import (
    Message,
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
)
from ...constants import SUPPORT_DEVELOPER_MESSAGE_MODELS, UTF8_ENFORCEMENT_MESSAGE
from ...enums import MessageRoles
from ...logging import debug, LogRecord, LogEvent
from ..type_utils import (
    is_string_content,
    is_list_content,
    is_text_block,
    is_image_block,
    is_tool_use_block,
    is_tool_result_block,
)


class AnthropicToOpenAIConverter(MessageConverter):
    """Converts Anthropic messages to OpenAI format."""

    def __init__(self, context: Optional[ConversionContext] = None):
        super().__init__(context)
        self.content_converter = ContentConverter()
        self.tool_converter = ToolConverter()

    def convert(self, source: Any) -> Any:
        """Convert from source format to target format."""
        if isinstance(source, list):
            return self.convert_messages(source)
        elif isinstance(source, Message):
            return self.convert_message(source)
        return source

    def convert_message(self, message: Message) -> Dict[str, Any]:
        """
        Convert a single Anthropic message to OpenAI format.

        Args:
            message: Anthropic message

        Returns:
            OpenAI format message dictionary
        """
        role = message.role
        content = message.content

        if is_string_content(content):
            return {"role": role, "content": content}

        if is_list_content(content):
            return self._convert_complex_message(message)

        # Fallback for unknown content types
        return {"role": role, "content": str(content)}

    def _convert_complex_message(self, message: Message) -> Dict[str, Any]:
        """Convert a message with complex content blocks."""
        role = message.role
        content = message.content

        if not content:
            return {"role": role, "content": ""}

        openai_parts = []
        tool_calls = []
        text_content = []
        tool_results = []

        for block_idx, block in enumerate(content):
            block_log_ctx = {
                "block_index": block_idx,
                "block_type": getattr(block, "type", type(block).__name__),
            }

            if isinstance(block, ContentBlockText) or is_text_block(block):
                if role == "user":
                    openai_parts.append({"type": "text", "text": block.text})
                elif role == "assistant":
                    text_content.append(block.text)

            elif (
                isinstance(block, ContentBlockImage) or is_image_block(block)
            ) and role == "user":
                openai_parts.append(
                    self.content_converter.convert_image_block_to_openai(block)
                )

            elif (
                isinstance(block, ContentBlockToolUse) or is_tool_use_block(block)
            ) and role == "assistant":
                tool_call = self.content_converter.convert_tool_use_to_openai(
                    block, len(tool_calls)
                )
                tool_calls.append(tool_call)

            elif (
                isinstance(block, ContentBlockToolResult) or is_tool_result_block(block)
            ) and role == "user":
                tool_results.append(self._convert_tool_result(block, block_log_ctx))

            else:
                debug(
                    LogRecord(
                        event=LogEvent.CONVERSION_EVENT.value,
                        message=f"Skipping unsupported content block type for {role} role",
                        request_id=self.context.request_id,
                        data=block_log_ctx,
                    )
                )

        return self._assemble_openai_message(
            role, openai_parts, text_content, tool_calls, tool_results
        )

    def _convert_tool_result(
        self, block: ContentBlockToolResult, log_context: Dict
    ) -> Dict[str, Any]:
        """Convert a tool result block to OpenAI format."""
        content_str = self.content_converter.serialize_tool_result_content(
            block.content, self.context.request_id, log_context
        )
        return {
            "role": "tool",
            "content": content_str,
            "tool_call_id": block.tool_use_id,
        }

    def _assemble_openai_message(
        self,
        role: str,
        openai_parts: List,
        text_content: List,
        tool_calls: List,
        tool_results: List,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Assemble the final OpenAI message(s) from converted parts."""
        messages = []

        if role == "user" and openai_parts:
            # For simple text-only messages, use string format
            if len(openai_parts) == 1 and openai_parts[0].get("type") == "text":
                messages.append({"role": "user", "content": openai_parts[0]["text"]})
            else:
                # For multi-modal or complex content, use array format
                messages.append({"role": "user", "content": openai_parts})
        elif role == "assistant":
            assistant_msg = {"role": "assistant"}
            if text_content:
                assistant_msg["content"] = "\n".join(text_content)
            else:
                assistant_msg["content"] = None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

        # Tool results become separate messages
        messages.extend(tool_results)

        return messages[0] if len(messages) == 1 else messages

    def convert_messages(
        self,
        messages: List[Message],
        system_prompt: Optional[Union[str, List]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert a list of Anthropic messages to OpenAI format.

        Args:
            messages: List of Anthropic messages
            system_prompt: Optional system prompt

        Returns:
            List of OpenAI format messages
        """
        openai_messages = []

        # Handle system prompt
        if system_prompt:
            system_messages = self._convert_system_prompt(system_prompt)
            openai_messages.extend(system_messages)

        # Convert each message
        for msg in messages:
            converted = self.convert_message(msg)
            if isinstance(converted, list):
                openai_messages.extend(converted)
            else:
                openai_messages.append(converted)

        # Validate tool calls have corresponding results
        self._validate_tool_consistency(openai_messages)

        return openai_messages

    def _validate_tool_consistency(self, messages: List[Dict[str, Any]]) -> None:
        """
        Validate that all tool calls have corresponding tool results.
        Remove orphaned tool calls only if there are subsequent messages that should contain results.

        Args:
            messages: List of converted OpenAI format messages
        """
        # Track tool call IDs that need results
        pending_tool_calls = {}
        last_assistant_with_tools_idx = -1

        for i, msg in enumerate(messages):
            # Track assistant messages with tool calls
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tool_call in msg["tool_calls"]:
                    tool_id = tool_call.get("id")
                    if tool_id:
                        pending_tool_calls[tool_id] = i
                        last_assistant_with_tools_idx = i

            # Check for tool results that match pending calls
            elif msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id in pending_tool_calls:
                    del pending_tool_calls[tool_call_id]

        # Only validate if there are messages after the last tool call
        # (indicating the conversation continues and results are expected)
        has_subsequent_messages = (
            last_assistant_with_tools_idx >= 0
            and last_assistant_with_tools_idx < len(messages) - 1
        )

        # If there are orphaned tool calls AND subsequent messages, fix the issue
        if pending_tool_calls and has_subsequent_messages:
            debug(
                LogRecord(
                    event=LogEvent.CONVERSION_EVENT.value,
                    message=f"Found {len(pending_tool_calls)} orphaned tool calls with subsequent messages",
                    request_id=self.context.request_id,
                    data={
                        "orphaned_tool_ids": list(pending_tool_calls.keys()),
                        "message_indices": list(pending_tool_calls.values()),
                        "has_subsequent_messages": has_subsequent_messages,
                    },
                )
            )

            # Remove tool_calls from messages with orphaned calls
            for msg_idx in set(pending_tool_calls.values()):
                if msg_idx < len(messages):
                    msg = messages[msg_idx]
                    if "tool_calls" in msg:
                        # Remove only the orphaned tool calls
                        msg["tool_calls"] = [
                            tc
                            for tc in msg["tool_calls"]
                            if tc.get("id") not in pending_tool_calls
                        ]
                        # If no tool calls remain, remove the field entirely
                        if not msg["tool_calls"]:
                            del msg["tool_calls"]
                            # Ensure the message has some content
                            if not msg.get("content"):
                                msg["content"] = ""

    def _convert_system_prompt(
        self, system_prompt: Union[str, List]
    ) -> List[Dict[str, Any]]:
        """Convert system prompt to OpenAI format."""
        system_text = self.content_converter.extract_system_text(
            system_prompt, self.context.request_id
        )

        if not system_text:
            return []

        # Check if target model supports developer role
        supports_developer = (
            self.context.target_model
            and self.context.target_model in SUPPORT_DEVELOPER_MESSAGE_MODELS
        )

        messages = []
        if supports_developer:
            # Combine system content with UTF-8 enforcement
            combined_content = system_text + "\n\n" + UTF8_ENFORCEMENT_MESSAGE
            messages.append(
                {"role": MessageRoles.Developer.value, "content": combined_content}
            )
        else:
            # Use system role for models that don't support developer role
            messages.append({"role": MessageRoles.System.value, "content": system_text})

        return messages
