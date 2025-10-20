"""Content block conversion utilities."""

import json
from typing import Any, Dict, List, Optional, Union, Tuple
from functools import lru_cache

from ...domain.models import (
    ContentBlockImage,
    ContentBlockToolUse,
    SystemContent,
)
from ..type_utils import (
    is_text_block,
    is_string_content,
    is_list_content,
    is_dict_content,
    is_serializable_primitive,
)
from ...logging import warning, LogRecord, LogEvent
from ..._cython import CYTHON_ENABLED

# Try to import Cython-optimized functions
if CYTHON_ENABLED:
    try:
        from ..._cython.json_ops import (
            json_dumps_compact,
        )

        _USING_CYTHON = True
    except ImportError:
        _USING_CYTHON = False
else:
    _USING_CYTHON = False

# Fallback to pure Python implementation if Cython not available
if not _USING_CYTHON:

    def json_dumps_compact(obj: Any) -> str:
        """Compact JSON serialization with minimal separators."""
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class ContentConverter:
    """Handles conversion of content blocks between formats."""

    @staticmethod
    @lru_cache(maxsize=512)
    def _serialize_tool_result_cached(key: Tuple[str, str]) -> str:
        """Cached helper for tool result serialization."""
        content_json, _ = key
        items = json.loads(content_json)
        parts = []
        for item in items:
            if is_text_block(item):
                parts.append(str(item["text"] if isinstance(item, dict) else item.text))
            else:
                try:
                    parts.append(json.dumps(item))
                except (TypeError, ValueError):
                    parts.append(f"<unserializable_item type='{type(item).__name__}'>")
        return "\n".join(parts)

    @classmethod
    def serialize_tool_result_content(
        cls,
        anthropic_tool_result_content: object,
        request_id: Optional[str] = None,
        log_context: Optional[Dict] = None,
    ) -> str:
        """
        Serializes Anthropic tool result content into a single string
        as expected by OpenAI for the 'content' field of a 'tool' role message.

        Args:
            anthropic_tool_result_content: Content to serialize
            request_id: Optional request ID for logging
            log_context: Optional logging context

        Returns:
            Serialized string representation
        """
        if is_string_content(anthropic_tool_result_content):
            return anthropic_tool_result_content

        if is_list_content(anthropic_tool_result_content):
            try:
                key = (
                    json.dumps(
                        anthropic_tool_result_content,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "1",
                )
                return cls._serialize_tool_result_cached(key)
            except TypeError:
                processed_parts = []
                for item in anthropic_tool_result_content:
                    if is_text_block(item):
                        processed_parts.append(
                            str(item["text"] if isinstance(item, dict) else item.text)
                        )
                    else:
                        try:
                            processed_parts.append(json.dumps(item))
                        except TypeError:
                            processed_parts.append(
                                f"<unserializable_item type='{type(item).__name__}'>"
                            )
                return "\n".join(processed_parts)

        # Handle None type
        if anthropic_tool_result_content is None:
            return "null"

        # Handle primitive types
        if is_serializable_primitive(
            anthropic_tool_result_content
        ) and not is_string_content(anthropic_tool_result_content):
            return json.dumps(anthropic_tool_result_content)

        # Handle dict type
        if is_dict_content(anthropic_tool_result_content):
            try:
                return json.dumps(anthropic_tool_result_content, sort_keys=True)
            except TypeError:
                return f"<unserializable_dict with keys: {list(anthropic_tool_result_content.keys())}>"

        # For any other type, try JSON serialization first, then fallback to string
        try:
            return json.dumps(anthropic_tool_result_content)
        except TypeError:
            # Last resort: convert to string
            warning(
                LogRecord(
                    event=LogEvent.CONVERSION_EVENT.value,
                    message=f"Fallback to str() for tool result content of type {type(anthropic_tool_result_content).__name__}",
                    request_id=request_id,
                    data=log_context or {},
                )
            )
            return str(anthropic_tool_result_content)

    @staticmethod
    def extract_system_text(
        anthropic_system: Union[str, List[SystemContent], None],
        request_id: Optional[str] = None,
    ) -> str:
        """
        Extract text content from Anthropic system prompt.

        Args:
            anthropic_system: System prompt (string or list of SystemContent)
            request_id: Optional request ID for logging

        Returns:
            Extracted text content
        """
        if is_string_content(anthropic_system):
            return anthropic_system
        elif is_list_content(anthropic_system):
            from ..type_utils import is_system_text_block

            system_texts = [
                block.text for block in anthropic_system if is_system_text_block(block)
            ]
            if len(system_texts) < len(anthropic_system):
                warning(
                    LogRecord(
                        event=LogEvent.SYSTEM_PROMPT_ADJUSTED.value,
                        message="Non-text content blocks in Anthropic system prompt were ignored.",
                        request_id=request_id,
                    )
                )
            return "\n".join(system_texts)
        return ""

    @staticmethod
    def convert_image_block_to_openai(block: ContentBlockImage) -> Dict[str, Any]:
        """
        Convert Anthropic image block to OpenAI format.

        Args:
            block: Anthropic image block

        Returns:
            OpenAI format image dictionary
        """
        if block.source.type == "base64":
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{block.source.media_type};base64,{block.source.data}"
                },
            }
        else:
            # URL type or other
            return {"type": "image_url", "image_url": {"url": block.source.data}}

    @staticmethod
    def convert_tool_use_to_openai(
        block: ContentBlockToolUse, tool_call_index: int = 0
    ) -> Dict[str, Any]:
        """
        Convert Anthropic tool use block to OpenAI format.

        Args:
            block: Anthropic tool use block
            tool_call_index: Index for the tool call

        Returns:
            OpenAI format tool call dictionary
        """
        # Handle JSON serialization errors gracefully
        try:
            arguments_json = json.dumps(block.input, ensure_ascii=False)
        except (TypeError, ValueError):
            # Fallback to string representation if JSON serialization fails
            arguments_json = json.dumps({"error": "Failed to serialize tool input"})

        return {
            "id": block.id,
            "type": "function",
            "function": {
                "name": block.name,
                "arguments": arguments_json,
            },
            "index": tool_call_index,
        }
