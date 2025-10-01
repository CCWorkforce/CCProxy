"""Tool-related conversion utilities."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache

from ...domain.models import Tool, ToolChoice
from ...logging import debug, LogRecord, LogEvent


class ToolConverter:
    """Handles conversion of tools and tool choices between formats."""

    @staticmethod
    @lru_cache(maxsize=128)
    def _convert_tools_cached(
        key: Tuple[Tuple[str, str, str], ...],
    ) -> List[Dict[str, Any]]:
        """Cached tool conversion for performance."""
        openai_tools = []
        for tool_name, tool_desc, input_schema_json in key:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc or "",
                    "parameters": json.loads(input_schema_json)
                    if input_schema_json
                    else {},
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    @classmethod
    def convert_tools_to_openai(
        cls,
        anthropic_tools: Optional[List[Tool]],
        request_id: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Convert Anthropic tools to OpenAI format.

        Args:
            anthropic_tools: List of Anthropic Tool objects
            request_id: Optional request ID for logging

        Returns:
            List of OpenAI format tool dictionaries, or None if no tools
        """
        if not anthropic_tools:
            return None

        # Create cache key
        cache_key = tuple(
            (
                tool.name,
                tool.description or "",
                json.dumps(tool.input_schema, sort_keys=True)
                if tool.input_schema
                else "",
            )
            for tool in anthropic_tools
        )

        openai_tools = cls._convert_tools_cached(cache_key)

        debug(
            LogRecord(
                event=LogEvent.CONVERSION_EVENT.value,
                message=f"Converted {len(anthropic_tools)} Anthropic tools to OpenAI format",
                request_id=request_id,
                data={"tool_count": len(anthropic_tools)},
            )
        )

        return openai_tools

    @staticmethod
    @lru_cache(maxsize=32)
    def _convert_tool_choice_cached(
        tool_choice_type: str,
        tool_name: Optional[str] = None,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Cached tool choice conversion."""
        if tool_choice_type == "none":
            return "none"
        elif tool_choice_type == "auto":
            return "auto"
        elif tool_choice_type == "any":
            return "required"
        elif tool_choice_type == "tool" and tool_name:
            return {"type": "function", "function": {"name": tool_name}}
        return None

    @classmethod
    def convert_tool_choice_to_openai(
        cls,
        anthropic_tool_choice: Optional[ToolChoice],
        request_id: Optional[str] = None,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Convert Anthropic tool choice to OpenAI format.

        Args:
            anthropic_tool_choice: Anthropic ToolChoice object
            request_id: Optional request ID for logging

        Returns:
            OpenAI format tool choice (string or dict), defaults to "auto" if None
        """
        if not anthropic_tool_choice:
            return "auto"  # Default to auto when no tool choice specified

        tool_choice_type = anthropic_tool_choice.type
        tool_name = (
            anthropic_tool_choice.name
            if hasattr(anthropic_tool_choice, "name")
            else None
        )

        result = cls._convert_tool_choice_cached(tool_choice_type, tool_name)

        if result is None:
            debug(
                LogRecord(
                    event=LogEvent.CONVERSION_EVENT.value,
                    message=f"Unknown tool choice type: {tool_choice_type}",
                    request_id=request_id,
                    data={"tool_choice_type": tool_choice_type},
                )
            )

        return result

    @staticmethod
    def map_stop_reason(openai_finish_reason: Optional[str]) -> str:
        """
        Map OpenAI finish reason to Anthropic stop reason.

        Args:
            openai_finish_reason: OpenAI finish reason

        Returns:
            Anthropic stop reason
        """
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "content_filter": "stop_sequence",
            None: "end_turn",
        }
        return stop_reason_map.get(openai_finish_reason, "end_turn")
