"""Main converter functions for backward compatibility."""

from typing import Any, Dict, List, Optional, Union

from openai.types.chat.chat_completion import ChatCompletion

from .base import ConversionContext
from .anthropic_to_openai import AnthropicToOpenAIConverter
from .openai_to_anthropic import OpenAIToAnthropicConverter
from .tool_converter import ToolConverter
from ...domain.models import (
    Message,
    MessagesResponse,
    Tool,
    ToolChoice,
    SystemContent,
)


def convert_anthropic_to_openai_messages(
    anthropic_messages: List[Message],
    anthropic_system: Optional[Union[str, List[SystemContent]]] = None,
    request_id: Optional[str] = None,
    target_model: Optional[str] = None,
    settings: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Convert Anthropic messages to OpenAI format.

    This function maintains backward compatibility with the original implementation.

    Args:
        anthropic_messages: List of Anthropic Message objects
        anthropic_system: Optional system prompt (string or list of SystemContent)
        request_id: Optional request ID for logging
        target_model: Target OpenAI model name

    Returns:
        List of OpenAI format message dictionaries
    """
    context = ConversionContext(
        request_id=request_id,
        target_model=target_model,
    )
    converter = AnthropicToOpenAIConverter(context)
    return converter.convert_messages(anthropic_messages, anthropic_system)


def convert_anthropic_tools_to_openai(
    anthropic_tools: Optional[List[Tool]],
    request_id: Optional[str] = None,
    settings: Optional[Any] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert Anthropic tools to OpenAI format.

    Args:
        anthropic_tools: List of Anthropic Tool objects
        request_id: Optional request ID for logging

    Returns:
        List of OpenAI format tool dictionaries, or None if no tools
    """
    converter = ToolConverter()
    return converter.convert_tools_to_openai(anthropic_tools, request_id)


def convert_anthropic_tool_choice_to_openai(
    anthropic_tool_choice: Optional[ToolChoice],
    request_id: Optional[str] = None,
    settings: Optional[Any] = None,
) -> Optional[Union[str, Dict[str, Any]]]:
    """
    Convert Anthropic tool choice to OpenAI format.

    Args:
        anthropic_tool_choice: Anthropic ToolChoice object
        request_id: Optional request ID for logging

    Returns:
        OpenAI format tool choice (string or dict), or None
    """
    converter = ToolConverter()
    return converter.convert_tool_choice_to_openai(anthropic_tool_choice, request_id)


def convert_openai_to_anthropic_response(
    openai_response: ChatCompletion,
    original_anthropic_model_name: str,
    request_id: Optional[str] = None,
    settings: Optional[Any] = None,
) -> MessagesResponse:
    """
    Convert an OpenAI ChatCompletion response to Anthropic MessagesResponse.

    Args:
        openai_response: OpenAI ChatCompletion response
        original_anthropic_model_name: The model name originally requested
        request_id: Optional request ID for logging

    Returns:
        Anthropic MessagesResponse
    """
    context = ConversionContext(
        request_id=request_id,
        original_model=original_anthropic_model_name,
    )
    converter = OpenAIToAnthropicConverter(context)
    return converter.convert_response(openai_response)