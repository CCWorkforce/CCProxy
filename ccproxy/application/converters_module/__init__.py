"""Message converters for Anthropic-OpenAI bidirectional conversion."""

from .anthropic_to_openai import AnthropicToOpenAIConverter
from .openai_to_anthropic import OpenAIToAnthropicConverter
from .base import BaseConverter, ConversionContext
from .tool_converter import ToolConverter
from .content_converter import ContentConverter

# Main converter functions for backward compatibility
from .main import (
    convert_anthropic_to_openai_messages,
    convert_anthropic_tools_to_openai,
    convert_anthropic_tool_choice_to_openai,
    convert_openai_to_anthropic_response,
)

__all__ = [
    "AnthropicToOpenAIConverter",
    "OpenAIToAnthropicConverter",
    "BaseConverter",
    "ConversionContext",
    "ToolConverter",
    "ContentConverter",
    # Legacy functions
    "convert_anthropic_to_openai_messages",
    "convert_anthropic_tools_to_openai",
    "convert_anthropic_tool_choice_to_openai",
    "convert_openai_to_anthropic_response",
]