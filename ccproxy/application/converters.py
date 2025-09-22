"""Message converters - imports from modular converter implementation."""

# Re-export all converter functionality for backward compatibility
from .converters_module import (
    # Main conversion functions
    convert_anthropic_to_openai_messages,
    convert_anthropic_tools_to_openai,
    convert_anthropic_tool_choice_to_openai,
    convert_openai_to_anthropic_response,
    # Converter classes
    AnthropicToOpenAIConverter,
    OpenAIToAnthropicConverter,
    ToolConverter,
    ContentConverter,
    # Base classes
    BaseConverter,
    ConversionContext,
    # Async converters
    AsyncMessageConverter,
    AsyncResponseConverter,
    convert_messages_async,
    convert_response_async,
)

# Legacy internal functions (if needed by other modules)
from .converters_module.content_converter import (
    ContentConverter as _ContentConverterImpl,
)
from .converters_module.tool_converter import ToolConverter as _ToolConverterImpl

_content_converter = _ContentConverterImpl()
_tool_converter = _ToolConverterImpl()

# Expose internal functions for backward compatibility
_serialize_tool_result_content_for_openai = (
    _content_converter.serialize_tool_result_content
)

# Expose cached functions for monitoring
_serialize_tool_result_content_for_openai_cached = (
    _content_converter._serialize_tool_result_cached
)
_tools_cache = _tool_converter._convert_tools_cached
_tool_choice_cache = _tool_converter._convert_tool_choice_cached

__all__ = [
    # Main functions
    "convert_anthropic_to_openai_messages",
    "convert_anthropic_tools_to_openai",
    "convert_anthropic_tool_choice_to_openai",
    "convert_openai_to_anthropic_response",
    # Classes
    "AnthropicToOpenAIConverter",
    "OpenAIToAnthropicConverter",
    "ToolConverter",
    "ContentConverter",
    "BaseConverter",
    "ConversionContext",
    # Async converters
    "AsyncMessageConverter",
    "AsyncResponseConverter",
    "convert_messages_async",
    "convert_response_async",
    # Legacy/Internal
    "_serialize_tool_result_content_for_openai",
    "_serialize_tool_result_content_for_openai_cached",
    "_tools_cache",
    "_tool_choice_cache",
]
