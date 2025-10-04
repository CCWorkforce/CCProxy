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
from ...application.error_tracker import error_tracker, ErrorType
from ...logging import error as log_error, LogRecord, LogEvent


def _fire_and_forget_error_tracking(
    exception: Exception,
    error_type: ErrorType,
    request_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Fire and forget error tracking - best effort only.

    Since we're in a sync context but the error tracker is async, we can't
    directly call it. Instead, we'll just log the error synchronously.
    This is acceptable because error tracking is a best-effort operation.
    """
    # Log the error synchronously since we're in a sync context
    log_error(
        LogRecord(
            event=LogEvent.CONVERSION_EVENT.value,
            message=f"Conversion error: {str(exception)}",
            request_id=request_id,
            data={
                **metadata,
                "error_type": error_type.value,
            } if metadata else {"error_type": error_type.value},
        )
    )


def convert_anthropic_to_openai_messages(
    anthropic_messages: List[Message],
    anthropic_system: Optional[Union[str, List[SystemContent]]] = None,
    request_id: Optional[str] = None,
    target_model: Optional[str] = None,
    settings: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Convert Anthropic messages to OpenAI format with comprehensive error tracking.

    This function maintains backward compatibility with the original implementation.

    Args:
        anthropic_messages: List of Anthropic Message objects
        anthropic_system: Optional system prompt (string or list of SystemContent)
        request_id: Optional request ID for logging
        target_model: Target OpenAI model name
        settings: Optional settings object

    Returns:
        List of OpenAI format message dictionaries
    """
    try:
        context = ConversionContext(
            request_id=request_id,
            target_model=target_model,
            settings=settings,
        )
        converter = AnthropicToOpenAIConverter(context)
        return converter.convert_messages(anthropic_messages, anthropic_system)
    except Exception as e:
        # Track conversion error with detailed context
        metadata = {
            "conversion_type": "anthropic_to_openai_messages",
            "target_model": target_model,
            "message_count": len(anthropic_messages) if anthropic_messages else 0,
            "has_system": anthropic_system is not None,
            "error_message": str(e),
            "error_type": type(e).__name__,
        }

        # Capture sample of input for debugging (truncated)
        if anthropic_messages and len(anthropic_messages) > 0:
            first_msg = anthropic_messages[0]
            metadata["first_message_role"] = getattr(first_msg, "role", "unknown")
            if hasattr(first_msg, "content") and first_msg.content:
                content_preview = str(first_msg.content)[:200]
                metadata["first_message_content_preview"] = content_preview

        # Log and track error asynchronously
        _fire_and_forget_error_tracking(
            exception=e,
            error_type=ErrorType.CONVERSION_ERROR,
            request_id=request_id,
            metadata=metadata,
        )

        log_error(
            LogRecord(
                event=LogEvent.CONVERSION_EVENT.value,
                message=f"Failed to convert Anthropic messages to OpenAI format: {str(e)}",
                request_id=request_id,
                data=metadata,
            )
        )

        # Re-raise to maintain original behavior
        raise


def convert_anthropic_tools_to_openai(
    anthropic_tools: Optional[List[Tool]],
    request_id: Optional[str] = None,
    settings: Optional[Any] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert Anthropic tools to OpenAI format with error tracking.

    Args:
        anthropic_tools: List of Anthropic Tool objects
        request_id: Optional request ID for logging
        settings: Optional settings object

    Returns:
        List of OpenAI format tool dictionaries, or None if no tools
    """
    try:
        converter = ToolConverter()
        return converter.convert_tools_to_openai(anthropic_tools, request_id)
    except Exception as e:
        metadata = {
            "conversion_type": "anthropic_to_openai_tools",
            "tool_count": len(anthropic_tools) if anthropic_tools else 0,
            "error_message": str(e),
        }

        if anthropic_tools and len(anthropic_tools) > 0:
            metadata["first_tool_name"] = getattr(anthropic_tools[0], "name", "unknown")

        _fire_and_forget_error_tracking(
            exception=e,
            error_type=ErrorType.CONVERSION_ERROR,
            request_id=request_id,
            metadata=metadata,
        )

        log_error(
            LogRecord(
                event=LogEvent.CONVERSION_EVENT.value,
                message=f"Failed to convert Anthropic tools: {str(e)}",
                request_id=request_id,
                data=metadata,
            )
        )
        raise


def convert_anthropic_tool_choice_to_openai(
    anthropic_tool_choice: Optional[ToolChoice],
    request_id: Optional[str] = None,
    settings: Optional[Any] = None,
) -> Optional[Union[str, Dict[str, Any]]]:
    """
    Convert Anthropic tool choice to OpenAI format with error tracking.

    Args:
        anthropic_tool_choice: Anthropic ToolChoice object
        request_id: Optional request ID for logging
        settings: Optional settings object

    Returns:
        OpenAI format tool choice (string or dict), or None
    """
    try:
        converter = ToolConverter()
        return converter.convert_tool_choice_to_openai(
            anthropic_tool_choice, request_id
        )
    except Exception as e:
        metadata = {
            "conversion_type": "anthropic_to_openai_tool_choice",
            "tool_choice_type": getattr(anthropic_tool_choice, "type", "unknown")
            if anthropic_tool_choice
            else None,
            "error_message": str(e),
        }

        _fire_and_forget_error_tracking(
            exception=e,
            error_type=ErrorType.CONVERSION_ERROR,
            request_id=request_id,
            metadata=metadata,
        )

        log_error(
            LogRecord(
                event=LogEvent.CONVERSION_EVENT.value,
                message=f"Failed to convert tool choice: {str(e)}",
                request_id=request_id,
                data=metadata,
            )
        )
        raise


def convert_openai_to_anthropic_response(
    openai_response: ChatCompletion,
    original_anthropic_model_name: str,
    request_id: Optional[str] = None,
    settings: Optional[Any] = None,
) -> MessagesResponse:
    """
    Convert an OpenAI ChatCompletion response to Anthropic MessagesResponse with error tracking.

    Args:
        openai_response: OpenAI ChatCompletion response
        original_anthropic_model_name: The model name originally requested
        request_id: Optional request ID for logging
        settings: Optional settings object

    Returns:
        Anthropic MessagesResponse
    """
    try:
        context = ConversionContext(
            request_id=request_id,
            original_model=original_anthropic_model_name,
            settings=settings,
        )
        converter = OpenAIToAnthropicConverter(context)
        return converter.convert_response(openai_response)
    except Exception as e:
        metadata = {
            "conversion_type": "openai_to_anthropic_response",
            "original_model": original_anthropic_model_name,
            "openai_model": getattr(openai_response, "model", "unknown")
            if openai_response
            else None,
            "finish_reason": getattr(openai_response.choices[0], "finish_reason", None)
            if openai_response and openai_response.choices
            else None,
            "error_message": str(e),
            "error_type": type(e).__name__,
        }

        # Capture usage info if available
        if (
            openai_response
            and hasattr(openai_response, "usage")
            and openai_response.usage
        ):
            metadata["input_tokens"] = getattr(
                openai_response.usage, "prompt_tokens", None
            )
            metadata["output_tokens"] = getattr(
                openai_response.usage, "completion_tokens", None
            )

        _fire_and_forget_error_tracking(
            exception=e,
            error_type=ErrorType.CONVERSION_ERROR,
            request_id=request_id,
            metadata=metadata,
        )

        log_error(
            LogRecord(
                event=LogEvent.CONVERSION_EVENT.value,
                message=f"Failed to convert OpenAI response to Anthropic format: {str(e)}",
                request_id=request_id,
                data=metadata,
            )
        )
        raise
