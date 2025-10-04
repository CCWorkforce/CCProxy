"""
Response handling utilities for provider implementations.
Handles response decoding, UTF-8 processing, and error recovery.
"""

import logging
from typing import Any, Dict, Optional, cast


def safe_decode_response(response_bytes: bytes, context: str = "API response") -> str:
    """
    Safely decode response bytes to UTF-8 string with error handling.

    This function attempts to decode byte responses from APIs, with fallback
    strategies for handling malformed UTF-8 sequences.

    Args:
        response_bytes: Raw bytes from HTTP response
        context: Description of the response context for logging

    Returns:
        Decoded UTF-8 string

    Raises:
        UnicodeDecodeError: If decoding fails even with error handling
    """
    try:
        # First try strict UTF-8 decoding
        return response_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        # Log the issue
        logging.warning(
            f"Malformed UTF-8 bytes detected in {context}. "
            f"Attempting recovery with byte replacement. Error: {str(e)}"
        )

        try:
            # Try with error replacement - replaces malformed bytes with replacement character
            decoded = response_bytes.decode("utf-8", errors="replace")

            # Log successful recovery
            logging.info(
                f"Successfully recovered {context} by replacing malformed UTF-8 bytes"
            )
            return decoded
        except Exception as recovery_error:
            # If even replacement fails, raise the original error
            logging.error(
                f"Failed to recover {context} even with byte replacement. "
                f"Recovery error: {str(recovery_error)}"
            )
            raise e


class ResponseProcessor:
    """Processes and validates API responses."""

    @staticmethod
    def process_chat_completion_response(response: Any) -> Any:
        """
        Process a chat completion response, handling byte content.

        Args:
            response: The API response object

        Returns:
            Processed response with decoded content
        """
        if hasattr(response, "choices") and response.choices:
            for choice in response.choices:
                if (
                    hasattr(choice, "message")
                    and hasattr(choice.message, "content")
                    and isinstance(choice.message.content, bytes)
                ):
                    choice.message.content = safe_decode_response(
                        choice.message.content, "chat completion response"
                    )
        return response

    @staticmethod
    def extract_usage_info(response: Any) -> Optional[Dict[str, int]]:
        """
        Extract usage information from API response.

        Args:
            response: The API response object

        Returns:
            Dictionary with usage information or None
        """
        if hasattr(response, "usage") and response.usage:
            return {
                "total_tokens": getattr(response.usage, "total_tokens", 0),
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            }
        return None

    @staticmethod
    def extract_model_info(response: Any) -> Optional[str]:
        """
        Extract model information from API response.

        Args:
            response: The API response object

        Returns:
            Model name or None
        """
        if hasattr(response, "model"):
            return str(response.model)
        return None

    @staticmethod
    def is_streaming_response(response: Any) -> bool:
        """
        Check if response is a streaming response.

        Args:
            response: The API response object

        Returns:
            True if response is streaming, False otherwise
        """
        # Check for common async generator patterns
        return hasattr(response, "__aiter__") or hasattr(response, "__anext__")


class ErrorResponseHandler:
    """Handles error responses and provides meaningful error messages."""

    @staticmethod
    def create_error_response(
        error: Exception,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a structured error response.

        Args:
            error: The exception that occurred
            correlation_id: Optional request correlation ID

        Returns:
            Dictionary with error information
        """
        error_info = {
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            }
        }

        if correlation_id:
            error_info["correlation_id"] = correlation_id

        return error_info

    @staticmethod
    def classify_error(error: Exception) -> str:
        """
        Classify an error into categories for better handling.

        Args:
            error: The exception to classify

        Returns:
            Error category string
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check for specific error patterns
        if "utf-8" in error_str or "codec can't decode" in error_str:
            return "conversion_error"
        elif "timeout" in error_str:
            return "timeout_error"
        elif "rate" in error_str and "limit" in error_str:
            return "rate_limit_error"
        elif "authentication" in error_str or "unauthorized" in error_str:
            return "auth_error"
        elif "connection" in error_str or "network" in error_str:
            return "network_error"
        elif error_type == "UnicodeDecodeError":
            return "conversion_error"
        else:
            return "api_error"

    @staticmethod
    def should_retry(error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: The exception to check

        Returns:
            True if error is retryable, False otherwise
        """
        error_category = ErrorResponseHandler.classify_error(error)
        retryable_categories = {
            "network_error",
            "timeout_error",
            "rate_limit_error",
        }
        return error_category in retryable_categories


class StreamResponseHandler:
    """Handles streaming responses from API."""

    @staticmethod
    async def process_stream_chunk(chunk: Any) -> Any:
        """
        Process a single chunk from a streaming response.

        Args:
            chunk: Stream chunk to process

        Returns:
            Processed chunk
        """
        # If chunk contains choices with byte content, decode it
        if hasattr(chunk, "choices") and chunk.choices:
            for choice in chunk.choices:
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    if isinstance(choice.delta.content, bytes):
                        choice.delta.content = safe_decode_response(
                            choice.delta.content, "stream chunk"
                        )
        return chunk

    @staticmethod
    def extract_content_from_chunk(chunk: Any) -> Optional[str]:
        """
        Extract content from a stream chunk.

        Args:
            chunk: Stream chunk

        Returns:
            Content string or None
        """
        if hasattr(chunk, "choices") and chunk.choices:
            for choice in chunk.choices:
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    return cast(Optional[str], choice.delta.content)
        return None
