"""
Response handling utilities for provider implementations.
Handles response decoding, UTF-8 processing, error recovery, and JSON validation.
"""

import json
import logging
from typing import Any, Dict, Optional, Union, cast


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
            error_info["correlation_id"] = correlation_id  # type: ignore[assignment]

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
        elif error_type == "JSONDecodeError":
            return "json_parse_error"
        elif "expecting value" in error_str and "json" in error_str.lower():
            return "json_parse_error"
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
            "json_parse_error",
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


class ResponseValidator:
    """Validates API responses for JSON structure and content integrity."""

    @staticmethod
    def validate_json_response(
        response_content: Union[str, bytes], context: str = "API response"
    ) -> bool:
        """
        Validate that response content is valid JSON.

        Args:
            response_content: Raw response content (string or bytes)
            context: Description of the response context for logging

        Returns:
            True if JSON is valid, False otherwise

        Raises:
            ValueError: If response_content is not a valid type
        """
        if not isinstance(response_content, (str, bytes)):
            raise ValueError(
                f"Response content must be str or bytes, got {type(response_content)}"
            )

        try:
            # Convert bytes to string if needed
            if isinstance(response_content, bytes):
                content_str = response_content.decode("utf-8")
            else:
                content_str = response_content

            # Attempt to parse JSON
            json.loads(content_str)
            return True

        except json.JSONDecodeError as e:
            logging.warning(
                f"Invalid JSON detected in {context}: {e}. "
                f"Response preview: {content_str[:200]}{'...' if len(content_str) > 200 else ''}"
            )
            return False
        except UnicodeDecodeError as e:
            logging.error(f"Failed to decode response bytes in {context}: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error validating JSON in {context}: {e}")
            return False

    @staticmethod
    def safe_parse_json(
        response_content: Union[str, bytes], context: str = "API response"
    ) -> Optional[Dict[str, Any]]:
        """
        Safely parse JSON response with comprehensive error handling.

        Args:
            response_content: Raw response content (string or bytes)
            context: Description of the response context for logging

        Returns:
            Parsed JSON as dictionary, or None if parsing fails
        """
        if not ResponseValidator.validate_json_response(response_content, context):
            return None

        try:
            # Convert bytes to string if needed
            if isinstance(response_content, bytes):
                content_str = response_content.decode("utf-8")
            else:
                content_str = response_content

            # Parse JSON
            return json.loads(content_str)

        except Exception as e:
            # This should rarely happen since we validated first, but handle just in case
            logging.error(f"Unexpected error parsing validated JSON in {context}: {e}")
            return None

    @staticmethod
    def detect_json_corruption_patterns(
        response_content: Union[str, bytes],
    ) -> list[str]:
        """
        Detect common patterns of JSON corruption.

        Args:
            response_content: Raw response content

        Returns:
            List of detected corruption patterns
        """
        issues = []

        if isinstance(response_content, bytes):
            content_str = response_content.decode("utf-8", errors="replace")
        else:
            content_str = response_content

        # Check for common corruption patterns
        if not content_str.strip():
            issues.append("empty_response")

        if content_str.startswith("<!DOCTYPE") or content_str.startswith("<html"):
            issues.append("html_response_instead_of_json")

        if content_str.startswith("<?xml"):
            issues.append("xml_response_instead_of_json")

        if "error" in content_str.lower() and (
            "500" in content_str or "502" in content_str or "503" in content_str
        ):
            issues.append("server_error_response")

        if len(content_str) < 10 and not content_str.strip().startswith("{"):
            issues.append("response_too_short_for_valid_json")

        # Check for truncated JSON
        if content_str.count("{") != content_str.count("}"):
            issues.append("unmatched_braces_truncated_json")

        if content_str.count("[") != content_str.count("]"):
            issues.append("unmatched_brackets_truncated_json")

        # Check for common network error patterns
        if "timeout" in content_str.lower():
            issues.append("timeout_error_in_response")

        if "connection" in content_str.lower() and "refused" in content_str.lower():
            issues.append("connection_refused_error")

        return issues

    @staticmethod
    def create_validation_error_response(
        original_error: Exception,
        response_content: Optional[Union[str, bytes]] = None,
        context: str = "API response",
    ) -> Dict[str, Any]:
        """
        Create a structured error response for JSON validation failures.

        Args:
            original_error: The original JSON decode error
            response_content: The invalid response content (optional)
            context: Description of the response context

        Returns:
            Dictionary with detailed error information
        """
        error_response = {
            "error": {
                "type": "json_validation_error",
                "message": f"Failed to parse {context}: {str(original_error)}",
                "original_error_type": type(original_error).__name__,
                "validation_context": context,
            }
        }

        # Add corruption pattern detection if response content is available
        if response_content is not None:
            corruption_patterns = ResponseValidator.detect_json_corruption_patterns(
                response_content
            )
            if corruption_patterns:
                error_response["error"]["corruption_patterns"] = corruption_patterns

            # Add response preview (first 300 characters) for debugging
            if isinstance(response_content, bytes):
                content_preview = response_content[:300].decode(
                    "utf-8", errors="replace"
                )
            else:
                content_preview = response_content[:300]

            error_response["error"]["response_preview"] = content_preview

        return error_response
