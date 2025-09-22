"""Custom exception hierarchy for CCProxy application.

This module defines specific exception types for different error scenarios,
replacing generic Exception handling with more granular error types.
"""

from typing import Optional, Dict, Any


class CCProxyException(Exception):
    """Base exception for all CCProxy-specific exceptions."""

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.details = details or {}


class ConversionError(CCProxyException):
    """Raised when message conversion between Anthropic and OpenAI formats fails."""

    pass


class SerializationError(ConversionError):
    """Raised when serializing/deserializing content fails."""

    def __init__(
        self,
        message: str,
        content_type: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, request_id, details)
        self.content_type = content_type


class ValidationError(CCProxyException):
    """Raised when request or response validation fails."""

    pass


class CacheError(CCProxyException):
    """Base exception for cache-related errors."""

    pass


class CacheValidationError(CacheError):
    """Raised when cached response validation fails."""

    def __init__(
        self,
        message: str,
        validation_failures: int = 0,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, request_id, details)
        self.validation_failures = validation_failures


class CacheMemoryError(CacheError):
    """Raised when cache memory limits are exceeded."""

    def __init__(
        self,
        message: str,
        memory_usage_bytes: int = 0,
        max_memory_bytes: int = 0,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, request_id, details)
        self.memory_usage_bytes = memory_usage_bytes
        self.max_memory_bytes = max_memory_bytes


class TokenizationError(CCProxyException):
    """Raised when tokenization operations fail."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, request_id, details)
        self.model = model


class TruncationError(TokenizationError):
    """Raised when message truncation fails or results in invalid state."""

    def __init__(
        self,
        message: str,
        strategy: Optional[str] = None,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, model, request_id, details)
        self.strategy = strategy


class ProviderError(CCProxyException):
    """Base exception for provider-related errors."""

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, request_id, details)
        self.provider_name = provider_name


class UpstreamError(ProviderError):
    """Raised when upstream API calls fail."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider_name, request_id, details)
        self.status_code = status_code


class UpstreamTimeoutError(UpstreamError):
    """Raised when upstream API call times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, None, provider_name, request_id, details)
        self.timeout_seconds = timeout_seconds


class RateLimitError(UpstreamError):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, 429, provider_name, request_id, details)
        self.retry_after = retry_after


class AuthenticationError(UpstreamError):
    """Raised when authentication with upstream provider fails."""

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, 401, provider_name, request_id, details)


class StreamingError(CCProxyException):
    """Raised when streaming operations fail."""

    def __init__(
        self,
        message: str,
        chunk_index: Optional[int] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, request_id, details)
        self.chunk_index = chunk_index


class UTF8Error(CCProxyException):
    """Raised when UTF-8 encoding/decoding fails."""

    def __init__(
        self,
        message: str,
        raw_bytes: Optional[bytes] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, request_id, details)
        self.raw_bytes = raw_bytes


class ConfigurationError(CCProxyException):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, request_id, details)
        self.config_key = config_key


class ModelSelectionError(CCProxyException):
    """Raised when model selection fails."""

    def __init__(
        self,
        message: str,
        requested_model: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, request_id, details)
        self.requested_model = requested_model


class ResourceExhaustedError(CCProxyException):
    """Raised when system resources are exhausted."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, request_id, details)
        self.resource_type = resource_type
