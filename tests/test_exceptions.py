"""Tests for CCProxy domain exceptions."""

from ccproxy.domain.exceptions import (
    CCProxyException,
    ConversionError,
    SerializationError,
    ValidationError,
    CacheError,
    CacheValidationError,
    CacheMemoryError,
    TokenizationError,
    TruncationError,
    ProviderError,
    UpstreamError,
    UpstreamTimeoutError,
    RateLimitError,
    AuthenticationError,
    StreamingError,
    UTF8Error,
    ConfigurationError,
    ModelSelectionError,
    ResourceExhaustedError,
)


class TestCCProxyException:
    """Test base CCProxyException."""

    def test_basic_exception(self):
        """Test basic exception with message only."""
        exc = CCProxyException("Test error")
        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.request_id is None
        assert exc.details == {}

    def test_exception_with_request_id(self):
        """Test exception with request ID."""
        exc = CCProxyException("Test error", request_id="req-123")
        assert exc.message == "Test error"
        assert exc.request_id == "req-123"

    def test_exception_with_details(self):
        """Test exception with details dict."""
        details = {"key": "value", "count": 42}
        exc = CCProxyException("Test error", details=details)
        assert exc.details == details

    def test_exception_with_all_params(self):
        """Test exception with all parameters."""
        exc = CCProxyException(
            "Test error", request_id="req-456", details={"error_code": "TEST_001"}
        )
        assert exc.message == "Test error"
        assert exc.request_id == "req-456"
        assert exc.details["error_code"] == "TEST_001"


class TestConversionError:
    """Test ConversionError."""

    def test_conversion_error(self):
        """Test conversion error inherits from CCProxyException."""
        exc = ConversionError("Conversion failed")
        assert isinstance(exc, CCProxyException)
        assert exc.message == "Conversion failed"


class TestSerializationError:
    """Test SerializationError."""

    def test_serialization_error_basic(self):
        """Test basic serialization error."""
        exc = SerializationError("JSON serialization failed")
        assert isinstance(exc, ConversionError)
        assert exc.message == "JSON serialization failed"
        assert exc.content_type is None

    def test_serialization_error_with_content_type(self):
        """Test serialization error with content type."""
        exc = SerializationError("Failed to serialize", content_type="application/json")
        assert exc.content_type == "application/json"

    def test_serialization_error_full_params(self):
        """Test serialization error with all parameters."""
        exc = SerializationError(
            "Serialization error",
            content_type="image/png",
            request_id="req-789",
            details={"field": "image_data"},
        )
        assert exc.content_type == "image/png"
        assert exc.request_id == "req-789"
        assert exc.details["field"] == "image_data"


class TestCacheErrors:
    """Test cache-related exceptions."""

    def test_cache_error(self):
        """Test base cache error."""
        exc = CacheError("Cache operation failed")
        assert isinstance(exc, CCProxyException)
        assert exc.message == "Cache operation failed"

    def test_cache_validation_error(self):
        """Test cache validation error."""
        exc = CacheValidationError(
            "Validation failed", validation_failures=5, request_id="req-cache"
        )
        assert isinstance(exc, CacheError)
        assert exc.validation_failures == 5
        assert exc.request_id == "req-cache"

    def test_cache_memory_error(self):
        """Test cache memory error."""
        exc = CacheMemoryError(
            "Memory limit exceeded",
            memory_usage_bytes=600_000_000,
            max_memory_bytes=500_000_000,
        )
        assert isinstance(exc, CacheError)
        assert exc.memory_usage_bytes == 600_000_000
        assert exc.max_memory_bytes == 500_000_000


class TestTokenizationErrors:
    """Test tokenization-related exceptions."""

    def test_tokenization_error(self):
        """Test basic tokenization error."""
        exc = TokenizationError("Tokenization failed", model="gpt-4")
        assert isinstance(exc, CCProxyException)
        assert exc.model == "gpt-4"

    def test_truncation_error(self):
        """Test truncation error."""
        exc = TruncationError(
            "Truncation failed",
            strategy="auto",
            model="gpt-3.5-turbo",
            request_id="req-trunc",
        )
        assert isinstance(exc, TokenizationError)
        assert exc.strategy == "auto"
        assert exc.model == "gpt-3.5-turbo"
        assert exc.request_id == "req-trunc"


class TestProviderErrors:
    """Test provider-related exceptions."""

    def test_provider_error(self):
        """Test base provider error."""
        exc = ProviderError("Provider failed", provider_name="openai")
        assert isinstance(exc, CCProxyException)
        assert exc.provider_name == "openai"

    def test_upstream_error(self):
        """Test upstream error."""
        exc = UpstreamError(
            "Upstream API failed", status_code=500, provider_name="openai"
        )
        assert isinstance(exc, ProviderError)
        assert exc.status_code == 500
        assert exc.provider_name == "openai"

    def test_upstream_timeout_error(self):
        """Test upstream timeout error."""
        exc = UpstreamTimeoutError(
            "Request timed out",
            timeout_seconds=30.0,
            provider_name="openai",
            request_id="req-timeout",
        )
        assert isinstance(exc, UpstreamError)
        assert exc.timeout_seconds == 30.0
        assert exc.provider_name == "openai"
        assert exc.status_code is None  # Timeout doesn't have status code

    def test_rate_limit_error(self):
        """Test rate limit error."""
        exc = RateLimitError(
            "Rate limit exceeded", retry_after=60.0, provider_name="openai"
        )
        assert isinstance(exc, UpstreamError)
        assert exc.status_code == 429
        assert exc.retry_after == 60.0

    def test_authentication_error(self):
        """Test authentication error."""
        exc = AuthenticationError(
            "Invalid API key", provider_name="openai", request_id="req-auth"
        )
        assert isinstance(exc, UpstreamError)
        assert exc.status_code == 401
        assert exc.provider_name == "openai"


class TestStreamingError:
    """Test streaming error."""

    def test_streaming_error_basic(self):
        """Test basic streaming error."""
        exc = StreamingError("Stream interrupted")
        assert isinstance(exc, CCProxyException)
        assert exc.chunk_index is None

    def test_streaming_error_with_chunk_index(self):
        """Test streaming error with chunk index."""
        exc = StreamingError("Invalid chunk", chunk_index=42, request_id="req-stream")
        assert exc.chunk_index == 42
        assert exc.request_id == "req-stream"


class TestUTF8Error:
    """Test UTF-8 error."""

    def test_utf8_error_basic(self):
        """Test basic UTF-8 error."""
        exc = UTF8Error("Encoding failed")
        assert isinstance(exc, CCProxyException)
        assert exc.raw_bytes is None

    def test_utf8_error_with_bytes(self):
        """Test UTF-8 error with raw bytes."""
        raw = b"\xff\xfe"
        exc = UTF8Error("Invalid UTF-8", raw_bytes=raw, request_id="req-utf8")
        assert exc.raw_bytes == raw
        assert exc.request_id == "req-utf8"


class TestConfigurationError:
    """Test configuration error."""

    def test_configuration_error_basic(self):
        """Test basic configuration error."""
        exc = ConfigurationError("Missing config")
        assert isinstance(exc, CCProxyException)
        assert exc.config_key is None

    def test_configuration_error_with_key(self):
        """Test configuration error with config key."""
        exc = ConfigurationError("Invalid value", config_key="OPENAI_API_KEY")
        assert exc.config_key == "OPENAI_API_KEY"


class TestModelSelectionError:
    """Test model selection error."""

    def test_model_selection_error_basic(self):
        """Test basic model selection error."""
        exc = ModelSelectionError("Model not found")
        assert isinstance(exc, CCProxyException)
        assert exc.requested_model is None

    def test_model_selection_error_with_model(self):
        """Test model selection error with requested model."""
        exc = ModelSelectionError(
            "Model unavailable", requested_model="gpt-5", request_id="req-model"
        )
        assert exc.requested_model == "gpt-5"
        assert exc.request_id == "req-model"


class TestResourceExhaustedError:
    """Test resource exhausted error."""

    def test_resource_exhausted_error_basic(self):
        """Test basic resource exhausted error."""
        exc = ResourceExhaustedError("Resources exhausted")
        assert isinstance(exc, CCProxyException)
        assert exc.resource_type is None

    def test_resource_exhausted_error_with_type(self):
        """Test resource exhausted error with resource type."""
        exc = ResourceExhaustedError(
            "Out of memory", resource_type="memory", request_id="req-resource"
        )
        assert exc.resource_type == "memory"
        assert exc.request_id == "req-resource"


class TestExceptionHierarchy:
    """Test exception hierarchy relationships."""

    def test_conversion_hierarchy(self):
        """Test conversion exception hierarchy."""
        exc = SerializationError("Test")
        assert isinstance(exc, SerializationError)
        assert isinstance(exc, ConversionError)
        assert isinstance(exc, CCProxyException)
        assert isinstance(exc, Exception)

    def test_cache_hierarchy(self):
        """Test cache exception hierarchy."""
        exc = CacheValidationError("Test")
        assert isinstance(exc, CacheValidationError)
        assert isinstance(exc, CacheError)
        assert isinstance(exc, CCProxyException)

    def test_tokenization_hierarchy(self):
        """Test tokenization exception hierarchy."""
        exc = TruncationError("Test")
        assert isinstance(exc, TruncationError)
        assert isinstance(exc, TokenizationError)
        assert isinstance(exc, CCProxyException)

    def test_provider_hierarchy(self):
        """Test provider exception hierarchy."""
        exc = RateLimitError("Test")
        assert isinstance(exc, RateLimitError)
        assert isinstance(exc, UpstreamError)
        assert isinstance(exc, ProviderError)
        assert isinstance(exc, CCProxyException)

    def test_all_inherit_from_base(self):
        """Test that all exceptions inherit from CCProxyException."""
        exceptions = [
            ConversionError("test"),
            SerializationError("test"),
            ValidationError("test"),
            CacheError("test"),
            CacheValidationError("test"),
            CacheMemoryError("test"),
            TokenizationError("test"),
            TruncationError("test"),
            ProviderError("test"),
            UpstreamError("test"),
            UpstreamTimeoutError("test"),
            RateLimitError("test"),
            AuthenticationError("test"),
            StreamingError("test"),
            UTF8Error("test"),
            ConfigurationError("test"),
            ModelSelectionError("test"),
            ResourceExhaustedError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, CCProxyException)
            assert isinstance(exc, Exception)
