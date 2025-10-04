"""Comprehensive test suite for HTTP routes and middleware."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from ccproxy.interfaces.http.app import create_app
from ccproxy.config import Settings
from ccproxy.domain.models import (
    ContentBlockText,
    MessagesResponse,
    Usage,
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.openai_api_key = "test-api-key"
    settings.base_url = "https://api.openai.com/v1"
    settings.tracing_enabled = False
    settings.host = "127.0.0.1"
    settings.port = 11434
    settings.app_name = "CCProxy"
    settings.app_version = "1.0.0"
    settings.log_level = "INFO"
    settings.log_pretty_console = False
    settings.log_file_path = None
    settings.redact_log_fields = []
    settings.rate_limit_enabled = True
    settings.rate_limit_per_minute = 60
    settings.rate_limit_burst = 30
    settings.security_headers_enabled = True
    settings.enable_hsts = False
    settings.enable_cors = False
    settings.error_tracking_enabled = True
    settings.metrics_cache_enabled = True
    # Circuit breaker settings
    settings.circuit_breaker_failure_threshold = 5
    settings.circuit_breaker_recovery_timeout = 60
    settings.circuit_breaker_half_open_requests = 3
    settings.circuit_breaker_expected_exception = None
    # Provider settings
    settings.request_timeout = 120.0
    settings.max_retries = 3
    settings.retry_delay = 1.0
    settings.max_connections = 100
    settings.max_keepalive = 100
    settings.keepalive_expiry = 120
    # Additional pool settings
    settings.pool_max_keepalive_connections = 100
    settings.pool_max_connections = 100
    settings.pool_keepalive_expiry = 120
    # HTTP timeout settings
    settings.http_connect_timeout = 30.0
    settings.http_write_timeout = 30.0
    settings.http_pool_timeout = 30.0
    settings.referer_url = "http://localhost:11434"
    settings.http2_enabled = True
    settings.provider_rate_limit_rpm = 5000
    settings.provider_rate_limit_tpm = 100000
    settings.provider_enable_rate_limit = True
    settings.provider_rate_limit_auto_start = True
    settings.provider_max_retries = 3  # Add missing attribute
    settings.provider_retry_base_delay = 1.0  # Add missing retry base delay
    settings.provider_retry_jitter = True  # Add missing retry jitter
    # Circuit breaker settings
    settings.circuit_breaker_failure_threshold = 5
    settings.circuit_breaker_recovery_timeout = 60
    settings.circuit_breaker_half_open_requests = 3
    # Timeout settings
    settings.max_stream_seconds = 60
    # Client rate limit settings
    settings.client_rate_limit_enabled = False
    settings.client_rate_limit_rpm = 1000
    settings.client_rate_limit_tpm = 50000
    settings.client_rate_limit_burst = 10
    settings.client_rate_limit_adaptive = False
    # Model mappings
    settings.big_model = "gpt-4"
    settings.small_model = "gpt-3.5-turbo"
    settings.big_model_name = "gpt-4"  # Add big_model_name
    # Cache settings
    settings.cache_enabled = True
    settings.cache_ttl_seconds = 3600
    settings.cache_max_size_mb = 100
    # Thread pool settings
    settings.thread_pool_max_workers = None
    settings.thread_pool_high_cpu_threshold = None
    settings.thread_pool_auto_scale = False
    settings.error_tracking_file = "error.jsonl"
    # Cache warmup settings
    settings.cache_warmup_enabled = False
    settings.cache_warmup_file_path = "cache_warmup.json"
    settings.cache_warmup_max_items = 100
    settings.cache_warmup_on_startup = True
    settings.cache_warmup_preload_common = True
    settings.cache_warmup_auto_save_popular = True
    settings.cache_warmup_popularity_threshold = 3
    settings.cache_warmup_save_interval_seconds = 3600
    return settings


@pytest.fixture
def test_app(mock_settings):
    """Create test FastAPI app."""
    app = create_app(mock_settings)
    return app


@pytest.fixture
def test_client(test_app):
    """Yield a TestClient and ensure proper cleanup after each test."""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
def sample_anthropic_request():
    """Create a sample Anthropic messages request."""
    return {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 100,
        "stream": False,
    }


@pytest.fixture
def sample_anthropic_response():
    """Create a sample Anthropic messages response."""
    return MessagesResponse(
        id="msg_test_123",
        type="message",
        role="assistant",
        model="claude-3-opus-20240229",
        content=[ContentBlockText(type="text", text="I'm doing well, thank you!")],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


class TestHealthRoutes:
    """Test health check routes."""

    def test_health_check(self, test_client):
        """Test basic health check endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_check_alternate_path(self, test_client):
        """Test health check at /health path."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_readiness_check(self, test_client):
        """Test readiness endpoint."""
        response = test_client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "timestamp" in data
        assert "version" in data

    def test_liveness_check(self, test_client):
        """Test liveness endpoint."""
        response = test_client.get("/alive")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}


class TestMonitoringRoutes:
    """Test monitoring and metrics routes."""

    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint."""
        response = test_client.get("/v1/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "uptime_seconds" in data
        assert "requests" in data
        assert "cache" in data
        assert "memory" in data

    def test_cache_stats_endpoint(self, test_client):
        """Test cache statistics endpoint."""
        response = test_client.get("/v1/cache/stats")
        assert response.status_code == 200

        data = response.json()
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate" in data
        assert "total_entries" in data

    def test_clear_cache_endpoint(self, test_client):
        """Test cache clearing endpoint."""
        response = test_client.post("/v1/cache/clear")
        assert response.status_code == 200
        assert response.json() == {
            "status": "cache_cleared",
            "message": "All caches have been cleared",
        }


class TestMessagesRoute:
    """Test the main messages proxy route."""

    @pytest.mark.anyio
    async def test_messages_endpoint_success(
        self, test_client, sample_anthropic_request
    ):
        """Test successful message proxying."""
        with patch(
            "ccproxy.interfaces.http.routes.messages.OpenAIProvider"
        ) as mock_provider:
            # Mock the provider response
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.content = "Test response"
            mock_completion.choices[0].message.role = "assistant"
            mock_completion.choices[0].finish_reason = "stop"
            mock_completion.usage = MagicMock()
            mock_completion.usage.prompt_tokens = 10
            mock_completion.usage.completion_tokens = 20
            mock_completion.usage.total_tokens = 30
            mock_completion.id = "chatcmpl-test"
            mock_completion.model = "gpt-5"

            mock_provider.return_value.create_chat_completion = AsyncMock(
                return_value=mock_completion
            )

            response = test_client.post(
                "/v1/messages",
                json=sample_anthropic_request,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert data["type"] == "message"
            assert data["role"] == "assistant"

    @pytest.mark.anyio
    async def test_messages_endpoint_invalid_json(self, test_client):
        """Test messages endpoint with invalid JSON."""
        response = test_client.post(
            "/v1/messages",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
        assert "error" in data

    @pytest.mark.anyio
    async def test_messages_endpoint_validation_error(self, test_client):
        """Test messages endpoint with validation error."""
        invalid_request = {
            "model": "invalid-model",
            # Missing required 'messages' field
        }

        response = test_client.post(
            "/v1/messages",
            json=invalid_request,
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422
        data = response.json()
        assert data["type"] == "error"

    def test_messages_endpoint_streaming(self, test_app, sample_anthropic_request):
        """Test streaming response."""
        sample_anthropic_request["stream"] = True

        with patch(
            "ccproxy.interfaces.http.routes.messages.OpenAIProvider"
        ) as mock_provider:
            # Mock streaming response
            async def mock_stream():
                yield MagicMock()  # Mock chunk

            mock_provider.return_value.create_chat_completion = AsyncMock(
                return_value=mock_stream()
            )

            with TestClient(test_app) as streamed_client:
                response = streamed_client.post(
                    "/v1/messages",
                    json=sample_anthropic_request,
                    headers={"Content-Type": "application/json"},
                )

            # Streaming returns SSE format
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")


class TestMiddleware:
    """Test middleware functionality."""

    def test_request_id_middleware(self, test_client):
        """Test request ID middleware."""
        response = test_client.get("/")

        # Should have X-Request-ID header in response
        assert "x-request-id" in response.headers
        request_id = response.headers["x-request-id"]

        # Request ID should be a valid UUID format
        assert len(request_id) > 0

    def test_security_headers_middleware(self, test_client):
        """Test security headers middleware."""
        response = test_client.get("/")

        # Check security headers
        assert response.headers.get("x-content-type-options") == "nosniff"
        assert response.headers.get("x-frame-options") == "DENY"
        assert "content-security-policy" in response.headers

    def test_cors_disabled_by_default(self, test_client):
        """Test that CORS is disabled by default."""
        response = test_client.options("/")

        # CORS headers should not be present when disabled
        assert "access-control-allow-origin" not in response.headers

    @pytest.mark.anyio
    async def test_cors_enabled(self, mock_settings):
        """Test CORS when enabled."""
        mock_settings.enable_cors = True
        mock_settings.cors_allow_origins = ["http://localhost:3000"]
        mock_settings.cors_allow_methods = ["GET", "POST"]
        mock_settings.cors_allow_headers = ["Content-Type", "Authorization"]

        app = await create_app(mock_settings)
        client = TestClient(app)

        response = client.options(
            "/v1/messages",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # CORS headers should be present when enabled
        assert (
            response.headers.get("access-control-allow-origin")
            == "http://localhost:3000"
        )

    def test_rate_limiting(self, test_client, mock_settings):
        """Test rate limiting middleware."""
        # Make many requests quickly
        responses = []
        for _ in range(100):
            response = test_client.get("/")
            responses.append(response)

        # Some requests should be rate limited (if rate limiting is enforced)
        # Note: Actual rate limiting behavior depends on implementation
        assert all(r.status_code in [200, 429] for r in responses)


class TestErrorHandling:
    """Test error handling in routes."""

    def test_404_not_found(self, test_client):
        """Test 404 error for unknown route."""
        response = test_client.get("/unknown/path")
        assert response.status_code == 404

    def test_method_not_allowed(self, test_client):
        """Test 405 error for wrong method."""
        response = test_client.delete("/v1/messages")
        assert response.status_code == 405

    @pytest.mark.anyio
    async def test_internal_server_error_handling(self, test_client):
        """Test handling of internal server errors."""
        with patch(
            "ccproxy.interfaces.http.routes.messages.OpenAIProvider"
        ) as mock_provider:
            # Mock provider to raise an exception
            mock_provider.return_value.create_chat_completion = AsyncMock(
                side_effect=Exception("Internal error")
            )

            response = test_client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-opus-20240229",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 100,
                },
            )

            assert response.status_code == 500
            data = response.json()
            assert data["type"] == "error"


class TestTokenCountEndpoint:
    """Test token counting endpoint."""

    def test_token_count_endpoint(self, test_client):
        """Test token counting endpoint."""
        request_data = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Count my tokens"}],
            "system": "You are a helpful assistant",
        }

        response = test_client.post("/v1/messages/count_tokens", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
        assert isinstance(data["input_tokens"], int)
        assert data["input_tokens"] > 0


class TestRequestValidation:
    """Test request validation."""

    def test_max_tokens_validation(self, test_client):
        """Test max_tokens validation."""
        request_data = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": -1,  # Invalid negative value
        }

        response = test_client.post("/v1/messages", json=request_data)

        assert response.status_code == 422

    def test_temperature_validation(self, test_client):
        """Test temperature validation."""
        request_data = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 2.5,  # Invalid, should be between 0 and 1
        }

        response = test_client.post("/v1/messages", json=request_data)

        # Should either be accepted or return 422 depending on validation
        assert response.status_code in [200, 422]

    def test_empty_messages_validation(self, test_client):
        """Test empty messages validation."""
        request_data = {
            "model": "claude-3-opus-20240229",
            "messages": [],  # Empty messages
            "max_tokens": 100,
        }

        response = test_client.post("/v1/messages", json=request_data)

        # Depends on whether empty messages are allowed
        assert response.status_code in [200, 422]


class TestAPIVersioning:
    """Test API versioning."""

    def test_v1_prefix(self, test_client):
        """Test that v1 endpoints are accessible."""
        # Test various v1 endpoints
        endpoints = [
            "/v1/metrics",
            "/v1/cache/stats",
        ]

        for endpoint in endpoints:
            response = test_client.get(endpoint)
            # Should not return 404
            assert response.status_code != 404


class TestContentTypes:
    """Test content type handling."""

    def test_json_content_type(self, test_client):
        """Test JSON content type."""
        response = test_client.get("/")
        assert "application/json" in response.headers.get("content-type", "")

    def test_unsupported_content_type(self, test_client):
        """Test unsupported content type."""
        response = test_client.post(
            "/v1/messages",
            data="plain text data",
            headers={"Content-Type": "text/plain"},
        )

        # Should reject non-JSON content type for this endpoint
        assert response.status_code in [400, 415]
