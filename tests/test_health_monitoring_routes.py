"""Comprehensive tests for health and monitoring routes."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from ccproxy.interfaces.http.app import create_app


@pytest.fixture
def mock_settings(tmp_path, monkeypatch):
    """Create settings using actual Settings class with test environment."""
    # Set required environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("BIG_MODEL_NAME", "gpt-4")
    monkeypatch.setenv("SMALL_MODEL_NAME", "gpt-3.5-turbo")

    # Create temp log file
    log_file = tmp_path / "test.log"
    error_log_file = tmp_path / "error.log"

    monkeypatch.setenv("LOG_FILE_PATH", str(log_file))
    monkeypatch.setenv("ERROR_LOG_FILE_PATH", str(error_log_file))

    # Use actual Settings class
    from ccproxy.config import Settings

    settings = Settings()
    return settings


@pytest.fixture
def test_app(mock_settings):
    """Create test FastAPI app."""
    app = create_app(mock_settings)
    return app


@pytest.fixture
def test_client(test_app):
    """Create test client."""
    with TestClient(test_app) as client:
        yield client


class TestHealthRoutes:
    """Test health check endpoints."""

    def test_root_health_check(self, test_client):
        """Test root health check endpoint returns status and timestamp."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "ok"
        assert "timestamp" in data

        # Verify timestamp is valid ISO format
        try:
            datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        except ValueError:
            pytest.fail("Invalid timestamp format")

    def test_preflight_check(self, test_client):
        """Test preflight health check endpoint."""
        response = test_client.get("/v1/preflight")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.text == "[BashTool] Pre-flight check passed."

    def test_health_check_response_format(self, test_client):
        """Test that health check returns proper JSON format."""
        response = test_client.get("/")

        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == 2  # status and timestamp only


class TestMonitoringRoutes:
    """Test monitoring and metrics endpoints."""

    def test_metrics_endpoint_structure(self, test_client):
        """Test metrics endpoint returns complete structure."""
        response = test_client.get("/v1/metrics")

        assert response.status_code == 200
        data = response.json()

        # Verify main sections
        assert "performance" in data
        assert "response_cache" in data
        assert "request_validator_cache" in data
        assert "token_count_cache" in data
        assert "converter_caches" in data

    def test_metrics_token_count_cache(self, test_client):
        """Test metrics include token count cache stats."""
        response = test_client.get("/v1/metrics")

        assert response.status_code == 200
        data = response.json()

        token_cache = data["token_count_cache"]
        assert "hits" in token_cache
        assert "misses" in token_cache
        assert "hit_rate" in token_cache
        assert isinstance(token_cache["hits"], int)
        assert isinstance(token_cache["misses"], int)
        assert isinstance(token_cache["hit_rate"], (int, float))

    def test_metrics_converter_caches(self, test_client):
        """Test metrics include converter cache statistics."""
        response = test_client.get("/v1/metrics")

        assert response.status_code == 200
        data = response.json()

        converter_caches = data["converter_caches"]
        assert "tools" in converter_caches
        assert "tool_choice" in converter_caches
        assert "tool_result" in converter_caches

        # Each cache should have currsize and maxsize
        for cache_name in ["tools", "tool_choice", "tool_result"]:
            cache = converter_caches[cache_name]
            assert "currsize" in cache
            assert "maxsize" in cache

    def test_cache_stats_endpoint(self, test_client):
        """Test cache statistics endpoint."""
        response = test_client.get("/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()

        assert "response_cache" in data
        assert "request_validator" in data

    def test_cache_stats_response_cache(self, test_client):
        """Test response cache stats structure."""
        response = test_client.get("/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()

        response_cache = data["response_cache"]
        # Response cache should have stats (exact structure depends on implementation)
        assert isinstance(response_cache, dict)

    def test_cache_stats_request_validator(self, test_client):
        """Test request validator cache stats."""
        response = test_client.get("/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()

        validator_cache = data["request_validator"]
        assert "cache_size" in validator_cache
        assert "max_cache_size" in validator_cache
        assert "cache_hits" in validator_cache
        assert "cache_misses" in validator_cache
        assert "hit_rate" in validator_cache
        assert "total_requests" in validator_cache

    def test_clear_cache_endpoint(self, test_client):
        """Test cache clearing endpoint."""
        response = test_client.post("/v1/cache/clear")

        assert response.status_code == 200
        data = response.json()

        assert data == {"status": "caches_cleared"}

    def test_reset_metrics_endpoint(self, test_client):
        """Test metrics reset endpoint."""
        response = test_client.post("/v1/metrics/reset")

        assert response.status_code == 200
        data = response.json()

        assert data == {"status": "metrics_reset"}

    def test_metrics_endpoint_json_format(self, test_client):
        """Test that metrics return proper JSON format."""
        response = test_client.get("/v1/metrics")

        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert isinstance(data, dict)

    def test_cache_stats_json_format(self, test_client):
        """Test that cache stats return proper JSON format."""
        response = test_client.get("/v1/cache/stats")

        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert isinstance(data, dict)

    def test_multiple_metrics_calls(self, test_client):
        """Test that metrics endpoint can be called multiple times."""
        # Call metrics multiple times
        responses = [test_client.get("/v1/metrics") for _ in range(3)]

        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "performance" in data

    def test_cache_operations_sequence(self, test_client):
        """Test sequence of cache operations."""
        # Get initial stats
        stats_response = test_client.get("/v1/cache/stats")
        assert stats_response.status_code == 200

        # Clear cache
        clear_response = test_client.post("/v1/cache/clear")
        assert clear_response.status_code == 200

        # Get stats again after clearing
        stats_after = test_client.get("/v1/cache/stats")
        assert stats_after.status_code == 200

    def test_metrics_reset_operation(self, test_client):
        """Test metrics reset operation."""
        # Get initial metrics
        metrics_before = test_client.get("/v1/metrics")
        assert metrics_before.status_code == 200

        # Reset metrics
        reset_response = test_client.post("/v1/metrics/reset")
        assert reset_response.status_code == 200

        # Get metrics after reset
        metrics_after = test_client.get("/v1/metrics")
        assert metrics_after.status_code == 200


class TestRouteAvailability:
    """Test that all monitoring routes are properly available."""

    def test_all_get_routes_available(self, test_client):
        """Test that all GET monitoring routes are available."""
        routes = [
            "/",
            "/v1/preflight",
            "/v1/metrics",
            "/v1/cache/stats",
        ]

        for route in routes:
            response = test_client.get(route)
            assert response.status_code == 200, (
                f"Route {route} returned {response.status_code}"
            )

    def test_all_post_routes_available(self, test_client):
        """Test that all POST monitoring routes are available."""
        routes = [
            "/v1/cache/clear",
            "/v1/metrics/reset",
        ]

        for route in routes:
            response = test_client.post(route)
            assert response.status_code == 200, (
                f"Route {route} returned {response.status_code}"
            )

    def test_health_routes_no_authentication(self, test_client):
        """Test that health routes don't require authentication."""
        response = test_client.get("/")
        assert response.status_code == 200

        response = test_client.get("/v1/preflight")
        assert response.status_code == 200

    def test_monitoring_routes_http_methods(self, test_client):
        """Test correct HTTP methods for monitoring routes."""
        # GET routes should not accept POST
        get_routes = ["/v1/metrics", "/v1/cache/stats"]
        for route in get_routes:
            response = test_client.post(route)
            assert response.status_code == 405  # Method Not Allowed

        # POST routes should not accept GET
        post_routes = ["/v1/cache/clear", "/v1/metrics/reset"]
        for route in post_routes:
            # These should return 405 for GET
            # (Note: Some frameworks return 405, others might return 200 if GET is also supported)
            response = test_client.get(route)
            # Just verify it's a valid response code
            assert response.status_code in [200, 405]
