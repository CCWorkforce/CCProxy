"""Comprehensive tests for security guardrail middleware."""

import pytest
import time
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from starlette.responses import Response

from ccproxy.interfaces.http.guardrails import (
    BodySizeLimitMiddleware,
    InjectionGuardMiddleware,
    _MemoryRateLimiter,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = MagicMock()
    request.headers = {}
    request.method = "POST"
    request.client = MagicMock()
    request.client.host = "192.168.1.1"
    request.url = MagicMock()
    request.url.scheme = "https"
    return request


@pytest.fixture
def mock_call_next():
    """Create a mock call_next function."""
    async def _call_next(request):
        response = Response(content="OK", status_code=200)
        return response

    return AsyncMock(side_effect=_call_next)


@pytest.fixture
def mock_log_event():
    """Mock LogEvent to avoid AttributeError on REQUEST_VALIDATION_ERROR."""
    with patch("ccproxy.interfaces.http.guardrails.LogEvent") as mock_event:
        mock_event.REQUEST_VALIDATION_ERROR = PropertyMock(return_value=MagicMock(value="validation_error"))
        yield mock_event


class TestBodySizeLimitMiddleware:
    """Test BodySizeLimitMiddleware."""

    @pytest.mark.anyio
    async def test_request_exceeds_limit_via_content_length(self, mock_request, mock_call_next):
        """Test request blocked when content-length exceeds limit."""
        mock_request.headers = {"content-length": "11000"}

        middleware = BodySizeLimitMiddleware(app=MagicMock(), max_bytes=10000)

        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            mock_error.return_value = Response(status_code=413)
            response = await middleware.dispatch(mock_request, mock_call_next)

            assert response.status_code == 413
            mock_error.assert_called_once()
            mock_call_next.assert_not_called()

    @pytest.mark.anyio
    async def test_request_within_limit(self, mock_request, mock_call_next):
        """Test request allowed when within size limit."""
        mock_request.headers = {"content-length": "5000"}

        middleware = BodySizeLimitMiddleware(app=MagicMock(), max_bytes=10000)
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200
        mock_call_next.assert_called_once()

    @pytest.mark.anyio
    async def test_request_exceeds_limit_no_content_length(self, mock_request, mock_call_next):
        """Test request blocked when actual body exceeds limit (no content-length header)."""
        large_body = b"x" * 11000
        mock_request.headers = {}
        mock_request.body = AsyncMock(return_value=large_body)

        middleware = BodySizeLimitMiddleware(app=MagicMock(), max_bytes=10000)

        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            mock_error.return_value = Response(status_code=413)
            response = await middleware.dispatch(mock_request, mock_call_next)

            assert response.status_code == 413
            mock_error.assert_called_once()

    @pytest.mark.anyio
    async def test_request_within_limit_no_content_length(self, mock_request, mock_call_next):
        """Test request allowed when body within limit (no content-length header)."""
        small_body = b"x" * 5000
        mock_request.headers = {}
        mock_request.body = AsyncMock(return_value=small_body)

        middleware = BodySizeLimitMiddleware(app=MagicMock(), max_bytes=10000)
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200
        mock_call_next.assert_called_once()

    @pytest.mark.anyio
    async def test_request_exactly_at_limit(self, mock_request, mock_call_next):
        """Test request at exactly the limit is allowed."""
        mock_request.headers = {"content-length": "10000"}

        middleware = BodySizeLimitMiddleware(app=MagicMock(), max_bytes=10000)
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200


class TestInjectionGuardMiddleware:
    """Test InjectionGuardMiddleware."""

    @pytest.mark.anyio
    async def test_sql_injection_in_body(self, mock_request, mock_call_next, mock_log_event):
        """Test SQL injection detected in request body."""
        malicious_json = {"query": "SELECT * FROM users WHERE id=1"}
        mock_request.headers = {"content-type": "application/json", "X-Request-Id": "test-123"}
        mock_request.body = AsyncMock(return_value=json.dumps(malicious_json).encode())

        middleware = InjectionGuardMiddleware(app=MagicMock())

        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            with patch("ccproxy.interfaces.http.guardrails.warning"):
                mock_error.return_value = Response(status_code=400)
                response = await middleware.dispatch(mock_request, mock_call_next)

                assert response.status_code == 400
                mock_call_next.assert_not_called()

    @pytest.mark.anyio
    async def test_xss_injection_in_body(self, mock_request, mock_call_next, mock_log_event):
        """Test XSS detected in request body."""
        malicious_json = {"content": "<script>alert('XSS')</script>"}
        mock_request.headers = {"content-type": "application/json", "X-Request-Id": "test-123"}
        mock_request.body = AsyncMock(return_value=json.dumps(malicious_json).encode())

        middleware = InjectionGuardMiddleware(app=MagicMock())

        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            with patch("ccproxy.interfaces.http.guardrails.warning"):
                mock_error.return_value = Response(status_code=400)
                response = await middleware.dispatch(mock_request, mock_call_next)

                assert response.status_code == 400

    @pytest.mark.anyio
    async def test_command_injection_in_body(self, mock_request, mock_call_next, mock_log_event):
        """Test command injection detected in request body."""
        malicious_json = {"cmd": "$(rm -rf /)"}
        mock_request.headers = {"content-type": "application/json", "X-Request-Id": "test-123"}
        mock_request.body = AsyncMock(return_value=json.dumps(malicious_json).encode())

        middleware = InjectionGuardMiddleware(app=MagicMock())

        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            with patch("ccproxy.interfaces.http.guardrails.warning"):
                mock_error.return_value = Response(status_code=400)
                response = await middleware.dispatch(mock_request, mock_call_next)

                assert response.status_code == 400

    @pytest.mark.anyio
    async def test_path_traversal_in_body(self, mock_request, mock_call_next, mock_log_event):
        """Test path traversal detected in request body."""
        malicious_json = {"file": "../../etc/passwd"}
        mock_request.headers = {"content-type": "application/json", "X-Request-Id": "test-123"}
        mock_request.body = AsyncMock(return_value=json.dumps(malicious_json).encode())

        middleware = InjectionGuardMiddleware(app=MagicMock())

        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            with patch("ccproxy.interfaces.http.guardrails.warning"):
                mock_error.return_value = Response(status_code=400)
                response = await middleware.dispatch(mock_request, mock_call_next)

                assert response.status_code == 400

    @pytest.mark.anyio
    async def test_injection_in_header(self, mock_request, mock_call_next, mock_log_event):
        """Test injection detected in request header."""
        mock_request.headers = {
            "User-Agent": "<script>alert('XSS')</script>",
            "X-Request-Id": "test-123"
        }
        mock_request.method = "GET"

        middleware = InjectionGuardMiddleware(app=MagicMock(), check_headers=True)

        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            with patch("ccproxy.interfaces.http.guardrails.warning"):
                mock_error.return_value = Response(status_code=400)
                response = await middleware.dispatch(mock_request, mock_call_next)

                assert response.status_code == 400

    @pytest.mark.anyio
    async def test_clean_request_allowed(self, mock_request, mock_call_next):
        """Test clean request is allowed through."""
        clean_json = {"message": "Hello, world!"}
        mock_request.headers = {"content-type": "application/json", "X-Request-Id": "test-123"}
        mock_request.body = AsyncMock(return_value=json.dumps(clean_json).encode())

        middleware = InjectionGuardMiddleware(app=MagicMock())
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200
        mock_call_next.assert_called_once()

    @pytest.mark.anyio
    async def test_nested_json_injection(self, mock_request, mock_call_next, mock_log_event):
        """Test injection detected in nested JSON structures."""
        malicious_json = {
            "data": {
                "nested": {
                    "deep": "SELECT * FROM users"
                }
            }
        }
        mock_request.headers = {"content-type": "application/json", "X-Request-Id": "test-123"}
        mock_request.body = AsyncMock(return_value=json.dumps(malicious_json).encode())

        middleware = InjectionGuardMiddleware(app=MagicMock())

        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            with patch("ccproxy.interfaces.http.guardrails.warning"):
                mock_error.return_value = Response(status_code=400)
                response = await middleware.dispatch(mock_request, mock_call_next)

                assert response.status_code == 400

    @pytest.mark.anyio
    async def test_injection_in_array(self, mock_request, mock_call_next, mock_log_event):
        """Test injection detected in array elements."""
        malicious_json = {
            "items": ["safe", "javascript:alert('XSS')"]
        }
        mock_request.headers = {"content-type": "application/json", "X-Request-Id": "test-123"}
        mock_request.body = AsyncMock(return_value=json.dumps(malicious_json).encode())

        middleware = InjectionGuardMiddleware(app=MagicMock())

        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            with patch("ccproxy.interfaces.http.guardrails.warning"):
                mock_error.return_value = Response(status_code=400)
                response = await middleware.dispatch(mock_request, mock_call_next)

                assert response.status_code == 400

    @pytest.mark.anyio
    async def test_malformed_json_fallback(self, mock_request, mock_call_next, mock_log_event):
        """Test malformed JSON is checked as plain text."""
        malicious_text = "SELECT * FROM users"
        mock_request.headers = {"content-type": "application/json", "X-Request-Id": "test-123"}
        mock_request.body = AsyncMock(return_value=malicious_text.encode())

        middleware = InjectionGuardMiddleware(app=MagicMock())

        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            with patch("ccproxy.interfaces.http.guardrails.warning"):
                mock_error.return_value = Response(status_code=400)
                response = await middleware.dispatch(mock_request, mock_call_next)

                assert response.status_code == 400

    @pytest.mark.anyio
    async def test_check_headers_disabled(self, mock_request, mock_call_next):
        """Test header checking can be disabled."""
        mock_request.headers = {
            "User-Agent": "<script>alert('XSS')</script>",
            "X-Request-Id": "test-123"
        }
        mock_request.method = "GET"

        middleware = InjectionGuardMiddleware(app=MagicMock(), check_headers=False)
        response = await middleware.dispatch(mock_request, mock_call_next)

        # Should pass through since header checking is disabled
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_get_request_no_body_check(self, mock_request, mock_call_next):
        """Test GET requests skip body checking."""
        mock_request.method = "GET"
        mock_request.headers = {"X-Request-Id": "test-123"}

        middleware = InjectionGuardMiddleware(app=MagicMock(), check_headers=False)
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200


class TestMemoryRateLimiter:
    """Test _MemoryRateLimiter."""

    @pytest.mark.anyio
    async def test_allow_new_key(self):
        """Test new key is allowed."""
        limiter = _MemoryRateLimiter(per_minute=10, burst=0)
        allowed = await limiter.allow("key1")
        assert allowed is True

    @pytest.mark.anyio
    async def test_rate_limit_exceeded(self):
        """Test rate limit exceeded scenario."""
        limiter = _MemoryRateLimiter(per_minute=2, burst=0)

        # First two requests should pass
        assert await limiter.allow("key1") is True
        assert await limiter.allow("key1") is True

        # Third request should be blocked
        assert await limiter.allow("key1") is False

    @pytest.mark.anyio
    async def test_burst_capacity(self):
        """Test burst capacity allows extra requests."""
        limiter = _MemoryRateLimiter(per_minute=2, burst=3)

        # Should allow per_minute + burst requests
        for _ in range(5):
            assert await limiter.allow("key1") is True

        # Next request should be blocked
        assert await limiter.allow("key1") is False

    @pytest.mark.anyio
    async def test_sliding_window_pruning(self):
        """Test old timestamps are pruned from sliding window."""
        limiter = _MemoryRateLimiter(per_minute=2, burst=0)

        # Add requests
        await limiter.allow("key1")
        await limiter.allow("key1")

        # Should be rate limited now
        assert await limiter.allow("key1") is False

        # Mock time advancement by directly manipulating store
        # In real scenario, old entries would be pruned after 60s
        with patch('time.monotonic', return_value=time.monotonic() + 61):
            # After 60s, old entries should be pruned and new request allowed
            assert await limiter.allow("key1") is True

    @pytest.mark.anyio
    async def test_multiple_keys_independent(self):
        """Test different keys have independent rate limits."""
        limiter = _MemoryRateLimiter(per_minute=1, burst=0)

        assert await limiter.allow("key1") is True
        assert await limiter.allow("key2") is True

        # Both keys should be at limit
        assert await limiter.allow("key1") is False
        assert await limiter.allow("key2") is False


class TestRateLimitMiddleware:
    """Test RateLimitMiddleware."""

    @pytest.mark.anyio
    async def test_request_allowed_within_limit(self, mock_request, mock_call_next):
        """Test request allowed when within rate limit."""
        mock_request.headers = {"Authorization": "Bearer token123"}

        middleware = RateLimitMiddleware(
            app=MagicMock(),
            per_minute=10,
            burst=0
        )
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200
        mock_call_next.assert_called_once()

    @pytest.mark.anyio
    async def test_request_blocked_over_limit(self, mock_request, mock_call_next):
        """Test request blocked when rate limit exceeded."""
        mock_request.headers = {"Authorization": "Bearer token123"}

        middleware = RateLimitMiddleware(
            app=MagicMock(),
            per_minute=1,
            burst=0
        )

        # First request should pass
        response1 = await middleware.dispatch(mock_request, mock_call_next)
        assert response1.status_code == 200

        # Second request should be rate limited
        with patch("ccproxy.interfaces.http.guardrails.log_and_return_error_response") as mock_error:
            mock_error.return_value = Response(status_code=429)
            response2 = await middleware.dispatch(mock_request, mock_call_next)
            assert response2.status_code == 429

    @pytest.mark.anyio
    async def test_uses_client_ip_when_no_auth_header(self, mock_request, mock_call_next):
        """Test uses client IP when Authorization header is missing."""
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.100"

        middleware = RateLimitMiddleware(
            app=MagicMock(),
            per_minute=10,
            burst=0
        )
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_anonymous_when_no_client(self, mock_request, mock_call_next):
        """Test uses 'anonymous' key when no client info."""
        mock_request.headers = {}
        mock_request.client = None

        middleware = RateLimitMiddleware(
            app=MagicMock(),
            per_minute=10,
            burst=0
        )
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200


class TestSecurityHeadersMiddleware:
    """Test SecurityHeadersMiddleware."""

    @pytest.mark.anyio
    async def test_security_headers_added(self, mock_request, mock_call_next):
        """Test all security headers are added to response."""
        middleware = SecurityHeadersMiddleware(app=MagicMock(), enable_hsts=False)
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("Referrer-Policy") == "same-origin"
        assert "Content-Security-Policy" in response.headers

    @pytest.mark.anyio
    async def test_hsts_added_on_https_when_enabled(self, mock_request, mock_call_next):
        """Test HSTS header added on HTTPS when enabled."""
        mock_request.url.scheme = "https"

        middleware = SecurityHeadersMiddleware(app=MagicMock(), enable_hsts=True)
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert "Strict-Transport-Security" in response.headers
        assert "max-age=63072000" in response.headers["Strict-Transport-Security"]

    @pytest.mark.anyio
    async def test_hsts_not_added_on_http(self, mock_request, mock_call_next):
        """Test HSTS header not added on HTTP."""
        mock_request.url.scheme = "http"

        middleware = SecurityHeadersMiddleware(app=MagicMock(), enable_hsts=True)
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert "Strict-Transport-Security" not in response.headers

    @pytest.mark.anyio
    async def test_hsts_not_added_when_disabled(self, mock_request, mock_call_next):
        """Test HSTS header not added when disabled."""
        mock_request.url.scheme = "https"

        middleware = SecurityHeadersMiddleware(app=MagicMock(), enable_hsts=False)
        response = await middleware.dispatch(mock_request, mock_call_next)

        assert "Strict-Transport-Security" not in response.headers

    @pytest.mark.anyio
    async def test_csp_header_strict(self, mock_request, mock_call_next):
        """Test Content-Security-Policy is strict."""
        middleware = SecurityHeadersMiddleware(app=MagicMock())
        response = await middleware.dispatch(mock_request, mock_call_next)

        csp = response.headers.get("Content-Security-Policy")
        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "sandbox" in csp


class TestInjectionPatterns:
    """Test specific injection pattern detection."""

    def test_sql_union_select(self):
        """Test UNION SELECT pattern detection."""
        middleware = InjectionGuardMiddleware(app=MagicMock())
        is_malicious, attack_type = middleware._check_for_injection("UNION SELECT username FROM users")
        assert is_malicious is True
        assert "SQL" in attack_type

    def test_sql_or_equals(self):
        """Test OR 1=1 pattern detection."""
        middleware = InjectionGuardMiddleware(app=MagicMock())
        is_malicious, attack_type = middleware._check_for_injection("admin' OR 1=1--")
        assert is_malicious is True

    def test_xss_iframe(self):
        """Test iframe injection pattern."""
        middleware = InjectionGuardMiddleware(app=MagicMock())
        is_malicious, attack_type = middleware._check_for_injection("<iframe src='evil.com'>")
        assert is_malicious is True
        assert "XSS" in attack_type

    def test_xss_event_handler(self):
        """Test event handler injection."""
        middleware = InjectionGuardMiddleware(app=MagicMock())
        is_malicious, attack_type = middleware._check_for_injection("<img onload='alert(1)'>")
        assert is_malicious is True

    def test_cmd_backtick(self):
        """Test backtick command substitution."""
        middleware = InjectionGuardMiddleware(app=MagicMock())
        is_malicious, attack_type = middleware._check_for_injection("`whoami`")
        assert is_malicious is True
        assert "Command" in attack_type

    def test_cmd_pipe(self):
        """Test pipe to command."""
        middleware = InjectionGuardMiddleware(app=MagicMock())
        is_malicious, attack_type = middleware._check_for_injection("| cat /etc/passwd")
        assert is_malicious is True

    def test_path_windows(self):
        """Test Windows path traversal."""
        middleware = InjectionGuardMiddleware(app=MagicMock())
        is_malicious, attack_type = middleware._check_for_injection("C:\\Windows\\System32")
        assert is_malicious is True
        assert "Path" in attack_type

    def test_clean_text(self):
        """Test clean text is not flagged."""
        middleware = InjectionGuardMiddleware(app=MagicMock())
        is_malicious, attack_type = middleware._check_for_injection("Hello, world! This is safe text.")
        assert is_malicious is False
