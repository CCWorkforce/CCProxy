"""Tests for upstream limits utilities."""

import pytest
import anyio
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ccproxy.interfaces.http.upstream_limits import (
    UpstreamTimeoutError,
    enforce_timeout,
)


class TestUpstreamTimeoutError:
    """Test UpstreamTimeoutError exception."""

    def test_error_with_default_message(self):
        """Test error creation with default message."""
        app = FastAPI()
        client = TestClient(app)
        
        @app.get("/test")
        async def endpoint(request: Request):
            return {"ok": True}
        
        with client:
            response = client.get("/test")
            assert response.status_code == 200
            
        # Create error with a mock request
        request = Request({"type": "http", "method": "GET", "url": "http://test"})
        error = UpstreamTimeoutError(request)
        
        assert error.request == request
        assert error.message == "Upstream request timeout exceeded."
        assert str(error) == "Upstream request timeout exceeded."

    def test_error_with_custom_message(self):
        """Test error creation with custom message."""
        request = Request({"type": "http", "method": "POST", "url": "http://test"})
        custom_msg = "Custom timeout message"
        error = UpstreamTimeoutError(request, message=custom_msg)
        
        assert error.request == request
        assert error.message == custom_msg
        assert str(error) == custom_msg

    def test_error_is_exception(self):
        """Test that UpstreamTimeoutError is an Exception."""
        request = Request({"type": "http", "method": "GET", "url": "http://test"})
        error = UpstreamTimeoutError(request)
        
        assert isinstance(error, Exception)


class TestEnforceTimeout:
    """Test enforce_timeout context manager."""

    @pytest.mark.anyio
    async def test_successful_execution_within_timeout(self):
        """Test that operations completing within timeout succeed."""
        request = Request({"type": "http", "method": "GET", "url": "http://test"})
        result = []
        
        async with enforce_timeout(request, seconds=1):
            # Quick operation that should complete
            await anyio.sleep(0.01)
            result.append("completed")
        
        assert result == ["completed"]

    @pytest.mark.anyio
    async def test_timeout_exceeded_raises_error(self):
        """Test that exceeding timeout raises UpstreamTimeoutError."""
        request = Request({"type": "http", "method": "GET", "url": "http://test"})
        
        with pytest.raises(UpstreamTimeoutError) as exc_info:
            async with enforce_timeout(request, seconds=0.1):
                # Operation that takes longer than timeout
                await anyio.sleep(1.0)
        
        error = exc_info.value
        assert error.request == request
        assert error.message == "Upstream request timeout exceeded."

    @pytest.mark.anyio
    async def test_timeout_with_zero_seconds(self):
        """Test timeout with very short duration."""
        request = Request({"type": "http", "method": "GET", "url": "http://test"})
        
        with pytest.raises(UpstreamTimeoutError):
            async with enforce_timeout(request, seconds=0):
                # Any operation should timeout immediately
                await anyio.sleep(0.001)

    @pytest.mark.anyio
    async def test_multiple_operations_within_timeout(self):
        """Test multiple quick operations within timeout."""
        request = Request({"type": "http", "method": "GET", "url": "http://test"})
        results = []
        
        async with enforce_timeout(request, seconds=2):
            for i in range(5):
                await anyio.sleep(0.01)
                results.append(i)
        
        assert results == [0, 1, 2, 3, 4]

    @pytest.mark.anyio
    async def test_context_manager_cleanup_on_timeout(self):
        """Test that context manager cleans up properly on timeout."""
        request = Request({"type": "http", "method": "GET", "url": "http://test"})
        executed = []
        
        try:
            async with enforce_timeout(request, seconds=0.05):
                executed.append("started")
                await anyio.sleep(1.0)
                executed.append("finished")  # Should not reach here
        except UpstreamTimeoutError:
            executed.append("timeout")
        
        assert "started" in executed
        assert "timeout" in executed
        assert "finished" not in executed

    @pytest.mark.anyio
    async def test_timeout_preserves_request_context(self):
        """Test that timeout error preserves the original request."""
        request = Request({
            "type": "http",
            "method": "POST",
            "url": "http://example.com/api",
            "headers": [(b"content-type", b"application/json")],
        })
        
        with pytest.raises(UpstreamTimeoutError) as exc_info:
            async with enforce_timeout(request, seconds=0.05):
                await anyio.sleep(1.0)
        
        error = exc_info.value
        assert error.request == request
        assert error.request.method == "POST"

    @pytest.mark.anyio
    async def test_immediate_completion(self):
        """Test immediate completion without any async operations."""
        request = Request({"type": "http", "method": "GET", "url": "http://test"})
        executed = False
        
        async with enforce_timeout(request, seconds=1):
            executed = True
        
        assert executed is True

    @pytest.mark.anyio
    async def test_timeout_with_exception_inside_block(self):
        """Test that other exceptions are not caught by timeout handler."""
        request = Request({"type": "http", "method": "GET", "url": "http://test"})
        
        with pytest.raises(ValueError, match="test error"):
            async with enforce_timeout(request, seconds=1):
                raise ValueError("test error")

    @pytest.mark.anyio
    async def test_nested_timeouts(self):
        """Test nested timeout contexts."""
        request1 = Request({"type": "http", "method": "GET", "url": "http://test1"})
        request2 = Request({"type": "http", "method": "GET", "url": "http://test2"})
        
        # Outer timeout is longer
        async with enforce_timeout(request1, seconds=2):
            # Inner timeout is shorter and should trigger first
            with pytest.raises(UpstreamTimeoutError) as exc_info:
                async with enforce_timeout(request2, seconds=0.05):
                    await anyio.sleep(1.0)
            
            # Verify inner timeout triggered with correct request
            assert exc_info.value.request == request2

    @pytest.mark.anyio
    async def test_timeout_with_large_value(self):
        """Test timeout with large timeout value completes successfully."""
        request = Request({"type": "http", "method": "GET", "url": "http://test"})
        result = []
        
        async with enforce_timeout(request, seconds=3600):  # 1 hour
            await anyio.sleep(0.01)
            result.append("done")
        
        assert result == ["done"]
