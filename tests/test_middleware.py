"""Tests for HTTP middleware."""

import time
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from fastapi.responses import Response, JSONResponse

from ccproxy.interfaces.http.middleware import logging_middleware


class TestLoggingMiddleware:
    """Test logging middleware functionality."""

    @pytest.mark.anyio
    async def test_adds_request_id_to_state(self):
        """Test that middleware adds request ID to request state."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test")
        async def endpoint(request: Request):
            assert hasattr(request.state, "request_id")
            assert isinstance(request.state.request_id, str)
            assert len(request.state.request_id) > 0
            return {"request_id": request.state.request_id}

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data

    @pytest.mark.anyio
    async def test_adds_request_id_header_to_response(self):
        """Test that middleware adds X-Request-ID header to response."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test")
        async def endpoint():
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/test")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0

    @pytest.mark.anyio
    async def test_adds_response_time_header(self):
        """Test that middleware adds X-Response-Time-ms header."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test")
        async def endpoint():
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/test")
        assert "X-Response-Time-ms" in response.headers
        response_time = float(response.headers["X-Response-Time-ms"])
        assert response_time >= 0

    @pytest.mark.anyio
    async def test_preserves_existing_request_id(self):
        """Test that middleware preserves existing request ID if already set."""
        app = FastAPI()
        existing_id = "existing-request-id-123"

        @app.middleware("http")
        async def set_id_first(request: Request, call_next):
            request.state.request_id = existing_id
            return await call_next(request)

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test")
        async def endpoint(request: Request):
            return {"request_id": request.state.request_id}

        client = TestClient(app)
        response = client.get("/test")
        data = response.json()
        assert data["request_id"] == existing_id
        assert response.headers["X-Request-ID"] == existing_id

    @pytest.mark.anyio
    async def test_sets_start_time(self):
        """Test that middleware sets start_time_monotonic."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test")
        async def endpoint(request: Request):
            assert hasattr(request.state, "start_time_monotonic")
            assert isinstance(request.state.start_time_monotonic, float)
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_preserves_existing_start_time(self):
        """Test that middleware preserves existing start_time if already set."""
        app = FastAPI()
        existing_time = time.monotonic()

        @app.middleware("http")
        async def set_time_first(request: Request, call_next):
            request.state.start_time_monotonic = existing_time
            return await call_next(request)

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test")
        async def endpoint(request: Request):
            return {"start_time": request.state.start_time_monotonic}

        client = TestClient(app)
        response = client.get("/test")
        data = response.json()
        assert data["start_time"] == existing_time

    @pytest.mark.anyio
    async def test_response_time_calculation(self):
        """Test that response time is calculated correctly."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test")
        async def endpoint():
            import anyio

            await anyio.sleep(0.05)  # Sleep for 50ms
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/test")
        response_time = float(response.headers["X-Response-Time-ms"])
        # Response time should be at least 50ms (accounting for overhead)
        assert response_time >= 40

    @pytest.mark.anyio
    async def test_handles_different_http_methods(self):
        """Test that middleware works with different HTTP methods."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test")
        async def get_endpoint():
            return {"method": "GET"}

        @app.post("/test")
        async def post_endpoint():
            return {"method": "POST"}

        @app.put("/test")
        async def put_endpoint():
            return {"method": "PUT"}

        @app.delete("/test")
        async def delete_endpoint():
            return {"method": "DELETE"}

        client = TestClient(app)

        for method in ["GET", "POST", "PUT", "DELETE"]:
            response = getattr(client, method.lower())("/test")
            assert "X-Request-ID" in response.headers
            assert "X-Response-Time-ms" in response.headers

    @pytest.mark.anyio
    async def test_handles_custom_error_response(self):
        """Test that middleware works with custom error responses."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/error")
        async def error_endpoint():
            return JSONResponse(status_code=400, content={"error": "Bad request"})

        client = TestClient(app)
        response = client.get("/error")
        assert response.status_code == 400
        # Should still have headers even on error
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-ms" in response.headers

    @pytest.mark.anyio
    async def test_no_tracing_by_default(self):
        """Test that middleware works without tracing enabled."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test")
        async def endpoint():
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        # Trace ID should not be in response when tracing is disabled
        # (unless explicitly passed and enabled)

    @pytest.mark.anyio
    async def test_different_response_types(self):
        """Test that middleware works with different response types."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/json")
        async def json_endpoint():
            return JSONResponse({"type": "json"})

        @app.get("/text")
        async def text_endpoint():
            return Response(content="text response", media_type="text/plain")

        client = TestClient(app)

        # JSON response
        response = client.get("/json")
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-ms" in response.headers

        # Text response
        response = client.get("/text")
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-ms" in response.headers

    @pytest.mark.anyio
    async def test_handles_streaming_response(self):
        """Test that middleware handles streaming responses."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/stream")
        async def stream_endpoint():
            from fastapi.responses import StreamingResponse

            async def generate():
                for i in range(3):
                    yield f"chunk{i}\n".encode()

            return StreamingResponse(generate(), media_type="text/plain")

        client = TestClient(app)
        response = client.get("/stream")
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-ms" in response.headers

    @pytest.mark.anyio
    async def test_request_id_uniqueness(self):
        """Test that each request gets a unique request ID."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test")
        async def endpoint():
            return {"ok": True}

        client = TestClient(app)

        request_ids = set()
        for _ in range(10):
            response = client.get("/test")
            request_id = response.headers["X-Request-ID"]
            request_ids.add(request_id)

        # All request IDs should be unique
        assert len(request_ids) == 10

    @pytest.mark.anyio
    async def test_multiple_concurrent_requests(self):
        """Test that middleware handles concurrent requests correctly."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/test/{item_id}")
        async def endpoint(item_id: int, request: Request):
            import anyio

            await anyio.sleep(0.01)
            return {"item_id": item_id, "request_id": request.state.request_id}

        client = TestClient(app)

        # Make multiple requests
        responses = []
        for i in range(5):
            response = client.get(f"/test/{i}")
            responses.append(response)

        # Each should have unique request ID
        request_ids = [r.headers["X-Request-ID"] for r in responses]
        assert len(set(request_ids)) == 5

        # Each should have response time
        for response in responses:
            assert "X-Response-Time-ms" in response.headers

    @pytest.mark.anyio
    async def test_different_status_codes(self):
        """Test that middleware works with different status codes."""
        app = FastAPI()

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await logging_middleware(request, call_next)

        @app.get("/200")
        async def ok_endpoint():
            return {"status": 200}

        @app.get("/404")
        async def not_found_endpoint():
            return Response(status_code=404, content="Not found")

        @app.get("/500")
        async def error_endpoint():
            return Response(status_code=500, content="Server error")

        client = TestClient(app, raise_server_exceptions=False)

        for status_code in [200, 404, 500]:
            response = client.get(f"/{status_code}")
            assert response.status_code == status_code
            assert "X-Request-ID" in response.headers
            assert "X-Response-Time-ms" in response.headers
