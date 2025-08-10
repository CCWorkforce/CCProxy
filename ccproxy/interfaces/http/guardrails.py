from __future__ import annotations

"""Security guardrail middleware for CCProxy."""

import time
import asyncio
from typing import Callable, Awaitable, Dict, List

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp
from fastapi import Request

from ..http.errors import log_and_return_error_response
from ...domain.models import AnthropicErrorType


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests whose body exceeds *max_bytes*."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:  # noqa: D401
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:  # noqa: D401
        # Check content-length header first for efficiency
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self._max_bytes:
            return await log_and_return_error_response(
                request,
                413,
                AnthropicErrorType.REQUEST_TOO_LARGE,
                f"Request body too large: {content_length} bytes (limit {self._max_bytes}).",
            )
        
        # For body size checking, we need to read the body but avoid breaking ASGI
        # Only check the actual body size if content-length wasn't sufficient
        if not content_length:
            body = await request.body()
            if len(body) > self._max_bytes:
                return await log_and_return_error_response(
                    request,
                    413,
                    AnthropicErrorType.REQUEST_TOO_LARGE,
                    f"Request body too large: {len(body)} bytes (limit {self._max_bytes}).",
                )

        return await call_next(request)


class _MemoryRateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, per_minute: int, burst: int):
        self.per_minute = per_minute
        self.burst = burst
        self._store: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        window_start = now - 60
        async with self._lock:
            timestamps = self._store.get(key, [])
            # prune
            timestamps = [ts for ts in timestamps if ts >= window_start]
            if len(timestamps) >= self.per_minute + self.burst:
                self._store[key] = timestamps  # update pruned
                return False
            timestamps.append(now)
            self._store[key] = timestamps
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Applies per-IP/API-key rate limiting using in-memory store."""

    def __init__(
        self,
        app: ASGIApp,
        per_minute: int,
        burst: int = 0,
        header_key_name: str = "Authorization",
    ) -> None:
        super().__init__(app)
        self._limiter = _MemoryRateLimiter(per_minute, burst)
        self._header = header_key_name

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:  # noqa: D401
        key = request.headers.get(self._header) or (request.client.host if request.client else "anonymous")
        allowed = await self._limiter.allow(key)
        if not allowed:
            return await log_and_return_error_response(
                request,
                429,
                AnthropicErrorType.RATE_LIMIT,
                "Rate limit exceeded. Try again later.",
            )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds common security HTTP response headers."""

    def __init__(self, app: ASGIApp, enable_hsts: bool = False):
        super().__init__(app)
        self.enable_hsts = enable_hsts

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:  # noqa: D401
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; sandbox",
        )
        if self.enable_hsts and request.url.scheme == "https":
            response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
        return response
