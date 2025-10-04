"""Security guardrail middleware for CCProxy."""

from __future__ import annotations

import time
import anyio
import re
import json
from typing import Dict, List, Pattern, Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp
from fastapi import Request

from ..http.errors import log_and_return_error_response
from ...domain.models import AnthropicErrorType
from ...logging import warning, LogRecord, LogEvent


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests whose body exceeds *max_bytes*."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        """Initializes body size limiting middleware.

        Args:
            app: Downstream ASGI application instance
            max_bytes: Maximum allowed size of request body in bytes
        """
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Processes incoming requests, validating body size before forwarding.

        Checks request body size against configured limit and returns 413 error
        if limit is exceeded. Otherwise delegates processing to downstream middleware.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in processing chain

        Returns:
            Response: Processed HTTP response
        """
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


class InjectionGuardMiddleware(BaseHTTPMiddleware):
    """WAF-like middleware to detect and block injection attacks in request payloads."""

    # Common SQL injection patterns
    SQL_PATTERNS = [
        r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER|EXEC|EXECUTE)\b.*\b(FROM|WHERE|TABLE|INTO)\b",
        r"\b(OR|AND)\b.*=.*",  # OR 1=1 style
        r"--.*$",  # SQL comments
        r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP)",  # Command chaining
        r"\bUNION\s+(ALL\s+)?SELECT\b",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # Script tags
        r"javascript:",  # Javascript protocol
        r"on\w+\s*=",  # Event handlers (onclick, onload, etc.)
        r"<iframe",  # Iframe injection
        r"<embed",  # Embed tags
        r"<object",  # Object tags
    ]

    # Command injection patterns
    CMD_PATTERNS = [
        r"\$\([^)]+\)",  # Command substitution $(cmd)
        r"`[^`]+`",  # Backtick command substitution
        r"\|\s*[a-zA-Z]+",  # Pipe to command
        r";\s*[a-zA-Z]+",  # Command chaining
        r"&&\s*[a-zA-Z]+",  # Command chaining with &&
        r"\|\|\s*[a-zA-Z]+",  # Command chaining with ||
    ]

    # Path traversal patterns
    PATH_PATTERNS = [
        r"\.\./\.\.",  # Multiple path traversals
        r"/etc/passwd",  # Common target files
        r"/etc/shadow",
        r"C:\\Windows\\",  # Windows paths
        r"\.\.\\\.\.\\",  # Windows traversal
    ]

    def __init__(self, app: ASGIApp, check_headers: bool = True):
        """Initialize injection guard middleware.

        Args:
            app: Downstream ASGI application instance
            check_headers: Whether to also check request headers for injections
        """
        super().__init__(app)
        self.check_headers = check_headers

        # Compile patterns for efficiency
        self.sql_regex: List[Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in self.SQL_PATTERNS
        ]
        self.xss_regex: List[Pattern[str]] = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.XSS_PATTERNS
        ]
        self.cmd_regex: List[Pattern[str]] = [re.compile(p) for p in self.CMD_PATTERNS]
        self.path_regex: List[Pattern[str]] = [
            re.compile(p) for p in self.PATH_PATTERNS
        ]

    def _check_for_injection(self, text: str) -> tuple[bool, str]:
        """Check text for injection patterns.

        Args:
            text: Text to check for injection patterns

        Returns:
            Tuple of (is_malicious, attack_type)
        """
        # Check SQL injection
        for pattern in self.sql_regex:
            if pattern.search(text):
                return True, "SQL Injection"

        # Check XSS
        for pattern in self.xss_regex:
            if pattern.search(text):
                return True, "Cross-Site Scripting (XSS)"

        # Check command injection
        for pattern in self.cmd_regex:
            if pattern.search(text):
                return True, "Command Injection"

        # Check path traversal
        for pattern in self.path_regex:
            if pattern.search(text):
                return True, "Path Traversal"

        return False, ""

    def _check_json_recursively(self, obj: Any) -> tuple[bool, str]:
        """Recursively check JSON object for injection patterns.

        Args:
            obj: JSON object to check

        Returns:
            Tuple of (is_malicious, attack_type)
        """
        if isinstance(obj, str):
            return self._check_for_injection(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                is_malicious, attack_type = self._check_json_recursively(value)
                if is_malicious:
                    return is_malicious, attack_type
        elif isinstance(obj, list):
            for item in obj:
                is_malicious, attack_type = self._check_json_recursively(item)
                if is_malicious:
                    return is_malicious, attack_type
        return False, ""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Check request for injection attacks before forwarding."""
        request_id = request.headers.get("X-Request-Id", "unknown")

        # Check headers if enabled
        if self.check_headers:
            suspicious_headers = ["User-Agent", "Referer", "X-Forwarded-For"]
            for header in suspicious_headers:
                header_value = request.headers.get(header, "")
                if header_value:
                    is_malicious, attack_type = self._check_for_injection(header_value)
                    if is_malicious:
                        warning(
                            LogRecord(
                                event=LogEvent.REQUEST_VALIDATION_ERROR.value,
                                message=f"Blocked {attack_type} attempt in header {header}",
                                request_id=request_id,
                                data={"header": header, "attack_type": attack_type},
                            )
                        )
                        return await log_and_return_error_response(
                            request,
                            400,
                            AnthropicErrorType.INVALID_REQUEST,
                            "Request blocked: Suspicious pattern detected in headers",
                        )

        # Check request body for JSON payloads
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body = await request.body()
                    if body:
                        # Parse JSON body
                        try:
                            json_body = json.loads(body)
                            is_malicious, attack_type = self._check_json_recursively(
                                json_body
                            )
                            if is_malicious:
                                warning(
                                    LogRecord(
                                        event=LogEvent.REQUEST_VALIDATION_ERROR.value,
                                        message=f"Blocked {attack_type} attempt in request body",
                                        request_id=request_id,
                                        data={"attack_type": attack_type},
                                    )
                                )
                                return await log_and_return_error_response(
                                    request,
                                    400,
                                    AnthropicErrorType.INVALID_REQUEST,
                                    "Request blocked: Suspicious pattern detected",
                                )
                        except json.JSONDecodeError:
                            # If it's not valid JSON, check as plain text
                            text = body.decode("utf-8", errors="ignore")
                            is_malicious, attack_type = self._check_for_injection(text)
                            if is_malicious:
                                warning(
                                    LogRecord(
                                        event=LogEvent.REQUEST_VALIDATION_ERROR.value,
                                        message=f"Blocked {attack_type} attempt in malformed JSON",
                                        request_id=request_id,
                                        data={"attack_type": attack_type},
                                    )
                                )
                                return await log_and_return_error_response(
                                    request,
                                    400,
                                    AnthropicErrorType.INVALID_REQUEST,
                                    "Request blocked: Suspicious pattern detected",
                                )
                except Exception as e:
                    # Log but don't block on unexpected errors
                    warning(
                        LogRecord(
                            event=LogEvent.REQUEST_VALIDATION_ERROR.value,
                            message=f"Error checking request body for injections: {e}",
                            request_id=request_id,
                        )
                    )

        return await call_next(request)


class _MemoryRateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, per_minute: int, burst: int):
        """Initialize rate limiter with sliding window configuration.

        Args:
            per_minute: Sustained requests allowed per 60-second window
            burst: Additional burst capacity beyond sustained rate
        """
        self.per_minute = per_minute
        self.burst = burst
        self._store: Dict[str, List[float]] = {}
        self._lock = anyio.Lock()

    async def allow(self, key: str) -> bool:
        """Check whether *key* is within its rate limit window."""
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
        """Initializes rate limiting middleware with specified parameters.

        Args:
            app: Downstream ASGI application instance
            per_minute: Sustained requests permitted per key
            burst: Extra burst capacity beyond sustained rate
            header_key_name: HTTP header containing API key used for rate limiting
        """
        super().__init__(app)
        self._limiter = _MemoryRateLimiter(per_minute, burst)
        self._header = header_key_name

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Apply rate limiting and forward the request if allowed."""
        key = request.headers.get(self._header) or (
            request.client.host if request.client else "anonymous"
        )
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
        """Initialize security headers middleware with configuration options.

        Args:
            app: Downstream ASGI application instance
            enable_hsts: Whether to add Strict-Transport-Security header on HTTPS responses
        """
        super().__init__(app)
        self.enable_hsts = enable_hsts

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Inject common security headers into HTTP responses."""
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; sandbox",
        )
        if self.enable_hsts and request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=63072000; includeSubDomains; preload",
            )
        return response
