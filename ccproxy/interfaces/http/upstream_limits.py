"""Utilities enforcing upstream call duration/host safety."""

from __future__ import annotations

import anyio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Request


class UpstreamTimeoutError(Exception):
    """Custom exception for upstream timeout with request context."""
    def __init__(self, request: Request, message: str = "Upstream request timeout exceeded."):
        self.request = request
        self.message = message
        super().__init__(message)


@asynccontextmanager
async def enforce_timeout(request: Request, seconds: int) -> AsyncGenerator[None, None]:
    """Enforce a timeout on upstream requests with graceful error handling.

    This async context manager sets a maximum duration for upstream API calls.
    If the timeout is exceeded, it raises an UpstreamTimeoutError that the caller
    should catch and handle by calling log_and_return_error_response.

    Args:
        request: The incoming FastAPI request object for error response context
        seconds: Maximum allowed duration for the upstream request in seconds

    Yields:
        None: Enters the context block where the upstream request should be made

    Raises:
        UpstreamTimeoutError: When the timeout is exceeded

    Example:
        try:
            async with enforce_timeout(request, 30):
                # Make upstream API call here
                response = await upstream_client.create_chat_completion(...)
        except UpstreamTimeoutError as e:
            return await log_and_return_error_response(
                e.request, 504, AnthropicErrorType.API_ERROR, e.message
            )
    """
    try:
        with anyio.fail_after(seconds):
            yield
    except TimeoutError:
        raise UpstreamTimeoutError(request)
