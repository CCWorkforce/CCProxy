"""Utilities enforcing upstream call duration/host safety."""
from __future__ import annotations

import anyio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Request
from ..http.errors import log_and_return_error_response
from ...domain.models import AnthropicErrorType


@asynccontextmanager
async def enforce_timeout(request: Request, seconds: int) -> AsyncGenerator[None, None]:
    """Enforce a timeout on upstream requests with graceful error handling.

    This async context manager sets a maximum duration for upstream API calls.
    If the timeout is exceeded, it returns a 504 Gateway Timeout error response
    to the client in a structured format matching Anthropic's error schema.

    Args:
        request: The incoming FastAPI request object for error response context
        seconds: Maximum allowed duration for the upstream request in seconds

    Yields:
        None: Enters the context block where the upstream request should be made

    Example:
        async with enforce_timeout(request, 30):
            # Make upstream API call here
            response = await upstream_client.create_chat_completion(...)
    """
    try:
        async with anyio.fail_after(seconds):
            yield
    except TimeoutError:
        yield await log_and_return_error_response(
            request,
            504,
            AnthropicErrorType.API_ERROR,
            "Upstream request timeout exceeded.",
        )
