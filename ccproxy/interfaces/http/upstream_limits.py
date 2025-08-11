"""Utilities enforcing upstream call duration/host safety."""
from __future__ import annotations

import anyio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Request
from ..http.errors import log_and_return_error_response
from ...domain.models import AnthropicErrorType


@asynccontextmanager
async def enforce_timeout(request: Request, seconds: int) -> AsyncGenerator[None, None]:  # noqa: D401
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
