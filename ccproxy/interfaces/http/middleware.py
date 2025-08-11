"""Common FastAPI middleware utilities for CCProxy HTTP interface."""

import time
import uuid
from typing import Awaitable, Callable

from fastapi import Request
from fastapi.responses import Response


async def logging_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach request ID and timing headers.

    This middleware generates a per-request UUID (``X-Request-ID``) and measures
    wall-clock latency from the start of ASGI processing to the completed
    response, recording it as ``X-Response-Time-ms``.  The identifiers are also
    stored on ``request.state`` for downstream handlers and log correlation.
    """
    if not hasattr(request.state, "request_id"):
        request.state.request_id = str(uuid.uuid4())
    if not hasattr(request.state, "start_time_monotonic"):
        request.state.start_time_monotonic = time.monotonic()

    response = await call_next(request)

    response.headers["X-Request-ID"] = request.state.request_id
    duration_ms = (time.monotonic() - request.state.start_time_monotonic) * 1000
    response.headers["X-Response-Time-ms"] = str(duration_ms)

    return response
