from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse
from datetime import datetime, timezone

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root_health_check() -> JSONResponse:
    """Check basic API health and availability.

    Returns a JSON response confirming the service is operational.

    Returns:
        JSONResponse: A response with status 'ok' and current UTC timestamp.
    """
    return JSONResponse(
        {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@router.get("/v1/preflight", include_in_schema=False)
async def preflight_check() -> PlainTextResponse:
    """Perform a pre-flight health check for service readiness.

    This endpoint is used to verify that the service is ready to handle requests.
    It returns a plain text confirmation message.

    Returns:
        PlainTextResponse: A response with the message "[BashTool] Pre-flight check passed."
    """
    return PlainTextResponse("[BashTool] Pre-flight check passed.")
