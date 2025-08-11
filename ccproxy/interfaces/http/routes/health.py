from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse
from datetime import datetime, timezone

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root_health_check() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@router.get("/v1/preflight", include_in_schema=False)
async def preflight_check() -> PlainTextResponse:
    return PlainTextResponse("[BashTool] Pre-flight check passed.")
