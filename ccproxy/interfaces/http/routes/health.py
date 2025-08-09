from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime, timezone

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root_health_check() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
