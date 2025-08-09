"""Monitoring and performance endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ....monitoring import performance_monitor

router = APIRouter()


@router.get("/v1/metrics")
async def get_metrics(request: Request) -> JSONResponse:
    """Get current performance metrics."""
    metrics = await performance_monitor.get_metrics()
    return JSONResponse(content=metrics)


@router.post("/v1/metrics/reset")
async def reset_metrics(request: Request) -> JSONResponse:
    """Reset performance metrics."""
    await performance_monitor.reset_metrics()
    return JSONResponse(content={"status": "metrics_reset"})