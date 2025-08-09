"""Monitoring and performance endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ....monitoring import performance_monitor
from ....application.response_cache import response_cache
from ....application.request_validator import request_validator

router = APIRouter()


@router.get("/v1/metrics")
async def get_metrics(request: Request) -> JSONResponse:
    """Get current performance metrics including cache statistics."""
    performance_metrics = await performance_monitor.get_metrics()
    response_cache_stats = response_cache.get_stats()
    request_validator_stats = request_validator.get_cache_stats()

    metrics = {
        "performance": performance_metrics,
        "response_cache": response_cache_stats,
        "request_validator_cache": request_validator_stats
    }
    return JSONResponse(content=metrics)


@router.get("/v1/cache/stats")
async def get_cache_stats(request: Request) -> JSONResponse:
    """Get detailed cache statistics."""
    return JSONResponse(content={
        "response_cache": response_cache.get_stats(),
        "request_validator": request_validator.get_cache_stats()
    })


@router.post("/v1/cache/clear")
async def clear_cache(request: Request) -> JSONResponse:
    """Clear all caches."""
    await response_cache.clear()
    return JSONResponse(content={"status": "caches_cleared"})


@router.post("/v1/metrics/reset")
async def reset_metrics(request: Request) -> JSONResponse:
    """Reset performance metrics."""
    await performance_monitor.reset_metrics()
    return JSONResponse(content={"status": "metrics_reset"})
