"""Monitoring and performance endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ....monitoring import performance_monitor
from ....application.tokenizer import _token_count_hits, _token_count_misses
from ....application.converters import _tools_cache, _tool_choice_cache, _serialize_tool_result_content_for_openai_cached
from ....application.request_validator import request_validator

router = APIRouter()


@router.get("/v1/metrics")
async def get_metrics(request: Request) -> JSONResponse:
    """Get current performance metrics including cache statistics."""
    performance_metrics = await performance_monitor.get_metrics()
    response_cache_stats = request.app.state.response_cache.get_stats()
    request_validator_stats = request_validator.get_cache_stats()

    metrics = {
        "performance": performance_metrics,
        "response_cache": response_cache_stats,
        "request_validator_cache": request_validator_stats,
        "token_count_cache": {
            "hits": _token_count_hits,
            "misses": _token_count_misses,
            "hit_rate": _token_count_hits / max(1, (_token_count_hits + _token_count_misses)),
        },
        "converter_caches": {
            "tools": {
                "currsize": getattr(_tools_cache, "cache_info")( ).currsize,
                "maxsize": getattr(_tools_cache, "cache_info")( ).maxsize,
            },
            "tool_choice": {
                "currsize": getattr(_tool_choice_cache, "cache_info")( ).currsize,
                "maxsize": getattr(_tool_choice_cache, "cache_info")( ).maxsize,
            },
            "tool_result": {
                "currsize": getattr(_serialize_tool_result_content_for_openai_cached, "cache_info")( ).currsize,
                "maxsize": getattr(_serialize_tool_result_content_for_openai_cached, "cache_info")( ).maxsize,
            },
        },
    }
    return JSONResponse(content=metrics)


@router.get("/v1/cache/stats")
async def get_cache_stats(request: Request) -> JSONResponse:
    """Get detailed cache statistics."""
    return JSONResponse(content={
        "response_cache": request.app.state.response_cache.get_stats(),
        "request_validator": request_validator.get_cache_stats()
    })


@router.post("/v1/cache/clear")
async def clear_cache(request: Request) -> JSONResponse:
    """Clear all caches."""
    await request.app.state.response_cache.clear()
    return JSONResponse(content={"status": "caches_cleared"})


@router.post("/v1/metrics/reset")
async def reset_metrics(request: Request) -> JSONResponse:
    """Reset performance metrics."""
    await performance_monitor.reset_metrics()
    return JSONResponse(content={"status": "metrics_reset"})
