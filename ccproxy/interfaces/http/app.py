import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import ValidationError
import openai

from ...config import Settings
from ...logging import init_logging
from ...infrastructure.providers.openai_provider import OpenAIProvider
from ...domain.models import AnthropicErrorType
from ...application.response_cache import response_cache
from .middleware import logging_middleware
from .errors import log_and_return_error_response, get_anthropic_error_details_from_execution
from .routes.messages import router as messages_router
from .routes.health import router as health_router
from .routes.monitoring import router as monitoring_router


def create_app(settings: Settings) -> FastAPI:
    init_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.response_cache.start_cleanup_task()
        try:
            yield
        finally:
            try:
                await app.state.response_cache.stop_cleanup_task()
            finally:
                provider = getattr(app.state, "provider", None)
                if provider and hasattr(provider, "close"):
                    try:
                        await provider.close()
                    except Exception:
                        pass

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url=None,
        redoc_url=None,
        description="Routes Anthropic API requests to an OpenAI-compatible API, selecting models dynamically.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.provider = OpenAIProvider(settings)
    app.state.response_cache = response_cache

    # Core middleware
    app.middleware("http")(logging_middleware)

    # Guardrail middleware (order: rate-limit → security headers)
    from .guardrails import RateLimitMiddleware, SecurityHeadersMiddleware

    settings = app.state.settings


    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            per_minute=settings.rate_limit_per_minute,
            burst=settings.rate_limit_burst,
        )

    if settings.enable_cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=settings.cors_allow_methods,
            allow_headers=settings.cors_allow_headers,
            allow_credentials=False,
            max_age=600,
        )

    if settings.security_headers_enabled:
        app.add_middleware(SecurityHeadersMiddleware, enable_hsts=settings.enable_hsts)

    app.include_router(messages_router, tags=["API"])
    app.include_router(health_router, tags=["Health"])
    app.include_router(monitoring_router, tags=["Monitoring"])


    @app.exception_handler(openai.APIError)
    async def openai_api_error_handler(request: Request, exc: openai.APIError):
        err_type, err_msg, err_status, prov_details = get_anthropic_error_details_from_execution(exc)
        return await log_and_return_error_response(request, err_status, err_type, err_msg, prov_details, exc)

    @app.exception_handler(ValidationError)
    async def pydantic_validation_error_handler(request: Request, exc: ValidationError):
        return await log_and_return_error_response(request, 422, AnthropicErrorType.INVALID_REQUEST, f"Validation error: {exc.errors()}", caught_exception=exc)

    @app.exception_handler(json.JSONDecodeError)
    async def json_decode_error_handler(request: Request, exc: json.JSONDecodeError):
        return await log_and_return_error_response(request, 400, AnthropicErrorType.INVALID_REQUEST, "Invalid JSON format.", caught_exception=exc)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return await log_and_return_error_response(request, 500, AnthropicErrorType.API_ERROR, "An unexpected internal server error occurred.", caught_exception=exc)

    return app
