import json
from fastapi import FastAPI, Request
from pydantic import ValidationError
import openai

from ...config import Settings
from ...logging import init_logging
from ...infrastructure.providers.openai_provider import OpenAIProvider
from ...domain.models import AnthropicErrorType
from .middleware import logging_middleware
from .errors import _log_and_return_error_response, _get_anthropic_error_details_from_exc
from .routes.messages import router as messages_router
from .routes.health import router as health_router


def create_app(settings: Settings) -> FastAPI:
    init_logging(settings)
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url=None,
        redoc_url=None,
        description="Routes Anthropic API requests to an OpenAI-compatible API, selecting models dynamically.",
    )
    app.state.settings = settings
    app.state.provider = OpenAIProvider(settings)

    app.middleware("http")(logging_middleware)

    app.include_router(messages_router, tags=["API"])
    app.include_router(health_router, tags=["Health"])

    @app.exception_handler(openai.APIError)
    async def openai_api_error_handler(request: Request, exc: openai.APIError):
        err_type, err_msg, err_status, prov_details = _get_anthropic_error_details_from_exc(exc)
        return await _log_and_return_error_response(request, err_status, err_type, err_msg, prov_details, exc)

    @app.exception_handler(ValidationError)
    async def pydantic_validation_error_handler(request: Request, exc: ValidationError):
        return await _log_and_return_error_response(request, 422, AnthropicErrorType.INVALID_REQUEST, f"Validation error: {exc.errors()}", caught_exception=exc)

    @app.exception_handler(json.JSONDecodeError)
    async def json_decode_error_handler(request: Request, exc: json.JSONDecodeError):
        return await _log_and_return_error_response(request, 400, AnthropicErrorType.INVALID_REQUEST, "Invalid JSON format.", caught_exception=exc)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return await _log_and_return_error_response(request, 500, AnthropicErrorType.API_ERROR, "An unexpected internal server error occurred.", caught_exception=exc)

    return app
