import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import ValidationError
import openai

from ...config import Settings
from ...logging import init_logging, info as log_info, LogRecord
from ...infrastructure.providers.openai_provider import OpenAIProvider
from ...domain.models import AnthropicErrorType
from ...application.response_cache import response_cache
from ...application.error_tracker import error_tracker
from ...application.cache.warmup import CacheWarmupManager, CacheWarmupConfig
from ...application.thread_pool import initialize_thread_pool, get_pool_stats
from .middleware import logging_middleware
from .errors import (
    log_and_return_error_response,
    get_anthropic_error_details_from_execution,
)

# Import tracing if available
try:
    from ...tracing import init_tracing

    tracing_available = True
except ImportError:
    tracing_available = False
    init_tracing = None
from .routes.messages import router as messages_router
from .routes.health import router as health_router
from .routes.monitoring import router as monitoring_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse


def create_app(settings: Settings) -> FastAPI:
    """Creates and configures the FastAPI application instance.

    Initializes logging, sets up middleware, and registers routes with comprehensive
    startup logging for operational visibility. Returns ready-to-use application.

    Args:
        settings: Configuration settings object

    Returns:
        Fully configured FastAPI application instance
    """
    init_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Initialize thread pool for CPU-bound operations
        logging.info("Initializing thread pool for CPU-bound operations")
        initialize_thread_pool(settings)
        pool_stats = get_pool_stats()
        logging.info(f"Thread pool initialized: {pool_stats}")

        # Check Cython optimization status
        try:
            from ..._cython import CYTHON_ENABLED

            if CYTHON_ENABLED:
                try:
                    # Try to import Cython modules to verify they're available
                    from ..._cython import type_checks  # noqa: F401

                    log_info(
                        LogRecord(
                            event="cython_status",
                            message="✓ Cython optimizations ACTIVE - using compiled C modules",
                            data={
                                "status": "enabled",
                                "modules": ["type_checks", "lru_ops", "cache_keys"],
                                "expected_improvement": "15-30% for CPU-bound operations",
                            },
                        )
                    )
                except ImportError:
                    log_info(
                        LogRecord(
                            event="cython_status",
                            message="Cython enabled but modules not found - using pure Python fallback",
                            data={
                                "status": "fallback",
                                "action_required": "run 'uv pip install -e .' to build",
                            },
                        )
                    )
            else:
                log_info(
                    LogRecord(
                        event="cython_status",
                        message="Cython optimizations DISABLED - using pure Python",
                        data={
                            "status": "disabled",
                            "reason": "CCPROXY_ENABLE_CYTHON=false",
                        },
                    )
                )
        except ImportError:
            log_info(
                LogRecord(
                    event="cython_status",
                    message="Cython not available - using pure Python",
                    data={"status": "unavailable"},
                )
            )

        # Initialize distributed tracing if enabled
        if tracing_available and settings.tracing_enabled:
            logging.info("Initializing distributed tracing")
            init_tracing(settings)
            logging.info(
                f"Distributed tracing initialized with {settings.tracing_exporter} exporter"
            )

        logging.info("Starting response cache cleanup task")
        await app.state.response_cache.start_cleanup_task()

        logging.info("Initializing error tracker")
        await app.state.error_tracker.initialize(settings)

        # Initialize cache warmup manager if enabled
        if settings.cache_warmup_enabled:
            logging.info("Initializing cache warmup manager")
            warmup_config = CacheWarmupConfig(
                enabled=settings.cache_warmup_enabled,
                warmup_file_path=settings.cache_warmup_file_path,
                max_warmup_items=settings.cache_warmup_max_items,
                warmup_on_startup=settings.cache_warmup_on_startup,
                preload_common_prompts=settings.cache_warmup_preload_common,
                auto_save_popular=settings.cache_warmup_auto_save_popular,
                popularity_threshold=settings.cache_warmup_popularity_threshold,
                save_interval_seconds=settings.cache_warmup_save_interval_seconds,
            )
            app.state.cache_warmup_manager = CacheWarmupManager(
                cache=app.state.response_cache,
                config=warmup_config,
            )
            await app.state.cache_warmup_manager.start()
            logging.info("Cache warmup manager started")

        try:
            yield
        finally:
            logging.info("Initiating application shutdown")
            try:
                try:
                    await app.state.response_cache.stop_cleanup_task()
                    logging.info("Response cache cleanup stopped")
                except Exception as e:
                    logging.error(f"Error stopping response cache: {str(e)}")

                try:
                    await app.state.error_tracker.shutdown()
                    logging.info("Error tracker shutdown")
                except Exception as e:
                    logging.error(f"Error shutting down error tracker: {str(e)}")

                # Stop cache warmup manager if it exists
                if hasattr(app.state, "cache_warmup_manager"):
                    try:
                        logging.info("Stopping cache warmup manager")
                        await app.state.cache_warmup_manager.stop()
                        logging.info("Cache warmup manager stopped")
                    except Exception as e:
                        logging.error(f"Error stopping cache warmup manager: {str(e)}")
            finally:
                provider = getattr(app.state, "provider", None)
                if provider and hasattr(provider, "close"):
                    logging.info("Closing OpenAI provider connection")
                    try:
                        await provider.close()
                    except Exception as e:
                        logging.error(f"Failed to close provider: {str(e)}")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url=None,
        redoc_url=None,
        description="Routes Anthropic API requests to an OpenAI-compatible API, selecting models dynamically.",
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
    )
    app.state.settings = settings
    try:
        app.state.provider = OpenAIProvider(settings)
        logging.info("OpenAI provider initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI provider: {str(e)}")
        raise
    app.state.response_cache = response_cache
    app.state.error_tracker = error_tracker

    # Core middleware
    app.middleware("http")(logging_middleware)

    # Guardrail middleware (order: rate-limit → security headers)
    from .guardrails import RateLimitMiddleware, SecurityHeadersMiddleware

    if settings.rate_limit_enabled:
        logging.info(
            f"Rate limiting enabled: {settings.rate_limit_per_minute}/minute with {settings.rate_limit_burst} burst"
        )
        app.add_middleware(
            RateLimitMiddleware,
            per_minute=settings.rate_limit_per_minute,
            burst=settings.rate_limit_burst,
        )

    if settings.enable_cors:
        logging.info(
            f"CORS enabled for origins: {settings.cors_allow_origins}, methods: {settings.cors_allow_methods}, headers: {settings.cors_allow_headers}"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=settings.cors_allow_methods,
            allow_headers=settings.cors_allow_headers,
            allow_credentials=False,
            max_age=600,
        )

    if settings.security_headers_enabled:
        logging.info(f"Security headers enabled (HSTS: {settings.enable_hsts})")
        app.add_middleware(SecurityHeadersMiddleware, enable_hsts=settings.enable_hsts)

    app.include_router(messages_router, tags=["API"])
    app.include_router(health_router, tags=["Health"])
    app.include_router(monitoring_router, tags=["Monitoring"])

    @app.exception_handler(openai.APIError)
    async def openai_api_error_handler(request: Request, exc: openai.APIError):
        err_type, err_msg, err_status, prov_details = (
            get_anthropic_error_details_from_execution(exc)
        )
        return await log_and_return_error_response(
            request, err_status, err_type, err_msg, prov_details, exc
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_error_handler(request: Request, exc: ValidationError):
        return await log_and_return_error_response(
            request,
            422,
            AnthropicErrorType.INVALID_REQUEST,
            f"Validation error: {exc.errors()}",
            caught_exception=exc,
        )

    @app.exception_handler(json.JSONDecodeError)
    async def json_decode_error_handler(request: Request, exc: json.JSONDecodeError):
        return await log_and_return_error_response(
            request,
            400,
            AnthropicErrorType.INVALID_REQUEST,
            "Invalid JSON format.",
            caught_exception=exc,
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return await log_and_return_error_response(
            request,
            500,
            AnthropicErrorType.API_ERROR,
            "An unexpected internal server error occurred.",
            caught_exception=exc,
        )

    return app
