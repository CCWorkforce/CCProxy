# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project: CCProxy – OpenAI-compatible proxy for Anthropic Messages API

Common commands
- Install deps: uv pip install -r requirements.txt (includes asyncer for async operations and aiofiles for async file I/O)
- Run dev (uvicorn): python main.py
- Run via script (env checks): ./run-ccproxy.sh
- Docker build/run (compose): ./docker-compose-run.sh up -d
- Docker logs: ./docker-compose-run.sh logs -f
- Lint (ruff check): ./start-lint.sh --check
- Lint fix + format: ./start-lint.sh --all or ./start-lint.sh --fix
- Typecheck: mypy . (strict mode enabled)
- Tests (all): ./run-tests.sh or uv run pytest -q
- Tests with coverage: ./run-tests.sh --coverage
- Single test file: uv run pytest -q test_optimized_client.py
- Single test by node: uv run pytest -q test_optimized_client.py::test_name

Environment configuration
Required (via .env or environment)
- OPENAI_API_KEY or OPENROUTER_API_KEY
- BIG_MODEL_NAME
- SMALL_MODEL_NAME
Optional
- OPENAI_BASE_URL (default https://api.openai.com/v1)
- HOST (default 127.0.0.1)
- PORT (default 11434)
- LOG_LEVEL (default INFO)
- LOG_FILE_PATH (default log.jsonl)
- ERROR_LOG_FILE_PATH (default error.jsonl)
- WEB_CONCURRENCY (for multi-worker Uvicorn deployments)
Thread Pool Configuration (all optional)
- THREAD_POOL_MAX_WORKERS (default None - auto-calculates based on CPU cores, max 40)
- THREAD_POOL_HIGH_CPU_THRESHOLD (default None - auto-calculates based on CPU count: 60% + 2.5% per core, max 90%)
- THREAD_POOL_AUTO_SCALE (default False - enable dynamic scaling based on CPU contention)
Cache Warmup (all optional)
- CACHE_WARMUP_ENABLED (default False)
- CACHE_WARMUP_FILE_PATH (default cache_warmup.json)
- CACHE_WARMUP_MAX_ITEMS (default 100)
- CACHE_WARMUP_ON_STARTUP (default True)
- CACHE_WARMUP_PRELOAD_COMMON (default True)
- CACHE_WARMUP_AUTO_SAVE_POPULAR (default True)
- CACHE_WARMUP_POPULARITY_THRESHOLD (default 3)
- CACHE_WARMUP_SAVE_INTERVAL_SECONDS (default 3600)
Cython Optimization (all optional)
- CCPROXY_ENABLE_CYTHON (default True - enable Cython-compiled modules for 15-35% performance improvement)
- CCPROXY_BUILD_CYTHON (default True - build Cython extensions during installation)
Scripts create .env.example and validate env where helpful.

Run options
- Local dev: python main.py (FastAPI with uvicorn; auto-reload per Settings.reload)
- Production: ./run-ccproxy.sh (Uvicorn with multi-worker support; workers = CPU × 2 + 1)
- Docker: docker build -t ccproxy:latest -f Dockerfile .; docker-compose up -d
Health/metrics
- Health: GET / (root) returns {status: ok}
- Metrics: GET /v1/metrics; cache stats: GET /v1/cache/stats; clear caches: POST /v1/cache/clear

Big-picture architecture (Hexagonal/Clean Architecture)

## Domain Layer (ccproxy/domain/)
- Domain models and core business logic
- ccproxy/domain/models.py: Core domain entities and data structures
- ccproxy/domain/exceptions.py: Domain-specific exceptions and error handling

## Application Layer (ccproxy/application/)
- Use cases and application services
- ccproxy/application/converters.py: Message format conversion between Anthropic and OpenAI (exports async converters)
- ccproxy/application/converters_module/: Modular converter implementations with specialized processors
  - async_converter.py: AsyncMessageConverter and AsyncResponseConverter for parallel processing
  - Uses Asyncer library for improved async operations (asyncify for CPU-bound operations, anyio.create_task_group for parallel execution)
  - Optimized for high-throughput with parallel message and tool call processing
- ccproxy/application/tokenizer.py: Advanced async-aware token counting with TTL-based cache (300s expiry); uses anyio.create_task_group for parallel token encoding with asyncified tiktoken operations; includes OpenAI request counting via count_tokens_for_openai_request for precise integration with tiktoken encoders.
- ccproxy/application/model_selection.py: Model mapping (opus/sonnet→BIG, haiku→SMALL)
- ccproxy/application/request_validator.py: LRU cache (10,000 capacity) with cryptographic hashing
- ccproxy/application/response_cache.py: Response caching abstraction (delegates to cache implementations)
- ccproxy/application/cache/: Advanced caching with circuit breaker pattern, memory management, streaming de-duplication
  - warmup.py: CacheWarmupManager for preloading popular requests and common prompts; uses anyio.Path for async file operations and parallel warmup item loading
- ccproxy/application/error_tracker.py: Comprehensive error tracking and monitoring system with async JSON serialization and parallel redaction processing using asyncer
- ccproxy/application/thread_pool.py: Intelligent thread pool management for CPU-bound operations
  - Auto-detects multi-worker deployment via WEB_CONCURRENCY and adjusts accordingly
  - Prevents resource exhaustion: reduces threads per worker in multi-worker mode
  - Target total threads = CPU_count × 5 (distributed across workers)
  - Single worker: up to 40 threads; Multi-worker: 4-20 threads per worker
- ccproxy/application/type_utils.py: Type utilities and helper functions (uses Cython optimizations for type checking)

## Infrastructure Layer (ccproxy/infrastructure/)
- External service integrations and infrastructure concerns
- ccproxy/infrastructure/providers/: Provider implementations for external services
  - base.py: ChatProvider protocol definition
  - openai_provider.py: High-performance HTTP/2 client with connection pooling (500 connections, 120s keepalive); includes circuit breaker (failure threshold=5, recovery=60s), comprehensive metrics (latency percentiles, health scoring), error tracking, adaptive timeouts, tiktoken for precise token estimation in rate limiting (via tokenizer.py), and request correlation IDs for resilience and monitoring
  - rate_limiter.py: Client-side adaptive rate limiter using sliding window (1-min tracking); supports RPM/TPM limits, auto-start, 429 backoff (80% reduction), success recovery (10% increase after 10 successes); uses asyncified list operations for non-blocking cleanup of request history; integrates with openai_provider for token estimation and release via precise count_tokens_for_openai_request for TPM accuracy.

## Interface Layer (ccproxy/interfaces/)
- External interfaces and delivery mechanisms
- ccproxy/interfaces/http/: HTTP/REST API interface
  - app.py: FastAPI application factory and dependency injection
  - routes/: HTTP route handlers and controllers
  - streaming.py: SSE streaming for real-time responses
  - errors.py: HTTP error handling and response formatting
  - middleware.py: Request/response middleware chain
  - guardrails.py: Input validation and security guards
  - http_status.py: HTTP status code utilities
  - upstream_limits.py: Upstream service rate limiting

## Cython Optimization Layer (ccproxy/_cython/)
- High-performance Cython-compiled modules for CPU-bound operations (15-35% performance improvement)
- ccproxy/_cython/type_checks.pyx: Optimized type checking and dispatch (30-50% improvement) - integrated
- ccproxy/_cython/lru_ops.pyx: LRU cache operations (20-40% improvement) - integrated
- ccproxy/_cython/cache_keys.pyx: Cache key generation (15-25% improvement) - integrated
- ccproxy/_cython/json_ops.pyx: JSON operations (10.7x faster for size estimation) - integrated
- ccproxy/_cython/string_ops.pyx: String and pattern matching (40-50% improvement) - integrated
- ccproxy/_cython/serialization.pyx: Content serialization (25-35% improvement) - integrated
- ccproxy/_cython/stream_state.pyx: SSE event formatting (20-30% improvement) - integrated
- ccproxy/_cython/dict_ops.pyx: Dictionary operations (7.83x faster for nested key counting) - integrated
- ccproxy/_cython/validation.pyx: Validation operations (30-40% improvement) - integrated
- See CYTHON_INTEGRATION.md for detailed documentation and benchmarks
- Automatic fallback to pure Python if Cython unavailable or disabled
- Control via CCPROXY_ENABLE_CYTHON environment variable (default: enabled)

## Cross-cutting Concerns
- ccproxy/config.py: Pydantic Settings with environment validation
- ccproxy/logging.py: Structured JSON logging with request tracing
- ccproxy/monitoring.py: Performance metrics and health monitoring
- ccproxy/constants.py: Global constants and configuration (includes reasoning effort model support)
- ccproxy/enums.py: Enumeration types used across layers

## Entry Points
- main.py: Development server (uvicorn with auto-reload)
- wsgi.py: Production ASGI application for Uvicorn
- App factory: ccproxy/interfaces/http/app.py:create_app(Settings) provides dependency injection

Development notes for Claude Code
- Always construct the FastAPI app through create_app(Settings); do not import globals directly
- Thread pool automatically adjusts for multi-worker deployment to prevent resource exhaustion
- Follow hexagonal architecture principles: domain models should not depend on external concerns
- Application layer orchestrates use cases; infrastructure layer handles external integrations
- When adding parameters, ensure OpenAI parity: warn or omit unsupported fields; map tool_choice carefully
- For non-stream requests, use application/cache layer to avoid duplicate upstream calls
- Use async converters (convert_messages_async, convert_response_async) for better performance
- Cache warmup runs on startup when enabled, preloading common prompts and popular requests
- Preserve UTF‑8 throughout; never assume ASCII; rely on provider handlers converting decode errors to APIError
- Follow existing logging events (LogEvent) and avoid logging secrets; Settings controls log file path
- Use dependency injection through the app factory for testability and loose coupling
- Error tracking is centralized in application/error_tracker.py for comprehensive monitoring
- Reasoning support: Implement provider-specific reasoning configurations (OpenRouter vs standard) based on base_url detection
- Cython optimizations: Enabled by default for 15-35% performance improvement; use CCPROXY_ENABLE_CYTHON=false to disable
- When integrating Cython modules, always provide pure Python fallback for compatibility
- Run benchmarks to verify Cython performance gains: pytest benchmarks/ --benchmark-only
- Run tests with uv: ./run-tests.sh or uv run pytest
- Always run linting after changes: ./start-lint.sh --check

Testing
- Pytest is configured via pyproject.toml (pythonpath and testpaths); tests live in tests/ (test_*.py)
- For async tests, use pytest-anyio (migrated from pytest-asyncio); respx is available for httpx mocking
- Test runner script: ./run-tests.sh (supports parallel execution, coverage, watch mode)
- Comprehensive test coverage: 120+ test cases across 27 test files covering error_tracker, converters, cache, routes, async components, rate_limiter, thread_pool, cache_warmup, guardrails, streaming, and more

CI/CD and tooling
- GitHub Actions workflows in .github/workflows/
  - ci.yml: Comprehensive CI pipeline (lint, test with/without Cython, benchmarks, Docker)
  - performance.yml: Performance regression detection on PRs
  - See .github/README.md for workflow documentation
- Ruff and mypy configured in pyproject.toml (strict type checking enabled)
- Mypy strict mode: disallow_untyped_defs=true, warn_return_any=true, strict_optional=true
- Dockerfile includes production (Debian) and Alpine targets; docker-compose.yml wires healthcheck and volumes
- start-lint.sh provides lint workflow; docker-compose-run.sh wraps common compose actions
- scripts/test-cython-build.sh: Local verification of Cython build and fallback behavior
- scripts/verify-cython-status.sh: Check Cython module availability and integration status

## Important Instruction Reminders
- Do what has been asked; nothing more, nothing less.
- NEVER create files unless they're absolutely necessary for achieving your goal.
- ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.