# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project: CCProxy – OpenAI-compatible proxy for Anthropic Messages API

Common commands
- Install deps: uv pip install -r requirements.txt
- Run dev (uvicorn): python main.py
- Run via script (env checks): ./run-ccproxy.sh
- Docker build/run (compose): ./docker-compose-run.sh up -d
- Docker logs: ./docker-compose-run.sh logs -f
- Lint (ruff check): ./start-lint.sh --check
- Lint fix + format: ./start-lint.sh --all
- Typecheck: mypy .
- Tests (all): pytest -q
- Single test file: pytest -q test_optimized_client.py
- Single test by node: pytest -q test_optimized_client.py::test_name

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
- WEB_CONCURRENCY (for Gunicorn)
Scripts create .env.example and validate env where helpful.

Run options
- Local dev: python main.py (FastAPI with uvicorn; auto-reload per Settings.reload)
- Gunicorn (prod): gunicorn --config gunicorn.conf.py wsgi:app
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
- ccproxy/application/converters.py: Message format conversion between Anthropic and OpenAI
- ccproxy/application/converters_module/: Modular converter implementations with specialized processors
- ccproxy/application/tokenizer.py: Advanced async-aware token counting with TTL-based cache (300s expiry)
- ccproxy/application/model_selection.py: Model mapping (opus/sonnet→BIG, haiku→SMALL)
- ccproxy/application/request_validator.py: LRU cache (10,000 capacity) with cryptographic hashing
- ccproxy/application/response_cache.py: Response caching abstraction (delegates to cache implementations)
- ccproxy/application/cache/: Advanced caching with circuit breaker pattern, memory management, streaming de-duplication
- ccproxy/application/error_tracker.py: Comprehensive error tracking and monitoring system
- ccproxy/application/type_utils.py: Type utilities and helper functions

## Infrastructure Layer (ccproxy/infrastructure/)
- External service integrations and infrastructure concerns
- ccproxy/infrastructure/providers/: Provider implementations for external services
  - base.py: ChatProvider protocol definition
  - openai_provider.py: High-performance HTTP/2 client with connection pooling (500 connections, 120s keepalive)

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

## Cross-cutting Concerns
- ccproxy/config.py: Pydantic Settings with environment validation
- ccproxy/logging.py: Structured JSON logging with request tracing
- ccproxy/monitoring.py: Performance metrics and health monitoring
- ccproxy/constants.py: Global constants and configuration (includes reasoning effort model support)
- ccproxy/enums.py: Enumeration types used across layers

## Entry Points
- main.py: Development server (uvicorn with auto-reload)
- wsgi.py: Production WSGI application for Gunicorn
- App factory: ccproxy/interfaces/http/app.py:create_app(Settings) provides dependency injection

Development notes for Claude Code
- Always construct the FastAPI app through create_app(Settings); do not import globals directly
- Follow hexagonal architecture principles: domain models should not depend on external concerns
- Application layer orchestrates use cases; infrastructure layer handles external integrations
- When adding parameters, ensure OpenAI parity: warn or omit unsupported fields; map tool_choice carefully
- For non-stream requests, use application/cache layer to avoid duplicate upstream calls
- Preserve UTF‑8 throughout; never assume ASCII; rely on provider handlers converting decode errors to APIError
- Follow existing logging events (LogEvent) and avoid logging secrets; Settings controls log file path
- Use dependency injection through the app factory for testability and loose coupling
- Error tracking is centralized in application/error_tracker.py for comprehensive monitoring
- Reasoning support: Implement provider-specific reasoning configurations (OpenRouter vs standard) based on base_url detection

Testing
- Pytest is configured via pyproject.toml (pythonpath and testpaths); tests live in tests/ (test_*.py)
- For async tests, use pytest-asyncio; respx is available for httpx mocking

CI/CD and tooling
- Ruff and mypy configured in pyproject.toml
- Dockerfile includes production (Debian) and Alpine targets; docker-compose.yml wires healthcheck and volumes
- start-lint.sh provides lint workflow; docker-compose-run.sh wraps common compose actions

## Important Instruction Reminders
- Do what has been asked; nothing more, nothing less.
- NEVER create files unless they're absolutely necessary for achieving your goal.
- ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.