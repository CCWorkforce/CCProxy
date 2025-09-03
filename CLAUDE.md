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

Big-picture architecture
- Entry points
  - main.py launches uvicorn; wsgi.py exposes app for Gunicorn
  - App factory: ccproxy/interfaces/http/app.py:create_app(Settings) wires dependencies and exception handlers
- HTTP surface (FastAPI routers)
  - /v1/messages (POST): ccproxy/interfaces/http/routes/messages.py:create_message_proxy handles Anthropic-compatible Messages requests; supports streaming via text/event-stream
  - /v1/messages/count_tokens (POST): returns token estimate
  - /v1/metrics, /v1/cache/stats, /v1/cache/clear, /v1/metrics/reset: operational endpoints
  - / (GET): basic health
- Provider abstraction
  - ccproxy/infrastructure/providers/base.py defines ChatProvider protocol
  - ccproxy/infrastructure/providers/openai_provider.py implements OpenAIProvider using openai.AsyncOpenAI
    - Optimized httpx AsyncClient (HTTP/2, connection pooling, timeouts)
    - Converts UnicodeDecodeError/JSON issues into openai.APIError; ensures UTF‑8 handling
- Request conversion and streaming bridge
  - ccproxy/application/converters.py maps Anthropic Messages + tools to OpenAI Chat Completions schema and back; enforces developer/system role rules and UTF‑8 developer hint for supported models
  - ccproxy/interfaces/http/streaming.py converts OpenAI ChatCompletionChunk streams into Anthropic SSE events, tracking content, tool-use, and thinking/signature blocks, with accurate stop-reason mapping
- Tokenization and model selection
  - ccproxy/application/tokenizer.py caches tiktoken encoders; counts tokens across message/content/tool structures with fallbacks
  - ccproxy/application/model_selection.py selects target OpenAI model based on requested Anthropic model (opus/sonnet→BIG, haiku→SMALL)
- Caching and validation
  - ccproxy/application/response_cache.py: LRU+TTL cache with memory budget, validation guard (JSON/UTF‑8), pending-request de‑duplication, background cleanup; exposed via app.state.response_cache
  - ccproxy/application/request_validator.py: LRU cache of validated MessagesRequest instances to avoid repeated Pydantic validation cost
- Config and logging
  - ccproxy/config.py: Pydantic Settings; reasoning-effort/temperature/developer message capability sets; env aliasing and validation
  - ccproxy/logging.py: JSON log formatting, structured events, optional file logging; middleware stamps X-Request-ID and timing
  - ccproxy/monitoring.py: rolling latency stats (avg/p95/p99), error rate

Development notes for Claude Code
- Always construct the FastAPI app through create_app(Settings); do not import globals directly
- When adding parameters, ensure OpenAI parity: warn or omit unsupported fields; map tool_choice carefully
- For non-stream requests, integrate with response_cache to avoid duplicate upstream calls; validate responses before caching
- Preserve UTF‑8 throughout; never assume ASCII; rely on provider handlers converting decode errors to APIError
- Follow existing logging events (LogEvent) and avoid logging secrets; Settings controls log file path

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