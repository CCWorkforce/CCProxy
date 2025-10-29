# CLAUDE.md

Scope: Python package root for CCProxy.

## Subdirectories:
- `_cython/`: Cython-optimized performance modules (9 .pyx files providing 15-35% performance improvement)
- `application/`: Application layer - use cases and business logic orchestration
- `domain/`: Domain layer - core business logic and domain models
- `infrastructure/`: Infrastructure layer - external service integrations
- `interfaces/`: Interface layer - external interfaces and delivery mechanisms (HTTP/REST API)

Guidelines:

- Import via relative package paths; avoid global singletons
- Construct apps with ccproxy.interfaces.http.app.create_app(Settings)
- Keep modules small and dependency direction inward (application <- interfaces)
- Do not log secrets; use ccproxy.logging helpers
- Preserve UTF-8 when handling bytes/strings
- JSON logging automatically omits null values for cleaner output
- Use async converters (convert_messages_async, convert_response_async) for better performance
- Async operations use Asyncer library (asyncify for CPU-bound tasks) and anyio (Path for async file I/O, create_task_group for parallel execution) for non-blocking operations
- Prefer anyio patterns over asyncio.gather for better resource management and exception handling
- Parallel execution uses anyio.create_task_group for concurrent processing of multiple items (messages, tokens, etc.)
- Configure cache warmup via environment variables for startup preloading
- Cython optimizations enabled by default (CCPROXY_ENABLE_CYTHON=true) for 15-35% performance improvement
- When using Cython modules from _cython/, always provide pure Python fallback for compatibility
- All 9 Cython modules fully integrated: type_checks, lru_ops, cache_keys, json_ops, string_ops, serialization, stream_state, dict_ops, validation
- Run tests with uv: ./run-tests.sh or uv run pytest
- Strict type checking enabled; all modules must have proper type annotations
- Run linting with uv: ./start-lint.sh whenever finishing the code change
- Ensure the server works properly after finishing the code change, run ./run-ccproxy.sh and then check its output.
