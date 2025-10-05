# CLAUDE.md

Scope: Python package root for CCProxy.

Guidelines:

- Import via relative package paths; avoid global singletons
- Construct apps with ccproxy.interfaces.http.app.create_app(Settings)
- Keep modules small and dependency direction inward (application <- interfaces)
- Do not log secrets; use ccproxy.logging helpers
- Preserve UTF-8 when handling bytes/strings
- JSON logging automatically omits null values for cleaner output
- Use async converters (convert_messages_async, convert_response_async) for better performance
- Async operations use Asyncer library (asyncify for CPU-bound tasks) and anyio (Path for async file I/O) for non-blocking operations
- Parallel execution uses anyio.create_task_group for concurrent processing of multiple items (messages, tokens, etc.)
- Configure cache warmup via environment variables for startup preloading
- Run tests with uv: ./run-tests.sh or uv run pytest
- Strict type checking enabled; all modules must have proper type annotations
- Run linting with uv: ./start-lint.sh whenever finishing the code change
- Ensure the server works properly after finishing the code change, run ./run-ccproxy.sh and then check its output.
