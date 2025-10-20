# HTTP Interface - CLAUDE.md

**Scope**: HTTP-specific implementations including FastAPI app, routing, streaming, and middleware

## Files in this layer:
- `app.py`: FastAPI application factory and dependency injection container
  - Integrates CacheWarmupManager during startup lifecycle
  - Handles graceful shutdown of cache warmup and async resources
- `routes/`: HTTP route handlers for different API endpoints
- `streaming.py`: Server-Sent Events (SSE) streaming implementation
- `errors.py`: HTTP error handling and response formatting
- `middleware.py`: Request/response middleware pipeline
- `guardrails.py`: Input validation and security protection including BodySizeLimitMiddleware and InjectionGuardMiddleware (SQLi/XSS/cmd/path traversal detection/blocking via regex patterns and recursive JSON checks)
- `http_status.py`: HTTP status code utilities and mappings
- `upstream_limits.py`: Rate limiting for upstream service calls

## Guidelines:
- **Anthropic compatibility**: Keep endpoints Anthropic-compatible; correctly map OpenAI finish reasons
- **SSE streaming**: Use StreamProcessor for SSE with race condition protection in subscriber management
- **Error logging**: Exception handlers must capture and log to error.jsonl via ccproxy.logging
- **Configuration**: Respect Settings for CORS, rate limits, security headers, and cache warmup
- **FastAPI patterns**: Use dependency injection and proper async patterns
- **Request lifecycle**: Proper middleware ordering and request/response handling
- **Startup/Shutdown**: Initialize CacheWarmupManager in lifespan context if enabled
- **Security**: Apply guardrails for input validation and protection against malicious requests; includes injection guards for SQLi/XSS in JSON bodies/headers (blocks on pattern match with logging)
- **Performance**: Optimize streaming and response handling for high throughput
- **Type safety**: Strict type checking enabled; use proper type annotations
- **Cython optimizations**:
  - streaming.py can leverage `ccproxy._cython.stream_state` for SSE event formatting (20-30% improvement) - integrated
  - guardrails.py can leverage `ccproxy._cython.string_ops` for regex pattern matching (40-50% improvement) - integrated
