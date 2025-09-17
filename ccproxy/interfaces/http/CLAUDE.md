# HTTP Interface - CLAUDE.md

**Scope**: HTTP-specific implementations including FastAPI app, routing, streaming, and middleware

## Files in this layer:
- `app.py`: FastAPI application factory and dependency injection container
- `routes/`: HTTP route handlers for different API endpoints
- `streaming.py`: Server-Sent Events (SSE) streaming implementation
- `errors.py`: HTTP error handling and response formatting
- `middleware.py`: Request/response middleware pipeline
- `guardrails.py`: Input validation and security protection
- `http_status.py`: HTTP status code utilities and mappings
- `upstream_limits.py`: Rate limiting for upstream service calls

## Guidelines:
- **Anthropic compatibility**: Keep endpoints Anthropic-compatible; correctly map OpenAI finish reasons
- **SSE streaming**: Use StreamProcessor for SSE with race condition protection in subscriber management
- **Error logging**: Exception handlers must capture and log to error.jsonl via ccproxy.logging
- **Configuration**: Respect Settings for CORS, rate limits, and security headers
- **FastAPI patterns**: Use dependency injection and proper async patterns
- **Request lifecycle**: Proper middleware ordering and request/response handling
- **Security**: Apply guardrails for input validation and protection against malicious requests
- **Performance**: Optimize streaming and response handling for high throughput
