# Interface Layer - CLAUDE.md

**Scope**: External interfaces, delivery mechanisms, and API controllers

## Files in this layer:
- `http/`: HTTP/REST API interface implementation
  - `app.py`: FastAPI application factory with dependency injection
  - `routes/`: HTTP route handlers and API controllers
  - `streaming.py`: SSE streaming for real-time response delivery
  - `errors.py`: HTTP error handling and response formatting
  - `middleware.py`: Request/response middleware chain
  - `guardrails.py`: Input validation and security guards
  - `http_status.py`: HTTP status code utilities
  - `upstream_limits.py`: Upstream service rate limiting

## Guidelines:
- **App construction**: Always build app via create_app(Settings) for proper dependency injection
- **Structured logging**: Use LogEvent and request_id from middleware for traceability
- **SSE streaming**: Use interfaces/http/streaming.py for Server-Sent Events with Cython stream_state integration for optimized SSE event formatting (20-30% improvement)
- **Error handling**: All exception paths must go through errors.log_and_return_error_response
- **Anthropic compatibility**: Keep endpoints Anthropic-compatible; map OpenAI finish reasons correctly
- **Stream safety**: Use StreamProcessor for SSE with race condition protection in subscriber management
- **Settings respect**: Honor CORS, rate limits, and security headers from Settings
- **Input validation**: Apply guardrails for security and data validation
- **HTTP standards**: Follow REST principles and proper HTTP status codes
