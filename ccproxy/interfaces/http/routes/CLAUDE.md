# HTTP Routes - CLAUDE.md

**Scope**: HTTP route handlers and API endpoint controllers

## Files in this module:
- `__init__.py`: Route module exports and router registration
- `messages.py`: Main messages endpoint handling Anthropic-compatible requests
- `health.py`: Health check and status endpoints
- `monitoring.py`: Metrics, cache stats, and operational endpoints

## Guidelines:
- **Provider proxying**: messages.py should proxy to provider using converters and async tokenizer
- **Anthropic compatibility**: Return Anthropic-compatible responses with proper error mapping
- **Streaming support**: Support SSE streaming via text/event-stream content type
- **Input validation**: Validate inputs and use request_validator where applicable
- **Async tokenization**: Await tokenizer functions (count_tokens_for_anthropic_request, truncate_request)
- **Security**: No secrets in logs; always include request_id in LogRecord for traceability
- **Error handling**: Use structured error responses through the error handling system
- **Performance**: Optimize for high throughput and low latency
- **Rate limiting**: Implement proper rate limiting for API endpoints
