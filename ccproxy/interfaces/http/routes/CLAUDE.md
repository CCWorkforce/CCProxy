# HTTP Routes - CLAUDE.md

**Scope**: HTTP route handlers and API endpoint controllers

## Files in this module:
- `__init__.py`: Route module exports and router registration
- `messages.py`: Main messages endpoint handling Anthropic-compatible requests
- `health.py`: Health check and status endpoints
- `monitoring.py`: Metrics, cache stats, and operational endpoints

## Guidelines:
- **Provider proxying**: messages.py should proxy to provider using async converters and async tokenizer
- **Async converters**: Use `convert_messages_async()` and `convert_response_async()` for better performance
- **Anthropic compatibility**: Return Anthropic-compatible responses with proper error mapping
- **Streaming support**: Support SSE streaming via text/event-stream content type
- **Input validation**: Validate inputs and use request_validator where applicable
- **Async tokenization**: Await tokenizer functions (count_tokens_for_anthropic_request, truncate_request)
- **Security**: No secrets in logs; always include request_id in LogRecord for traceability
- **Error handling**: Use structured error responses through the error handling system
- **Performance**: Optimize for high throughput and low latency with parallel processing
- **Rate limiting**: Implement proper rate limiting for API endpoints
- **Reasoning support**: Handle OpenRouter and standard reasoning configurations based on provider detection
- **Type safety**: All route handlers must have proper type annotations

## OpenRouter Reasoning Support:
- **Provider detection**: Automatically detects OpenRouter by checking if "openrouter" is in base_url
- **Dual format support**: Uses OpenRouter's `reasoning` object for OpenRouter providers, standard `reasoning_effort` for others
- **Model filtering**: Uses `OPENROUTER_SUPPORT_REASONING_EFFORT_MODELS` for OpenRouter-specific reasoning models
- **Token limits**: Enforces OpenRouter's 1024-32000 reasoning token limits with proper validation
- **Configuration**: OpenRouter reasoning object includes `effort`, `max_tokens`, `enabled`, and `exclude` parameters
