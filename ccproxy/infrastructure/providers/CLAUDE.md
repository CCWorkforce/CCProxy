# Infrastructure Providers - CLAUDE.md

**Scope**: External service provider implementations and protocol contracts

## Files in this module:
- `base.py`: ChatProvider protocol definition and interface contracts
- `openai_provider.py`: High-performance OpenAI API client with HTTP/2 optimization

## Guidelines:
- **Protocol compliance**: Implement ChatProvider protocol contracts correctly
- **Error normalization**: Normalize upstream errors to Anthropic-style via error helpers
- **UTF-8 integrity**: Ensure UTF-8 integrity on all request/response bodies
- **Resource cleanup**: Implement proper resource cleanup with async context managers
- **Configuration**: Keep provider-specific config under Settings
- **HTTP optimization**: Use HTTP/2, connection pooling (500 connections, 120s keepalive)
- **Retry logic**: Implement exponential backoff with jitter for retries
- **Rate limiting**: Respect upstream rate limits and implement backoff strategies
- **Security**: Never leak API keys or secrets in logs or error messages
