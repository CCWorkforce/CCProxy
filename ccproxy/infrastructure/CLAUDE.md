# Infrastructure Layer - CLAUDE.md

**Scope**: External service integrations, third-party adapters, and infrastructure concerns

## Files in this layer:
- `providers/`: External service provider implementations
  - `base.py`: ChatProvider protocol definition and contracts
  - `openai_provider.py`: High-performance HTTP/2 client with connection pooling (500 connections, 120s keepalive); includes circuit breaker (failure threshold=5, recovery=60s), comprehensive metrics (latency percentiles, health scoring 0-100), error tracking via ErrorTracker, adaptive timeouts (p95*2), request correlation IDs, and client rate limiter integration for resilience and monitoring
  - `rate_limiter.py`: Client-side rate limiter with sliding window and adaptive strategy (backoff on 429, recovery on successes); uses `asyncify()` for non-blocking list cleanup operations on request history.

## Guidelines:
- **External integrations**: Handle all communication with external services (OpenAI, OpenRouter, etc.)
- **Error wrapping**: OpenAIProvider must wrap errors into openai.APIError for consistency
- **HTTP optimization**: Use shared httpx client patterns (HTTP/2, timeouts, connection pooling)
- **Resource management**: Providers implement proper cleanup via async context managers
- **Security**: Never leak secrets in logs; rely on ccproxy.logging redaction
- **Resilience**: Keep adapters async and resilient with retries per Settings configuration
- **Performance**: Optimize connection pooling (500 connections, 120s keepalive) and implement exponential backoff
- **Protocol compliance**: Implement the ChatProvider protocol correctly
- **UTF-8 handling**: Comprehensive UTF-8 error handling and recovery mechanisms
- **Client-side rate limiting**: Use ClientRateLimiter in providers for proactive RPM/TPM control; estimate tokens roughly (~4 chars/token) before acquire().
