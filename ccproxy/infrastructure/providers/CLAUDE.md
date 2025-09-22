# Infrastructure Providers - CLAUDE.md

**Scope**: External service provider implementations and protocol contracts

## Files in this module:
- `base.py`: ChatProvider protocol definition and interface contracts
- `openai_provider.py`: High-performance OpenAI API client with dynamic HTTP client selection (aiohttp for prod, httpx for local) and HTTP/2 optimization; integrates CircuitBreaker, PerformanceMonitor, ErrorTracker; adaptive timeouts, UTF-8 safe decoding, tracing propagation (if available), and rate limiter for estimated token-based limiting.
- `rate_limiter.py`: Implements ClientRateLimiter with RateLimitStrategy (adaptive default); sliding window for rates, metrics for hits/rejections, handles 429 with backoff/recovery multipliers.

## Guidelines:
- **Protocol compliance**: Implement ChatProvider protocol contracts correctly
- **Error normalization**: Normalize upstream errors to Anthropic-style via error helpers
- **UTF-8 integrity**: Ensure UTF-8 integrity on all request/response bodies
- **Resource cleanup**: Implement proper resource cleanup with async context managers
- **Configuration**: Keep provider-specific config under Settings
- **HTTP optimization**: Use HTTP/2 where supported, dynamic connection pooling (500 local / 1000 prod connections, 120s keepalive)
- **Retry logic**: Implement exponential backoff with jitter for retries
- **Rate limiting**: Client-side rate limiting: Implement via rate_limiter.py with adaptive logic (no background refill; uses acquire/release with estimation); respect upstream 429s by reducing limits dynamically.
- **Security**: Never leak API keys or secrets in logs or error messages
