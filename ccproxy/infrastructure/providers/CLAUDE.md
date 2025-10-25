# Infrastructure Providers - CLAUDE.md

**Scope**: External service provider implementations and protocol contracts

## Module Architecture (Modularized)
The provider infrastructure has been refactored into specialized modules for better maintainability and reusability.

## Core Provider Files:
- `base.py`: ChatProvider protocol definition and interface contracts
- `openai_provider.py`: Orchestrates high-performance OpenAI API client (420 lines, reduced from 858) with modular components for resilience, metrics, logging, and HTTP client management
- `rate_limiter.py`: Implements ClientRateLimiter with RateLimitStrategy (adaptive default); sliding window for rates, metrics for hits/rejections, handles 429 with backoff/recovery multipliers; uses `asyncify()` for non-blocking list cleanup of request history

## Supporting Modules:
- `resilience.py`: Circuit breaker pattern (CircuitBreaker), retry logic (RetryHandler), and unified resilient execution (ResilientExecutor) with exponential backoff and jitter; adaptive backoff reduces limits by 80% on 429 errors, recovers by 10% after 10 consecutive successes
- `metrics.py`: Provider metrics collection (MetricsCollector), health monitoring (HealthMonitor), adaptive timeout calculation, and comprehensive performance tracking with percentiles
- `http_client_factory.py`: HTTP client factory for dynamic selection (aiohttp for production, httpx for local), HTTP/2 optimization, connection pooling configuration (100-300 connections)
- `response_handlers.py`: Response processing (ResponseProcessor), UTF-8 safe decoding, error classification (ErrorResponseHandler), and streaming response handling
- `request_logger.py`: Request/response logging with correlation IDs (RequestLogger), performance tracking (PerformanceTracker), distributed tracing support, and metadata management

## Guidelines:
- **Protocol compliance**: Implement ChatProvider protocol contracts correctly
- **Error normalization**: Normalize upstream errors to Anthropic-style via error helpers
- **UTF-8 integrity**: Ensure UTF-8 integrity on all request/response bodies (handled by response_handlers.py)
- **Resource cleanup**: Implement proper resource cleanup with async context managers
- **Configuration**: Keep provider-specific config under Settings
- **HTTP optimization**: Use HTTP/2 where supported, dynamic connection pooling via http_client_factory.py
- **Retry logic**: Implement exponential backoff with jitter for retries (handled by resilience.py)
- **Rate limiting**: Client-side rate limiting via rate_limiter.py uses precise tiktoken counts from tokenizer.py for RPM/TPM (OpenAI format); fallback to rough estimate on failure
- **Security**: Never leak API keys or secrets in logs or error messages
- **Modularity**: Use specialized modules for specific concerns (resilience, metrics, logging, etc.)
- **Monitoring**: Leverage metrics.py for comprehensive health checks and performance tracking
- **Tracing**: Support distributed tracing via request_logger.py when available
- **Cython optimizations**: rate_limiter.py integrates `ccproxy._cython.lru_ops` for request history cleanup (20-40% improvement) and `ccproxy._cython.dict_ops` for optimized request tracking
