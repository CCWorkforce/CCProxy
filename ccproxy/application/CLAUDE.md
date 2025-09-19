# Application Layer - CLAUDE.md

**Scope**: Use cases, application services, and business logic orchestration

## Files in this layer:
- `converters.py`: Message format conversion between Anthropic and OpenAI APIs (exports async converters)
- `converters_module/`: Modular converter implementations with specialized processors
  - `async_converter.py`: AsyncMessageConverter and AsyncResponseConverter for parallel processing
  - Uses ThreadPoolExecutor for CPU-bound operations, optimized for high throughput
  - Provides `convert_messages_async()` and `convert_response_async()` functions
- `tokenizer.py`: Advanced async-aware token counting with TTL-based cache (300s expiry)
- `model_selection.py`: Model mapping logic (opus/sonnet→BIG, haiku→SMALL)
- `request_validator.py`: Request validation with LRU cache (10,000 capacity) and cryptographic hashing
- `response_cache.py`: Response caching abstraction (delegates to cache implementations)
- `cache/`: Advanced caching with circuit breaker pattern, memory management, streaming de-duplication
  - `warmup.py`: CacheWarmupManager for preloading popular requests and common prompts on startup
  - Tracks request popularity and auto-saves frequently used prompts
- `error_tracker.py`: Comprehensive error tracking and monitoring system
- `type_utils.py`: Type utilities and helper functions for the application

## Guidelines:
- **Pure application logic**: No FastAPI or direct I/O side effects
- **OpenAI↔Anthropic parity**: Maintain schema compatibility in converters
- **Async-aware**: All functions support async operations where appropriate
- **Performance optimization**: Use async converters (`convert_messages_async`, `convert_response_async`) for better throughput
- **Cache integration**: Use response_cache for non-stream flows; validate JSON/UTF‑8
- **Cache warmup**: Configure CacheWarmupManager via environment variables for startup preloading
- **Error handling**: Centralized error tracking via error_tracker.py
- **Secret safety**: Never log secrets; use ccproxy.logging for structured logging
- **Dependency injection**: Accept dependencies as parameters for testability
- **Memory management**: Respect cache limits and TTL configurations
- **Type safety**: Strict type checking enabled; all functions must have type annotations
