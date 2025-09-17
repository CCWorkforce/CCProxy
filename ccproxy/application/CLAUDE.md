# Application Layer - CLAUDE.md

**Scope**: Use cases, application services, and business logic orchestration

## Files in this layer:
- `converters.py`: Message format conversion between Anthropic and OpenAI APIs
- `converters_module/`: Modular converter implementations with specialized processors
- `tokenizer.py`: Advanced async-aware token counting with TTL-based cache (300s expiry)
- `model_selection.py`: Model mapping logic (opus/sonnet→BIG, haiku→SMALL)
- `request_validator.py`: Request validation with LRU cache (10,000 capacity) and cryptographic hashing
- `response_cache.py`: Response caching abstraction (delegates to cache implementations)
- `cache/`: Advanced caching with circuit breaker pattern, memory management, streaming de-duplication
- `error_tracker.py`: Comprehensive error tracking and monitoring system
- `type_utils.py`: Type utilities and helper functions for the application

## Guidelines:
- **Pure application logic**: No FastAPI or direct I/O side effects
- **OpenAI↔Anthropic parity**: Maintain schema compatibility in converters
- **Async-aware**: All functions support async operations where appropriate
- **Cache integration**: Use response_cache for non-stream flows; validate JSON/UTF‑8
- **Error handling**: Centralized error tracking via error_tracker.py
- **Secret safety**: Never log secrets; use ccproxy.logging for structured logging
- **Dependency injection**: Accept dependencies as parameters for testability
- **Memory management**: Respect cache limits and TTL configurations
