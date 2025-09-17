# Application Cache Module - CLAUDE.md

**Scope**: Advanced caching implementation with circuit breaker patterns, memory management, and stream deduplication

## Files in this module:
- `__init__.py`: Cache module exports and public API
- `response_cache.py`: Main response caching implementation with circuit breaker pattern
- `memory_manager.py`: Memory management and cleanup for cache operations (500MB limit)
- `circuit_breaker.py`: Circuit breaker pattern for validation failure protection
- `stream_deduplication.py`: Publisher-subscriber patterns for streaming de-duplication
- `statistics.py`: Cache performance metrics and statistics tracking
- `models.py`: Cache-related data models and configurations

## Guidelines:
- **Memory limits**: Respect 500MB cache memory limit with automatic cleanup
- **Circuit breaker**: Use circuit breaker pattern to protect against validation failures
- **Stream safety**: Implement race condition protection in subscriber management
- **Performance metrics**: Track hits, misses, evictions, and performance statistics
- **Data redaction**: Ensure sensitive data is properly redacted before caching
- **Background cleanup**: Implement background processes for cache maintenance
- **Thread safety**: All cache operations must be thread-safe and async-compatible
- **TTL management**: Proper handling of time-to-live for cached entries