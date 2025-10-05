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
- `warmup.py`: CacheWarmupManager for preloading popular requests and common prompts
  - Uses `anyio.Path` for async file I/O operations and parallel warmup item loading
  - Tracks request popularity and auto-saves frequently used prompts
  - Configurable via environment variables (CACHE_WARMUP_*)
  - Supports warmup from log files and predefined common prompts

## Synchronization Primitives:
- **anyio.Lock**: Used for thread-safe access to cache data structures and pending requests
- **anyio.Event**: Used for signaling completion of pending cache operations
- **anyio.create_task_group**: Used for structured concurrency in background cleanup tasks
- **anyio.Semaphore**: Used in stream deduplication for controlling concurrent access

## Guidelines:
- **Memory limits**: Respect 500MB cache memory limit with automatic cleanup
- **Circuit breaker**: Use circuit breaker pattern to protect against validation failures
- **Stream safety**: Implement race condition protection in subscriber management
- **Performance metrics**: Track hits, misses, evictions, and performance statistics
- **Data redaction**: Ensure sensitive data is properly redacted before caching
- **Background cleanup**: Implement background processes for cache maintenance
- **Thread safety**: All cache operations must be thread-safe and async-compatible using anyio primitives
- **TTL management**: Proper handling of time-to-live for cached entries
- **Cache warmup**: Configure via environment to preload popular requests on startup
- **Popularity tracking**: Auto-save popular requests based on configurable thresholds