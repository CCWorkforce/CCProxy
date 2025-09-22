"""Response caching system - imports from modular cache implementation."""

# Re-export from the modular cache implementation for backward compatibility
from .cache import ResponseCache, CachedResponse, CacheStatistics

# Create singleton instance for backward compatibility
response_cache = ResponseCache()

__all__ = ["ResponseCache", "CachedResponse", "CacheStatistics", "response_cache"]
