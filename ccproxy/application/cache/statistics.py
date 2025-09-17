"""Cache statistics tracking and reporting."""

from typing import Dict, Any
import time


class CacheStatistics:
    """Tracks and manages cache performance statistics."""

    def __init__(self):
        """Initialize cache statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_usage_bytes = 0
        self.evictions = 0
        self.timeouts_prevented = 0
        self.validation_failures = 0
        self.total_cache_attempts = 0
        self.start_time = time.time()

    def record_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1

    def record_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1

    def record_eviction(self, count: int = 1):
        """Record cache eviction(s)."""
        self.evictions += count

    def record_timeout_prevented(self):
        """Record a timeout that was prevented by cache."""
        self.timeouts_prevented += 1

    def record_validation_failure(self):
        """Record a validation failure."""
        self.validation_failures += 1

    def record_cache_attempt(self):
        """Record a cache attempt."""
        self.total_cache_attempts += 1

    def update_memory_usage(self, bytes_delta: int):
        """Update memory usage statistics."""
        self.memory_usage_bytes += bytes_delta

    def set_memory_usage(self, bytes_total: int):
        """Set total memory usage."""
        self.memory_usage_bytes = bytes_total

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def uptime_seconds(self) -> float:
        """Get cache uptime in seconds."""
        return time.time() - self.start_time

    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics as a dictionary."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 3),
            "memory_usage_mb": round(self.memory_usage_bytes / (1024 * 1024), 2),
            "evictions": self.evictions,
            "timeouts_prevented": self.timeouts_prevented,
            "validation_failures": self.validation_failures,
            "total_cache_attempts": self.total_cache_attempts,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }

    def reset(self):
        """Reset all statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_usage_bytes = 0
        self.evictions = 0
        self.timeouts_prevented = 0
        self.validation_failures = 0
        self.total_cache_attempts = 0
        self.start_time = time.time()