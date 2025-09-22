"""Memory management for cache with eviction strategies."""

import asyncio
from typing import Dict, List, Optional
from collections import OrderedDict

from .models import CachedResponse
from ...constants import DEFAULT_CACHE_MAX_MEMORY_MB
from ...logging import debug, info, LogRecord, LogEvent


class CacheMemoryManager:
    """
    Manages cache memory usage and eviction policies.

    Implements LRU eviction and memory-based constraints to ensure
    the cache doesn't exceed configured memory limits.
    """

    def __init__(
        self,
        max_memory_mb: int = DEFAULT_CACHE_MAX_MEMORY_MB,
        max_size: int = 1000,
    ):
        """Initialize memory manager."""
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_size = max_size
        self.memory_usage_bytes = 0
        self.cache: OrderedDict[str, CachedResponse] = OrderedDict()
        self.lock = asyncio.Lock()
        self.eviction_count = 0

    async def add(
        self, key: str, response: CachedResponse, request_id: Optional[str] = None
    ) -> bool:
        """
        Add a response to the cache with memory management.

        Args:
            key: Cache key
            response: Cached response to add
            request_id: Optional request ID for logging

        Returns:
            True if added successfully, False if memory constraints prevent addition
        """
        async with self.lock:
            # Check if we need to evict entries
            if response.size_bytes > self.max_memory_bytes:
                debug(
                    LogRecord(
                        event=LogEvent.CACHE_EVENT.value,
                        message="Response too large to cache",
                        request_id=request_id,
                        data={
                            "response_size_mb": response.size_bytes / (1024 * 1024),
                            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                        },
                    )
                )
                return False

            # Evict if necessary
            await self._evict_if_needed(response.size_bytes, request_id)

            # Add to cache
            if key in self.cache:
                # Remove old entry's memory usage
                old_response = self.cache.pop(key)
                self.memory_usage_bytes -= old_response.size_bytes

            self.cache[key] = response
            self.memory_usage_bytes += response.size_bytes

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            return True

    async def get(self, key: str) -> Optional[CachedResponse]:
        """
        Get a cached response and update LRU order.

        Args:
            key: Cache key

        Returns:
            Cached response if found, None otherwise
        """
        async with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                response = self.cache[key]
                response.update_access()
                return response
            return None

    async def remove(self, key: str) -> Optional[CachedResponse]:
        """
        Remove a response from the cache.

        Args:
            key: Cache key

        Returns:
            Removed response if found, None otherwise
        """
        async with self.lock:
            if key in self.cache:
                response = self.cache.pop(key)
                self.memory_usage_bytes -= response.size_bytes
                return response
            return None

    async def _evict_if_needed(
        self, required_bytes: int, request_id: Optional[str] = None
    ):
        """
        Evict entries if needed to make room for new entry.

        Args:
            required_bytes: Bytes needed for new entry
            request_id: Optional request ID for logging
        """
        # Check size constraint
        while len(self.cache) >= self.max_size:
            self._evict_lru(request_id)

        # Check memory constraint
        while self.memory_usage_bytes + required_bytes > self.max_memory_bytes:
            if not self.cache:
                break
            self._evict_lru(request_id)

    def _evict_lru(self, request_id: Optional[str] = None):
        """Evict the least recently used entry."""
        if not self.cache:
            return

        # Pop the first item (least recently used)
        key, response = self.cache.popitem(last=False)
        self.memory_usage_bytes -= response.size_bytes
        self.eviction_count += 1

        debug(
            LogRecord(
                event=LogEvent.CACHE_EVENT.value,
                message="Evicted LRU cache entry",
                request_id=request_id,
                data={
                    "evicted_key": key[:8] + "...",
                    "size_bytes": response.size_bytes,
                    "access_count": response.access_count,
                },
            )
        )

    async def evict_expired(self, ttl_seconds: int) -> List[str]:
        """
        Evict all expired entries.

        Args:
            ttl_seconds: TTL in seconds

        Returns:
            List of evicted keys
        """
        async with self.lock:
            expired_keys = []

            for key, response in list(self.cache.items()):
                if response.is_expired(ttl_seconds):
                    expired_keys.append(key)
                    self.cache.pop(key)
                    self.memory_usage_bytes -= response.size_bytes

            if expired_keys:
                info(
                    LogRecord(
                        event=LogEvent.CACHE_EVENT.value,
                        message=f"Evicted {len(expired_keys)} expired entries",
                        request_id=None,
                        data={"evicted_count": len(expired_keys)},
                    )
                )

            return expired_keys

    async def clear(self):
        """Clear all cached entries."""
        async with self.lock:
            self.cache.clear()
            self.memory_usage_bytes = 0
            self.eviction_count = 0

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.memory_usage_bytes / (1024 * 1024)

    def get_size(self) -> int:
        """Get number of cached entries."""
        return len(self.cache)

    def get_keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self.cache.keys())

    def get_stats(self) -> Dict[str, any]:
        """Get memory manager statistics."""
        return {
            "cache_size": len(self.cache),
            "memory_usage_mb": round(self.get_memory_usage_mb(), 2),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "max_size": self.max_size,
            "eviction_count": self.eviction_count,
        }
