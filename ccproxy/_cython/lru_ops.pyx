# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=True
# cython: linetrace=True
"""Cython-optimized LRU cache operations for CCProxy.

This module provides high-performance LRU cache management operations including:
- Shard selection via consistent hashing
- TTL expiry checks with optimized timestamp arithmetic
- List filtering for request history cleanup
- Token sum calculations for rate limiting

Performance Targets:
    - Shard selection: < 0.05ms per operation
    - TTL check: < 0.05ms per check
    - List filtering: < 3ms for 1000 entries
    - Token sum: < 0.5ms for 500 entries
"""

from typing import Any, List, Tuple
cimport cython
from libc.time cimport time_t
from cpython.list cimport PyList_New, PyList_SET_ITEM, PyList_GET_SIZE, PyList_GET_ITEM
from cpython.ref cimport Py_INCREF


cpdef Py_ssize_t get_shard_index(str key, Py_ssize_t num_shards):
    """Get shard index for a key using consistent hashing.

    Uses Python's built-in hash() function modulo num_shards for
    consistent shard assignment.

    Args:
        key: Key to hash
        num_shards: Total number of shards

    Returns:
        Shard index (0 to num_shards-1)
    """
    return hash(key) % num_shards


cpdef bint is_expired(double timestamp, double current_time, double ttl_seconds):
    """Check if a timestamp has expired based on TTL.

    Optimized C-level arithmetic for timestamp comparison.

    Args:
        timestamp: Creation/access timestamp
        current_time: Current time
        ttl_seconds: Time-to-live in seconds

    Returns:
        True if (current_time - timestamp) > ttl_seconds
    """
    return (current_time - timestamp) > ttl_seconds


cpdef list filter_recent_timestamps(list timestamps, double cutoff_time):
    """Filter timestamps to only include those after cutoff time.

    Optimized replacement for:
        [t for t in timestamps if current_time - t < window]

    Args:
        timestamps: List of float timestamps
        cutoff_time: Cutoff time (current_time - window)

    Returns:
        New list containing only timestamps >= cutoff_time
    """
    cdef Py_ssize_t i, n = len(timestamps)
    cdef list result = []
    cdef double t

    for i in range(n):
        t = <double>timestamps[i]
        if t >= cutoff_time:
            result.append(t)

    return result


cpdef list filter_request_times(list request_times, double current_time, double window_seconds):
    """Filter request times to only include those within the time window.

    Optimized for rate limiter usage - filters out requests older than
    the specified time window.

    Args:
        request_times: List of (timestamp, request_id) tuples or just timestamps
        current_time: Current time
        window_seconds: Time window in seconds (e.g., 60 for RPM)

    Returns:
        Filtered list of request times within the window
    """
    cdef double cutoff_time = current_time - window_seconds
    cdef Py_ssize_t i, n = len(request_times)
    cdef list result = []
    cdef double t
    cdef object item

    for i in range(n):
        item = request_times[i]
        # Handle both float timestamps and tuples
        if isinstance(item, tuple):
            t = <double>(<tuple>item)[0]
        else:
            t = <double>item

        if t >= cutoff_time:
            result.append(item)

    return result


cpdef list filter_token_counts(list token_counts, double current_time, double window_seconds):
    """Filter token counts to only include those within the time window.

    Optimized for TPM (tokens per minute) rate limiting.

    Args:
        token_counts: List of (timestamp, token_count) tuples
        current_time: Current time
        window_seconds: Time window in seconds (e.g., 60 for TPM)

    Returns:
        Filtered list of token counts within the window
    """
    cdef double cutoff_time = current_time - window_seconds
    cdef Py_ssize_t i, n = len(token_counts)
    cdef list result = []
    cdef double t
    cdef tuple item

    for i in range(n):
        item = <tuple>token_counts[i]
        t = <double>item[0]
        if t >= cutoff_time:
            result.append(item)

    return result


cpdef long long sum_token_counts(list token_counts):
    """Sum token counts from a list of (timestamp, count) tuples.

    Optimized replacement for:
        sum(c for _, c in token_counts)

    Args:
        token_counts: List of (timestamp, token_count) tuples

    Returns:
        Total token count
    """
    cdef long long total = 0
    cdef Py_ssize_t i, n = len(token_counts)
    cdef tuple item
    cdef long long count

    for i in range(n):
        item = <tuple>token_counts[i]
        count = <long long>item[1]
        total += count

    return total


cpdef bint should_evict_lru(Py_ssize_t current_size, Py_ssize_t max_size):
    """Check if LRU eviction should be triggered.

    Simple size comparison optimized at C level.

    Args:
        current_size: Current cache size
        max_size: Maximum allowed cache size

    Returns:
        True if current_size >= max_size
    """
    return current_size >= max_size


cpdef list get_expired_keys(dict cache, double current_time, double ttl_seconds):
    """Get list of expired keys from cache with TTL-based entries.

    Assumes cache values are objects with a 'timestamp' attribute.

    Args:
        cache: Dictionary with timestamp-based entries
        current_time: Current time
        ttl_seconds: Time-to-live in seconds

    Returns:
        List of expired keys
    """
    cdef list expired = []
    cdef str key
    cdef object entry
    cdef double timestamp

    for key, entry in cache.items():
        # Assume entry has timestamp attribute (CacheEntry dataclass)
        try:
            timestamp = <double>entry.timestamp
            if is_expired(timestamp, current_time, ttl_seconds):
                expired.append(key)
        except (AttributeError, TypeError):
            # Skip entries without timestamp
            continue

    return expired


cpdef Py_ssize_t count_expired_entries(dict cache, double current_time, double ttl_seconds):
    """Count number of expired entries in cache without creating list.

    Memory-efficient version that only counts without allocating a list.

    Args:
        cache: Dictionary with timestamp-based entries
        current_time: Current time
        ttl_seconds: Time-to-live in seconds

    Returns:
        Number of expired entries
    """
    cdef Py_ssize_t count = 0
    cdef object entry
    cdef double timestamp

    for entry in cache.values():
        try:
            timestamp = <double>entry.timestamp
            if is_expired(timestamp, current_time, ttl_seconds):
                count += 1
        except (AttributeError, TypeError):
            continue

    return count


cpdef double calculate_hit_rate(long long hits, long long misses):
    """Calculate cache hit rate percentage.

    Args:
        hits: Number of cache hits
        misses: Number of cache misses

    Returns:
        Hit rate as percentage (0.0 to 100.0)
    """
    cdef long long total = hits + misses
    if total == 0:
        return 0.0
    return (100.0 * <double>hits) / <double>total


@cython.cdivision(True)
cpdef Py_ssize_t calculate_max_per_shard(Py_ssize_t max_total, Py_ssize_t num_shards):
    """Calculate maximum entries per shard.

    Uses C-level division for performance.

    Args:
        max_total: Total maximum entries
        num_shards: Number of shards

    Returns:
        Maximum entries per shard (at least 1)
    """
    cdef Py_ssize_t max_per_shard = max_total // num_shards
    if max_per_shard < 1:
        return 1
    return max_per_shard


cpdef list get_lru_eviction_candidates(list lru_order, Py_ssize_t num_to_evict):
    """Get keys to evict from LRU order (oldest first).

    Args:
        lru_order: List of keys in LRU order (oldest first)
        num_to_evict: Number of keys to evict

    Returns:
        List of keys to evict (oldest entries)
    """
    if num_to_evict <= 0:
        return []
    if num_to_evict >= len(lru_order):
        return list(lru_order)  # Return copy

    cdef Py_ssize_t i
    cdef list result = []

    for i in range(num_to_evict):
        result.append(lru_order[i])

    return result
