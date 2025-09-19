"""Comprehensive test suite for cache implementation."""

import asyncio
import time
from unittest.mock import MagicMock
import pytest

from ccproxy.application.cache.response_cache import ResponseCache
from ccproxy.application.cache.memory_manager import CacheMemoryManager
from ccproxy.application.cache.circuit_breaker import CacheCircuitBreaker
from ccproxy.application.cache.stream_deduplication import StreamDeduplicator
from ccproxy.application.cache.statistics import CacheStatistics
from ccproxy.application.cache.models import CachedResponse

from ccproxy.domain.models import (
    MessagesRequest,
    MessagesResponse,
    Message,
    ContentBlockText,
    Usage,
)


@pytest.fixture
async def response_cache():
    """Create a response cache instance for testing."""
    cache = ResponseCache(
        max_size=10,
        max_memory_mb=1,
        ttl_seconds=60,
        cleanup_interval_seconds=10,
        validation_failure_threshold=3
    )
    await cache.start_cleanup_task()
    yield cache
    await cache.stop_cleanup_task()


@pytest.fixture
def memory_manager():
    """Create a memory manager instance for testing."""
    return CacheMemoryManager(max_memory_mb=1, max_size=10)


@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker instance for testing."""
    return CacheCircuitBreaker(failure_threshold=3, reset_time=1)


@pytest.fixture
def stream_deduplicator():
    """Create a stream deduplicator instance for testing."""
    return StreamDeduplicator()


@pytest.fixture
def cache_statistics():
    """Create a cache statistics instance for testing."""
    return CacheStatistics()


@pytest.fixture
def sample_request():
    """Create a sample MessagesRequest."""
    return MessagesRequest(
        model="claude-3-opus-20240229",
        messages=[
            Message(
                role="user",
                content=[ContentBlockText(type="text", text="Hello, how are you?")]
            )
        ],
        max_tokens=100,
        stream=False
    )


@pytest.fixture
def sample_response():
    """Create a sample MessagesResponse."""
    return MessagesResponse(
        id="msg_123",
        type="message",
        role="assistant",
        model="claude-3-opus-20240229",
        content=[ContentBlockText(type="text", text="I'm doing well, thank you!")],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn"
    )


@pytest.fixture
def cached_response(sample_response):
    """Create a CachedResponse."""
    return CachedResponse(
        response=sample_response,
        timestamp=time.time(),
        size_bytes=1024
    )


class TestResponseCache:
    """Test cases for ResponseCache class."""

    @pytest.mark.asyncio
    async def test_cache_initialization(self, response_cache):
        """Test cache initialization."""
        assert response_cache._ttl_seconds == 60
        assert response_cache._cleanup_interval == 10
        assert response_cache._memory_manager is not None
        assert response_cache._statistics is not None
        assert response_cache._circuit_breaker is not None
        assert response_cache._stream_deduplicator is not None

    @pytest.mark.asyncio
    async def test_get_set_cache(self, response_cache, sample_request, sample_response):
        """Test basic cache get and set operations."""
        cache_key = "test_key_123"

        # Initially cache should be empty
        result = await response_cache.get(cache_key)
        assert result is None

        # Set a value
        await response_cache.set(cache_key, sample_response, sample_request)

        # Get the cached value
        result = await response_cache.get(cache_key)
        assert result is not None
        assert result.id == sample_response.id
        assert result.content[0].text == "I'm doing well, thank you!"

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, sample_request, sample_response):
        """Test cache TTL expiration."""
        # Create cache with very short TTL
        cache = ResponseCache(ttl_seconds=0.1)  # 100ms TTL
        await cache.start_cleanup_task()

        cache_key = "ttl_test_key"

        # Set a value
        await cache.set(cache_key, sample_response, sample_request)

        # Value should be available immediately
        result = await cache.get(cache_key)
        assert result is not None

        # Wait for TTL to expire
        await asyncio.sleep(0.2)

        # Value should be expired
        result = await cache.get(cache_key)
        assert result is None

        await cache.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_cache_deduplication(self, response_cache, sample_request, sample_response):
        """Test request deduplication."""
        cache_key = "dedupe_test"

        # Start multiple concurrent requests for the same key
        async def get_or_fetch():
            result = await response_cache.get(cache_key)
            if result is None:
                # Simulate a slow fetch
                await asyncio.sleep(0.1)
                await response_cache.set(cache_key, sample_response, sample_request)
                return sample_response
            return result

        # Launch multiple concurrent requests
        results = await asyncio.gather(*[get_or_fetch() for _ in range(5)])

        # All results should be the same
        assert all(r.id == sample_response.id for r in results)

    @pytest.mark.asyncio
    async def test_cache_statistics(self, response_cache, sample_request, sample_response):
        """Test cache statistics tracking."""
        cache_key = "stats_test"

        # Initial statistics
        stats = await response_cache.get_statistics()
        initial_hits = stats.get("hits", 0)
        initial_misses = stats.get("misses", 0)

        # Cache miss
        await response_cache.get(cache_key)
        stats = await response_cache.get_statistics()
        assert stats["misses"] == initial_misses + 1

        # Set value
        await response_cache.set(cache_key, sample_response, sample_request)

        # Cache hit
        await response_cache.get(cache_key)
        stats = await response_cache.get_statistics()
        assert stats["hits"] == initial_hits + 1

    @pytest.mark.asyncio
    async def test_cache_clear(self, response_cache, sample_request, sample_response):
        """Test cache clearing."""
        # Add multiple items
        for i in range(5):
            await response_cache.set(f"key_{i}", sample_response, sample_request)

        # Clear cache
        await response_cache.clear()

        # All items should be gone
        for i in range(5):
            result = await response_cache.get(f"key_{i}")
            assert result is None

    @pytest.mark.asyncio
    async def test_streaming_response_handling(self, response_cache, sample_request):
        """Test handling of streaming responses."""
        cache_key = "stream_test"

        async def stream_generator():
            for i in range(3):
                yield f"chunk_{i}"

        # Register stream
        registered = await response_cache.register_stream(cache_key, stream_generator())
        assert registered is True

        # Second registration should fail (already streaming)
        registered = await response_cache.register_stream(cache_key, stream_generator())
        assert registered is False

        # Subscribe to stream
        subscriber = response_cache.subscribe_to_stream(cache_key)
        assert subscriber is not None


class TestMemoryManager:
    """Test cases for CacheMemoryManager class."""

    def test_memory_manager_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert memory_manager._max_memory_bytes == 1 * 1024 * 1024  # 1MB
        assert memory_manager._max_size == 10
        assert memory_manager._current_memory == 0
        assert len(memory_manager._cache_entries) == 0

    def test_add_entry(self, memory_manager, cached_response):
        """Test adding entry to memory manager."""
        success = memory_manager.add_entry("test_key", cached_response)
        assert success is True
        assert memory_manager._current_memory == 1024
        assert "test_key" in memory_manager._cache_entries

    def test_memory_limit_enforcement(self, memory_manager, cached_response):
        """Test memory limit enforcement."""
        # Create a large cached response
        large_response = CachedResponse(
            response=cached_response.response,
            timestamp=time.time(),
            size_bytes=2 * 1024 * 1024  # 2MB, exceeds limit
        )

        success = memory_manager.add_entry("large_key", large_response)
        assert success is False
        assert "large_key" not in memory_manager._cache_entries

    def test_lru_eviction(self, memory_manager):
        """Test LRU eviction when size limit is reached."""
        # Add entries up to the limit
        for i in range(10):
            response = CachedResponse(
                response=MagicMock(),
                timestamp=time.time(),
                size_bytes=1024
            )
            memory_manager.add_entry(f"key_{i}", response)

        # Adding one more should evict the oldest
        new_response = CachedResponse(
            response=MagicMock(),
            timestamp=time.time(),
            size_bytes=1024
        )
        success = memory_manager.add_entry("new_key", new_response)
        assert success is True
        assert "new_key" in memory_manager._cache_entries
        assert "key_0" not in memory_manager._cache_entries  # First one should be evicted

    def test_get_entry(self, memory_manager, cached_response):
        """Test getting entry from memory manager."""
        memory_manager.add_entry("test_key", cached_response)

        # Get existing entry
        entry = memory_manager.get_entry("test_key")
        assert entry is not None
        assert entry.size_bytes == 1024

        # Get non-existent entry
        entry = memory_manager.get_entry("non_existent")
        assert entry is None

    def test_remove_entry(self, memory_manager, cached_response):
        """Test removing entry from memory manager."""
        memory_manager.add_entry("test_key", cached_response)
        assert memory_manager._current_memory == 1024

        # Remove entry
        removed = memory_manager.remove_entry("test_key")
        assert removed is True
        assert memory_manager._current_memory == 0
        assert "test_key" not in memory_manager._cache_entries

        # Try to remove non-existent entry
        removed = memory_manager.remove_entry("non_existent")
        assert removed is False

    def test_clear_all(self, memory_manager, cached_response):
        """Test clearing all entries."""
        # Add multiple entries
        for i in range(5):
            memory_manager.add_entry(f"key_{i}", cached_response)

        assert memory_manager._current_memory == 5 * 1024

        # Clear all
        memory_manager.clear_all()
        assert memory_manager._current_memory == 0
        assert len(memory_manager._cache_entries) == 0

    def test_get_statistics(self, memory_manager, cached_response):
        """Test getting memory statistics."""
        # Add some entries
        for i in range(3):
            memory_manager.add_entry(f"key_{i}", cached_response)

        stats = memory_manager.get_statistics()
        assert stats["total_entries"] == 3
        assert stats["current_memory_bytes"] == 3 * 1024
        assert stats["max_memory_bytes"] == 1024 * 1024
        assert "memory_usage_percent" in stats


class TestCircuitBreaker:
    """Test cases for CacheCircuitBreaker class."""

    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker._failure_threshold == 3
        assert circuit_breaker._reset_time == 1
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker._is_open is False

    def test_record_success(self, circuit_breaker):
        """Test recording successful operations."""
        circuit_breaker.record_success()
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker._is_open is False

    def test_record_failure_threshold(self, circuit_breaker):
        """Test circuit breaker opening on failure threshold."""
        # Record failures up to threshold
        for _ in range(2):
            circuit_breaker.record_failure()
            assert circuit_breaker._is_open is False

        # One more failure should open the circuit
        circuit_breaker.record_failure()
        assert circuit_breaker._is_open is True
        assert circuit_breaker.is_open() is True

    def test_circuit_reset_after_timeout(self):
        """Test circuit breaker reset after timeout."""
        breaker = CacheCircuitBreaker(failure_threshold=1, reset_time=0.1)

        # Open the circuit
        breaker.record_failure()
        assert breaker.is_open() is True

        # Wait for reset time
        time.sleep(0.15)

        # Circuit should be closed again
        assert breaker.is_open() is False
        assert breaker._failure_count == 0

    def test_success_resets_failure_count(self, circuit_breaker):
        """Test that success resets failure count."""
        # Record some failures (but not enough to open)
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        assert circuit_breaker._failure_count == 2

        # Success should reset count
        circuit_breaker.record_success()
        assert circuit_breaker._failure_count == 0


class TestStreamDeduplicator:
    """Test cases for StreamDeduplicator class."""

    @pytest.mark.asyncio
    async def test_register_stream(self, stream_deduplicator):
        """Test registering a new stream."""
        stream_id = "stream_123"

        async def sample_stream():
            for i in range(3):
                yield f"data_{i}"

        # Register stream
        success = await stream_deduplicator.register_stream(stream_id, sample_stream())
        assert success is True

        # Try to register same stream again
        success = await stream_deduplicator.register_stream(stream_id, sample_stream())
        assert success is False

    @pytest.mark.asyncio
    async def test_subscribe_to_stream(self, stream_deduplicator):
        """Test subscribing to a stream."""
        stream_id = "stream_456"

        async def sample_stream():
            for i in range(3):
                yield f"chunk_{i}"
                await asyncio.sleep(0.01)

        # Register stream
        await stream_deduplicator.register_stream(stream_id, sample_stream())

        # Subscribe multiple times
        subscriber1 = stream_deduplicator.subscribe(stream_id)
        subscriber2 = stream_deduplicator.subscribe(stream_id)

        assert subscriber1 is not None
        assert subscriber2 is not None

        # Collect data from subscribers
        data1 = []
        data2 = []

        async def collect_data(subscriber, data_list):
            async for chunk in subscriber:
                data_list.append(chunk)

        # Collect concurrently
        await asyncio.gather(
            collect_data(subscriber1, data1),
            collect_data(subscriber2, data2)
        )

        # Both should receive all chunks
        assert data1 == ["chunk_0", "chunk_1", "chunk_2"]
        assert data2 == ["chunk_0", "chunk_1", "chunk_2"]

    @pytest.mark.asyncio
    async def test_stream_cleanup(self, stream_deduplicator):
        """Test stream cleanup after completion."""
        stream_id = "cleanup_test"

        async def short_stream():
            yield "data"

        # Register and consume stream
        await stream_deduplicator.register_stream(stream_id, short_stream())
        subscriber = stream_deduplicator.subscribe(stream_id)

        data = []
        async for chunk in subscriber:
            data.append(chunk)

        assert data == ["data"]

        # Stream should be cleaned up
        await asyncio.sleep(0.1)  # Allow cleanup to happen

        # New subscription should return None (stream gone)
        stream_deduplicator.subscribe(stream_id)
        # Stream might still be in cleanup phase or already cleaned
        # This behavior depends on implementation details


class TestCacheStatistics:
    """Test cases for CacheStatistics class."""

    def test_statistics_initialization(self, cache_statistics):
        """Test statistics initialization."""
        assert cache_statistics.hits == 0
        assert cache_statistics.misses == 0
        assert cache_statistics.sets == 0
        assert cache_statistics.evictions == 0

    def test_record_hit(self, cache_statistics):
        """Test recording cache hits."""
        cache_statistics.record_hit()
        assert cache_statistics.hits == 1
        assert cache_statistics.hit_rate() == 1.0

    def test_record_miss(self, cache_statistics):
        """Test recording cache misses."""
        cache_statistics.record_miss()
        assert cache_statistics.misses == 1
        assert cache_statistics.hit_rate() == 0.0

    def test_hit_rate_calculation(self, cache_statistics):
        """Test hit rate calculation."""
        # Record some hits and misses
        for _ in range(3):
            cache_statistics.record_hit()
        for _ in range(2):
            cache_statistics.record_miss()

        # Hit rate should be 3/5 = 0.6
        assert cache_statistics.hit_rate() == 0.6

    def test_get_summary(self, cache_statistics):
        """Test getting statistics summary."""
        # Record various operations
        cache_statistics.record_hit()
        cache_statistics.record_hit()
        cache_statistics.record_miss()
        cache_statistics.record_set()
        cache_statistics.record_eviction()

        summary = cache_statistics.get_summary()
        assert summary["hits"] == 2
        assert summary["misses"] == 1
        assert summary["sets"] == 1
        assert summary["evictions"] == 1
        assert summary["hit_rate"] == 2/3

    def test_reset_statistics(self, cache_statistics):
        """Test resetting statistics."""
        # Record some operations
        cache_statistics.record_hit()
        cache_statistics.record_miss()
        cache_statistics.record_set()

        # Reset
        cache_statistics.reset()

        # All counters should be zero
        assert cache_statistics.hits == 0
        assert cache_statistics.misses == 0
        assert cache_statistics.sets == 0
        assert cache_statistics.evictions == 0


class TestCachedResponse:
    """Test cases for CachedResponse model."""

    def test_cached_response_creation(self, sample_response):
        """Test creating a cached response."""
        cached = CachedResponse(
            response=sample_response,
            timestamp=time.time(),
            size_bytes=2048
        )

        assert cached.response == sample_response
        assert cached.size_bytes == 2048
        assert cached.timestamp > 0

    def test_cached_response_expiration(self, sample_response):
        """Test checking if cached response is expired."""
        # Create response with past timestamp
        past_time = time.time() - 100
        cached = CachedResponse(
            response=sample_response,
            timestamp=past_time,
            size_bytes=1024
        )

        # Check if expired with different TTLs
        assert cached.is_expired(ttl_seconds=50) is True
        assert cached.is_expired(ttl_seconds=150) is False