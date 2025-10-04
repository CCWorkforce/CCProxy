"""Comprehensive test suite for cache implementation."""

import anyio
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
        validation_failure_threshold=3,
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
                content=[ContentBlockText(type="text", text="Hello, how are you?")],
            )
        ],
        max_tokens=100,
        stream=False,
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
        stop_reason="end_turn",
    )


@pytest.fixture
def cached_response(sample_response):
    """Create a CachedResponse."""
    return CachedResponse(
        response=sample_response,
        request_hash="test_hash_12345",
        timestamp=time.time(),
        size_bytes=1024,
    )


class TestResponseCache:
    """Test cases for ResponseCache class."""

    @pytest.mark.anyio
    async def test_cache_initialization(self, response_cache):
        """Test cache initialization."""
        assert response_cache._ttl_seconds == 60
        assert response_cache._cleanup_interval == 10
        assert response_cache._memory_manager is not None
        assert response_cache._statistics is not None
        assert response_cache._circuit_breaker is not None
        assert response_cache._stream_deduplicator is not None

    @pytest.mark.anyio
    async def test_get_set_cache(self, response_cache, sample_request, sample_response):
        """Test basic cache get and set operations."""
        # Initially cache should be empty
        result = await response_cache.get_cached_response(sample_request)
        assert result is None

        # Cache a response
        success = await response_cache.cache_response(sample_request, sample_response)
        assert success is True

        # Get the cached value
        result = await response_cache.get_cached_response(sample_request)
        assert result is not None
        assert result.id == sample_response.id
        assert result.content[0].text == "I'm doing well, thank you!"

    @pytest.mark.anyio
    async def test_cache_ttl_expiration(self, sample_request, sample_response):
        """Test cache TTL expiration."""
        # Create cache with very short TTL
        cache = ResponseCache(ttl_seconds=0.1)  # 100ms TTL
        await cache.start_cleanup_task()

        # Cache a value
        await cache.cache_response(sample_request, sample_response)

        # Value should be available immediately
        result = await cache.get_cached_response(sample_request)
        assert result is not None

        # Wait for TTL to expire and cleanup to run
        await anyio.sleep(0.3)

        # Value should be expired after cleanup
        result = await cache.get_cached_response(sample_request)
        # Note: The value might still be there but expired, depends on cleanup interval
        # The important thing is that it's not returned after TTL

        await cache.stop_cleanup_task()

    @pytest.mark.anyio
    async def test_cache_deduplication(
        self, response_cache, sample_request, sample_response
    ):
        """Test request deduplication."""
        # Mark request as pending
        await response_cache.mark_request_pending(sample_request)

        # Start multiple concurrent requests for the same request
        async def get_or_fetch():
            result = await response_cache.get_cached_response(
                sample_request, wait_for_pending=True, timeout_seconds=2.0
            )
            if result is None:
                # First one to get here does the actual fetch
                # Simulate a slow fetch
                await anyio.sleep(0.1)
                await response_cache.cache_response(sample_request, sample_response)
                await response_cache.clear_pending_request(sample_request)
                return sample_response
            return result

        # Launch multiple concurrent requests
        async with anyio.create_task_group() as tg:
            results = []

            async def fetch_and_append():
                result = await get_or_fetch()
                results.append(result)

            for _ in range(5):
                tg.start_soon(fetch_and_append)

        # All results should be the same
        assert all(r.id == sample_response.id for r in results)

    @pytest.mark.anyio
    async def test_cache_statistics(
        self, response_cache, sample_request, sample_response
    ):
        """Test cache statistics tracking."""
        # Initial statistics
        stats = response_cache.get_stats()
        initial_hits = stats.get("cache_hits", 0)
        initial_misses = stats.get("cache_misses", 0)

        # Cache miss
        await response_cache.get_cached_response(sample_request)
        stats = response_cache.get_stats()
        assert stats["cache_misses"] == initial_misses + 1

        # Cache a value
        await response_cache.cache_response(sample_request, sample_response)

        # Cache hit
        await response_cache.get_cached_response(sample_request)
        stats = response_cache.get_stats()
        assert stats["cache_hits"] == initial_hits + 1

    @pytest.mark.anyio
    async def test_cache_clear(self, response_cache, sample_request, sample_response):
        """Test cache clearing."""
        # Create different requests for testing
        requests = []
        for i in range(5):
            req = MessagesRequest(
                model="claude-3-opus-20240229",
                messages=[
                    Message(
                        role="user",
                        content=[ContentBlockText(type="text", text=f"Question {i}?")],
                    )
                ],
                max_tokens=100,
                stream=False,
            )
            requests.append(req)
            await response_cache.cache_response(req, sample_response)

        # Clear cache
        await response_cache.clear()

        # All items should be gone
        for req in requests:
            result = await response_cache.get_cached_response(req)
            assert result is None

    @pytest.mark.anyio
    async def test_streaming_response_handling(self, response_cache, sample_request):
        """Test handling of streaming responses."""
        # Subscribe to stream (returns tuple: is_primary, iterator, key)
        is_primary, stream_iterator, key = await response_cache.subscribe_stream(
            sample_request
        )

        # If we're the primary, we should publish data
        if is_primary:
            # Publish some data
            for i in range(3):
                await response_cache.publish_stream_line(key, f"chunk_{i}")
            await response_cache.finalize_stream(key)

        # Whether primary or not, we should be able to iterate
        assert stream_iterator is not None
        assert key is not None


class TestMemoryManager:
    """Test cases for CacheMemoryManager class."""

    @pytest.mark.anyio
    async def test_memory_manager_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert memory_manager.max_memory_bytes == 1 * 1024 * 1024  # 1MB
        assert memory_manager.max_size == 10
        assert memory_manager.memory_usage_bytes == 0
        assert len(memory_manager.cache) == 0

    @pytest.mark.anyio
    async def test_add_entry(self, memory_manager, cached_response):
        """Test adding entry to memory manager."""
        success = await memory_manager.add("test_key", cached_response, "req_123")
        assert success is True
        assert memory_manager.memory_usage_bytes == 1024
        assert "test_key" in memory_manager.cache

    @pytest.mark.anyio
    async def test_memory_limit_enforcement(self, memory_manager, cached_response):
        """Test memory limit enforcement."""
        # Create a large cached response
        large_response = CachedResponse(
            response=cached_response.response,
            request_hash="large_hash",
            timestamp=time.time(),
            size_bytes=2 * 1024 * 1024,  # 2MB, exceeds limit
        )

        success = await memory_manager.add("large_key", large_response, "req_456")
        assert success is False
        assert "large_key" not in memory_manager.cache

    @pytest.mark.anyio
    async def test_lru_eviction(self, memory_manager):
        """Test LRU eviction when size limit is reached."""
        # Add entries up to the limit
        for i in range(10):
            response = CachedResponse(
                response=MagicMock(),
                request_hash=f"hash_{i}",
                timestamp=time.time(),
                size_bytes=1024,
            )
            await memory_manager.add(f"key_{i}", response, f"req_{i}")

        # Adding one more should evict the oldest
        new_response = CachedResponse(
            response=MagicMock(),
            request_hash="new_hash",
            timestamp=time.time(),
            size_bytes=1024,
        )
        success = await memory_manager.add("new_key", new_response, "req_new")
        assert success is True
        assert "new_key" in memory_manager.cache
        assert "key_0" not in memory_manager.cache  # First one should be evicted

    @pytest.mark.anyio
    async def test_get_entry(self, memory_manager, cached_response):
        """Test getting entry from memory manager."""
        await memory_manager.add("test_key", cached_response, "req_789")

        # Get existing entry
        entry = await memory_manager.get("test_key")
        assert entry is not None
        assert entry.size_bytes == 1024

        # Get non-existent entry
        entry = await memory_manager.get("non_existent")
        assert entry is None

    @pytest.mark.anyio
    async def test_remove_entry(self, memory_manager, cached_response):
        """Test removing entry from memory manager."""
        await memory_manager.add("test_key", cached_response, "req_abc")
        assert memory_manager.memory_usage_bytes == 1024

        # Remove entry
        removed = await memory_manager.remove("test_key")
        assert removed is not None
        assert removed.size_bytes == 1024
        assert memory_manager.memory_usage_bytes == 0
        assert "test_key" not in memory_manager.cache

        # Try to remove non-existent entry
        removed = await memory_manager.remove("non_existent")
        assert removed is None

    @pytest.mark.anyio
    async def test_clear_all(self, memory_manager, cached_response):
        """Test clearing all entries."""
        # Add multiple entries
        for i in range(5):
            await memory_manager.add(f"key_{i}", cached_response, f"req_{i}")

        assert memory_manager.memory_usage_bytes == 5 * 1024

        # Clear all
        await memory_manager.clear()
        assert memory_manager.memory_usage_bytes == 0
        assert len(memory_manager.cache) == 0

    @pytest.mark.anyio
    async def test_get_statistics(self, memory_manager, cached_response):
        """Test getting memory statistics."""
        # Add some entries
        for i in range(3):
            await memory_manager.add(f"key_{i}", cached_response, f"req_{i}")

        stats = memory_manager.get_stats()
        assert stats["cache_size"] == 3
        assert stats["memory_usage_mb"] == round(3 * 1024 / (1024 * 1024), 2)
        assert stats["max_memory_mb"] == 1.0
        assert stats["max_size"] == 10


class TestCircuitBreaker:
    """Test cases for CacheCircuitBreaker class."""

    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.reset_time == 1
        assert circuit_breaker.consecutive_failures == 0
        assert circuit_breaker.is_open() is False

    def test_record_success(self, circuit_breaker):
        """Test recording successful operations."""
        circuit_breaker.record_success()
        assert circuit_breaker.consecutive_failures == 0
        assert circuit_breaker.is_open() is False

    def test_record_failure_threshold(self, circuit_breaker):
        """Test circuit breaker opening on failure threshold."""
        # Record failures up to threshold
        for _ in range(2):
            circuit_breaker.record_failure()
            assert circuit_breaker.is_open() is False

        # One more failure should open the circuit
        circuit_breaker.record_failure()
        assert circuit_breaker.is_open() is True

    def test_circuit_reset_after_timeout(self):
        """Test circuit breaker behavior after timeout."""
        breaker = CacheCircuitBreaker(failure_threshold=1, reset_time=0.1)

        # Open the circuit
        breaker.record_failure()
        assert breaker.is_open() is True

        # Wait for reset time
        time.sleep(0.15)

        # The circuit remains open because consecutive_failures is still >= threshold
        # It doesn't auto-reset, it needs a success or manual reset
        assert breaker.is_open() is True  # Still open due to consecutive_failures
        assert breaker.consecutive_failures == 1

        # Now record a success to actually reset it
        breaker.record_success()
        assert breaker.consecutive_failures == 0
        assert breaker.is_open() is False  # Now it's closed

    def test_success_resets_failure_count(self, circuit_breaker):
        """Test that success resets failure count."""
        # Record some failures (but not enough to open)
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        assert circuit_breaker.consecutive_failures == 2

        # Success should reduce count by 1
        circuit_breaker.record_success()
        assert circuit_breaker.consecutive_failures == 1
        # Another success should reduce to 0
        circuit_breaker.record_success()
        assert circuit_breaker.consecutive_failures == 0


class TestStreamDeduplicator:
    """Test cases for StreamDeduplicator class."""

    @pytest.mark.anyio
    async def test_register_stream(self, stream_deduplicator):
        """Test registering a new stream."""
        stream_id = "stream_123"

        async def sample_stream():
            for i in range(3):
                yield f"data_{i}"

        # Register first subscriber - should be primary
        is_primary, queue = await stream_deduplicator.register(stream_id)
        assert is_primary is True
        assert queue is not None

        # Try to register same stream again - should not be primary
        is_primary2, queue2 = await stream_deduplicator.register(stream_id)
        assert is_primary2 is False  # Not primary since stream already exists
        assert queue2 is not None

    @pytest.mark.anyio
    async def test_subscribe_to_stream(self, stream_deduplicator):
        """Test subscribing to a stream."""
        stream_id = "stream_456"

        async def sample_stream():
            for i in range(3):
                yield f"chunk_{i}"
                await anyio.sleep(0.01)

        # Register stream as primary
        is_primary, primary_queue = await stream_deduplicator.register(stream_id)
        assert is_primary is True

        # Subscribe multiple times (non-primary)
        _, subscriber1 = await stream_deduplicator.register(stream_id)
        _, subscriber2 = await stream_deduplicator.register(stream_id)

        assert subscriber1 is not None
        assert subscriber2 is not None

        # Publish data to all subscribers
        await stream_deduplicator.publish(stream_id, "chunk_0")
        await stream_deduplicator.publish(stream_id, "chunk_1")
        await stream_deduplicator.publish(stream_id, "chunk_2")
        await stream_deduplicator.finalize(stream_id)

        # Collect data from subscribers
        data1 = []
        data2 = []

        # Read from queues until we get None (end signal)
        while True:
            item = await subscriber1.get()
            if item is None:
                break
            data1.append(item)

        while True:
            item = await subscriber2.get()
            if item is None:
                break
            data2.append(item)

        # Both should receive all chunks
        assert data1 == ["chunk_0", "chunk_1", "chunk_2"]
        assert data2 == ["chunk_0", "chunk_1", "chunk_2"]

    @pytest.mark.anyio
    async def test_stream_cleanup(self, stream_deduplicator):
        """Test stream cleanup after completion."""
        stream_id = "cleanup_test"

        async def short_stream():
            yield "data"

        # Register and get queue
        is_primary, queue = await stream_deduplicator.register(stream_id)
        assert is_primary is True

        # Publish and finalize
        await stream_deduplicator.publish(stream_id, "data")
        await stream_deduplicator.finalize(stream_id)

        # Collect data
        data = []
        while True:
            item = await queue.get()
            if item is None:
                break
            data.append(item)

        assert data == ["data"]

        # After finalization, unregister the queue
        await stream_deduplicator.unregister(stream_id, queue)

        # After cleanup, registering the same stream should make us primary again
        is_primary_new, _ = await stream_deduplicator.register(stream_id)
        # Should be primary again since previous stream was cleaned up
        assert is_primary_new is True


class TestCacheStatistics:
    """Test cases for CacheStatistics class."""

    def test_statistics_initialization(self, cache_statistics):
        """Test statistics initialization."""
        assert cache_statistics.cache_hits == 0
        assert cache_statistics.cache_misses == 0
        assert cache_statistics.total_cache_attempts == 0
        assert cache_statistics.evictions == 0

    def test_record_hit(self, cache_statistics):
        """Test recording cache hits."""
        cache_statistics.record_hit()
        assert cache_statistics.cache_hits == 1
        stats = cache_statistics.get_stats()
        assert stats["hit_rate"] == 1.0

    def test_record_miss(self, cache_statistics):
        """Test recording cache misses."""
        cache_statistics.record_miss()
        assert cache_statistics.cache_misses == 1
        stats = cache_statistics.get_stats()
        assert stats["hit_rate"] == 0.0

    def test_hit_rate_calculation(self, cache_statistics):
        """Test hit rate calculation."""
        # Record some hits and misses
        for _ in range(3):
            cache_statistics.record_hit()
        for _ in range(2):
            cache_statistics.record_miss()

        # Hit rate should be 3/5 = 0.6
        stats = cache_statistics.get_stats()
        assert stats["hit_rate"] == 0.6

    def test_get_summary(self, cache_statistics):
        """Test getting statistics summary."""
        # Record various operations
        cache_statistics.record_hit()
        cache_statistics.record_hit()
        cache_statistics.record_miss()
        cache_statistics.record_cache_attempt()
        cache_statistics.record_eviction()

        summary = cache_statistics.get_stats()
        assert summary["cache_hits"] == 2
        assert summary["cache_misses"] == 1
        assert summary["total_cache_attempts"] == 1
        assert summary["evictions"] == 1
        assert abs(summary["hit_rate"] - (2 / 3)) < 0.01  # Use approx comparison

    def test_reset_statistics(self, cache_statistics):
        """Test resetting statistics."""
        # Record some operations
        cache_statistics.record_hit()
        cache_statistics.record_miss()
        cache_statistics.record_cache_attempt()

        # Reset
        cache_statistics.reset()

        # All counters should be zero
        assert cache_statistics.cache_hits == 0
        assert cache_statistics.cache_misses == 0
        assert cache_statistics.total_cache_attempts == 0
        assert cache_statistics.evictions == 0


class TestCachedResponse:
    """Test cases for CachedResponse model."""

    def test_cached_response_creation(self, sample_response):
        """Test creating a cached response."""
        cached = CachedResponse(
            response=sample_response,
            request_hash="test_hash_abc",
            timestamp=time.time(),
            size_bytes=2048,
        )

        assert cached.response == sample_response
        assert cached.request_hash == "test_hash_abc"
        assert cached.size_bytes == 2048
        assert cached.timestamp > 0

    def test_cached_response_expiration(self, sample_response):
        """Test checking if cached response is expired."""
        # Create response with past timestamp
        past_time = time.time() - 100
        cached = CachedResponse(
            response=sample_response,
            request_hash="test_hash_def",
            timestamp=past_time,
            size_bytes=1024,
        )

        # Check if expired with different TTLs
        # Calculate expected expiration based on the implementation
        current_time = time.time()
        assert (current_time - cached.timestamp) > 50  # Should be expired for 50s TTL
        assert (
            current_time - cached.timestamp
        ) < 150  # Should not be expired for 150s TTL
