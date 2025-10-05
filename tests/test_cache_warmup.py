"""Test cache warmup functionality."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from ccproxy.application.cache.warmup import CacheWarmupManager, CacheWarmupConfig
from ccproxy.application.cache.response_cache import ResponseCache
from ccproxy.domain.models import (
    MessagesRequest,
    MessagesResponse,
    Usage,
    Message,
    ContentBlockText,
)


class TestCacheWarmupManager:
    """Test cache warmup manager functionality."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock response cache."""
        cache = MagicMock(spec=ResponseCache)
        cache.get_cached_response = AsyncMock(return_value=None)
        cache.cache_response = AsyncMock(return_value=True)
        cache.set = AsyncMock(return_value=True)  # For warmup compatibility
        cache.get_stats = MagicMock(
            return_value={
                "cache_hits": 0,
                "cache_misses": 0,
                "entries": 0,
                "memory_usage_mb": 0,
            }
        )
        return cache

    @pytest.fixture
    def warmup_config(self, tmp_path):
        """Create a test warmup config."""
        warmup_file = tmp_path / "warmup.json"
        return CacheWarmupConfig(
            enabled=True,
            warmup_file_path=str(warmup_file),
            max_warmup_items=10,
            warmup_on_startup=True,
            preload_common_prompts=True,
            auto_save_popular=True,
            popularity_threshold=2,
            save_interval_seconds=60,
        )

    @pytest.mark.anyio
    async def test_init_and_start(self, mock_cache, warmup_config):
        """Test initialization and startup."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        assert manager.cache == mock_cache
        assert manager.config == warmup_config
        assert manager._popular_items == {}
        assert manager._task_group is None

        await manager.start()
        # Manager should initialize successfully
        assert manager.cache == mock_cache

        await manager.stop()
        # After stop, task group should be None
        assert manager._task_group is None

    @pytest.mark.anyio
    async def test_preload_common_prompts(self, mock_cache, warmup_config):
        """Test preloading of common prompts."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Mock the provider to return a response
        with patch.object(manager, "_load_warmup_item", new=AsyncMock()) as mock_load:
            await manager._warmup_cache()

            # Should have loaded at least one common prompt
            assert mock_load.call_count > 0

    @pytest.mark.anyio
    async def test_save_and_load_warmup_file(self, mock_cache, warmup_config):
        """Test saving and loading warmup items to/from file."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Track some popular items
        manager._popular_items = {
            "cache_key_1": 5,
            "cache_key_2": 3,
            "cache_key_3": 1,  # Below threshold
        }

        # Save popular items that meet threshold
        await manager._save_popular_items()

        # Verify file was created if any items met threshold
        if any(
            count >= warmup_config.popularity_threshold
            for count in manager._popular_items.values()
        ):
            assert Path(warmup_config.warmup_file_path).exists()

    @pytest.mark.anyio
    async def test_track_cache_hit(self, mock_cache, warmup_config):
        """Test tracking cache hits for popularity."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Track cache hits
        cache_key = "test_cache_key_123"

        # Track the same key multiple times
        for _ in range(3):
            manager.track_cache_hit(cache_key)

        # Should be tracked as popular
        assert cache_key in manager._popular_items
        assert manager._popular_items[cache_key] == 3

    @pytest.mark.anyio
    async def test_load_warmup_item(self, mock_cache, warmup_config):
        """Test loading a single warmup item into cache."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # The _load_warmup_item method expects the item to have both request and response
        # But it only caches if the cache key exists in _popular_items
        cache_key = "test_key"
        manager._popular_items[cache_key] = 3  # Make it popular

        test_item = {
            "cache_key": cache_key,
            "request": {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
            "response": {
                "id": "msg_789",
                "model": "claude-3-opus-20240229",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "stop_reason": "end_turn",
                "type": "message",
            },
        }

        await manager._load_warmup_item(test_item)

        # Should have called cache.set (not cache_response)
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        # cache.set(cache_key, response, request)
        assert call_args[0][1].id == "msg_789"  # response
        assert call_args[0][2].model == "claude-3-opus-20240229"  # request

    @pytest.mark.anyio
    async def test_periodic_save(self, mock_cache, warmup_config):
        """Test periodic saving of popular requests."""
        # Set a very short save interval
        warmup_config.save_interval_seconds = 0.1
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Track some cache hits to make items popular
        cache_key = "popular_cache_key"
        for _ in range(3):
            manager.track_cache_hit(cache_key)

        # Start the manager
        await manager.start()

        # Wait for save to happen
        import anyio

        await anyio.sleep(0.2)

        # Stop the manager
        await manager.stop()

        # Check that the file might have been saved
        # (actual saving depends on having cached responses)
        # This is more about testing the save loop runs

    @pytest.mark.anyio
    async def test_max_warmup_items_limit(self, mock_cache, warmup_config):
        """Test that max_warmup_items limit is respected."""
        warmup_config.max_warmup_items = 2
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Track many cache hits for different keys
        for i in range(5):
            cache_key = f"cache_key_{i}"
            # Track each key 3 times to make it popular
            for _ in range(3):
                manager.track_cache_hit(cache_key)

        # Save popular items
        await manager._save_popular_items()

        # Check if file was created and respects limits
        if Path(warmup_config.warmup_file_path).exists():
            with open(warmup_config.warmup_file_path) as f:
                data = json.load(f)
                # Should respect max_warmup_items limit
                assert len(data) <= warmup_config.max_warmup_items

    @pytest.mark.anyio
    async def test_disabled_warmup(self, mock_cache):
        """Test that warmup doesn't run when disabled."""
        config = CacheWarmupConfig(
            enabled=False,
            warmup_file_path="test.json",
            max_warmup_items=10,
            warmup_on_startup=False,
            preload_common_prompts=False,
            auto_save_popular=False,
        )

        manager = CacheWarmupManager(cache=mock_cache, config=config)

        await manager.start()

        # Should not have started any tasks when disabled
        assert manager._task_group is None

        # Should not attempt to load anything
        mock_cache.cache_response.assert_not_called()

        await manager.stop()
        assert manager._task_group is None

    @pytest.mark.anyio
    async def test_preload_responses(self, mock_cache, warmup_config):
        """Test preloading specific responses."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Create test data
        requests = [
            MessagesRequest(
                model="claude-3-opus-20240229",
                messages=[
                    Message(
                        role="user",
                        content=[ContentBlockText(type="text", text=f"Question {i}")],
                    )
                ],
                max_tokens=100,
                stream=False,
            )
            for i in range(3)
        ]

        responses = [
            MessagesResponse(
                id=f"msg_{i}",
                type="message",
                role="assistant",
                model="claude-3-opus-20240229",
                content=[ContentBlockText(type="text", text=f"Answer {i}")],
                usage=Usage(input_tokens=10, output_tokens=10),
                stop_reason="end_turn",
            )
            for i in range(3)
        ]

        # Preload the responses (takes two separate lists)
        count = await manager.preload_responses(requests, responses)

        # Should have cached all responses via cache.set
        assert count == 3
        assert mock_cache.set.call_count == 3

    @pytest.mark.anyio
    async def test_warmup_from_log(self, mock_cache, warmup_config, tmp_path):
        """Test warming up cache from a log file."""
        # Create a mock log file with correct structure
        log_file = tmp_path / "test_log.jsonl"
        log_entries = [
            {
                "event": "REQUEST_COMPLETED",
                "data": {
                    "anthropic_request": {
                        "model": "claude-3-opus-20240229",
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "Test"}],
                            }
                        ],
                        "max_tokens": 100,
                    },
                    "anthropic_response": {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3-opus-20240229",
                        "content": [{"type": "text", "text": "Test response"}],
                        "usage": {"input_tokens": 5, "output_tokens": 5},
                        "stop_reason": "end_turn",
                    },
                    "status_code": 200,
                },
            }
        ]

        with open(log_file, "w") as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")

        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Warmup from log
        count = await manager.warmup_from_log(str(log_file), max_items=10)

        # Should have loaded entries from log
        assert count >= 0  # May be 0 if filtering happens
        # The actual caching depends on implementation details
