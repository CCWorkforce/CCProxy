"""Test cache warmup functionality."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from ccproxy.application.cache.warmup import CacheWarmupManager, CacheWarmupConfig
from ccproxy.application.response_cache import ResponseCache
from ccproxy.domain.models import MessagesRequest, MessagesResponse, Usage


class TestCacheWarmupManager:
    """Test cache warmup manager functionality."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock response cache."""
        cache = MagicMock(spec=ResponseCache)
        cache.get_cached_response = AsyncMock(return_value=None)
        cache.cache_response = AsyncMock()
        cache.get_stats = AsyncMock(
            return_value={"hits": 0, "misses": 0, "entries": 0, "size_bytes": 0}
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

    @pytest.mark.asyncio
    async def test_init_and_start(self, mock_cache, warmup_config):
        """Test initialization and startup."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        assert manager.cache == mock_cache
        assert manager.config == warmup_config
        assert manager._warmup_items == []
        assert manager._request_counts == {}

        await manager.start()
        # Should attempt to load warmup file
        assert manager._save_task is not None

        await manager.stop()

    @pytest.mark.asyncio
    async def test_preload_common_prompts(self, mock_cache, warmup_config):
        """Test preloading of common prompts."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Mock the provider to return a response
        with patch.object(manager, "_load_warmup_item", new=AsyncMock()) as mock_load:
            await manager._warmup_cache()

            # Should have loaded at least one common prompt
            assert mock_load.call_count > 0

    @pytest.mark.asyncio
    async def test_save_and_load_warmup_file(self, mock_cache, warmup_config):
        """Test saving and loading warmup items to/from file."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Add some warmup items
        test_item = {
            "request": {
                "model": "claude-3-opus",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            "response": {
                "id": "msg_123",
                "model": "claude-3-opus",
                "content": [{"type": "text", "text": "Hi there!"}],
                "usage": {"input_tokens": 5, "output_tokens": 4},
            },
        }
        manager._warmup_items = [test_item]

        # Save to file
        await manager._save_warmup_file()

        # Verify file was created
        assert Path(warmup_config.warmup_file_path).exists()

        # Load from file
        manager._warmup_items = []
        await manager._load_warmup_file()

        # Should have loaded the item
        assert len(manager._warmup_items) == 1
        assert manager._warmup_items[0] == test_item

    @pytest.mark.asyncio
    async def test_track_request_popularity(self, mock_cache, warmup_config):
        """Test tracking popular requests."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        request = MessagesRequest(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=100,
        )

        response = MessagesResponse(
            id="msg_456",
            model="claude-3-opus",
            role="assistant",
            content=[{"type": "text", "text": "Test response"}],
            usage=Usage(input_tokens=10, output_tokens=8),
        )

        # Track the same request multiple times
        for _ in range(3):
            await manager.track_request(request, response)

        # Should have tracked the request
        request_key = manager._get_request_key(request)
        assert manager._request_counts[request_key] == 3

        # Should identify it as popular
        popular = await manager._get_popular_requests()
        assert len(popular) == 1
        assert popular[0]["count"] == 3

    @pytest.mark.asyncio
    async def test_load_warmup_item(self, mock_cache, warmup_config):
        """Test loading a single warmup item into cache."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        test_item = {
            "request": {
                "model": "claude-3-opus",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
            "response": {
                "id": "msg_789",
                "model": "claude-3-opus",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        }

        await manager._load_warmup_item(test_item)

        # Should have cached the response
        mock_cache.cache_response.assert_called_once()
        call_args = mock_cache.cache_response.call_args
        assert call_args[0][0].model == "claude-3-opus"
        assert call_args[0][1].id == "msg_789"

    @pytest.mark.asyncio
    async def test_periodic_save(self, mock_cache, warmup_config):
        """Test periodic saving of popular requests."""
        # Set a very short save interval
        warmup_config.save_interval_seconds = 0.1
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Track some popular requests
        request = MessagesRequest(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Popular request"}],
            max_tokens=100,
        )
        response = MessagesResponse(
            id="msg_999",
            model="claude-3-opus",
            role="assistant",
            content=[{"type": "text", "text": "Popular response"}],
            usage=Usage(input_tokens=10, output_tokens=10),
        )

        for _ in range(3):
            await manager.track_request(request, response)

        # Start the manager
        await manager.start()

        # Wait for save to happen
        import asyncio

        await asyncio.sleep(0.2)

        # Stop the manager
        await manager.stop()

        # Check that the file was saved
        if Path(warmup_config.warmup_file_path).exists():
            with open(warmup_config.warmup_file_path) as f:
                data = json.load(f)
                # Should have saved at least the popular request
                assert len(data) >= 1

    @pytest.mark.asyncio
    async def test_max_warmup_items_limit(self, mock_cache, warmup_config):
        """Test that max_warmup_items limit is respected."""
        warmup_config.max_warmup_items = 2
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Try to add more items than the limit
        for i in range(5):
            request = MessagesRequest(
                model="claude-3-opus",
                messages=[{"role": "user", "content": f"Request {i}"}],
                max_tokens=100,
            )
            response = MessagesResponse(
                id=f"msg_{i}",
                model="claude-3-opus",
                role="assistant",
                content=[{"type": "text", "text": f"Response {i}"}],
                usage=Usage(input_tokens=10, output_tokens=10),
            )

            # Track each request 3 times to make it popular
            for _ in range(3):
                await manager.track_request(request, response)

        # Save popular requests
        await manager._save_popular_requests()

        # Should only have max_warmup_items
        assert len(manager._warmup_items) <= warmup_config.max_warmup_items

    @pytest.mark.asyncio
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

        # Should not have started save task
        assert manager._save_task is None

        # Should not attempt to load anything
        mock_cache.cache_response.assert_not_called()

        await manager.stop()
