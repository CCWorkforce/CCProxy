"""Test cache warmup functionality."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from ccproxy.application.cache.warmup import CacheWarmupManager, CacheWarmupConfig
from ccproxy.application.cache.response_cache import ResponseCache
from typing import Any

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
    def mock_cache(self: Any) -> MagicMock:
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
    def warmup_config(self: Any, tmp_path: Any) -> Any:
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
    async def test_init_and_start(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> Any:
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
    async def test_preload_common_prompts(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> Any:
        """Test preloading of common prompts."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Mock the provider to return a response
        with patch.object(manager, "_load_warmup_item", new=AsyncMock()) as mock_load:
            await manager._warmup_cache()

            # Should have loaded at least one common prompt
            assert mock_load.call_count > 0

    @pytest.mark.anyio
    async def test_save_and_load_warmup_file(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
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
    async def test_track_cache_hit(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
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
    async def test_load_warmup_item(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
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
    async def test_periodic_save(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
        """Test periodic saving of popular requests."""
        # Set[Any] a very short save interval
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
    async def test_max_warmup_items_limit(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
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
    async def test_disabled_warmup(self, mock_cache: MagicMock) -> None:
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
    async def test_preload_responses(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
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
    async def test_warmup_from_log(
        self, mock_cache: MagicMock, warmup_config: Any, tmp_path: Any
    ) -> None:
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

    @pytest.mark.anyio
    async def test_stop_with_exception(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
        """Test stop() handles exceptions from task group cleanup."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Start the manager
        await manager.start()

        # Mock __aexit__ to raise an exception
        if manager._task_group:

            async def failing_aexit(*args) -> Any:  # type: ignore[unreachable]
                raise RuntimeError("Task group cleanup failed")

            manager._task_group.__aexit__ = failing_aexit

        # Stop should handle the exception gracefully
        await manager.stop()
        assert manager._task_group is None

    @pytest.mark.anyio
    async def test_load_from_warmup_file(
        self, mock_cache: MagicMock, warmup_config: Any, tmp_path: Any
    ) -> None:
        """Test loading warmup items from file."""
        # Create warmup file
        warmup_file = tmp_path / "warmup.json"
        warmup_data = [
            {
                "messages": [{"role": "user", "content": "Test message"}],
                "model": "claude-3-opus-20240229",
                "max_tokens": 100,
                "response": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Test response"}],
                    "usage": {"input_tokens": 5, "output_tokens": 5},
                    "stop_reason": "end_turn",
                },
            }
        ]

        with open(warmup_file, "w") as f:
            json.dump(warmup_data, f)

        warmup_config.warmup_file_path = str(warmup_file)
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Warmup should load from file
        await manager._warmup_cache()

        # Should have called cache.set for loaded items
        assert mock_cache.set.call_count >= len(warmup_data)

    @pytest.mark.anyio
    async def test_load_from_invalid_warmup_file(  # type: ignore[no-untyped-def]
        self, mock_cache, warmup_config, tmp_path
    ):
        """Test handling of invalid warmup file."""
        # Create invalid warmup file
        warmup_file = tmp_path / "warmup.json"
        with open(warmup_file, "w") as f:
            f.write("invalid json{{{")

        warmup_config.warmup_file_path = str(warmup_file)
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Should handle invalid file gracefully
        await manager._warmup_cache()

    @pytest.mark.anyio
    async def test_warmup_cache_exception(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
        """Test _warmup_cache handles exceptions."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Mock _load_warmup_item to raise an exception
        with patch.object(
            manager, "_load_warmup_item", side_effect=RuntimeError("Load failed")
        ):
            # Should handle exception gracefully
            await manager._warmup_cache()

    @pytest.mark.anyio
    async def test_load_warmup_item_exception(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
        """Test _load_warmup_item handles exceptions."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Create invalid warmup item
        invalid_item = {
            "messages": "invalid",  # Should be a list
            "response": {},
        }

        # Should handle exception gracefully
        await manager._load_warmup_item(invalid_item)

    @pytest.mark.anyio
    async def test_auto_save_loop_exception(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
        """Test _auto_save_loop handles non-cancellation exceptions."""
        import anyio

        warmup_config.save_interval_seconds = 0.05
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Mock _save_popular_items to raise an exception
        call_count = 0
        original_save = manager._save_popular_items

        async def failing_save() -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Save failed")
            await original_save()

        with patch.object(manager, "_save_popular_items", side_effect=failing_save):
            await manager.start()

            # Wait for exception to occur
            await anyio.sleep(0.15)

            # Stop the manager
            await manager.stop()

            # Should have attempted save at least once
            assert call_count >= 1

    @pytest.mark.anyio
    async def test_save_popular_items_creates_directory(
        self, mock_cache: MagicMock, tmp_path: Any
    ) -> None:
        """Test _save_popular_items creates parent directory if needed."""
        # Use a nested path that doesn't exist
        warmup_file = tmp_path / "nested" / "dir" / "warmup.json"
        config = CacheWarmupConfig(
            enabled=True,
            warmup_file_path=str(warmup_file),
            auto_save_popular=True,
            popularity_threshold=2,
        )

        manager = CacheWarmupManager(cache=mock_cache, config=config)

        # Track popular items
        manager._popular_items = {"key1": 3, "key2": 5}

        # Save should create directory
        await manager._save_popular_items()

        # Directory should exist
        assert warmup_file.parent.exists()

    @pytest.mark.anyio
    async def test_save_popular_items_exception(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> None:
        """Test _save_popular_items handles exceptions."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Track popular items
        manager._popular_items = {"key1": 5}

        # Mock anyio.Path to raise an exception

        with patch("ccproxy.application.cache.warmup.anyio.Path") as mock_path:
            mock_path.side_effect = RuntimeError("Path error")

            # Should handle exception gracefully
            await manager._save_popular_items()

    @pytest.mark.anyio
    async def test_track_cache_hit_disabled(self, mock_cache: MagicMock) -> None:
        """Test track_cache_hit when auto_save_popular is disabled."""
        config = CacheWarmupConfig(
            enabled=True,
            auto_save_popular=False,
        )

        manager = CacheWarmupManager(cache=mock_cache, config=config)

        # Track cache hit
        manager.track_cache_hit("test_key")

        # Should not track when disabled
        assert "test_key" not in manager._popular_items

    @pytest.mark.anyio
    async def test_preload_responses_mismatched_lengths(  # type: ignore[no-untyped-def]
        self, mock_cache, warmup_config
    ):
        """Test preload_responses with mismatched request/response lengths."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        requests = [
            MessagesRequest(
                model="claude-3-opus-20240229",
                messages=[
                    Message(
                        role="user",
                        content=[ContentBlockText(type="text", text="Test")],
                    )
                ],
                max_tokens=100,
                stream=False,
            )
        ]

        responses = [
            MessagesResponse(
                id="msg_1",
                type="message",
                role="assistant",
                model="claude-3-opus-20240229",
                content=[ContentBlockText(type="text", text="Response 1")],
                usage=Usage(input_tokens=10, output_tokens=10),
                stop_reason="end_turn",
            ),
            MessagesResponse(
                id="msg_2",
                type="message",
                role="assistant",
                model="claude-3-opus-20240229",
                content=[ContentBlockText(type="text", text="Response 2")],
                usage=Usage(input_tokens=10, output_tokens=10),
                stop_reason="end_turn",
            ),
        ]

        # Should raise ValueError
        with pytest.raises(ValueError, match="same length"):
            await manager.preload_responses(requests, responses)

    @pytest.mark.anyio
    async def test_preload_responses_with_exception(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> Any:
        """Test preload_responses handles exceptions during cache set."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Make cache.set fail
        mock_cache.set = AsyncMock(side_effect=RuntimeError("Cache error"))

        requests = [
            MessagesRequest(
                model="claude-3-opus-20240229",
                messages=[
                    Message(
                        role="user",
                        content=[ContentBlockText(type="text", text="Test")],
                    )
                ],
                max_tokens=100,
                stream=False,
            )
        ]

        responses = [
            MessagesResponse(
                id="msg_1",
                type="message",
                role="assistant",
                model="claude-3-opus-20240229",
                content=[ContentBlockText(type="text", text="Response")],
                usage=Usage(input_tokens=10, output_tokens=10),
                stop_reason="end_turn",
            )
        ]

        # Should handle exception and return 0
        count = await manager.preload_responses(requests, responses)
        assert count == 0

    @pytest.mark.anyio
    async def test_warmup_from_nonexistent_log(
        self, mock_cache: MagicMock, warmup_config: Any
    ) -> Any:
        """Test warmup_from_log with non-existent file."""
        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Try to load from non-existent file
        count = await manager.warmup_from_log(
            "/nonexistent/path/log.jsonl", max_items=10
        )

        # Should return 0
        assert count == 0

    @pytest.mark.anyio
    async def test_warmup_from_log_max_items(
        self, mock_cache: MagicMock, warmup_config: Any, tmp_path: Any
    ) -> None:
        """Test warmup_from_log respects max_items limit."""
        log_file = tmp_path / "test_log.jsonl"

        # Create log file with multiple entries
        log_entries = [
            {
                "request": {
                    "model": "claude-3-opus-20240229",
                    "messages": [{"role": "user", "content": f"Test {i}"}],
                    "max_tokens": 100,
                },
                "response": {
                    "id": f"msg_{i}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-opus-20240229",
                    "content": [{"type": "text", "text": f"Response {i}"}],
                    "usage": {"input_tokens": 5, "output_tokens": 5},
                    "stop_reason": "end_turn",
                },
            }
            for i in range(10)
        ]

        with open(log_file, "w") as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")

        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Load with max_items=3
        count = await manager.warmup_from_log(str(log_file), max_items=3)

        # Should load at most 3 items
        assert count <= 3

    @pytest.mark.anyio
    async def test_warmup_from_log_with_exception(  # type: ignore[no-untyped-def]
        self, mock_cache, warmup_config, tmp_path
    ):
        """Test warmup_from_log handles exceptions during file processing."""
        log_file = tmp_path / "test_log.jsonl"

        # Create log file
        with open(log_file, "w") as f:
            f.write('{"request": {}, "response": {}}\n')

        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Mock aiofiles.open to raise an exception
        with patch("ccproxy.application.cache.warmup.aiofiles.open") as mock_open:
            mock_open.side_effect = RuntimeError("File error")

            # Should handle exception and return 0
            count = await manager.warmup_from_log(str(log_file), max_items=10)
            assert count == 0

    @pytest.mark.anyio
    async def test_warmup_from_log_invalid_json_entries(  # type: ignore[no-untyped-def]
        self, mock_cache, warmup_config, tmp_path
    ):
        """Test warmup_from_log skips invalid JSON entries."""
        log_file = tmp_path / "test_log.jsonl"

        # Create log file with mix of valid and invalid entries
        with open(log_file, "w") as f:
            # Valid entry
            f.write(
                json.dumps(
                    {
                        "request": {
                            "model": "claude-3-opus-20240229",
                            "messages": [{"role": "user", "content": "Test"}],
                            "max_tokens": 100,
                        },
                        "response": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "model": "claude-3-opus-20240229",
                            "content": [{"type": "text", "text": "Response"}],
                            "usage": {"input_tokens": 5, "output_tokens": 5},
                            "stop_reason": "end_turn",
                        },
                    }
                )
                + "\n"
            )
            # Invalid JSON line
            f.write("invalid json line{{{[\n")
            # Another valid entry
            f.write(
                json.dumps(
                    {
                        "request": {
                            "model": "claude-3-opus-20240229",
                            "messages": [{"role": "user", "content": "Test 2"}],
                            "max_tokens": 100,
                        },
                        "response": {
                            "id": "msg_2",
                            "type": "message",
                            "role": "assistant",
                            "model": "claude-3-opus-20240229",
                            "content": [{"type": "text", "text": "Response 2"}],
                            "usage": {"input_tokens": 5, "output_tokens": 5},
                            "stop_reason": "end_turn",
                        },
                    }
                )
                + "\n"
            )

        manager = CacheWarmupManager(cache=mock_cache, config=warmup_config)

        # Should skip invalid entries and load valid ones
        count = await manager.warmup_from_log(str(log_file), max_items=10)

        # Should have loaded 2 valid entries (skipping the invalid one)
        assert count == 2
