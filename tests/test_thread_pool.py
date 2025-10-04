"""Tests for thread pool management module."""

import json
from unittest.mock import patch, MagicMock
import pytest

from ccproxy.application.thread_pool import (
    initialize_thread_pool,
    get_thread_limiter,
    asyncify,
    get_pool_stats,
)
from ccproxy.config import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.thread_pool_max_workers = 10
    settings.thread_pool_high_cpu_threshold = 80
    settings.thread_pool_auto_scale = False
    return settings


class TestThreadPool:
    """Test cases for thread pool management."""

    @pytest.mark.asyncio
    async def test_asyncify_with_kwargs(self):
        """Test that asyncify correctly handles functions with keyword arguments."""
        # This is a regression test for the bug where kwargs were passed
        # to run_sync() instead of the wrapped function

        def test_func(data, sort_keys=False, separators=None):
            """Test function that mimics json.dumps signature."""
            if sort_keys:
                data = dict(sorted(data.items()))
            if separators:
                sep1, sep2 = separators
                return json.dumps(data, separators=(sep1, sep2))
            return json.dumps(data)

        # Create async version
        async_func = asyncify(test_func)

        # Test with kwargs
        test_data = {"b": 2, "a": 1}
        result = await async_func(test_data, sort_keys=True, separators=(",", ":"))

        # Should produce sorted, compact JSON
        assert result == '{"a":1,"b":2}'

    @pytest.mark.asyncio
    async def test_asyncify_with_positional_args(self):
        """Test asyncify with positional arguments."""

        def add(a, b):
            return a + b

        async_add = asyncify(add)
        result = await async_add(5, 3)
        assert result == 8

    @pytest.mark.asyncio
    async def test_asyncify_with_mixed_args(self):
        """Test asyncify with both positional and keyword arguments."""

        def format_string(template, name, age=0, city="Unknown"):
            return template.format(name=name, age=age, city=city)

        async_format = asyncify(format_string)
        result = await async_format(
            "{name} is {age} years old from {city}", "Alice", age=30, city="NYC"
        )
        assert result == "Alice is 30 years old from NYC"

    def test_initialize_thread_pool(self, mock_settings):
        """Test thread pool initialization."""
        initialize_thread_pool(mock_settings)

        limiter = get_thread_limiter()
        assert limiter is not None

        stats = get_pool_stats()
        assert stats["max_workers"] == 10
        assert stats["cpu_threshold"] == 80

    def test_initialize_with_auto_calculation(self, mock_settings):
        """Test thread pool initialization with auto-calculated values."""
        mock_settings.thread_pool_max_workers = None
        mock_settings.thread_pool_high_cpu_threshold = None

        initialize_thread_pool(mock_settings)

        stats = get_pool_stats()
        # Should auto-calculate based on actual CPU count
        # The formula is min(40, max(4, cpu_count * 5))
        # Since we're not mocking CPU count, we just verify it's set
        assert stats["max_workers"] > 0
        assert stats["cpu_threshold"] > 0  # Should be auto-calculated

    def test_gunicorn_detection(self, mock_settings):
        """Test detection of Gunicorn multi-worker mode."""
        mock_settings.thread_pool_max_workers = None

        with patch.dict(
            "os.environ", {"SERVER_SOFTWARE": "gunicorn/12345", "WEB_CONCURRENCY": "4"}
        ):
            initialize_thread_pool(mock_settings)

            stats = get_pool_stats()
            # With 4 workers, should reduce threads per worker
            # The exact value depends on CPU count, but should be less than single-worker mode
            assert stats["max_workers"] > 0
            assert (
                stats["max_workers"] <= 20
            )  # Max allowed per worker in multi-worker mode

    @pytest.mark.asyncio
    async def test_json_dumps_async_compatibility(self):
        """Test that our asyncify works with json.dumps (the actual use case)."""
        # This test ensures the fix for the kwargs bug works with real json.dumps
        json_dumps_async = asyncify(json.dumps)

        data = {"z": 26, "a": 1, "m": 13}

        # Test with sort_keys
        result = await json_dumps_async(data, sort_keys=True)
        assert result == '{"a": 1, "m": 13, "z": 26}'

        # Test with custom separators
        result = await json_dumps_async(data, separators=(",", ":"))
        assert ":" in result and ", " not in result

        # Test with both
        result = await json_dumps_async(data, sort_keys=True, separators=(",", ":"))
        assert result == '{"a":1,"m":13,"z":26}'
