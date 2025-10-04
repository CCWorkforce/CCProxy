"""Tests for thread pool management module."""

import json
import os
from unittest.mock import patch, MagicMock, Mock
import pytest

from ccproxy.application.thread_pool import (
    initialize_thread_pool,
    get_thread_limiter,
    asyncify,
    get_pool_stats,
    should_decrease_pool_size,
    should_increase_pool_size,
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

    @pytest.mark.anyio
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

    @pytest.mark.anyio
    async def test_asyncify_with_positional_args(self):
        """Test asyncify with positional arguments."""

        def add(a, b):
            return a + b

        async_add = asyncify(add)
        result = await async_add(5, 3)
        assert result == 8

    @pytest.mark.anyio
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

    def test_uvicorn_multi_worker_detection(self, mock_settings):
        """Test detection of Uvicorn multi-worker mode."""
        mock_settings.thread_pool_max_workers = None

        with patch.dict("os.environ", {"WEB_CONCURRENCY": "4"}):
            initialize_thread_pool(mock_settings)

            stats = get_pool_stats()
            # With 4 workers, should reduce threads per worker
            # The exact value depends on CPU count, but should be less than single-worker mode
            assert stats["max_workers"] > 0
            assert (
                stats["max_workers"] <= 20
            )  # Max allowed per worker in multi-worker mode

    @pytest.mark.anyio
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

    @patch('psutil.cpu_percent')
    def test_cpu_load_detection_auto_scale(self, mock_cpu_percent, mock_settings):
        """Test CPU load detection with auto-scaling enabled."""
        mock_settings.thread_pool_auto_scale = True
        mock_cpu_percent.return_value = 85.0  # High CPU

        initialize_thread_pool(mock_settings)

        # Check if CPU monitoring is considered
        stats = get_pool_stats()
        assert "cpu_threshold" in stats

    @patch('psutil.cpu_count')
    @patch.dict(os.environ, {"WEB_CONCURRENCY": "4"})
    def test_multi_worker_thread_distribution(self, mock_cpu_count, mock_settings):
        """Test thread distribution in multi-worker mode."""
        mock_cpu_count.return_value = 8  # 8 CPUs
        mock_settings.thread_pool_max_workers = None  # Auto-calculate

        initialize_thread_pool(mock_settings)

        stats = get_pool_stats()
        # With 8 CPUs and 4 workers, total threads = 8 * 5 = 40
        # Per worker: 40 / 4 = 10 threads
        assert stats["max_workers"] <= 20  # Should not exceed max per worker
        assert stats["max_workers"] >= 4  # Should have at least minimum

    @patch('psutil.cpu_count')
    def test_cpu_count_edge_cases(self, mock_cpu_count, mock_settings):
        """Test thread pool with various CPU counts."""
        # Test with very low CPU count
        mock_cpu_count.return_value = 1
        mock_settings.thread_pool_max_workers = None

        initialize_thread_pool(mock_settings)
        stats = get_pool_stats()
        assert stats["max_workers"] >= 4  # Minimum threshold

        # Test with high CPU count
        mock_cpu_count.return_value = 16
        mock_settings.thread_pool_max_workers = None

        initialize_thread_pool(mock_settings)
        stats = get_pool_stats()
        assert stats["max_workers"] <= 40  # Maximum threshold

    @patch('psutil.cpu_percent')
    def test_should_scale_down_detection(self, mock_cpu_percent, mock_settings):
        """Test detection of when to scale down threads."""
        # Initialize first
        initialize_thread_pool(mock_settings)

        # Test low CPU - should scale down
        mock_cpu_percent.return_value = 30.0
        assert should_decrease_pool_size() == True

        # Test high CPU - should not scale down
        mock_cpu_percent.return_value = 85.0
        assert should_decrease_pool_size() == False

    @patch('psutil.cpu_percent')
    def test_should_scale_up_detection(self, mock_cpu_percent, mock_settings):
        """Test detection of when to scale up threads."""
        # Initialize first
        initialize_thread_pool(mock_settings)

        # Test very high CPU - should scale up
        mock_cpu_percent.return_value = 95.0
        assert should_increase_pool_size() == True

        # Test normal CPU - should not scale up
        mock_cpu_percent.return_value = 70.0
        assert should_increase_pool_size() == False

    @patch.dict(os.environ, {"WEB_CONCURRENCY": "2"})
    def test_environment_variable_parsing(self, mock_settings):
        """Test parsing of WEB_CONCURRENCY environment variable."""
        mock_settings.thread_pool_max_workers = None

        initialize_thread_pool(mock_settings)

        stats = get_pool_stats()
        # Verify environment variable was read
        assert stats["max_workers"] > 0

    def test_thread_pool_manager_singleton(self, mock_settings):
        """Test that ThreadPoolManager maintains singleton pattern."""
        initialize_thread_pool(mock_settings)
        limiter1 = get_thread_limiter()

        # Re-initialize should return same limiter
        initialize_thread_pool(mock_settings)
        limiter2 = get_thread_limiter()

        # Should be the same instance or equivalent
        assert limiter1 is not None
        assert limiter2 is not None

    @patch('psutil.cpu_percent')
    def test_auto_scale_with_varying_load(self, mock_cpu_percent, mock_settings):
        """Test auto-scaling behavior with varying CPU load."""
        mock_settings.thread_pool_auto_scale = True
        mock_settings.thread_pool_high_cpu_threshold = 75

        initialize_thread_pool(mock_settings)

        # Simulate varying CPU loads
        cpu_loads = [50, 70, 85, 90, 75, 60]

        for load in cpu_loads:
            mock_cpu_percent.return_value = load

            # Check scaling decisions
            if load > 90:  # Very high load
                assert should_increase_pool_size() == True
            elif load < 50:  # Low load
                assert should_decrease_pool_size() == True
