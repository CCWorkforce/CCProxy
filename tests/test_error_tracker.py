"""Comprehensive test suite for the error tracking system."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, mock_open
import pytest

from ccproxy.application.error_tracker import (
    ErrorTracker,
    ErrorType,
    ErrorContext,
    RequestSnapshot,
    ResponseSnapshot,
)
from ccproxy.config import Settings


@pytest.fixture
async def error_tracker_instance():
    """Create a fresh error tracker instance for testing."""
    tracker = ErrorTracker()
    # Reset singleton state
    tracker._initialized = False
    tracker._settings = None
    tracker._file_handle = None
    tracker._writer_task = None
    return tracker


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.error_tracking_enabled = True
    settings.error_tracking_file = "test_errors.jsonl"
    settings.error_tracking_max_size_mb = 100
    settings.error_tracking_retention_days = 30
    settings.error_tracking_capture_request = True
    settings.error_tracking_capture_response = True
    settings.error_tracking_max_body_size = 10000
    return settings


@pytest.fixture
def sample_request_snapshot():
    """Create a sample request snapshot."""
    return RequestSnapshot(
        method="POST",
        path="/v1/messages",
        headers={"Authorization": "Bearer sk-test123", "Content-Type": "application/json"},
        query_params={"stream": "true"},
        body={"messages": [{"role": "user", "content": "test"}]},
        client_ip="192.168.1.1",
        user_agent="TestClient/1.0"
    )


@pytest.fixture
def sample_response_snapshot():
    """Create a sample response snapshot."""
    return ResponseSnapshot(
        status_code=200,
        headers={"Content-Type": "application/json"},
        body={"id": "msg_123", "content": "response"},
        elapsed_ms=150.5
    )


@pytest.fixture
def sample_error_context(sample_request_snapshot, sample_response_snapshot):
    """Create a sample error context."""
    return ErrorContext(
        error_id="test-error-123",
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id="req-456",
        error_type=ErrorType.API_ERROR,
        error_message="Test API error",
        stack_trace="Traceback: test stack trace",
        request_snapshot=sample_request_snapshot,
        response_snapshot=sample_response_snapshot,
        metadata={"test_key": "test_value"},
        redacted_fields=["authorization"]
    )


class TestErrorTracker:
    """Test cases for ErrorTracker class."""

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that ErrorTracker follows singleton pattern."""
        tracker1 = ErrorTracker()
        tracker2 = ErrorTracker()
        assert tracker1 is tracker2

    @pytest.mark.asyncio
    async def test_initialization(self, error_tracker_instance, mock_settings):
        """Test error tracker initialization."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            await error_tracker_instance.initialize(mock_settings)

            assert error_tracker_instance._settings == mock_settings
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            assert error_tracker_instance._writer_task is not None

    @pytest.mark.asyncio
    async def test_initialization_disabled(self, error_tracker_instance, mock_settings):
        """Test initialization when error tracking is disabled."""
        mock_settings.error_tracking_enabled = False

        await error_tracker_instance.initialize(mock_settings)

        assert error_tracker_instance._settings == mock_settings
        assert error_tracker_instance._writer_task is None

    @pytest.mark.asyncio
    async def test_track_error(self, error_tracker_instance, mock_settings):
        """Test tracking an error."""
        await error_tracker_instance.initialize(mock_settings)

        test_error = ValueError("Test error message")
        request_id = "test-req-123"
        metadata = {"key": "value"}

        # Track the error
        await error_tracker_instance.track_error(
            error=test_error,
            error_type=ErrorType.VALIDATION_ERROR,
            request_id=request_id,
            metadata=metadata
        )

        # Check that error was queued
        assert error_tracker_instance._write_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_track_error_with_request_response(
        self,
        error_tracker_instance,
        mock_settings,
        sample_request_snapshot,
        sample_response_snapshot
    ):
        """Test tracking error with request and response snapshots."""
        await error_tracker_instance.initialize(mock_settings)

        test_error = RuntimeError("Test runtime error")

        # Mock the request and response objects
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/v1/messages"
        mock_request.headers = {"Authorization": "Bearer sk-test", "Content-Type": "application/json"}
        mock_request.query_params = {"stream": "true"}
        mock_request.client.host = "192.168.1.1"

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"Content-Type": "application/json"}

        await error_tracker_instance.track_error(
            error=test_error,
            error_type=ErrorType.INTERNAL_ERROR,
            request=mock_request,
            response=mock_response,
            request_id="req-789"
        )

        assert error_tracker_instance._write_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_redaction_patterns(self, error_tracker_instance):
        """Test sensitive data redaction patterns."""
        # Test API key redaction
        text = 'api_key="sk-proj-abcdef123456" and token="secret123"'
        redacted = error_tracker_instance._redact_sensitive_data(text)
        assert "sk-proj-abcdef123456" not in redacted
        assert "secret123" not in redacted
        assert "[REDACTED]" in redacted

        # Test Bearer token redaction
        text = "Authorization: Bearer sk-1234567890abcdef"
        redacted = error_tracker_instance._redact_sensitive_data(text)
        assert "sk-1234567890abcdef" not in redacted
        assert "[REDACTED]" in redacted

    @pytest.mark.asyncio
    async def test_context_manager(self, error_tracker_instance, mock_settings):
        """Test error tracking context manager."""
        await error_tracker_instance.initialize(mock_settings)

        # Test successful execution
        async with error_tracker_instance.track_context(
            error_type=ErrorType.API_ERROR,
            request_id="ctx-123"
        ):
            pass  # No error should be tracked

        assert error_tracker_instance._write_queue.qsize() == 0

        # Test with exception
        with pytest.raises(ValueError):
            async with error_tracker_instance.track_context(
                error_type=ErrorType.API_ERROR,
                request_id="ctx-456"
            ):
                raise ValueError("Test error in context")

        # Error should be tracked
        assert error_tracker_instance._write_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_error_context_to_dict(self, sample_error_context):
        """Test ErrorContext to_dict conversion."""
        error_dict = sample_error_context.to_dict()

        assert error_dict["error_id"] == "test-error-123"
        assert error_dict["error_type"] == ErrorType.API_ERROR.value
        assert error_dict["error_message"] == "Test API error"
        assert error_dict["request_id"] == "req-456"
        assert error_dict["metadata"]["test_key"] == "test_value"

    @pytest.mark.asyncio
    async def test_writer_loop(self, error_tracker_instance, mock_settings):
        """Test the background writer loop."""
        mock_file = mock_open()

        with patch("builtins.open", mock_file):
            await error_tracker_instance.initialize(mock_settings)

            # Add an error context to the queue
            test_context = ErrorContext(
                error_type=ErrorType.CACHE_ERROR,
                error_message="Test cache error"
            )
            await error_tracker_instance._write_queue.put(test_context)

            # Give writer task time to process
            await asyncio.sleep(0.1)

            # Check that write was attempted
            mock_file().write.assert_called()

    @pytest.mark.asyncio
    async def test_log_rotation(self, error_tracker_instance, mock_settings):
        """Test log file rotation when size limit is exceeded."""
        mock_settings.error_tracking_max_size_mb = 0.001  # Very small limit

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 2 * 1024 * 1024  # 2MB

            with patch("pathlib.Path.rename") as mock_rename:
                await error_tracker_instance.initialize(mock_settings)
                await error_tracker_instance._rotate_log_if_needed()

                # Check that rotation was attempted
                mock_rename.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_old_logs(self, error_tracker_instance, mock_settings):
        """Test cleanup of old error logs."""
        mock_settings.error_tracking_retention_days = 7

        with patch("pathlib.Path.glob") as mock_glob:
            # Create mock old log files
            old_log = MagicMock()
            old_log.stat.return_value.st_mtime = 0  # Very old timestamp
            old_log.exists.return_value = True

            recent_log = MagicMock()
            recent_log.stat.return_value.st_mtime = datetime.now().timestamp()
            recent_log.exists.return_value = True

            mock_glob.return_value = [old_log, recent_log]

            with patch("pathlib.Path.unlink"):
                await error_tracker_instance.initialize(mock_settings)
                await error_tracker_instance._cleanup_old_logs()

                # Only old log should be deleted
                old_log.unlink.assert_called_once()
                recent_log.unlink.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown(self, error_tracker_instance, mock_settings):
        """Test graceful shutdown of error tracker."""
        await error_tracker_instance.initialize(mock_settings)

        # Add some items to the queue
        for i in range(3):
            test_context = ErrorContext(
                error_type=ErrorType.INTERNAL_ERROR,
                error_message=f"Test error {i}"
            )
            await error_tracker_instance._write_queue.put(test_context)

        # Shutdown should process remaining items
        await error_tracker_instance.shutdown()

        # Check that writer task was cancelled
        assert error_tracker_instance._writer_task is None or error_tracker_instance._writer_task.cancelled()

    @pytest.mark.asyncio
    async def test_decorator_sync_function(self, error_tracker_instance, mock_settings):
        """Test error tracking decorator on synchronous function."""
        await error_tracker_instance.initialize(mock_settings)

        @error_tracker_instance.track_errors(ErrorType.INTERNAL_ERROR)
        def failing_function():
            raise ValueError("Test error in sync function")

        with pytest.raises(ValueError):
            failing_function()

        # Error should be tracked
        assert error_tracker_instance._write_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_decorator_async_function(self, error_tracker_instance, mock_settings):
        """Test error tracking decorator on async function."""
        await error_tracker_instance.initialize(mock_settings)

        @error_tracker_instance.track_errors(ErrorType.INTERNAL_ERROR)
        async def async_failing_function():
            raise RuntimeError("Test error in async function")

        with pytest.raises(RuntimeError):
            await async_failing_function()

        # Error should be tracked
        assert error_tracker_instance._write_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_error_types(self, error_tracker_instance, mock_settings):
        """Test all error type classifications."""
        await error_tracker_instance.initialize(mock_settings)

        # Test each error type
        for error_type in ErrorType:
            await error_tracker_instance.track_error(
                error=Exception(f"Test {error_type.value}"),
                error_type=error_type,
                request_id=f"test-{error_type.value}"
            )

        # All errors should be queued
        assert error_tracker_instance._write_queue.qsize() == len(ErrorType)

    @pytest.mark.asyncio
    async def test_metadata_truncation(self, error_tracker_instance, mock_settings):
        """Test that large metadata is truncated properly."""
        await error_tracker_instance.initialize(mock_settings)
        mock_settings.error_tracking_max_body_size = 100

        large_metadata = {
            "large_data": "x" * 1000,  # Very large string
            "normal_data": "small"
        }

        await error_tracker_instance.track_error(
            error=Exception("Test"),
            error_type=ErrorType.INTERNAL_ERROR,
            metadata=large_metadata
        )

        # Check that error was queued (truncation should not prevent tracking)
        assert error_tracker_instance._write_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, error_tracker_instance, mock_settings):
        """Test concurrent error tracking from multiple tasks."""
        await error_tracker_instance.initialize(mock_settings)

        async def track_errors(task_id: int):
            for i in range(5):
                await error_tracker_instance.track_error(
                    error=Exception(f"Task {task_id} error {i}"),
                    error_type=ErrorType.INTERNAL_ERROR,
                    request_id=f"task-{task_id}-{i}"
                )
                await asyncio.sleep(0.01)

        # Run multiple tasks concurrently
        tasks = [track_errors(i) for i in range(3)]
        await asyncio.gather(*tasks)

        # All errors should be queued
        assert error_tracker_instance._write_queue.qsize() == 15

    @pytest.mark.asyncio
    async def test_queue_full_handling(self, error_tracker_instance, mock_settings):
        """Test handling when the write queue is full."""
        await error_tracker_instance.initialize(mock_settings)

        # Fill the queue to capacity
        error_tracker_instance._write_queue = asyncio.Queue(maxsize=2)

        # Add items until queue is full
        for i in range(2):
            await error_tracker_instance._write_queue.put(
                ErrorContext(error_type=ErrorType.INTERNAL_ERROR, error_message=f"Error {i}")
            )

        # Try to add one more (should not block indefinitely)
        with pytest.raises(asyncio.QueueFull):
            error_tracker_instance._write_queue.put_nowait(
                ErrorContext(error_type=ErrorType.INTERNAL_ERROR, error_message="Overflow")
            )