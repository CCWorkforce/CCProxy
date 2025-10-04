import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from ccproxy.infrastructure.providers.request_lifecycle_observer import (
    RequestLifecycleObserver,
)
from ccproxy.monitoring import PerformanceMonitor
from ccproxy.application.error_tracker import ErrorTracker, ErrorType
from ccproxy.infrastructure.providers.resilience import CircuitBreaker
from ccproxy.infrastructure.providers.metrics import MetricsCollector
from ccproxy.infrastructure.providers.response_handlers import ResponseProcessor
from ccproxy.infrastructure.providers.request_logger import (
    RequestLogger,
    PerformanceTracker,
)


@pytest.fixture
def mock_components():
    return {
        "performance_monitor": Mock(spec=PerformanceMonitor),
        "performance_tracker": Mock(spec=PerformanceTracker),
        "metrics_collector": AsyncMock(spec=MetricsCollector),
        "error_tracker": AsyncMock(spec=ErrorTracker),
        "circuit_breaker": Mock(spec=CircuitBreaker),
        "request_logger": Mock(spec=RequestLogger),
        "response_processor": Mock(spec=ResponseProcessor),
    }


def test_init(mock_components):
    observer = RequestLifecycleObserver(**mock_components)

    assert observer._performance_monitor == mock_components["performance_monitor"]
    assert observer._performance_tracker == mock_components["performance_tracker"]
    assert observer._metrics_collector == mock_components["metrics_collector"]
    assert observer._error_tracker == mock_components["error_tracker"]
    assert observer._circuit_breaker == mock_components["circuit_breaker"]
    assert observer._request_logger == mock_components["request_logger"]
    assert observer._response_processor == mock_components["response_processor"]


@pytest.mark.asyncio
async def test_on_success(mock_components, mocker):
    mocker.patch("time.monotonic", return_value=1.0)
    correlation_id = "test-id"
    request_start = time.monotonic()
    response = {"usage": {"total_tokens": 100}}

    mock_components["response_processor"].extract_usage_info.return_value = {
        "total_tokens": 100
    }
    mock_components["metrics_collector"].record_success = AsyncMock()
    mock_components["metrics_collector"].update_circuit_state = AsyncMock()
    mock_components["request_logger"].log_response = Mock()
    mock_components["performance_monitor"].end_request = AsyncMock()
    mock_components["performance_tracker"].end_request = AsyncMock()

    observer = RequestLifecycleObserver(**mock_components)
    await observer.on_success(correlation_id, request_start, response)

    latency_ms = (time.monotonic() - request_start) * 1000
    mock_components["performance_monitor"].end_request.assert_called_once_with(
        correlation_id, success=True
    )
    mock_components["performance_tracker"].end_request.assert_called_once_with(
        correlation_id
    )
    mock_components["response_processor"].extract_usage_info.assert_called_once_with(
        response
    )
    await mock_components["metrics_collector"].record_success.assert_called_once_with(
        latency_ms, 100
    )
    await mock_components[
        "metrics_collector"
    ].update_circuit_state.assert_called_once_with(
        mock_components["circuit_breaker"].state
    )
    mock_components["request_logger"].log_response.assert_called_once_with(
        correlation_id=correlation_id,
        latency_ms=latency_ms,
        success=True,
        response=response,
    )


@pytest.mark.asyncio
async def test_on_failure(mock_components, mocker):
    mocker.patch("time.monotonic", return_value=1.0)
    correlation_id = "test-id"
    request_start = time.monotonic()
    error = Exception("Test error")
    error_type = ErrorType.API_ERROR

    mock_components["error_tracker"].track_error = AsyncMock()
    mock_components["metrics_collector"].record_failure = AsyncMock()
    mock_components["metrics_collector"].update_circuit_state = AsyncMock()
    mock_components["request_logger"].log_response = Mock()
    mock_components["performance_monitor"].end_request = AsyncMock()
    mock_components["performance_tracker"].end_request = AsyncMock()

    observer = RequestLifecycleObserver(**mock_components)
    await observer.on_failure(correlation_id, request_start, error, error_type)

    latency_ms = (time.monotonic() - request_start) * 1000
    mock_components["performance_monitor"].end_request.assert_called_once_with(
        correlation_id, success=False
    )
    mock_components["performance_tracker"].end_request.assert_called_once_with(
        correlation_id
    )
    await mock_components["error_tracker"].track_error.assert_called_once_with(
        error=error,
        error_type=error_type,
        request_id=correlation_id,
    )
    await mock_components["metrics_collector"].record_failure.assert_called_once_with(
        latency_ms, mock_components["circuit_breaker"].consecutive_failures
    )
    await mock_components[
        "metrics_collector"
    ].update_circuit_state.assert_called_once_with(
        mock_components["circuit_breaker"].state
    )
    mock_components["request_logger"].log_response.assert_called_once_with(
        correlation_id=correlation_id,
        latency_ms=latency_ms,
        success=False,
        error=error,
    )


@pytest.mark.asyncio
async def test_on_request_start(mock_components):
    correlation_id = "test-id"
    params = {"model": "gpt-4"}
    trace_id = "trace-123"

    mock_components["performance_monitor"].start_request = AsyncMock()
    mock_components["performance_tracker"].start_request = AsyncMock()

    observer = RequestLifecycleObserver(**mock_components)
    await observer.on_request_start(correlation_id, params, trace_id)

    mock_components["request_logger"].log_request.assert_called_once_with(
        correlation_id, params, trace_id
    )
    await mock_components["performance_monitor"].start_request.assert_called_once_with(
        correlation_id
    )
    await mock_components["performance_tracker"].start_request.assert_called_once_with(
        correlation_id
    )


@patch(
    "ccproxy.infrastructure.providers.request_lifecycle_observer.ErrorResponseHandler"
)
def test_classify_error(mock_error_handler_class, mock_components):
    mock_handler = Mock()
    mock_error_handler_class.return_value = mock_handler
    mock_handler.classify_error.return_value = "timeout_error"
    error = Exception("Test")

    observer = RequestLifecycleObserver(**mock_components)
    result = observer.classify_error(error)

    mock_error_handler_class.assert_called_once()
    mock_handler.classify_error.assert_called_once_with(error)
    assert result == ErrorType.TIMEOUT_ERROR

    # Test default
    mock_handler.classify_error.return_value = "unknown"
    result = observer.classify_error(error)
    assert result == ErrorType.API_ERROR

    # Test mapping edge case
    mock_handler.classify_error.return_value = "conversion_error"
    result = observer.classify_error(error)
    assert result == ErrorType.CONVERSION_ERROR


@pytest.mark.asyncio
async def test_concurrent_on_success_calls(mock_components):
    """Test concurrent on_success calls with unique correlation IDs."""
    observer = RequestLifecycleObserver(**mock_components)

    # Setup unique latency tracking
    latencies = []

    async def track_latency(*args, **kwargs):
        # Extract latency from the call
        if args:
            latencies.append(args[0])

    mock_components["metrics_collector"].record_success.side_effect = track_latency
    mock_components["response_processor"].extract_usage_info.side_effect = lambda r: {
        "total_tokens": r.get("usage", {}).get("total_tokens", 100)
    }

    # Create unique correlation IDs and responses
    correlation_ids = [f"test-id-{i}" for i in range(5)]
    responses = [{"usage": {"total_tokens": 100 + i * 10}} for i in range(5)]

    # Run concurrent requests
    tasks = [
        observer.on_success(cid, time.monotonic(), response)
        for cid, response in zip(correlation_ids, responses)
    ]

    await asyncio.gather(*tasks)

    # Verify all calls were made with unique correlation IDs
    assert mock_components["performance_monitor"].end_request.call_count == 5
    assert mock_components["performance_tracker"].end_request.call_count == 5
    assert mock_components["metrics_collector"].record_success.call_count == 5

    # Verify no shared state overlap
    called_ids = [
        call[0][0]
        for call in mock_components["performance_monitor"].end_request.call_args_list
    ]
    assert len(set(called_ids)) == 5  # All unique
    assert set(called_ids) == set(correlation_ids)

    # Verify latencies are positive and unique
    assert len(latencies) == 5
    assert all(lat > 0 for lat in latencies)


@pytest.mark.asyncio
async def test_concurrent_mixed_success_failure(mock_components):
    """Test concurrent mix of success and failure calls."""
    observer = RequestLifecycleObserver(**mock_components)

    # Track error types
    tracked_errors = []

    async def track_error(*args, **kwargs):
        if "error_type" in kwargs:
            tracked_errors.append(kwargs["error_type"])

    mock_components["error_tracker"].track_error.side_effect = track_error
    mock_components["response_processor"].extract_usage_info.return_value = {
        "total_tokens": 100
    }

    # Create mix of success and failure tasks
    success_ids = [f"success-{i}" for i in range(3)]
    failure_ids = [f"failure-{i}" for i in range(2)]

    success_tasks = [
        observer.on_success(cid, time.monotonic(), {"usage": {"total_tokens": 100}})
        for cid in success_ids
    ]

    failure_tasks = [
        observer.on_failure(
            cid,
            time.monotonic(),
            Exception(f"Error {i}"),
            ErrorType.API_ERROR if i == 0 else ErrorType.TIMEOUT_ERROR
        )
        for i, cid in enumerate(failure_ids)
    ]

    # Run all tasks concurrently
    await asyncio.gather(*(success_tasks + failure_tasks))

    # Verify counts
    assert mock_components["metrics_collector"].record_success.call_count == 3
    assert mock_components["metrics_collector"].record_failure.call_count == 2
    assert mock_components["error_tracker"].track_error.call_count == 2

    # Verify error types were tracked correctly
    assert ErrorType.API_ERROR in tracked_errors
    assert ErrorType.TIMEOUT_ERROR in tracked_errors


@pytest.mark.asyncio
async def test_concurrent_request_starts(mock_components):
    """Test concurrent on_request_start calls with unique trace IDs."""
    observer = RequestLifecycleObserver(**mock_components)

    # Create unique request data
    requests = [
        {
            "correlation_id": f"req-{i}",
            "params": {"model": f"model-{i}"},
            "trace_id": f"trace-{i}"
        }
        for i in range(5)
    ]

    # Run concurrent request starts
    tasks = [
        observer.on_request_start(
            req["correlation_id"],
            req["params"],
            req["trace_id"]
        )
        for req in requests
    ]

    await asyncio.gather(*tasks)

    # Verify all calls were made
    assert mock_components["request_logger"].log_request.call_count == 5
    assert mock_components["performance_monitor"].start_request.call_count == 5
    assert mock_components["performance_tracker"].start_request.call_count == 5

    # Verify unique trace IDs were used
    logged_traces = [
        call[0][2]  # Third argument is trace_id
        for call in mock_components["request_logger"].log_request.call_args_list
    ]
    assert len(set(logged_traces)) == 5
    assert all(f"trace-{i}" in logged_traces for i in range(5))


@pytest.mark.asyncio
async def test_concurrent_with_varying_error_types(mock_components):
    """Test concurrent failures with different error types via classify_error."""
    observer = RequestLifecycleObserver(**mock_components)

    # Setup error classification
    with patch.object(observer, 'classify_error') as mock_classify:
        error_types = [
            ErrorType.API_ERROR,
            ErrorType.TIMEOUT_ERROR,
            ErrorType.CONVERSION_ERROR,
            ErrorType.RATE_LIMIT,
            ErrorType.NETWORK_ERROR
        ]

        # Return different error types for different calls
        mock_classify.side_effect = error_types

        # Create failure tasks with different errors
        tasks = [
            observer.on_failure(
                f"fail-{i}",
                time.monotonic(),
                Exception(f"Error {i}"),
                error_type
            )
            for i, error_type in enumerate(error_types)
        ]

        await asyncio.gather(*tasks)

        # Verify all failures were recorded
        assert mock_components["metrics_collector"].record_failure.call_count == 5
        assert mock_components["error_tracker"].track_error.call_count == 5

        # Verify error types were passed correctly
        error_type_calls = [
            call[1]["error_type"]
            for call in mock_components["error_tracker"].track_error.call_args_list
        ]
        assert set(error_type_calls) == set(error_types)
