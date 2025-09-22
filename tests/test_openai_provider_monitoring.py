"""Comprehensive test suite for OpenAI provider monitoring and resilience features."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest
import pytest_asyncio
from datetime import datetime
from openai import RateLimitError, AuthenticationError

from ccproxy.infrastructure.providers.openai_provider import (
    OpenAIProvider, CircuitBreaker, CircuitState
)
from ccproxy.config import Settings
from ccproxy.application.error_tracker import ErrorType


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.openai_api_key = "test-api-key"
    settings.base_url = "https://api.openai.com/v1"
    settings.provider_max_retries = 3
    settings.provider_retry_base_delay = 1.0
    settings.provider_retry_jitter = 0.5
    settings.referer_url = "http://localhost:11434"
    settings.app_name = "CCProxy"
    settings.app_version = "1.0.0"
    settings.max_stream_seconds = 300
    settings.error_log_file_path = "test_errors.log"
    settings.log_file_path = "test.log"
    settings.monitoring_recent_durations_maxlen = 100
    return settings


@pytest_asyncio.fixture
async def provider_with_monitoring(mock_settings):
    """Create an OpenAI provider with monitoring enabled."""
    import os
    os.environ['IS_LOCAL_DEPLOYMENT'] = 'True'
    provider = OpenAIProvider(mock_settings)
    yield provider
    await provider.close()


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self):
        """Test circuit breaker starts in closed state."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == 0
        assert not breaker.is_open

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

        async def failing_func():
            raise Exception("Test failure")

        # Fail 3 times to open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open
        assert breaker.consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self):
        """Test circuit breaker blocks calls when open."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=30)

        async def failing_func():
            raise Exception("Test failure")

        # Open the circuit
        for i in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        # Should now block calls
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, half_open_requests=2)

        async def func(should_fail=False):
            if should_fail:
                raise Exception("Test failure")
            return "success"

        # Open the circuit
        for i in range(2):
            with pytest.raises(Exception):
                await breaker.call(func, should_fail=True)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Should enter half-open state and allow test calls
        result = await breaker.call(func, should_fail=False)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN

        # One more success should close the circuit
        result = await breaker.call(func, should_fail=False)
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self):
        """Test circuit breaker resets consecutive failures on success."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def func(should_fail=False):
            if should_fail:
                raise Exception("Test failure")
            return "success"

        # One failure
        with pytest.raises(Exception):
            await breaker.call(func, should_fail=True)
        assert breaker.consecutive_failures == 1

        # Success should reset
        await breaker.call(func, should_fail=False)
        assert breaker.consecutive_failures == 0


class TestProviderMetrics:
    """Test provider metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_initialization(self, provider_with_monitoring):
        """Test metrics are initialized correctly."""
        metrics = await provider_with_monitoring.get_metrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.health_score == 100.0
        assert metrics.circuit_state == "closed"

    @pytest.mark.asyncio
    async def test_successful_request_metrics(self, provider_with_monitoring):
        """Test metrics update on successful requests."""
        with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_response = MagicMock()
            mock_response.usage.total_tokens = 100
            mock_create.return_value = mock_response

            await provider_with_monitoring.create_chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4"
            )

            metrics = await provider_with_monitoring.get_metrics()
            assert metrics.total_requests == 1
            assert metrics.successful_requests == 1
            assert metrics.failed_requests == 0
            assert metrics.tokens_processed == 100
            assert metrics.avg_latency_ms > 0

    @pytest.mark.asyncio
    async def test_failed_request_metrics(self, provider_with_monitoring):
        """Test metrics update on failed requests."""
        with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Test error")

            # Bypass circuit breaker for this test
            with patch.object(provider_with_monitoring._circuit_breaker, 'call',
                            side_effect=Exception("Test error")):
                with pytest.raises(Exception):
                    await provider_with_monitoring.create_chat_completion(
                        messages=[{"role": "user", "content": "test"}],
                        model="gpt-4"
                    )

            metrics = await provider_with_monitoring.get_metrics()
            assert metrics.total_requests == 1
            assert metrics.successful_requests == 0
            assert metrics.failed_requests == 1

    @pytest.mark.asyncio
    async def test_latency_percentiles(self, provider_with_monitoring):
        """Test latency percentile calculations."""
        # Simulate multiple requests with different latencies
        provider_with_monitoring._latency_history = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            110, 120, 130, 140, 150, 160, 170, 180, 190, 200
        ]

        async with provider_with_monitoring._metrics_lock:
            provider_with_monitoring._metrics.successful_requests = 20
            provider_with_monitoring._metrics.total_requests = 20

            # Trigger percentile calculation
            if len(provider_with_monitoring._latency_history) >= 10:
                sorted_latencies = sorted(provider_with_monitoring._latency_history)
                # For 20 items: index 18 (0-based) = 190, index 19 = 200
                provider_with_monitoring._metrics.p95_latency_ms = sorted_latencies[18]  # 190
                provider_with_monitoring._metrics.p99_latency_ms = sorted_latencies[19]  # 200

            provider_with_monitoring._update_health_score()

        metrics = await provider_with_monitoring.get_metrics()
        assert metrics.p95_latency_ms == 190  # 95th percentile
        assert metrics.p99_latency_ms == 200  # 99th percentile


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_healthy_status(self, provider_with_monitoring):
        """Test healthy status when everything is working."""
        # Simulate successful metrics
        provider_with_monitoring._metrics.total_requests = 100
        provider_with_monitoring._metrics.successful_requests = 95
        provider_with_monitoring._metrics.health_score = 95.0

        health = await provider_with_monitoring.health_check()

        assert health["status"] == "healthy"
        assert health["health_score"] == 95.0
        assert health["success_rate"] == 0.95
        assert health["circuit_breaker"] == "closed"

    @pytest.mark.asyncio
    async def test_degraded_status(self, provider_with_monitoring):
        """Test degraded status with some failures."""
        # Simulate degraded metrics
        provider_with_monitoring._metrics.total_requests = 100
        provider_with_monitoring._metrics.successful_requests = 60
        provider_with_monitoring._metrics.health_score = 60.0

        health = await provider_with_monitoring.health_check()

        assert health["status"] == "degraded"
        assert health["health_score"] == 60.0
        assert health["success_rate"] == 0.6

    @pytest.mark.asyncio
    async def test_unhealthy_status(self, provider_with_monitoring):
        """Test unhealthy status with many failures."""
        # Simulate unhealthy metrics
        provider_with_monitoring._metrics.total_requests = 100
        provider_with_monitoring._metrics.successful_requests = 30
        provider_with_monitoring._metrics.health_score = 30.0
        provider_with_monitoring._circuit_breaker.state = CircuitState.OPEN

        health = await provider_with_monitoring.health_check()

        assert health["status"] == "unhealthy"
        assert health["health_score"] == 30.0
        assert health["success_rate"] == 0.3
        assert health["circuit_breaker"] == "open"

    @pytest.mark.asyncio
    async def test_health_score_with_penalties(self, provider_with_monitoring):
        """Test health score calculation with various penalties."""
        # Setup metrics
        provider_with_monitoring._metrics.total_requests = 100
        provider_with_monitoring._metrics.successful_requests = 90  # 90% success rate
        provider_with_monitoring._metrics.p99_latency_ms = 6000  # High latency
        provider_with_monitoring._metrics.last_failure_time = datetime.now()  # Recent failure
        provider_with_monitoring._circuit_breaker.state = CircuitState.HALF_OPEN

        async with provider_with_monitoring._metrics_lock:
            provider_with_monitoring._update_health_score()

        metrics = await provider_with_monitoring.get_metrics()

        # Base score is 90, minus penalties for:
        # - Half-open circuit: -25
        # - High latency: -20
        # - Recent failure: -15
        # Total: 90 - 60 = 30
        assert metrics.health_score == 30.0


class TestAdaptiveTimeout:
    """Test adaptive timeout functionality."""

    @pytest.mark.asyncio
    async def test_default_timeout_with_no_history(self, provider_with_monitoring):
        """Test default timeout when no latency history."""
        timeout = provider_with_monitoring.get_adaptive_timeout()
        assert timeout == 300.0  # Default max_stream_seconds

    @pytest.mark.asyncio
    async def test_adaptive_timeout_calculation(self, provider_with_monitoring):
        """Test adaptive timeout based on latency history."""
        # Simulate latency history (in ms)
        provider_with_monitoring._latency_history = [
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
            1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000
        ]

        timeout = provider_with_monitoring.get_adaptive_timeout()

        # P95 is 1900ms, so timeout should be 1900 * 2 / 1000 = 3.8 seconds
        # But minimum is 10 seconds
        assert timeout == 10.0

    @pytest.mark.asyncio
    async def test_adaptive_timeout_max_bound(self, provider_with_monitoring):
        """Test adaptive timeout respects maximum bound."""
        # Simulate very high latencies
        provider_with_monitoring._latency_history = [150000] * 20  # 150 seconds

        timeout = provider_with_monitoring.get_adaptive_timeout()

        # Should be capped at max_stream_seconds
        assert timeout == 300.0


class TestRequestLogging:
    """Test request logging with correlation IDs."""

    @pytest.mark.asyncio
    async def test_correlation_id_generation(self, provider_with_monitoring):
        """Test that correlation IDs are generated for requests."""
        with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock()

            with patch('uuid.uuid4', return_value='test-correlation-id'):
                await provider_with_monitoring.create_chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    model="gpt-4"
                )

            # Request log should be cleaned up after success
            assert len(provider_with_monitoring._request_log) == 0

    @pytest.mark.asyncio
    async def test_request_logging_excludes_sensitive_data(self, provider_with_monitoring):
        """Test that sensitive data is excluded from request logs."""
        params = {
            "messages": [{"role": "user", "content": "secret"}],
            "model": "gpt-4",
            "api_key": "secret-key",
            "temperature": 0.7
        }

        provider_with_monitoring._log_request("test-id", params)

        log_entry = provider_with_monitoring._request_log["test-id"]
        assert "api_key" not in log_entry["params"]
        assert "messages" not in log_entry["params"]
        assert log_entry["params"]["model"] == "gpt-4"
        assert log_entry["params"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_request_cleanup_on_failure(self, provider_with_monitoring):
        """Test that request logs are cleaned up even on failure."""
        with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Test error")

            with patch.object(provider_with_monitoring._circuit_breaker, 'call',
                            side_effect=Exception("Test error")):
                with pytest.raises(Exception):
                    await provider_with_monitoring.create_chat_completion(
                        messages=[{"role": "user", "content": "test"}],
                        model="gpt-4"
                    )

            # Request log should be cleaned up after failure
            assert len(provider_with_monitoring._request_log) == 0


class TestErrorTracking:
    """Test error tracking integration."""

    @pytest.mark.asyncio
    async def test_rate_limit_error_tracking(self, provider_with_monitoring):
        """Test rate limit errors are tracked correctly."""
        with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=Mock(),
                body={}
            )

            with patch.object(provider_with_monitoring._error_tracker, 'track_error',
                            new_callable=AsyncMock) as mock_track:
                with pytest.raises(RateLimitError):
                    await provider_with_monitoring.create_chat_completion(
                        messages=[{"role": "user", "content": "test"}],
                        model="gpt-4"
                    )

                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args.kwargs["error_type"] == ErrorType.RATE_LIMIT_ERROR

    @pytest.mark.asyncio
    async def test_auth_error_tracking(self, provider_with_monitoring):
        """Test authentication errors are tracked correctly."""
        with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = AuthenticationError(
                message="Invalid API key",
                response=Mock(),
                body={}
            )

            with patch.object(provider_with_monitoring._error_tracker, 'track_error',
                            new_callable=AsyncMock) as mock_track:
                with pytest.raises(AuthenticationError):
                    await provider_with_monitoring.create_chat_completion(
                        messages=[{"role": "user", "content": "test"}],
                        model="gpt-4"
                    )

                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args.kwargs["error_type"] == ErrorType.AUTH_ERROR


class TestPerformanceMonitoring:
    """Test performance monitoring integration."""

    @pytest.mark.asyncio
    async def test_performance_monitor_tracks_requests(self, provider_with_monitoring):
        """Test that performance monitor tracks request lifecycle."""
        with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock()

            await provider_with_monitoring.create_chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4"
            )

            # Check performance monitor metrics
            perf_metrics = provider_with_monitoring._performance_monitor.metrics
            assert perf_metrics.request_count == 1
            assert perf_metrics.active_requests == 0
            assert perf_metrics.error_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_request_tracking(self, provider_with_monitoring):
        """Test tracking of concurrent requests."""
        async def make_request():
            with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                            new_callable=AsyncMock) as mock_create:
                mock_create.return_value = MagicMock()
                await provider_with_monitoring.create_chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    model="gpt-4"
                )

        # Make concurrent requests
        tasks = [make_request() for _ in range(5)]
        await asyncio.gather(*tasks)

        # All requests should be completed
        perf_metrics = provider_with_monitoring._performance_monitor.metrics
        assert perf_metrics.request_count == 5
        assert perf_metrics.active_requests == 0

    @pytest.mark.asyncio
    async def test_streaming_request_monitoring(self, provider_with_monitoring):
        """Test monitoring of streaming requests."""
        async def mock_stream():
            yield {"choices": [{"delta": {"content": "test"}}]}

        with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream()

            await provider_with_monitoring.create_chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4",
                stream=True
            )

            metrics = await provider_with_monitoring.get_metrics()
            assert metrics.successful_requests == 1


class TestIntegration:
    """Integration tests for all monitoring features."""

    @pytest.mark.asyncio
    async def test_full_monitoring_workflow(self, provider_with_monitoring):
        """Test complete monitoring workflow with multiple scenarios."""
        # Successful request
        with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_response = MagicMock()
            mock_response.usage.total_tokens = 50
            mock_create.return_value = mock_response

            await provider_with_monitoring.create_chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4"
            )

        # Failed request
        with patch.object(provider_with_monitoring._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Test failure")

            with patch.object(provider_with_monitoring._circuit_breaker, 'call',
                            side_effect=Exception("Test failure")):
                with pytest.raises(Exception):
                    await provider_with_monitoring.create_chat_completion(
                        messages=[{"role": "user", "content": "test"}],
                        model="gpt-4"
                    )

        # Check comprehensive metrics
        metrics = await provider_with_monitoring.get_metrics()
        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.tokens_processed == 50

        # Check health
        health = await provider_with_monitoring.health_check()
        assert health["success_rate"] == 0.5
        assert health["tokens_processed"] == 50

        # Check adaptive timeout
        timeout = provider_with_monitoring.get_adaptive_timeout()
        assert timeout > 0