import pytest
from unittest.mock import Mock

from ccproxy.infrastructure.providers.provider_components_factory import (
    ProviderComponentsFactory,
    ResilienceComponents,
    MonitoringComponents,
    LoggingComponents,
    ProcessingComponents,
)
from ccproxy.config import Settings
from ccproxy.infrastructure.providers.rate_limiter import ClientRateLimiter
from ccproxy.infrastructure.providers.resilience import (
    CircuitBreaker,
    RetryHandler,
    ResilientExecutor,
)
from ccproxy.infrastructure.providers.metrics import (
    MetricsCollector,
    HealthMonitor,
    AdaptiveTimeoutCalculator,
)
from ccproxy.monitoring import PerformanceMonitor
from ccproxy.application.error_tracker import ErrorTracker
from ccproxy.infrastructure.providers.response_handlers import (
    ResponseProcessor,
    ErrorResponseHandler,
)
from ccproxy.infrastructure.providers.request_logger import (
    RequestLogger,
    PerformanceTracker,
)


@pytest.fixture
def mock_settings():
    settings = Mock(spec=Settings)
    settings.circuit_breaker_failure_threshold = 5
    settings.circuit_breaker_recovery_timeout = 60
    settings.circuit_breaker_half_open_requests = 3
    settings.provider_max_retries = 3
    settings.provider_retry_base_delay = 1.0
    settings.provider_retry_jitter = 0.5
    settings.max_stream_seconds = 300.0
    settings.client_rate_limit_enabled = True
    settings.client_rate_limit_rpm = 100
    settings.client_rate_limit_tpm = 100000
    settings.client_rate_limit_burst = 10
    settings.client_rate_limit_adaptive = True
    return settings


def test_create_resilience_components(mock_settings):
    components = ProviderComponentsFactory.create_resilience_components(mock_settings)

    assert isinstance(components, ResilienceComponents)
    assert isinstance(components.circuit_breaker, CircuitBreaker)
    assert (
        components.circuit_breaker.failure_threshold
        == mock_settings.circuit_breaker_failure_threshold
    )
    assert (
        components.circuit_breaker.recovery_timeout
        == mock_settings.circuit_breaker_recovery_timeout
    )
    assert (
        components.circuit_breaker.half_open_requests
        == mock_settings.circuit_breaker_half_open_requests
    )

    assert isinstance(components.retry_handler, RetryHandler)
    assert components.retry_handler.max_retries == mock_settings.provider_max_retries
    assert (
        components.retry_handler.base_delay == mock_settings.provider_retry_base_delay
    )
    assert components.retry_handler.jitter == mock_settings.provider_retry_jitter

    assert isinstance(components.resilient_executor, ResilientExecutor)
    assert components.resilient_executor.circuit_breaker == components.circuit_breaker
    assert components.resilient_executor.retry_handler == components.retry_handler


def test_create_monitoring_components(mock_settings):
    components = ProviderComponentsFactory.create_monitoring_components(mock_settings)

    assert isinstance(components, MonitoringComponents)
    assert isinstance(components.metrics_collector, MetricsCollector)
    assert isinstance(components.health_monitor, HealthMonitor)
    assert components.health_monitor.metrics_collector == components.metrics_collector
    assert isinstance(components.timeout_calculator, AdaptiveTimeoutCalculator)
    assert (
        components.timeout_calculator.metrics_collector == components.metrics_collector
    )
    assert components.timeout_calculator.max_timeout == float(
        mock_settings.max_stream_seconds
    )
    assert isinstance(components.performance_monitor, PerformanceMonitor)
    assert isinstance(components.error_tracker, ErrorTracker)


def test_create_logging_components():
    components = ProviderComponentsFactory.create_logging_components()

    assert isinstance(components, LoggingComponents)
    assert isinstance(components.request_logger, RequestLogger)
    assert isinstance(components.performance_tracker, PerformanceTracker)


def test_create_processing_components():
    components = ProviderComponentsFactory.create_processing_components()

    assert isinstance(components, ProcessingComponents)
    assert isinstance(components.response_processor, ResponseProcessor)
    assert isinstance(components.error_handler, ErrorResponseHandler)


def test_create_rate_limiter_enabled(mock_settings):
    mock_settings.client_rate_limit_enabled = True
    rate_limiter = ProviderComponentsFactory.create_rate_limiter(mock_settings)

    assert isinstance(rate_limiter, ClientRateLimiter)
    assert (
        rate_limiter.config.requests_per_minute == mock_settings.client_rate_limit_rpm
    )
    assert rate_limiter.config.tokens_per_minute == mock_settings.client_rate_limit_tpm
    assert rate_limiter.config.burst_size == mock_settings.client_rate_limit_burst
    assert (
        rate_limiter.config.adaptive_enabled == mock_settings.client_rate_limit_adaptive
    )


def test_create_rate_limiter_disabled(mock_settings):
    mock_settings.client_rate_limit_enabled = False
    rate_limiter = ProviderComponentsFactory.create_rate_limiter(mock_settings)

    assert rate_limiter is None


@pytest.mark.parametrize(
    "create_method, invalid_attr, invalid_value, expected_error",
    [
        # Resilience component invalid values
        (
            "create_resilience_components",
            "circuit_breaker_failure_threshold",
            -1,
            ValueError,
        ),
        (
            "create_resilience_components",
            "circuit_breaker_failure_threshold",
            0,
            ValueError,
        ),
        (
            "create_resilience_components",
            "circuit_breaker_recovery_timeout",
            -1,
            ValueError,
        ),
        (
            "create_resilience_components",
            "circuit_breaker_recovery_timeout",
            0,
            ValueError,
        ),
        (
            "create_resilience_components",
            "circuit_breaker_half_open_requests",
            0,
            ValueError,
        ),
        (
            "create_resilience_components",
            "circuit_breaker_half_open_requests",
            -1,
            ValueError,
        ),
        ("create_resilience_components", "provider_max_retries", -1, ValueError),
        ("create_resilience_components", "provider_retry_base_delay", -0.5, ValueError),
        ("create_resilience_components", "provider_retry_base_delay", 0, ValueError),
        ("create_resilience_components", "provider_retry_jitter", -0.1, ValueError),
        ("create_resilience_components", "provider_retry_jitter", 1.5, ValueError),
        # Rate limiter invalid values
        ("create_rate_limiter", "client_rate_limit_rpm", 0, ValueError),
        ("create_rate_limiter", "client_rate_limit_tpm", 0, ValueError),
        ("create_rate_limiter", "client_rate_limit_rpm", -1, ValueError),
        ("create_rate_limiter", "client_rate_limit_tpm", -1, ValueError),
        (
            "create_rate_limiter",
            "client_rate_limit_rpm",
            0.5,
            ValueError,
        ),  # Fractional RPM
        (
            "create_rate_limiter",
            "client_rate_limit_tpm",
            1.5,
            ValueError,
        ),  # Fractional TPM
        ("create_rate_limiter", "client_rate_limit_burst", 0, ValueError),
        ("create_rate_limiter", "client_rate_limit_burst", -1, ValueError),
        # Monitoring component invalid values
        ("create_monitoring_components", "max_stream_seconds", -1.0, ValueError),
        ("create_monitoring_components", "max_stream_seconds", 0, ValueError),
    ],
)
def test_invalid_settings_raise_value_error(
    create_method, invalid_attr, invalid_value, expected_error
):
    """Test that invalid settings raise appropriate errors in component creation."""

    settings = Mock()

    # Set default valid values
    settings.circuit_breaker_failure_threshold = 5
    settings.circuit_breaker_recovery_timeout = 60
    settings.circuit_breaker_half_open_requests = 3
    settings.provider_max_retries = 3
    settings.provider_retry_base_delay = 1.0
    settings.provider_retry_jitter = 0.1
    settings.client_rate_limit_rpm = 1500
    settings.client_rate_limit_tpm = 270000
    settings.client_rate_limit_burst = 100
    settings.client_rate_limit_enabled = True
    settings.client_rate_limit_adaptive = True
    settings.max_stream_seconds = 300.0

    # Set the invalid value
    setattr(settings, invalid_attr, invalid_value)

    method = getattr(ProviderComponentsFactory, create_method)

    with pytest.raises(expected_error):
        method(settings)
