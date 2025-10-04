"""
Factory for creating and initializing provider components.
Reduces complexity of provider initialization by extracting component creation logic.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ...config import Settings
from ...monitoring import PerformanceMonitor
from ...application.error_tracker import ErrorTracker
from .rate_limiter import ClientRateLimiter, RateLimitConfig
from .resilience import CircuitBreaker, RetryHandler, ResilientExecutor
from .metrics import (
    MetricsCollector,
    HealthMonitor,
    AdaptiveTimeoutCalculator,
)
from .response_handlers import ResponseProcessor, ErrorResponseHandler
from .request_logger import RequestLogger, PerformanceTracker


@dataclass
class ResilienceComponents:
    """Bundle of resilience-related components."""

    circuit_breaker: CircuitBreaker
    retry_handler: RetryHandler
    resilient_executor: ResilientExecutor


@dataclass
class MonitoringComponents:
    """Bundle of monitoring-related components."""

    metrics_collector: MetricsCollector
    health_monitor: HealthMonitor
    timeout_calculator: AdaptiveTimeoutCalculator
    performance_monitor: PerformanceMonitor
    error_tracker: ErrorTracker


@dataclass
class LoggingComponents:
    """Bundle of logging-related components."""

    request_logger: RequestLogger
    performance_tracker: PerformanceTracker


@dataclass
class ProcessingComponents:
    """Bundle of processing-related components."""

    response_processor: ResponseProcessor
    error_handler: ErrorResponseHandler


class ProviderComponentsFactory:
    """Factory for creating provider components with reduced initialization complexity."""

    @staticmethod
    def create_resilience_components(settings: Settings) -> ResilienceComponents:
        """
        Create and configure resilience components.

        Args:
            settings: Application settings

        Returns:
            ResilienceComponents bundle

        Raises:
            ValueError: If settings contain invalid values
        """
        # Validate retry settings
        if settings.provider_retry_base_delay <= 0:
            raise ValueError(
                f"provider_retry_base_delay must be positive, got {settings.provider_retry_base_delay}"
            )
        if not (0 <= settings.provider_retry_jitter <= 1):
            raise ValueError(
                f"provider_retry_jitter must be between 0 and 1, got {settings.provider_retry_jitter}"
            )

        circuit_breaker = CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            recovery_timeout=settings.circuit_breaker_recovery_timeout,
            half_open_requests=settings.circuit_breaker_half_open_requests,
        )

        retry_handler = RetryHandler(
            max_retries=settings.provider_max_retries,
            base_delay=settings.provider_retry_base_delay,
            jitter=settings.provider_retry_jitter,
        )

        resilient_executor = ResilientExecutor(
            circuit_breaker=circuit_breaker,
            retry_handler=retry_handler,
        )

        return ResilienceComponents(
            circuit_breaker=circuit_breaker,
            retry_handler=retry_handler,
            resilient_executor=resilient_executor,
        )

    @staticmethod
    def create_monitoring_components(settings: Settings) -> MonitoringComponents:
        """
        Create and configure monitoring components.

        Args:
            settings: Application settings

        Returns:
            MonitoringComponents bundle

        Raises:
            ValueError: If settings contain invalid values
        """
        # Validate monitoring settings
        if settings.max_stream_seconds <= 0:
            raise ValueError(
                f"max_stream_seconds must be positive, got {settings.max_stream_seconds}"
            )

        metrics_collector = MetricsCollector()
        health_monitor = HealthMonitor(metrics_collector)
        timeout_calculator = AdaptiveTimeoutCalculator(
            metrics_collector,
            max_timeout=float(settings.max_stream_seconds),
        )
        performance_monitor = PerformanceMonitor()
        error_tracker = ErrorTracker()  # Singleton, no parameters

        return MonitoringComponents(
            metrics_collector=metrics_collector,
            health_monitor=health_monitor,
            timeout_calculator=timeout_calculator,
            performance_monitor=performance_monitor,
            error_tracker=error_tracker,
        )

    @staticmethod
    def create_logging_components() -> LoggingComponents:
        """
        Create and configure logging components.

        Returns:
            LoggingComponents bundle
        """
        request_logger = RequestLogger()
        performance_tracker = PerformanceTracker()

        return LoggingComponents(
            request_logger=request_logger,
            performance_tracker=performance_tracker,
        )

    @staticmethod
    def create_processing_components() -> ProcessingComponents:
        """
        Create and configure processing components.

        Returns:
            ProcessingComponents bundle
        """
        response_processor = ResponseProcessor()
        error_handler = ErrorResponseHandler()

        return ProcessingComponents(
            response_processor=response_processor,
            error_handler=error_handler,
        )

    @staticmethod
    def create_rate_limiter(settings: Settings) -> Optional[ClientRateLimiter]:
        """
        Create and configure client-side rate limiter if enabled.

        Args:
            settings: Application settings

        Returns:
            Configured ClientRateLimiter or None if disabled

        Raises:
            ValueError: If settings contain invalid values
        """
        if not settings.client_rate_limit_enabled:
            logging.info("Client rate limiting disabled")
            return None

        # Validate rate limit settings
        if (
            not isinstance(settings.client_rate_limit_tpm, int)
            or settings.client_rate_limit_tpm <= 0
        ):
            raise ValueError(
                f"client_rate_limit_tpm must be a positive integer, got {settings.client_rate_limit_tpm}"
            )

        rate_limit_config = RateLimitConfig(
            requests_per_minute=settings.client_rate_limit_rpm,
            tokens_per_minute=settings.client_rate_limit_tpm,
            burst_size=settings.client_rate_limit_burst,
            adaptive_enabled=settings.client_rate_limit_adaptive,
        )

        rate_limiter = ClientRateLimiter(rate_limit_config)
        logging.info(
            f"Client rate limiter initialized: {settings.client_rate_limit_rpm} RPM, "
            f"{settings.client_rate_limit_tpm} TPM"
        )

        return rate_limiter
