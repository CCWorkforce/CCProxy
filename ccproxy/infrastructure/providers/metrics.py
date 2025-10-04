"""
Provider metrics and health monitoring.
Tracks performance, errors, and health scores for provider instances.
"""

import anyio
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .resilience import CircuitState


@dataclass
class ProviderMetrics:
    """Detailed provider metrics for monitoring and observability."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    avg_latency_ms: float = 0
    p95_latency_ms: float = 0
    p99_latency_ms: float = 0
    tokens_processed: int = 0
    circuit_state: str = CircuitState.CLOSED.value
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    health_score: float = 100.0  # 0-100 scale


@dataclass
class MetricsCollector:
    """Collects and analyzes provider performance metrics."""

    metrics: ProviderMetrics = field(default_factory=ProviderMetrics)
    latency_history: List[float] = field(default_factory=list)
    _metrics_lock: anyio.Lock = field(default_factory=anyio.Lock)
    max_history_size: int = 1000

    async def record_success(self, latency_ms: float, tokens: int = 0) -> None:
        """
        Record a successful request.

        Args:
            latency_ms: Request latency in milliseconds
            tokens: Number of tokens processed
        """
        async with self._metrics_lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.total_latency_ms += latency_ms
            self.metrics.tokens_processed += tokens

            # Update average latency
            self.metrics.avg_latency_ms = self.metrics.total_latency_ms / max(
                self.metrics.successful_requests, 1
            )

            # Track latency history for percentiles
            self.latency_history.append(latency_ms)
            if len(self.latency_history) > self.max_history_size:
                self.latency_history = self.latency_history[-self.max_history_size :]

            # Calculate percentiles if we have enough data
            self._update_percentiles()

    async def record_failure(
        self, latency_ms: float, consecutive_failures: int = 0
    ) -> None:
        """
        Record a failed request.

        Args:
            latency_ms: Request latency in milliseconds
            consecutive_failures: Number of consecutive failures
        """
        async with self._metrics_lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures = consecutive_failures
            self.metrics.last_failure_time = datetime.now()

    def _update_percentiles(self) -> None:
        """Update latency percentiles. Call with lock held."""
        if len(self.latency_history) >= 10:
            sorted_latencies = sorted(self.latency_history)
            self.metrics.p95_latency_ms = sorted_latencies[
                int(len(sorted_latencies) * 0.95)
            ]
            self.metrics.p99_latency_ms = sorted_latencies[
                int(len(sorted_latencies) * 0.99)
            ]

    async def update_circuit_state(self, state: CircuitState) -> None:
        """Update circuit breaker state in metrics."""
        async with self._metrics_lock:
            self.metrics.circuit_state = state.value

    async def get_snapshot(self) -> ProviderMetrics:
        """Get current metrics snapshot."""
        async with self._metrics_lock:
            return ProviderMetrics(**self.metrics.__dict__)

    async def reset(self) -> None:
        """Reset all metrics."""
        async with self._metrics_lock:
            self.metrics = ProviderMetrics()
            self.latency_history.clear()


class HealthMonitor:
    """Monitors provider health and calculates health scores."""

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize health monitor.

        Args:
            metrics_collector: MetricsCollector instance to monitor
        """
        self.metrics_collector = metrics_collector

    async def calculate_health_score(
        self, circuit_state: Optional[CircuitState] = None
    ) -> float:
        """
        Calculate health score based on metrics.

        Args:
            circuit_state: Optional circuit breaker state

        Returns:
            Health score from 0 to 100
        """
        metrics = await self.metrics_collector.get_snapshot()

        if metrics.total_requests == 0:
            return 100.0

        # Calculate base score from success rate
        success_rate = metrics.successful_requests / metrics.total_requests
        base_score = success_rate * 100

        # Apply penalties
        penalties = 0

        # Circuit breaker state penalty
        if circuit_state:
            if circuit_state == CircuitState.OPEN:
                penalties += 50
            elif circuit_state == CircuitState.HALF_OPEN:
                penalties += 25

        # High latency penalty (if p99 > 5 seconds)
        if metrics.p99_latency_ms > 5000:
            penalties += 20

        # Recent failures penalty
        if metrics.last_failure_time:
            minutes_since_failure = (
                datetime.now() - metrics.last_failure_time
            ).total_seconds() / 60
            if minutes_since_failure < 1:
                penalties += 15
            elif minutes_since_failure < 5:
                penalties += 10

        health_score = max(0, min(100, base_score - penalties))

        # Update the health score in metrics
        metrics.health_score = health_score
        await self._update_health_score(health_score)

        return health_score

    async def _update_health_score(self, score: float) -> None:
        """Update health score in metrics collector."""
        async with self.metrics_collector._metrics_lock:
            self.metrics_collector.metrics.health_score = score

    async def get_health_check(
        self, circuit_state: Optional[CircuitState] = None, active_requests: int = 0
    ) -> Dict[str, Any]:
        """
        Perform health check and return detailed status.

        Args:
            circuit_state: Optional circuit breaker state
            active_requests: Number of currently active requests

        Returns:
            Health check results with status and metrics
        """
        metrics = await self.metrics_collector.get_snapshot()
        health_score = await self.calculate_health_score(circuit_state)

        # Determine health status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 50:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "health_score": health_score,
            "circuit_breaker": circuit_state.value if circuit_state else "unknown",
            "success_rate": metrics.successful_requests
            / max(metrics.total_requests, 1),
            "avg_latency_ms": metrics.avg_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "p99_latency_ms": metrics.p99_latency_ms,
            "active_requests": active_requests,
            "tokens_processed": metrics.tokens_processed,
            "total_requests": metrics.total_requests,
            "failed_requests": metrics.failed_requests,
            "last_failure_time": metrics.last_failure_time.isoformat()
            if metrics.last_failure_time
            else None,
        }


class AdaptiveTimeoutCalculator:
    """Calculates adaptive timeouts based on recent performance."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        max_timeout: float = 120.0,
        min_timeout: float = 10.0,
    ):
        """
        Initialize timeout calculator.

        Args:
            metrics_collector: MetricsCollector instance for performance data
            max_timeout: Maximum timeout in seconds
            min_timeout: Minimum timeout in seconds
        """
        self.metrics_collector = metrics_collector
        self.max_timeout = max_timeout
        self.min_timeout = min_timeout

    def calculate_timeout(self) -> float:
        """
        Calculate adaptive timeout based on recent performance.

        Returns:
            Timeout in seconds
        """
        if not self.metrics_collector.latency_history:
            return self.max_timeout

        # Use p95 latency * 2 as timeout, with bounds
        try:
            p95_latency = statistics.quantiles(
                self.metrics_collector.latency_history, n=20
            )[18]  # 95th percentile

            adaptive_timeout_s = min(
                self.max_timeout,
                max(self.min_timeout, p95_latency * 2 / 1000),  # Convert ms to seconds
            )

            return adaptive_timeout_s
        except (statistics.StatisticsError, IndexError):
            # If we can't calculate percentiles, use max timeout
            return self.max_timeout
