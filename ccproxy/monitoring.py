"""Performance monitoring utilities."""

import time
import anyio
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from typing import Deque
import statistics

from .constants import MONITORING_RECENT_DURATIONS_MAXLEN


@dataclass
class PerformanceMetrics:
    """Stores performance metrics for monitoring."""

    request_count: int = 0
    total_duration_ms: float = 0
    avg_duration_ms: float = 0
    p95_duration_ms: float = 0
    p99_duration_ms: float = 0
    error_count: int = 0
    active_requests: int = 0
    recent_durations: Deque[float] = field(
        default_factory=lambda: deque(maxlen=MONITORING_RECENT_DURATIONS_MAXLEN)
    )


class PerformanceMonitor:
    """Monitor and track application performance metrics."""

    def __init__(self) -> None:
        """Initialize the performance monitor with default metrics and synchronization primitives.

        Sets up:
        - Performance metrics tracking object
        - Async lock for thread-safe metric updates
        - Dictionary to track request start times for duration calculation

        The monitor is ready to use immediately after initialization and can
        safely handle concurrent requests in an async environment.
        """
        self.metrics = PerformanceMetrics()
        self._lock = anyio.Lock()
        self._request_start_times: Dict[str, float] = {}

    async def start_request(self, request_id: str) -> float:
        """Mark the start of a request."""
        start_time = time.monotonic()
        async with self._lock:
            self._request_start_times[request_id] = start_time
            self.metrics.active_requests += 1
        return start_time

    async def end_request(
        self, request_id: str, success: bool = True
    ) -> Optional[float]:
        """Mark the end of a request and calculate duration."""
        end_time = time.monotonic()

        async with self._lock:
            start_time = self._request_start_times.pop(request_id, None)
            if start_time is None:
                return None

            duration_ms = (end_time - start_time) * 1000

            self.metrics.request_count += 1
            self.metrics.total_duration_ms += duration_ms
            self.metrics.recent_durations.append(duration_ms)
            self.metrics.active_requests = max(0, self.metrics.active_requests - 1)

            if not success:
                self.metrics.error_count += 1

            # Update statistics
            if self.metrics.recent_durations:
                durations = list(self.metrics.recent_durations)
                self.metrics.avg_duration_ms = statistics.mean(durations)

                if len(durations) >= 10:
                    sorted_durations = sorted(durations)
                    p95_index = int(len(sorted_durations) * 0.95)
                    p99_index = int(len(sorted_durations) * 0.99)
                    self.metrics.p95_duration_ms = sorted_durations[p95_index]
                    self.metrics.p99_duration_ms = sorted_durations[p99_index]

            return duration_ms

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        async with self._lock:
            return {
                "request_count": self.metrics.request_count,
                "active_requests": self.metrics.active_requests,
                "error_count": self.metrics.error_count,
                "error_rate": self.metrics.error_count
                / max(1, self.metrics.request_count),
                "avg_duration_ms": round(self.metrics.avg_duration_ms, 2),
                "p95_duration_ms": round(self.metrics.p95_duration_ms, 2),
                "p99_duration_ms": round(self.metrics.p99_duration_ms, 2),
            }

    async def reset_metrics(self) -> None:
        """Reset all metrics."""
        async with self._lock:
            self.metrics = PerformanceMetrics()
            self._request_start_times.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
