"""Thread pool management for async CPU-bound operations.

This module provides a centralized thread pool configuration for offloading
CPU-bound operations (like JSON serialization, token encoding) to thread pools,
preventing event loop blocking.
"""

import os
import psutil
from typing import Optional, Any, TypeVar, Callable, Coroutine
from functools import wraps, partial
from anyio import CapacityLimiter, to_thread
from asyncer import asyncify as _asyncify_orig

from ..config import Settings
from ..logging import info, LogRecord

# Type variable for generic return types
T = TypeVar("T")

# Global thread pool limiter
_thread_limiter: Optional[CapacityLimiter] = None
_settings: Optional[Settings] = None
_initialized: bool = False
_cpu_count: int = os.cpu_count() or 4
_cpu_threshold: int = 80  # Default threshold

# Thread pool shrinking metrics
_shrink_events: int = 0
_last_low_cpu_time: Optional[float] = None
_pool_utilization_history: list[float] = []
_pool_resize_events: list[dict] = []  # Track resize events for metrics


def initialize_thread_pool(settings: Settings) -> None:
    """Initialize the thread pool with configuration from settings.

    Args:
        settings: Application settings containing thread pool configuration
    """
    global _thread_limiter, _settings, _initialized, _cpu_threshold

    if _initialized:
        return

    _settings = settings

    # Check if running with multiple workers (Uvicorn)
    worker_count = 1
    web_concurrency = os.environ.get("WEB_CONCURRENCY")
    if web_concurrency and int(web_concurrency) > 1:
        # Multiple workers detected via WEB_CONCURRENCY
        worker_count = int(web_concurrency)
        info(
            LogRecord(
                event="thread_pool_init",
                message=f"Running with {worker_count} workers (multi-worker mode)",
                data={"worker_count": worker_count},
            )
        )

    # Determine the number of worker threads
    max_workers = settings.thread_pool_max_workers

    if max_workers is None:
        # Use default but adjust for multiple workers
        cpu_count = _cpu_count

        if worker_count > 1:
            # When running with multiple workers, reduce threads per worker
            # Formula: Total threads = workers * threads_per_worker
            # Target total threads: cpu_count * 5 (distributed across workers)
            target_total = cpu_count * 5
            max_workers = max(4, min(20, target_total // worker_count))
            info(
                LogRecord(
                    event="thread_pool_init",
                    message=f"Multi-worker mode: {max_workers} threads per worker",
                    data={
                        "cpu_count": cpu_count,
                        "worker_count": worker_count,
                        "threads_per_worker": max_workers,
                        "total_threads": worker_count * max_workers,
                    },
                )
            )
        else:
            # Single worker mode (development or explicit single-worker)
            max_workers = min(40, max(4, cpu_count * 5))
            info(
                LogRecord(
                    event="thread_pool_init",
                    message=f"Single-worker mode: {max_workers} threads",
                    data={"cpu_count": cpu_count, "max_workers": max_workers},
                )
            )
    else:
        info(
            LogRecord(
                event="thread_pool_init",
                message=f"Using configured thread pool size: {max_workers} workers",
                data={"max_workers": max_workers},
            )
        )

    # Determine CPU threshold
    if settings.thread_pool_high_cpu_threshold is None:
        # Auto-calculate based on CPU count
        # More cores = can handle higher overall CPU usage
        # Formula: base 60% + (2.5% per core), capped at 90%
        _cpu_threshold = min(90, int(60 + (_cpu_count * 2.5)))
        info(
            LogRecord(
                event="thread_pool_init",
                message=f"Auto-calculated CPU threshold: {_cpu_threshold}% (based on {_cpu_count} cores)",
                data={"cpu_count": _cpu_count, "cpu_threshold": _cpu_threshold},
            )
        )
    else:
        _cpu_threshold = settings.thread_pool_high_cpu_threshold
        info(
            LogRecord(
                event="thread_pool_init",
                message=f"Using configured CPU threshold: {_cpu_threshold}%",
                data={"cpu_threshold": _cpu_threshold},
            )
        )

    # Create the capacity limiter for thread pool
    _thread_limiter = CapacityLimiter(max_workers)
    _initialized = True


def get_thread_limiter() -> Optional[CapacityLimiter]:
    """Get the configured thread pool limiter.

    Returns:
        The configured CapacityLimiter or None if not initialized
    """
    return _thread_limiter


def check_cpu_contention() -> dict:
    """Check current CPU utilization to detect contention.

    Returns:
        Dictionary with CPU metrics including overall and per-core usage
    """
    try:
        # Get overall CPU percent over a short interval
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Get per-core CPU usage
        per_core = psutil.cpu_percent(interval=0.1, percpu=True)

        # Calculate average core usage and find hottest core
        avg_core = sum(per_core) / len(per_core) if per_core else cpu_percent
        max_core = max(per_core) if per_core else cpu_percent

        return {
            "overall": cpu_percent,
            "average_core": avg_core,
            "max_core": max_core,
            "per_core": per_core,
            "core_count": len(per_core) if per_core else _cpu_count,
        }
    except Exception:
        return {
            "overall": 0.0,
            "average_core": 0.0,
            "max_core": 0.0,
            "per_core": [],
            "core_count": _cpu_count,
        }


def should_increase_pool_size() -> bool:
    """Determine if thread pool size should be increased based on CPU contention.

    Uses intelligent metrics including per-core utilization to make decisions.

    Returns:
        True if pool size should be increased, False otherwise
    """
    if not _settings or not _settings.thread_pool_auto_scale:
        return False

    cpu_metrics = check_cpu_contention()

    # Use the configured or auto-calculated threshold
    threshold = _cpu_threshold

    # Consider both overall CPU and per-core hotspots
    # If overall is below threshold AND no single core is maxed out
    if cpu_metrics["overall"] < threshold and cpu_metrics["max_core"] < 95:
        # Also check if we have reasonable headroom on average cores
        if cpu_metrics["average_core"] < (threshold - 10):
            return True

    return False


def should_decrease_pool_size() -> bool:
    """Determine if thread pool size should be decreased based on low CPU usage.

    Implements hysteresis: requires sustained low CPU for a configurable period
    before shrinking the pool to avoid thrashing.

    Returns:
        True if pool size should be decreased, False otherwise
    """
    import time

    global _last_low_cpu_time

    if not _settings or not _settings.thread_pool_auto_scale:
        return False

    # Get shrink configuration from settings (with defaults)
    shrink_threshold = getattr(
        _settings, "thread_pool_shrink_threshold", 30
    )  # Default 30%
    shrink_delay = getattr(
        _settings, "thread_pool_shrink_delay_seconds", 60
    )  # Default 60s

    cpu_metrics = check_cpu_contention()
    current_time = time.time()

    # Check if CPU is below shrink threshold
    if cpu_metrics["overall"] < shrink_threshold and cpu_metrics["max_core"] < (
        shrink_threshold + 10
    ):
        # Track how long CPU has been low
        if _last_low_cpu_time is None:
            _last_low_cpu_time = current_time
            return False

        # Check if we've been in low CPU state long enough
        if (current_time - _last_low_cpu_time) >= shrink_delay:
            # Check pool utilization - only shrink if pool is underutilized
            if _thread_limiter:
                utilization = (
                    _thread_limiter.borrowed_tokens / _thread_limiter.total_tokens
                )
                if utilization < 0.5:  # Less than 50% pool utilization
                    return True
    else:
        # Reset the timer if CPU is no longer low
        _last_low_cpu_time = None

    return False


def asyncify(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """Enhanced asyncify that uses configured thread pool.

    This wraps asyncer's asyncify to use our configured thread pool limiter,
    allowing for better control over thread pool size and CPU contention.

    Args:
        func: Synchronous function to wrap

    Returns:
        Async version of the function that runs in thread pool
    """
    # Get the async version using original asyncify
    async_func = _asyncify_orig(func)

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        # If we have a configured limiter, use it
        limiter = get_thread_limiter()

        if limiter:
            # Run with our configured limiter using anyio's to_thread
            # Use partial to bind args and kwargs to the function
            return await to_thread.run_sync(
                partial(func, *args, **kwargs), limiter=limiter
            )
        else:
            # Fall back to default asyncify behavior
            return await async_func(*args, **kwargs)

    return wrapper


def configure_for_high_load() -> None:
    """Adjust thread pool configuration for high load scenarios.

    This can be called when high CPU contention is detected to dynamically
    adjust the thread pool size up or down based on current metrics.
    """
    global _thread_limiter, _shrink_events, _pool_resize_events
    import time

    if not _settings or not _settings.thread_pool_auto_scale:
        return

    current_limit = _thread_limiter.total_tokens if _thread_limiter else 40
    cpu_metrics = check_cpu_contention()

    # Track pool utilization
    if _thread_limiter:
        utilization = _thread_limiter.borrowed_tokens / _thread_limiter.total_tokens
        _pool_utilization_history.append(utilization)
        # Keep only last 100 samples
        if len(_pool_utilization_history) > 100:
            _pool_utilization_history.pop(0)

    # Check if we should increase
    if should_increase_pool_size():
        # Increase by 25% but cap at reasonable limit based on CPU cores
        # Max threads = cores * 10 (but at least 100)
        max_limit = max(100, _cpu_count * 10)
        new_limit = min(int(current_limit * 1.25), max_limit)

        if new_limit > current_limit:
            _thread_limiter = CapacityLimiter(new_limit)
            resize_event = {
                "timestamp": time.time(),
                "action": "increase",
                "old_size": current_limit,
                "new_size": new_limit,
                "cpu_overall": cpu_metrics["overall"],
                "cpu_avg_core": cpu_metrics["average_core"],
                "cpu_max_core": cpu_metrics["max_core"],
                "utilization": utilization if _thread_limiter else 0,
            }
            _pool_resize_events.append(resize_event)
            info(
                LogRecord(
                    event="thread_pool_resize",
                    message=f"Increased thread pool size: {current_limit} -> {new_limit}",
                    data=resize_event,
                )
            )

    # Check if we should decrease (shrink)
    elif should_decrease_pool_size():
        # Decrease by 20% but keep a minimum based on CPU cores
        min_limit = max(4, _cpu_count * 2)  # At least 2 threads per core
        new_limit = max(int(current_limit * 0.8), min_limit)

        if new_limit < current_limit:
            _thread_limiter = CapacityLimiter(new_limit)
            _shrink_events += 1
            resize_event = {
                "timestamp": time.time(),
                "action": "decrease",
                "old_size": current_limit,
                "new_size": new_limit,
                "cpu_overall": cpu_metrics["overall"],
                "cpu_avg_core": cpu_metrics["average_core"],
                "cpu_max_core": cpu_metrics["max_core"],
                "utilization": _thread_limiter.borrowed_tokens
                / _thread_limiter.total_tokens,
                "shrink_event_count": _shrink_events,
            }
            _pool_resize_events.append(resize_event)
            info(
                LogRecord(
                    event="thread_pool_resize",
                    message=f"Decreased thread pool size: {current_limit} -> {new_limit}",
                    data=resize_event,
                )
            )
            # Reset low CPU timer after shrinking
            global _last_low_cpu_time
            _last_low_cpu_time = None


def get_pool_stats() -> dict:
    """Get current thread pool statistics including shrink metrics.

    Returns:
        Dictionary containing pool statistics including CPU metrics and resize history
    """
    cpu_metrics = check_cpu_contention()

    stats = {
        "initialized": _initialized,
        "max_workers": None,
        "available_tokens": None,
        "borrowed_tokens": None,
        "utilization": 0.0,
        "cpu_overall": cpu_metrics["overall"],
        "cpu_avg_core": cpu_metrics["average_core"],
        "cpu_max_core": cpu_metrics["max_core"],
        "cpu_threshold": _cpu_threshold,
        "core_count": cpu_metrics["core_count"],
        "auto_scale_enabled": _settings.thread_pool_auto_scale if _settings else False,
        "shrink_events": _shrink_events,
        "resize_history": len(_pool_resize_events),
        "recent_resizes": _pool_resize_events[-5:] if _pool_resize_events else [],
    }

    if _thread_limiter:
        stats["max_workers"] = _thread_limiter.total_tokens
        stats["available_tokens"] = _thread_limiter.available_tokens
        stats["borrowed_tokens"] = _thread_limiter.borrowed_tokens
        stats["utilization"] = (
            _thread_limiter.borrowed_tokens / _thread_limiter.total_tokens
        )

    # Add pool utilization statistics if we have history
    if _pool_utilization_history:
        stats["avg_utilization"] = sum(_pool_utilization_history) / len(
            _pool_utilization_history
        )
        stats["max_utilization"] = max(_pool_utilization_history)
        stats["min_utilization"] = min(_pool_utilization_history)

    # Add shrink configuration if available
    if _settings:
        stats["shrink_threshold"] = getattr(
            _settings, "thread_pool_shrink_threshold", 30
        )
        stats["shrink_delay_seconds"] = getattr(
            _settings, "thread_pool_shrink_delay_seconds", 60
        )

    return stats


# Export the enhanced asyncify as the primary interface
__all__ = [
    "initialize_thread_pool",
    "asyncify",
    "get_thread_limiter",
    "configure_for_high_load",
    "get_pool_stats",
]
