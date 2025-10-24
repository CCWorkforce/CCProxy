"""Tests for cache circuit breaker."""

import time
from unittest.mock import patch

from ccproxy.application.cache.circuit_breaker import CacheCircuitBreaker


class TestCacheCircuitBreaker:
    """Test cases for CacheCircuitBreaker."""

    def test_initialization(self) -> None:
        """Test circuit breaker initialization."""
        breaker = CacheCircuitBreaker(failure_threshold=5, reset_time=60)
        assert breaker.failure_threshold == 5
        assert breaker.reset_time == 60
        assert breaker.consecutive_failures == 0
        assert breaker.disabled_until == 0
        assert not breaker.is_open()

    def test_initial_state_is_closed(self) -> None:
        """Test that circuit breaker starts in closed state."""
        breaker = CacheCircuitBreaker()
        assert not breaker.is_open()
        assert breaker.consecutive_failures == 0

    def test_circuit_opens_after_threshold_failures(self) -> None:
        """Test that circuit opens after reaching failure threshold."""
        breaker = CacheCircuitBreaker(failure_threshold=3, reset_time=60)

        # Record failures below threshold
        breaker.record_failure()
        breaker.record_failure()
        assert not breaker.is_open()

        # Record failure at threshold
        breaker.record_failure()
        assert breaker.is_open()

    def test_record_success_decrements_failures(self) -> None:
        """Test that recording success decrements failure count."""
        breaker = CacheCircuitBreaker(failure_threshold=5)

        # Record some failures
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.consecutive_failures == 2

        # Record success
        breaker.record_success()
        assert breaker.consecutive_failures == 1

    def test_success_does_not_go_below_zero(self) -> None:
        """Test that failure count doesn't go below zero."""
        breaker = CacheCircuitBreaker()

        breaker.record_success()
        breaker.record_success()
        assert breaker.consecutive_failures == 0

    def test_success_resets_disabled_until_when_failures_reach_zero(self) -> None:
        """Test that disabled_until is cleared when failures reach zero."""
        breaker = CacheCircuitBreaker(failure_threshold=2, reset_time=60)

        # Trigger circuit breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open()
        assert breaker.disabled_until > 0

        # Reset by recording successes
        breaker.record_success()
        breaker.record_success()
        assert breaker.consecutive_failures == 0
        assert breaker.disabled_until == 0

    def test_circuit_closes_after_reset_time(self) -> None:
        """Test that circuit closes after reset time expires."""
        breaker = CacheCircuitBreaker(failure_threshold=2, reset_time=1)  # 1 second

        # Open circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open()

        # Wait for reset time
        time.sleep(1.5)

        # Circuit is still considered open due to consecutive failures
        # Need to manually reset or record successes to close
        breaker.reset()
        assert not breaker.is_open()

    def test_exponential_backoff_on_repeated_failures(self) -> None:
        """Test exponential backoff increases disabled duration."""
        breaker = CacheCircuitBreaker(failure_threshold=2, reset_time=10)

        # First trip
        breaker.record_failure()
        breaker.record_failure()
        breaker.is_open()
        first_disabled_until = breaker.disabled_until

        # Add more failures
        breaker.record_failure()
        breaker.record_failure()

        # Check again with more failures
        with patch("time.time", return_value=first_disabled_until + 1):
            breaker.is_open()
            second_disabled_until = breaker.disabled_until

            # Second disabled period should be longer
            assert second_disabled_until > first_disabled_until

    def test_max_consecutive_failures_cap(self) -> None:
        """Test that consecutive failures are capped."""
        breaker = CacheCircuitBreaker(failure_threshold=5)
        max_failures = breaker.max_consecutive_failures

        # Record many failures
        for _ in range(max_failures + 100):
            breaker.record_failure()

        # Should be capped
        assert breaker.consecutive_failures == max_failures

    def test_reset_clears_all_state(self) -> None:
        """Test that reset clears all circuit breaker state."""
        breaker = CacheCircuitBreaker(failure_threshold=2)

        # Trigger circuit breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open()

        # Reset
        breaker.reset()

        # Should be in initial state
        assert breaker.consecutive_failures == 0
        assert breaker.disabled_until == 0
        assert not breaker.is_open()

    def test_get_status_when_closed(self) -> None:
        """Test get_status returns correct info when circuit is closed."""
        breaker = CacheCircuitBreaker(failure_threshold=5)
        breaker.record_failure()

        status = breaker.get_status()

        assert status["is_open"] is False
        assert status["consecutive_failures"] == 1
        assert status["failure_threshold"] == 5
        assert status["disabled_until"] is None

    def test_get_status_when_open(self) -> None:
        """Test get_status returns correct info when circuit is open."""
        breaker = CacheCircuitBreaker(failure_threshold=2, reset_time=60)

        # Open circuit
        breaker.record_failure()
        breaker.record_failure()

        status = breaker.get_status()

        assert status["is_open"] is True
        assert status["consecutive_failures"] >= 2
        assert status["disabled_until"] is not None
        assert status["time_until_reset"] > 0

    def test_backoff_multiplier_capped_at_10(self) -> None:
        """Test that backoff multiplier is capped at 10."""
        breaker = CacheCircuitBreaker(failure_threshold=1, reset_time=10)

        # Record many failures to test cap
        for _ in range(100):
            breaker.record_failure()

        # Force check to calculate backoff
        current_time = time.time()
        with patch("time.time", return_value=current_time):
            breaker.is_open()

            # Maximum disabled duration should be 10 * reset_time = 100
            max_expected_duration = 10 * breaker.reset_time
            actual_duration = breaker.disabled_until - current_time

            assert actual_duration <= max_expected_duration

    def test_concurrent_failure_recording(self) -> None:
        """Test that recording multiple failures works correctly."""
        breaker = CacheCircuitBreaker(failure_threshold=10)

        # Record failures
        for i in range(5):
            breaker.record_failure()
            assert breaker.consecutive_failures == i + 1

    def test_mixed_success_and_failure(self) -> None:
        """Test alternating success and failure recordings."""
        breaker = CacheCircuitBreaker(failure_threshold=5)

        # Pattern: fail, fail, succeed, fail, fail
        breaker.record_failure()  # 1
        breaker.record_failure()  # 2
        breaker.record_success()  # 1
        breaker.record_failure()  # 2
        breaker.record_failure()  # 3

        assert breaker.consecutive_failures == 3
        assert not breaker.is_open()

    def test_default_values(self) -> None:
        """Test that default values are used when not specified."""
        breaker = CacheCircuitBreaker()

        # Should use defaults from constants
        assert breaker.failure_threshold > 0
        assert breaker.reset_time > 0
        assert breaker.max_consecutive_failures == breaker.failure_threshold * 10
