"""
Unit tests for DSPy recovery systems.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from dspy_toolkit.exceptions import DSPyIntegrationError
from dspy_toolkit.recovery import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    ComponentHealth,
    FallbackConfig,
    FallbackManager,
    FallbackStrategy,
    HealthCheckConfig,
    HealthChecker,
    RetryConfig,
    RetryHandler,
)


class TestCircuitBreaker:
    """Test cases for circuit breaker."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        assert cb.name == "test"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.config.failure_threshold == 3

    def test_circuit_breaker_success(self):
        """Test successful function execution."""
        cb = CircuitBreaker("test")

        def success_func():
            return "success"

        result = cb.call(success_func)

        assert result == "success"
        assert cb.total_successes == 1
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opening after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)

        def failing_func():
            raise Exception("Test failure")

        # First failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitBreakerState.CLOSED

        # Second failure - should open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitBreakerState.OPEN

        # Third call should be blocked
        with pytest.raises(DSPyIntegrationError, match="Circuit breaker.*is OPEN"):
            cb.call(failing_func)

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,  # Short timeout for testing
            success_threshold=1,
        )
        cb = CircuitBreaker("test", config)

        def failing_func():
            raise Exception("Test failure")

        def success_func():
            return "success"

        # Trigger failure to open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should attempt reset and succeed
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        @cb
        def test_func(should_fail=False):
            if should_fail:
                raise Exception("Test failure")
            return "success"

        # Success
        assert test_func() == "success"

        # Failure
        with pytest.raises(Exception):
            test_func(should_fail=True)
        assert cb.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics."""
        cb = CircuitBreaker("test")

        def test_func():
            return "success"

        cb.call(test_func)
        stats = cb.get_stats()

        assert stats["name"] == "test"
        assert stats["total_requests"] == 1
        assert stats["total_successes"] == 1
        assert stats["success_rate"] == 1.0


class TestRetryHandler:
    """Test cases for retry handler."""

    def test_retry_handler_success(self):
        """Test successful execution without retries."""
        retry_handler = RetryHandler()

        def success_func():
            return "success"

        result = retry_handler.execute(success_func)

        assert result == "success"
        assert retry_handler.total_attempts == 1
        assert retry_handler.total_retries == 0

    def test_retry_handler_eventual_success(self):
        """Test eventual success after retries."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)  # Fast retry for testing
        retry_handler = RetryHandler(config)

        attempt_count = 0

        def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise DSPyIntegrationError("Temporary failure")
            return "success"

        result = retry_handler.execute(flaky_func)

        assert result == "success"
        assert retry_handler.total_attempts == 3
        assert retry_handler.total_retries == 2

    def test_retry_handler_max_attempts(self):
        """Test failure after max attempts."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        retry_handler = RetryHandler(config)

        def failing_func():
            raise DSPyIntegrationError("Persistent failure")

        with pytest.raises(DSPyIntegrationError, match="failed after 2 attempts"):
            retry_handler.execute(failing_func)

        assert retry_handler.total_attempts == 2
        assert retry_handler.total_failures == 1

    def test_retry_handler_non_retryable_exception(self):
        """Test non-retryable exception handling."""
        config = RetryConfig(retryable_exceptions=[DSPyIntegrationError])
        retry_handler = RetryHandler(config)

        def non_retryable_func():
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError):
            retry_handler.execute(non_retryable_func)

        assert retry_handler.total_attempts == 1
        assert retry_handler.total_retries == 0

    def test_retry_handler_decorator(self):
        """Test retry handler as decorator."""
        from dspy_toolkit.recovery.retry_handler import retry

        attempt_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise DSPyIntegrationError("Temporary failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_async_retry_handler(self):
        """Test async retry handler."""
        from dspy_toolkit.recovery.retry_handler import AsyncRetryHandler

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        retry_handler = AsyncRetryHandler(config)

        attempt_count = 0

        async def async_flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise DSPyIntegrationError("Temporary failure")
            return "async_success"

        result = await retry_handler.execute(async_flaky_func)

        assert result == "async_success"
        assert attempt_count == 2


class TestFallbackManager:
    """Test cases for fallback manager."""

    def test_fallback_manager_primary_success(self):
        """Test successful primary function execution."""
        manager = FallbackManager()

        def primary_func():
            return "primary_success"

        manager.set_primary(primary_func)
        result = manager.execute()

        assert result == "primary_success"
        assert manager.primary_successes == 1

    def test_fallback_manager_sequential_fallback(self):
        """Test sequential fallback execution."""
        config = FallbackConfig(strategy=FallbackStrategy.SEQUENTIAL)
        manager = FallbackManager(config)

        def failing_primary():
            raise Exception("Primary failed")

        def failing_fallback():
            raise Exception("Fallback failed")

        def success_fallback():
            return "fallback_success"

        manager.set_primary(failing_primary)
        manager.add_fallback(failing_fallback, "fallback1")
        manager.add_fallback(success_fallback, "fallback2")

        result = manager.execute()

        assert result == "fallback_success"
        assert manager.fallback_successes == 1

    def test_fallback_manager_all_fail(self):
        """Test when all fallbacks fail."""
        manager = FallbackManager()

        def failing_primary():
            raise Exception("Primary failed")

        def failing_fallback():
            raise Exception("Fallback failed")

        manager.set_primary(failing_primary)
        manager.add_fallback(failing_fallback, "fallback1")

        with pytest.raises(DSPyIntegrationError, match="All fallback options failed"):
            manager.execute()

        assert manager.total_failures == 1

    def test_fallback_manager_weighted_strategy(self):
        """Test weighted fallback strategy."""
        config = FallbackConfig(strategy=FallbackStrategy.WEIGHTED)
        manager = FallbackManager(config)

        def failing_primary():
            raise Exception("Primary failed")

        def success_fallback():
            return "weighted_success"

        manager.set_primary(failing_primary)
        manager.add_fallback(success_fallback, "weighted_fallback", weight=2.0)

        result = manager.execute()

        assert result == "weighted_success"

    def test_fallback_manager_enable_disable(self):
        """Test enabling/disabling fallback options."""
        manager = FallbackManager()

        def failing_primary():
            raise Exception("Primary failed")

        def success_fallback():
            return "fallback_success"

        manager.set_primary(failing_primary)
        manager.add_fallback(success_fallback, "test_fallback")

        # Disable fallback
        manager.disable_fallback("test_fallback")

        with pytest.raises(DSPyIntegrationError):
            manager.execute()

        # Enable fallback
        manager.enable_fallback("test_fallback")
        result = manager.execute()

        assert result == "fallback_success"


class TestHealthChecker:
    """Test cases for health checker."""

    @pytest.mark.asyncio
    async def test_health_checker_basic(self):
        """Test basic health checker functionality."""
        checker = HealthChecker()

        def healthy_check():
            return {"status": "healthy", "message": "All good"}

        checker.add_health_check("test_component", healthy_check)

        results = await checker.check_all()

        assert "test_component" in results
        assert results["test_component"].status == ComponentHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_health_checker_unhealthy(self):
        """Test unhealthy component detection."""
        checker = HealthChecker()

        def unhealthy_check():
            raise Exception("Component is down")

        checker.add_health_check("failing_component", unhealthy_check)

        results = await checker.check_all()

        assert "failing_component" in results
        assert results["failing_component"].status == ComponentHealth.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_checker_timeout(self):
        """Test health check timeout."""
        config = HealthCheckConfig(timeout=0.1)
        checker = HealthChecker(config)

        async def slow_check():
            await asyncio.sleep(0.2)  # Longer than timeout
            return {"status": "healthy"}

        checker.add_health_check("slow_component", slow_check)

        results = await checker.check_all()

        assert "slow_component" in results
        assert results["slow_component"].status == ComponentHealth.UNHEALTHY
        assert "timed out" in results["slow_component"].message.lower()

    def test_health_checker_overall_health(self):
        """Test overall health calculation."""
        checker = HealthChecker()

        def healthy_check():
            return True

        def unhealthy_check():
            return False

        checker.add_health_check("healthy_component", healthy_check)
        checker.add_health_check("unhealthy_component", unhealthy_check)

        # Manually set component statuses for testing
        checker.health_checks["healthy_component"].current_status = (
            ComponentHealth.HEALTHY
        )
        checker.health_checks["unhealthy_component"].current_status = (
            ComponentHealth.UNHEALTHY
        )

        overall_health = checker.get_overall_health()

        assert overall_health["status"] == ComponentHealth.UNHEALTHY.value
        assert overall_health["summary"]["total"] == 2
        assert overall_health["summary"]["healthy"] == 1
        assert overall_health["summary"]["unhealthy"] == 1

    @pytest.mark.asyncio
    async def test_health_checker_monitoring(self):
        """Test continuous health monitoring."""
        config = HealthCheckConfig(check_interval=0.1)
        checker = HealthChecker(config)

        check_count = 0

        def counting_check():
            nonlocal check_count
            check_count += 1
            return True

        checker.add_health_check("counting_component", counting_check)

        # Start monitoring
        checker.start_monitoring()

        # Wait for a few checks
        await asyncio.sleep(0.3)

        # Stop monitoring
        checker.stop_monitoring()

        # Should have performed multiple checks
        assert check_count >= 2

    def test_health_checker_stats(self):
        """Test health checker statistics."""
        checker = HealthChecker()

        def test_check():
            return True

        checker.add_health_check("test_component", test_check)

        # Manually update stats for testing
        health_check = checker.health_checks["test_component"]
        health_check.total_checks = 10
        health_check.total_successes = 8
        health_check.total_failures = 2

        stats = checker.get_all_stats()

        assert "test_component" in stats
        assert stats["test_component"]["total_checks"] == 10
        assert stats["test_component"]["success_rate"] == 0.8


@pytest.mark.integration
class TestRecoveryIntegration:
    """Integration tests for recovery systems."""

    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker combined with retry handler."""
        from dspy_toolkit.recovery.circuit_breaker import circuit_breaker
        from dspy_toolkit.recovery.retry_handler import retry

        attempt_count = 0

        @circuit_breaker("test_integration", CircuitBreakerConfig(failure_threshold=2))
        @retry(max_attempts=3, base_delay=0.01)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 4:
                raise DSPyIntegrationError("Temporary failure")
            return "success"

        # Should eventually succeed with retries
        result = flaky_function()
        assert result == "success"

    def test_fallback_with_circuit_breaker(self):
        """Test fallback manager with circuit breaker protection."""
        from dspy_toolkit.recovery.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker("fallback_test", CircuitBreakerConfig(failure_threshold=1))
        manager = FallbackManager()

        def failing_primary():
            raise Exception("Primary always fails")

        @cb
        def protected_fallback():
            return "protected_success"

        manager.set_primary(failing_primary)
        manager.add_fallback(protected_fallback, "protected_fallback")

        result = manager.execute()
        assert result == "protected_success"

    @pytest.mark.asyncio
    async def test_health_checker_with_recovery(self):
        """Test health checker triggering recovery actions."""
        checker = HealthChecker()

        component_healthy = False

        def component_check():
            return component_healthy

        def recovery_action():
            nonlocal component_healthy
            component_healthy = True
            return "recovered"

        checker.add_health_check("recoverable_component", component_check)

        # Initial check should be unhealthy
        results = await checker.check_all()
        assert results["recoverable_component"].status == ComponentHealth.UNHEALTHY

        # Trigger recovery
        recovery_result = recovery_action()
        assert recovery_result == "recovered"

        # Check should now be healthy
        results = await checker.check_all()
        assert results["recoverable_component"].status == ComponentHealth.HEALTHY
