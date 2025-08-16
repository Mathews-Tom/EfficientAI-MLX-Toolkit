"""
Circuit breaker pattern implementation for DSPy Integration Framework.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any

from ..exceptions import DSPyIntegrationError

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures to open circuit
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    success_threshold: int = 3  # Successes needed to close circuit in half-open
    timeout: float = 30.0  # Request timeout in seconds
    expected_exception: type[Exception] = Exception  # Exception type that triggers circuit


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """Initialize circuit breaker."""
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0

        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0

        logger.info("Circuit breaker '%s' initialized", name)

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        self.total_requests += 1

        # Check if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._attempt_reset()
            else:
                raise DSPyIntegrationError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Last failure: {time.time() - self.last_failure_time:.1f}s ago"
                )

        # Execute function
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Check for timeout
            if execution_time > self.config.timeout:
                self.total_timeouts += 1
                self._record_failure(f"Timeout after {execution_time:.2f}s")
                raise DSPyIntegrationError(f"Function timeout after {execution_time:.2f}s")

            # Record success
            self._record_success()
            return result

        except self.config.expected_exception as e:
            self._record_failure(str(e))
            raise
        except Exception as e:
            # Unexpected exceptions also trigger circuit breaker
            self._record_failure(f"Unexpected error: {e}")
            raise DSPyIntegrationError(f"Circuit breaker caught unexpected error: {e}") from e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout

    def _attempt_reset(self):
        """Attempt to reset circuit breaker to half-open state."""
        logger.info("Circuit breaker '%s' attempting reset to HALF_OPEN", self.name)
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0

    def _record_success(self):
        """Record successful execution."""
        self.total_successes += 1
        self.last_success_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _record_failure(self, error_message: str):
        """Record failed execution."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        logger.warning("Circuit breaker '%s' recorded failure: %s", self.name, error_message)

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._open_circuit()
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open state opens the circuit
            self._open_circuit()

    def _open_circuit(self):
        """Open the circuit breaker."""
        logger.error("Circuit breaker '%s' opened due to failures", self.name)
        self.state = CircuitBreakerState.OPEN
        self.failure_count = 0  # Reset for next attempt

    def _close_circuit(self):
        """Close the circuit breaker."""
        logger.info("Circuit breaker '%s' closed - service recovered", self.name)
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0

    def force_open(self):
        """Manually open the circuit breaker."""
        logger.warning("Circuit breaker '%s' manually opened", self.name)
        self.state = CircuitBreakerState.OPEN
        self.last_failure_time = time.time()

    def force_close(self):
        """Manually close the circuit breaker."""
        logger.info("Circuit breaker '%s' manually closed", self.name)
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0

    def get_stats(self) -> dict[str, str | int | float | bool]:
        """Get circuit breaker statistics."""
        uptime = time.time() - (self.last_failure_time or time.time())
        success_rate = (
            (self.total_successes / self.total_requests) if self.total_requests > 0 else 0.0
        )

        return {
            "name": self.name,
            "state": self.state.value,
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_timeouts": self.total_timeouts,
            "success_rate": success_rate,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "uptime_seconds": uptime if self.state != CircuitBreakerState.OPEN else 0,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }

    def reset_stats(self):
        """Reset circuit breaker statistics."""
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        logger.info("Circuit breaker '%s' statistics reset", self.name)


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""

    def __init__(self):
        """Initialize circuit breaker manager."""
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]

    def get_all_stats(self) -> dict[str, dict[str, str | int | float | bool]]:
        """Get statistics for all circuit breakers."""
        return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}

    def get_unhealthy_circuits(self) -> dict[str, CircuitBreaker]:
        """Get circuit breakers that are not in closed state."""
        return {
            name: cb
            for name, cb in self.circuit_breakers.items()
            if cb.state != CircuitBreakerState.CLOSED
        }

    def force_close_all(self):
        """Force close all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.force_close()
        logger.info("All circuit breakers forced closed")

    def reset_all_stats(self):
        """Reset statistics for all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset_stats()
        logger.info("All circuit breaker statistics reset")


# Global circuit breaker manager instance
circuit_breaker_manager = CircuitBreakerManager()


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator to add circuit breaker protection to a function."""

    def decorator(func: Callable) -> Callable:
        cb = circuit_breaker_manager.get_or_create(name, config)
        return cb(func)

    return decorator


# Predefined circuit breakers for common DSPy operations
def dspy_circuit_breaker(operation_name: str):
    """Circuit breaker specifically configured for DSPy operations."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=2,
        timeout=60.0,
        expected_exception=DSPyIntegrationError,
    )
    return circuit_breaker(f"dspy_{operation_name}", config)
