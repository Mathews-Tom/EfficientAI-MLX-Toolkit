"""
Retry handler with exponential backoff for DSPy Integration Framework.
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from ..exceptions import DSPyIntegrationError

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry handler."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # "exponential", "linear", "fixed"
    retryable_exceptions: list[type[Exception]] = None


class RetryHandler:
    """Retry handler with configurable backoff strategies."""

    def __init__(self, config: RetryConfig = None):
        """Initialize retry handler."""
        self.config = config or RetryConfig()

        if self.config.retryable_exceptions is None:
            self.config.retryable_exceptions = [
                DSPyIntegrationError,
                ConnectionError,
                TimeoutError,
            ]

        # Statistics
        self.total_attempts = 0
        self.total_retries = 0
        self.total_successes = 0
        self.total_failures = 0

    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to function."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)

        return wrapper

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            self.total_attempts += 1

            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info("Function succeeded on attempt %d", attempt)
                self.total_successes += 1
                return result

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.error("Non-retryable exception: %s", e)
                    self.total_failures += 1
                    raise

                # Don't retry on last attempt
                if attempt == self.config.max_attempts:
                    logger.error("All %d attempts failed", self.config.max_attempts)
                    self.total_failures += 1
                    break

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.warning("Attempt %d failed: %s. Retrying in %.2fs", attempt, e, delay)

                self.total_retries += 1
                time.sleep(delay)

        # All attempts failed
        raise DSPyIntegrationError(
            f"Function failed after {self.config.max_attempts} attempts. "
            f"Last error: {last_exception}"
        )

    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        return any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.config.backoff_strategy == "exponential":
            delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        elif self.config.backoff_strategy == "linear":
            delay = self.config.base_delay * attempt
        elif self.config.backoff_strategy == "fixed":
            delay = self.config.base_delay
        else:
            raise ValueError(f"Unknown backoff strategy: {self.config.backoff_strategy}")

        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def get_stats(self) -> dict:
        """Get retry handler statistics."""
        success_rate = (
            (self.total_successes / self.total_attempts) if self.total_attempts > 0 else 0.0
        )
        avg_retries = (
            (self.total_retries / self.total_successes) if self.total_successes > 0 else 0.0
        )

        return {
            "total_attempts": self.total_attempts,
            "total_retries": self.total_retries,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": success_rate,
            "average_retries_per_success": avg_retries,
            "config": {
                "max_attempts": self.config.max_attempts,
                "base_delay": self.config.base_delay,
                "max_delay": self.config.max_delay,
                "backoff_strategy": self.config.backoff_strategy,
                "jitter": self.config.jitter,
            },
        }

    def reset_stats(self):
        """Reset retry handler statistics."""
        self.total_attempts = 0
        self.total_retries = 0
        self.total_successes = 0
        self.total_failures = 0


class AsyncRetryHandler:
    """Async version of retry handler."""

    def __init__(self, config: RetryConfig = None):
        """Initialize async retry handler."""
        self.config = config or RetryConfig()

        if self.config.retryable_exceptions is None:
            self.config.retryable_exceptions = [
                DSPyIntegrationError,
                ConnectionError,
                TimeoutError,
            ]

        # Statistics
        self.total_attempts = 0
        self.total_retries = 0
        self.total_successes = 0
        self.total_failures = 0

    def __call__(self, func: Callable) -> Callable:
        """Decorator to add async retry logic to function."""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return wrapper

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        import asyncio

        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            self.total_attempts += 1

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                if attempt > 1:
                    logger.info("Async function succeeded on attempt %d", attempt)
                self.total_successes += 1
                return result

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.error("Non-retryable async exception: %s", e)
                    self.total_failures += 1
                    raise

                # Don't retry on last attempt
                if attempt == self.config.max_attempts:
                    logger.error("All %d async attempts failed", self.config.max_attempts)
                    self.total_failures += 1
                    break

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.warning("Async attempt %d failed: %s. Retrying in %.2fs", attempt, e, delay)

                self.total_retries += 1
                await asyncio.sleep(delay)

        # All attempts failed
        raise DSPyIntegrationError(
            f"Async function failed after {self.config.max_attempts} attempts. "
            f"Last error: {last_exception}"
        )

    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        return any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.config.backoff_strategy == "exponential":
            delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        elif self.config.backoff_strategy == "linear":
            delay = self.config.base_delay * attempt
        elif self.config.backoff_strategy == "fixed":
            delay = self.config.base_delay
        else:
            raise ValueError(f"Unknown backoff strategy: {self.config.backoff_strategy}")

        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def get_stats(self) -> dict:
        """Get async retry handler statistics."""
        success_rate = (
            (self.total_successes / self.total_attempts) if self.total_attempts > 0 else 0.0
        )
        avg_retries = (
            (self.total_retries / self.total_successes) if self.total_successes > 0 else 0.0
        )

        return {
            "total_attempts": self.total_attempts,
            "total_retries": self.total_retries,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": success_rate,
            "average_retries_per_success": avg_retries,
            "config": {
                "max_attempts": self.config.max_attempts,
                "base_delay": self.config.base_delay,
                "max_delay": self.config.max_delay,
                "backoff_strategy": self.config.backoff_strategy,
                "jitter": self.config.jitter,
            },
        }


# Convenience decorators
def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_strategy: str = "exponential",
    retryable_exceptions: list[type[Exception]] = None,
):
    """Decorator for adding retry logic with custom configuration."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        backoff_strategy=backoff_strategy,
        retryable_exceptions=retryable_exceptions,
    )
    return RetryHandler(config)


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_strategy: str = "exponential",
    retryable_exceptions: list[type[Exception]] = None,
):
    """Decorator for adding async retry logic with custom configuration."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        backoff_strategy=backoff_strategy,
        retryable_exceptions=retryable_exceptions,
    )
    return AsyncRetryHandler(config)


# Predefined retry handlers for common DSPy operations
def dspy_retry(max_attempts: int = 3):
    """Retry handler specifically configured for DSPy operations."""
    return retry(
        max_attempts=max_attempts,
        base_delay=2.0,
        backoff_strategy="exponential",
        retryable_exceptions=[DSPyIntegrationError, ConnectionError, TimeoutError],
    )


def dspy_async_retry(max_attempts: int = 3):
    """Async retry handler specifically configured for DSPy operations."""
    return async_retry(
        max_attempts=max_attempts,
        base_delay=2.0,
        backoff_strategy="exponential",
        retryable_exceptions=[DSPyIntegrationError, ConnectionError, TimeoutError],
    )
