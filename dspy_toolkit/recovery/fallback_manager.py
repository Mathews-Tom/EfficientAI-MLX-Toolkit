"""
Fallback management system for DSPy Integration Framework.
"""

import concurrent.futures
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any

from ..exceptions import DSPyIntegrationError

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategies."""

    SEQUENTIAL = "sequential"  # Try fallbacks in order
    PARALLEL = "parallel"  # Try all fallbacks simultaneously
    WEIGHTED = "weighted"  # Use weighted selection
    ADAPTIVE = "adaptive"  # Learn from success rates


@dataclass
class FallbackConfig:
    """Configuration for fallback manager."""

    strategy: FallbackStrategy = FallbackStrategy.SEQUENTIAL
    max_fallbacks: int = 3
    timeout_per_fallback: float = 30.0
    enable_learning: bool = True
    success_threshold: float = 0.8  # For adaptive strategy


class FallbackOption:
    """Represents a fallback option."""

    def __init__(self, name: str, func: Callable, weight: float = 1.0, enabled: bool = True):
        """Initialize fallback option."""
        self.name = name
        self.func = func
        self.weight = weight
        self.enabled = enabled

        # Statistics
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.avg_response_time = 0.0
        self.last_success_time = 0.0
        self.last_failure_time = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    def record_attempt(self, success: bool, response_time: float):
        """Record attempt result."""
        import time

        self.attempts += 1

        if success:
            self.successes += 1
            self.last_success_time = time.time()
        else:
            self.failures += 1
            self.last_failure_time = time.time()

        # Update average response time
        self.avg_response_time = (
            self.avg_response_time * (self.attempts - 1) + response_time
        ) / self.attempts

    def get_stats(self) -> dict[str, str | int | float | bool]:
        """Get fallback option statistics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "weight": self.weight,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "last_success_time": self.last_success_time,
            "last_failure_time": self.last_failure_time,
        }


class FallbackManager:
    """Manages fallback options and strategies."""

    def __init__(self, config: FallbackConfig = None):
        """Initialize fallback manager."""
        self.config = config or FallbackConfig()
        self.fallback_options: list[FallbackOption] = []
        self.primary_function: Callable | None = None

        # Statistics
        self.total_requests = 0
        self.primary_successes = 0
        self.fallback_successes = 0
        self.total_failures = 0

    def set_primary(self, func: Callable, name: str = "primary"):
        """Set the primary function."""
        self.primary_function = func
        logger.info("Set primary function: %s", name)

    def add_fallback(self, func: Callable, name: str, weight: float = 1.0, enabled: bool = True):
        """Add a fallback option."""
        fallback = FallbackOption(name, func, weight, enabled)
        self.fallback_options.append(fallback)
        logger.info("Added fallback option: %s (weight: %s)", name, weight)

    def remove_fallback(self, name: str):
        """Remove a fallback option by name."""
        self.fallback_options = [f for f in self.fallback_options if f.name != name]
        logger.info("Removed fallback option: %s", name)

    def enable_fallback(self, name: str):
        """Enable a fallback option."""
        for fallback in self.fallback_options:
            if fallback.name == name:
                fallback.enabled = True
                logger.info("Enabled fallback option: %s", name)
                break

    def disable_fallback(self, name: str):
        """Disable a fallback option."""
        for fallback in self.fallback_options:
            if fallback.name == name:
                fallback.enabled = False
                logger.info("Disabled fallback option: %s", name)
                break

    def execute(self, *args, **kwargs) -> Any:
        """Execute with fallback logic."""
        import time

        self.total_requests += 1

        # Try primary function first
        if self.primary_function:
            try:
                start_time = time.time()
                result = self.primary_function(*args, **kwargs)
                response_time = time.time() - start_time

                self.primary_successes += 1
                logger.debug("Primary function succeeded in %.3fs", response_time)
                return result

            except Exception as e:
                logger.warning("Primary function failed: %s", e)

        # Try fallbacks based on strategy
        if self.config.strategy == FallbackStrategy.SEQUENTIAL:
            return self._execute_sequential(*args, **kwargs)
        elif self.config.strategy == FallbackStrategy.PARALLEL:
            return self._execute_parallel(*args, **kwargs)
        elif self.config.strategy == FallbackStrategy.WEIGHTED:
            return self._execute_weighted(*args, **kwargs)
        elif self.config.strategy == FallbackStrategy.ADAPTIVE:
            return self._execute_adaptive(*args, **kwargs)
        else:
            raise DSPyIntegrationError(f"Unknown fallback strategy: {self.config.strategy}")

    def _execute_sequential(self, *args, **kwargs) -> Any:
        """Execute fallbacks sequentially."""
        import time

        enabled_fallbacks = [f for f in self.fallback_options if f.enabled]

        for fallback in enabled_fallbacks[: self.config.max_fallbacks]:
            try:
                start_time = time.time()
                result = fallback.func(*args, **kwargs)
                response_time = time.time() - start_time

                fallback.record_attempt(True, response_time)
                self.fallback_successes += 1

                logger.info("Fallback '%s' succeeded in %.3fs", fallback.name, response_time)
                return result

            except Exception as e:
                response_time = time.time() - start_time
                fallback.record_attempt(False, response_time)
                logger.warning("Fallback '%s' failed: %s", fallback.name, e)
                continue

        # All fallbacks failed
        self.total_failures += 1
        raise DSPyIntegrationError("All fallback options failed")

    def _execute_parallel(self, *args, **kwargs) -> Any:
        """Execute fallbacks in parallel."""
        enabled_fallbacks = [f for f in self.fallback_options if f.enabled]

        if not enabled_fallbacks:
            self.total_failures += 1
            raise DSPyIntegrationError("No enabled fallback options")

        # Use thread pool for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(enabled_fallbacks)) as executor:
            future_to_fallback = {}

            for fallback in enabled_fallbacks[: self.config.max_fallbacks]:
                future = executor.submit(fallback.func, *args, **kwargs)
                future_to_fallback[future] = fallback

            # Wait for first successful result
            for future in concurrent.futures.as_completed(
                future_to_fallback, timeout=self.config.timeout_per_fallback
            ):
                fallback = future_to_fallback[future]

                try:
                    result = future.result()
                    fallback.record_attempt(True, 0.0)  # Would need proper timing
                    self.fallback_successes += 1

                    logger.info("Parallel fallback '%s' succeeded", fallback.name)
                    return result

                except Exception as e:
                    fallback.record_attempt(False, 0.0)
                    logger.warning("Parallel fallback '%s' failed: %s", fallback.name, e)
                    continue

        # All parallel fallbacks failed
        self.total_failures += 1
        raise DSPyIntegrationError("All parallel fallback options failed")

    def _execute_weighted(self, *args, **kwargs) -> Any:
        """Execute fallbacks using weighted selection."""
        import random
        import time

        enabled_fallbacks = [f for f in self.fallback_options if f.enabled]

        if not enabled_fallbacks:
            self.total_failures += 1
            raise DSPyIntegrationError("No enabled fallback options")

        # Calculate weights (higher success rate = higher weight)
        weights = []
        for fallback in enabled_fallbacks:
            # Base weight plus success rate bonus
            weight = fallback.weight * (1 + fallback.success_rate)
            weights.append(weight)

        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            # All weights are zero, use equal probability
            selected_fallback = random.choice(enabled_fallbacks)
        else:
            rand_val = random.uniform(0, total_weight)
            cumulative = 0
            selected_fallback = enabled_fallbacks[0]  # Default

            for i, weight in enumerate(weights):
                cumulative += weight
                if rand_val <= cumulative:
                    selected_fallback = enabled_fallbacks[i]
                    break

        # Execute selected fallback
        try:
            start_time = time.time()
            result = selected_fallback.func(*args, **kwargs)
            response_time = time.time() - start_time

            selected_fallback.record_attempt(True, response_time)
            self.fallback_successes += 1

            logger.info("Weighted fallback '%s' succeeded", selected_fallback.name)
            return result

        except Exception as e:
            response_time = time.time() - start_time
            selected_fallback.record_attempt(False, response_time)
            self.total_failures += 1

            logger.error("Weighted fallback '%s' failed: %s", selected_fallback.name, e)
            raise DSPyIntegrationError(f"Weighted fallback failed: {e}")

    def _execute_adaptive(self, *args, **kwargs) -> Any:
        """Execute fallbacks using adaptive strategy."""
        import time

        enabled_fallbacks = [f for f in self.fallback_options if f.enabled]

        if not enabled_fallbacks:
            self.total_failures += 1
            raise DSPyIntegrationError("No enabled fallback options")

        # Sort by success rate (descending) and response time (ascending)
        sorted_fallbacks = sorted(
            enabled_fallbacks, key=lambda f: (-f.success_rate, f.avg_response_time)
        )

        # Try the best performing fallbacks first
        for fallback in sorted_fallbacks[: self.config.max_fallbacks]:
            # Skip fallbacks with very low success rates (unless no other option)
            if (
                fallback.success_rate < self.config.success_threshold
                and len(sorted_fallbacks) > 1
                and fallback != sorted_fallbacks[-1]
            ):
                continue

            try:
                start_time = time.time()
                result = fallback.func(*args, **kwargs)
                response_time = time.time() - start_time

                fallback.record_attempt(True, response_time)
                self.fallback_successes += 1

                logger.info("Adaptive fallback '%s' succeeded", fallback.name)
                return result

            except Exception as e:
                response_time = time.time() - start_time
                fallback.record_attempt(False, response_time)
                logger.warning("Adaptive fallback '%s' failed: %s", fallback.name, e)
                continue

        # All adaptive fallbacks failed
        self.total_failures += 1
        raise DSPyIntegrationError("All adaptive fallback options failed")

    def get_stats(self) -> dict[str, str | int | float | bool]:
        """Get fallback manager statistics."""
        success_rate = (
            (self.primary_successes + self.fallback_successes) / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

        fallback_usage_rate = (
            self.fallback_successes / self.total_requests if self.total_requests > 0 else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "primary_successes": self.primary_successes,
            "fallback_successes": self.fallback_successes,
            "total_failures": self.total_failures,
            "success_rate": success_rate,
            "fallback_usage_rate": fallback_usage_rate,
            "strategy": self.config.strategy.value,
            "fallback_options": [f.get_stats() for f in self.fallback_options],
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.total_requests = 0
        self.primary_successes = 0
        self.fallback_successes = 0
        self.total_failures = 0

        for fallback in self.fallback_options:
            fallback.attempts = 0
            fallback.successes = 0
            fallback.failures = 0
            fallback.avg_response_time = 0.0

        logger.info("Fallback manager statistics reset")


def fallback_manager(config: FallbackConfig = None):
    """Decorator to create a fallback manager for a function."""

    def decorator(func: Callable) -> Callable:
        manager = FallbackManager(config)
        manager.set_primary(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return manager.execute(*args, **kwargs)

        # Attach manager to wrapper for external access
        wrapper.fallback_manager = manager
        return wrapper

    return decorator


# Predefined fallback configurations
def dspy_fallback_manager():
    """Fallback manager specifically configured for DSPy operations."""
    config = FallbackConfig(
        strategy=FallbackStrategy.ADAPTIVE,
        max_fallbacks=2,
        timeout_per_fallback=60.0,
        enable_learning=True,
        success_threshold=0.7,
    )
    return fallback_manager(config)
