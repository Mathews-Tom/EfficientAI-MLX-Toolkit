"""
Health checking system for DSPy Integration Framework components.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComponentHealth(Enum):
    """Component health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""

    check_interval: float = 30.0  # Seconds between checks
    timeout: float = 10.0  # Timeout for individual checks
    failure_threshold: int = 3  # Failures before marking unhealthy
    recovery_threshold: int = 2  # Successes needed to recover
    enable_auto_recovery: bool = True
    alert_on_state_change: bool = True


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    component_name: str
    status: ComponentHealth
    response_time: float
    timestamp: float
    message: str
    details: dict[str, str | int | float | bool]
    error: str | None = None


class HealthCheck:
    """Individual health check for a component."""

    def __init__(self, name: str, check_func: Callable, config: HealthCheckConfig = None):
        """Initialize health check."""
        self.name = name
        self.check_func = check_func
        self.config = config or HealthCheckConfig()

        # State tracking
        self.current_status = ComponentHealth.UNKNOWN
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_check_time = 0.0
        self.last_success_time = 0.0
        self.last_failure_time = 0.0

        # Statistics
        self.total_checks = 0
        self.total_successes = 0
        self.total_failures = 0
        self.avg_response_time = 0.0

        # History
        self.check_history: list[HealthCheckResult] = []
        self.max_history = 100

    async def execute_check(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = time.time()
        self.total_checks += 1

        try:
            # Execute check function with timeout
            if asyncio.iscoroutinefunction(self.check_func):
                result = await asyncio.wait_for(self.check_func(), timeout=self.config.timeout)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, self.check_func),
                    timeout=self.config.timeout,
                )

            response_time = time.time() - start_time

            # Process result
            if isinstance(result, dict):
                status = ComponentHealth(result.get("status", "healthy"))
                message = result.get("message", "Health check passed")
                details = result.get("details", {})
            elif isinstance(result, bool):
                status = ComponentHealth.HEALTHY if result else ComponentHealth.UNHEALTHY
                message = "Health check passed" if result else "Health check failed"
                details = {}
            else:
                status = ComponentHealth.HEALTHY
                message = str(result)
                details = {}

            # Update statistics
            self._record_success(response_time)

            # Create result
            check_result = HealthCheckResult(
                component_name=self.name,
                status=status,
                response_time=response_time,
                timestamp=time.time(),
                message=message,
                details=details,
            )

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            self._record_failure(response_time)

            check_result = HealthCheckResult(
                component_name=self.name,
                status=ComponentHealth.UNHEALTHY,
                response_time=response_time,
                timestamp=time.time(),
                message="Health check timed out",
                details={},
                error="Timeout",
            )

        except Exception as e:
            response_time = time.time() - start_time
            self._record_failure(response_time)

            check_result = HealthCheckResult(
                component_name=self.name,
                status=ComponentHealth.UNHEALTHY,
                response_time=response_time,
                timestamp=time.time(),
                message=f"Health check failed: {e}",
                details={},
                error=str(e),
            )

        # Update component status
        self._update_status(check_result)

        # Add to history
        self.check_history.append(check_result)
        if len(self.check_history) > self.max_history:
            self.check_history.pop(0)

        return check_result

    def _record_success(self, response_time: float):
        """Record successful check."""
        self.total_successes += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()

        # Update average response time
        self.avg_response_time = (
            self.avg_response_time * (self.total_checks - 1) + response_time
        ) / self.total_checks

    def _record_failure(self, response_time: float):
        """Record failed check."""
        self.total_failures += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()

        # Update average response time
        self.avg_response_time = (
            self.avg_response_time * (self.total_checks - 1) + response_time
        ) / self.total_checks

    def _update_status(self, check_result: HealthCheckResult):
        """Update component status based on check result."""
        previous_status = self.current_status

        # Determine new status based on consecutive results
        if check_result.status == ComponentHealth.HEALTHY:
            if self.consecutive_successes >= self.config.recovery_threshold:
                self.current_status = ComponentHealth.HEALTHY
        elif check_result.status == ComponentHealth.UNHEALTHY:
            if self.consecutive_failures >= self.config.failure_threshold:
                self.current_status = ComponentHealth.UNHEALTHY
            elif self.current_status == ComponentHealth.HEALTHY:
                self.current_status = ComponentHealth.DEGRADED
        else:
            self.current_status = check_result.status

        # Log status changes
        if previous_status != self.current_status:
            logger.info(
                "Component '%s' status changed: %s -> %s",
                self.name,
                previous_status.value,
                self.current_status.value,
            )

            if self.config.alert_on_state_change:
                self._send_alert(previous_status, self.current_status)

    def _send_alert(self, old_status: ComponentHealth, new_status: ComponentHealth):
        """Send alert for status change."""
        # Placeholder for alert integration
        logger.warning(
            "HEALTH ALERT: %s status changed from %s to %s",
            self.name,
            old_status.value,
            new_status.value,
        )

    def get_stats(self) -> dict[str, str | int | float | bool]:
        """Get health check statistics."""
        success_rate = self.total_successes / self.total_checks if self.total_checks > 0 else 0.0
        uptime = time.time() - self.last_failure_time if self.last_failure_time > 0 else time.time()

        return {
            "name": self.name,
            "current_status": self.current_status.value,
            "total_checks": self.total_checks,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": success_rate,
            "avg_response_time": self.avg_response_time,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_check_time": self.last_check_time,
            "last_success_time": self.last_success_time,
            "last_failure_time": self.last_failure_time,
            "uptime_seconds": (uptime if self.current_status != ComponentHealth.UNHEALTHY else 0),
        }


class HealthChecker:
    """Main health checker that manages multiple health checks."""

    def __init__(self, config: HealthCheckConfig = None):
        """Initialize health checker."""
        self.config = config or HealthCheckConfig()
        self.health_checks: dict[str, HealthCheck] = {}
        self.monitoring_task: asyncio.Task | None = None
        self.is_monitoring = False

    def add_health_check(
        self,
        name: str,
        check_func: Callable,
        config: HealthCheckConfig | None = None,
    ):
        """Add a health check."""
        check_config = config or self.config
        health_check = HealthCheck(name, check_func, check_config)
        self.health_checks[name] = health_check
        logger.info("Added health check: %s", name)

    def remove_health_check(self, name: str):
        """Remove a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info("Removed health check: %s", name)

    async def check_all(self) -> dict[str, HealthCheckResult]:
        """Execute all health checks."""
        results = {}

        # Execute all checks concurrently
        tasks = []
        for name, health_check in self.health_checks.items():
            task = asyncio.create_task(health_check.execute_check())
            tasks.append((name, task))

        # Collect results
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                logger.error("Health check '%s' failed with exception: %s", name, e)
                results[name] = HealthCheckResult(
                    component_name=name,
                    status=ComponentHealth.UNHEALTHY,
                    response_time=0.0,
                    timestamp=time.time(),
                    message=f"Health check exception: {e}",
                    details={},
                    error=str(e),
                )

        return results

    async def check_component(self, name: str) -> HealthCheckResult | None:
        """Execute health check for specific component."""
        if name not in self.health_checks:
            logger.error("Health check '%s' not found", name)
            return None

        return await self.health_checks[name].execute_check()

    def start_monitoring(self):
        """Start continuous health monitoring."""
        if not self.is_monitoring:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.is_monitoring = True
            logger.info("Started health monitoring")

    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
            self.is_monitoring = False
            logger.info("Stopped health monitoring")

    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        try:
            while True:
                try:
                    await self.check_all()
                    await asyncio.sleep(self.config.check_interval)
                except Exception as e:
                    logger.error("Health monitoring error: %s", e)
                    await asyncio.sleep(self.config.check_interval)
        except asyncio.CancelledError:
            logger.info("Health monitoring cancelled")

    def get_overall_health(self) -> dict[str, str | int | float | bool]:
        """Get overall system health."""
        if not self.health_checks:
            return {
                "status": ComponentHealth.UNKNOWN.value,
                "message": "No health checks configured",
                "components": {},
                "summary": {
                    "total": 0,
                    "healthy": 0,
                    "degraded": 0,
                    "unhealthy": 0,
                    "unknown": 0,
                },
            }

        component_statuses = {}
        status_counts = {status.value: 0 for status in ComponentHealth}

        for name, health_check in self.health_checks.items():
            status = health_check.current_status
            component_statuses[name] = status.value
            status_counts[status.value] += 1

        # Determine overall status
        if status_counts["unhealthy"] > 0:
            overall_status = ComponentHealth.UNHEALTHY
        elif status_counts["degraded"] > 0:
            overall_status = ComponentHealth.DEGRADED
        elif status_counts["healthy"] == len(self.health_checks):
            overall_status = ComponentHealth.HEALTHY
        else:
            overall_status = ComponentHealth.UNKNOWN

        return {
            "status": overall_status.value,
            "message": f"System health: {overall_status.value}",
            "components": component_statuses,
            "summary": {
                "total": len(self.health_checks),
                "healthy": status_counts["healthy"],
                "degraded": status_counts["degraded"],
                "unhealthy": status_counts["unhealthy"],
                "unknown": status_counts["unknown"],
            },
            "timestamp": time.time(),
        }

    def get_all_stats(self) -> dict[str, dict[str, str | int | float | bool]]:
        """Get statistics for all health checks."""
        return {name: check.get_stats() for name, check in self.health_checks.items()}

    def get_unhealthy_components(self) -> list[str]:
        """Get list of unhealthy components."""
        return [
            name
            for name, check in self.health_checks.items()
            if check.current_status == ComponentHealth.UNHEALTHY
        ]

    def get_degraded_components(self) -> list[str]:
        """Get list of degraded components."""
        return [
            name
            for name, check in self.health_checks.items()
            if check.current_status == ComponentHealth.DEGRADED
        ]


# Predefined health check functions for DSPy components
async def dspy_framework_health_check(framework) -> dict[str, str | int | float | bool]:
    """Health check for DSPy framework."""
    try:
        health = framework.health_check()

        if health["overall_status"] == "healthy":
            return {
                "status": "healthy",
                "message": "DSPy framework is healthy",
                "details": health,
            }
        else:
            return {
                "status": "degraded",
                "message": f"DSPy framework issues: {len(health.get('issues', []))} problems",
                "details": health,
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"DSPy framework health check failed: {e}",
            "details": {"error": str(e)},
        }


async def mlx_provider_health_check(provider) -> dict[str, str | int | float | bool]:
    """Health check for MLX provider."""
    try:
        if provider and provider.is_available():
            # Try a simple benchmark
            benchmark_result = provider.benchmark_performance("Health check")

            return {
                "status": "healthy",
                "message": "MLX provider is healthy",
                "details": {"available": True, "benchmark": benchmark_result},
            }
        else:
            return {
                "status": "unhealthy",
                "message": "MLX provider is not available",
                "details": {"available": False},
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"MLX provider health check failed: {e}",
            "details": {"error": str(e)},
        }
