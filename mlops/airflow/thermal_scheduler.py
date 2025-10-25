"""Thermal-Aware Scheduler for Airflow

This module provides intelligent task scheduling that respects thermal constraints
on Apple Silicon hardware. It integrates with AppleSiliconMonitor to make scheduling
decisions based on real-time thermal state, memory pressure, and power mode.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from mlops.silicon.monitor import AppleSiliconMonitor

logger = logging.getLogger(__name__)


@dataclass
class ThermalThresholds:
    """Thermal threshold configuration for scheduling decisions

    Attributes:
        nominal: Threshold for nominal operation (0-3 scale)
        fair: Threshold for fair operation with some constraints
        serious: Threshold for serious thermal state with heavy throttling
        critical: Threshold for critical state where only essential tasks run
    """
    nominal: int = 0
    fair: int = 1
    serious: int = 2
    critical: int = 3


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision

    Attributes:
        should_run: Whether the task should run now
        reason: Human-readable reason for the decision
        retry_after_seconds: Suggested retry delay if task should not run
        throttle_level: Suggested throttle level (0.0-1.0)
        thermal_state: Current thermal state
        health_score: System health score (0-100)
    """
    should_run: bool
    reason: str
    retry_after_seconds: int | None = None
    throttle_level: float = 1.0
    thermal_state: int = 0
    health_score: float = 100.0


class ThermalAwareScheduler:
    """Scheduler that makes intelligent decisions based on thermal state

    This scheduler integrates with AppleSiliconMonitor to track real-time
    thermal conditions and makes scheduling decisions accordingly.

    Example:
        >>> from mlops.airflow.thermal_scheduler import ThermalAwareScheduler
        >>>
        >>> scheduler = ThermalAwareScheduler()
        >>> decision = scheduler.should_run_task(
        ...     task_id="train_model",
        ...     thermal_threshold=2,
        ...     memory_required_gb=8.0
        ... )
        >>> if decision.should_run:
        ...     print(f"Running task with throttle: {decision.throttle_level}")
        ... else:
        ...     print(f"Delaying task: {decision.reason}")
    """

    def __init__(
        self,
        monitor: AppleSiliconMonitor | None = None,
        thresholds: ThermalThresholds | None = None,
        memory_threshold_percent: float = 85.0,
        min_health_score: float = 40.0,
    ) -> None:
        """Initialize thermal-aware scheduler

        Args:
            monitor: AppleSiliconMonitor instance (created if not provided)
            thresholds: Thermal thresholds configuration
            memory_threshold_percent: Maximum memory utilization before throttling
            min_health_score: Minimum health score to allow task execution
        """
        self.monitor = monitor or AppleSiliconMonitor(project_name="airflow")
        self.thresholds = thresholds or ThermalThresholds()
        self.memory_threshold = memory_threshold_percent
        self.min_health_score = min_health_score

        logger.info(
            "ThermalAwareScheduler initialized (memory_threshold=%.1f%%, min_health=%.1f)",
            memory_threshold_percent,
            min_health_score,
        )

    def should_run_task(
        self,
        task_id: str,
        thermal_threshold: int = 2,
        memory_required_gb: float | None = None,
        priority: str = "normal",
    ) -> SchedulingDecision:
        """Determine if a task should run based on thermal state

        Args:
            task_id: Unique task identifier for logging
            thermal_threshold: Maximum thermal state allowed (0-3)
            memory_required_gb: Memory required for task (GB)
            priority: Task priority (low, normal, high, critical)

        Returns:
            SchedulingDecision with should_run flag and reasoning
        """
        try:
            # Collect current metrics
            metrics = self.monitor.collect()

            # Check thermal constraints
            if metrics.thermal_state > thermal_threshold:
                retry_seconds = self._calculate_retry_delay(metrics.thermal_state)
                return SchedulingDecision(
                    should_run=False,
                    reason=f"Thermal state {metrics.thermal_state} exceeds threshold {thermal_threshold}",
                    retry_after_seconds=retry_seconds,
                    thermal_state=metrics.thermal_state,
                    health_score=metrics.get_health_score(),
                )

            # Check health score
            health_score = metrics.get_health_score()
            if health_score < self.min_health_score:
                # Allow critical priority tasks even with low health
                if priority != "critical":
                    return SchedulingDecision(
                        should_run=False,
                        reason=f"System health {health_score:.1f} below minimum {self.min_health_score}",
                        retry_after_seconds=60,
                        thermal_state=metrics.thermal_state,
                        health_score=health_score,
                    )

            # Check memory constraints
            if memory_required_gb is not None:
                if metrics.memory_available_gb < memory_required_gb:
                    return SchedulingDecision(
                        should_run=False,
                        reason=f"Insufficient memory: {metrics.memory_available_gb:.1f}GB available, {memory_required_gb}GB required",
                        retry_after_seconds=30,
                        thermal_state=metrics.thermal_state,
                        health_score=health_score,
                    )

            # Check memory pressure
            if metrics.memory_utilization_percent > self.memory_threshold:
                # Allow high priority tasks with throttling
                if priority in ("high", "critical"):
                    throttle = 0.7  # Run at 70% capacity
                    logger.warning(
                        "Task %s running with throttle %.1f due to memory pressure",
                        task_id,
                        throttle,
                    )
                    return SchedulingDecision(
                        should_run=True,
                        reason="Running with throttle due to memory pressure",
                        throttle_level=throttle,
                        thermal_state=metrics.thermal_state,
                        health_score=health_score,
                    )
                else:
                    return SchedulingDecision(
                        should_run=False,
                        reason=f"Memory pressure: {metrics.memory_utilization_percent:.1f}% > {self.memory_threshold}%",
                        retry_after_seconds=20,
                        thermal_state=metrics.thermal_state,
                        health_score=health_score,
                    )

            # Check thermal throttling
            if metrics.is_thermal_throttling():
                # Critical tasks can run with heavy throttling
                if priority == "critical":
                    throttle = 0.5
                    logger.warning(
                        "Critical task %s running with throttle %.1f due to thermal throttling",
                        task_id,
                        throttle,
                    )
                    return SchedulingDecision(
                        should_run=True,
                        reason="Critical task running with heavy throttle",
                        throttle_level=throttle,
                        thermal_state=metrics.thermal_state,
                        health_score=health_score,
                    )
                else:
                    retry_seconds = self._calculate_retry_delay(metrics.thermal_state)
                    return SchedulingDecision(
                        should_run=False,
                        reason="System is thermal throttling",
                        retry_after_seconds=retry_seconds,
                        thermal_state=metrics.thermal_state,
                        health_score=health_score,
                    )

            # Calculate throttle level based on thermal state
            throttle = self._calculate_throttle_level(metrics.thermal_state)

            logger.info(
                "Task %s approved: thermal=%d, health=%.1f, throttle=%.2f",
                task_id,
                metrics.thermal_state,
                health_score,
                throttle,
            )

            return SchedulingDecision(
                should_run=True,
                reason="System conditions within acceptable limits",
                throttle_level=throttle,
                thermal_state=metrics.thermal_state,
                health_score=health_score,
            )

        except Exception as e:
            logger.error("Error in scheduling decision for task %s: %s", task_id, e)
            # Fail safe: allow task to run with throttling
            return SchedulingDecision(
                should_run=True,
                reason=f"Scheduling check failed, allowing with throttle: {e}",
                throttle_level=0.8,
            )

    def _calculate_retry_delay(self, thermal_state: int) -> int:
        """Calculate retry delay based on thermal state

        Args:
            thermal_state: Current thermal state (0-3)

        Returns:
            Retry delay in seconds
        """
        delays = {
            0: 10,   # nominal
            1: 30,   # fair
            2: 60,   # serious
            3: 120,  # critical
        }
        return delays.get(thermal_state, 60)

    def _calculate_throttle_level(self, thermal_state: int) -> float:
        """Calculate throttle level based on thermal state

        Args:
            thermal_state: Current thermal state (0-3)

        Returns:
            Throttle level (0.0-1.0)
        """
        throttle_map = {
            0: 1.0,   # nominal - full speed
            1: 0.9,   # fair - slight throttle
            2: 0.7,   # serious - moderate throttle
            3: 0.5,   # critical - heavy throttle
        }
        return throttle_map.get(thermal_state, 0.8)

    def wait_for_thermal_clearance(
        self,
        task_id: str,
        thermal_threshold: int = 2,
        timeout_seconds: int = 300,
        check_interval_seconds: int = 10,
    ) -> bool:
        """Wait for thermal conditions to improve

        Args:
            task_id: Task identifier for logging
            thermal_threshold: Desired thermal threshold
            timeout_seconds: Maximum time to wait
            check_interval_seconds: Interval between checks

        Returns:
            True if conditions improved, False if timeout
        """
        start_time = time.time()

        logger.info(
            "Task %s waiting for thermal clearance (threshold=%d, timeout=%ds)",
            task_id,
            thermal_threshold,
            timeout_seconds,
        )

        while time.time() - start_time < timeout_seconds:
            decision = self.should_run_task(task_id, thermal_threshold)

            if decision.should_run:
                logger.info(
                    "Task %s thermal clearance granted after %.1fs",
                    task_id,
                    time.time() - start_time,
                )
                return True

            logger.debug(
                "Task %s waiting: %s (retry in %ds)",
                task_id,
                decision.reason,
                check_interval_seconds,
            )

            time.sleep(check_interval_seconds)

        logger.warning(
            "Task %s thermal clearance timeout after %.1fs",
            task_id,
            time.time() - start_time,
        )
        return False

    def get_scheduling_stats(self) -> dict[str, Any]:
        """Get current scheduling statistics

        Returns:
            Dictionary with scheduling state and recommendations
        """
        try:
            metrics = self.monitor.collect()
            health = self.monitor.check_health()

            return {
                "timestamp": datetime.now().isoformat(),
                "thermal_state": metrics.thermal_state,
                "health_score": metrics.get_health_score(),
                "memory_utilization": metrics.memory_utilization_percent,
                "memory_available_gb": metrics.memory_available_gb,
                "cpu_percent": metrics.cpu_percent,
                "power_mode": metrics.power_mode,
                "thermal_throttling": metrics.is_thermal_throttling(),
                "memory_constrained": metrics.is_memory_constrained(),
                "recommendations": health["recommendations"],
                "can_schedule_normal": metrics.thermal_state <= self.thresholds.fair,
                "can_schedule_high": metrics.thermal_state <= self.thresholds.serious,
                "can_schedule_critical": metrics.thermal_state <= self.thresholds.critical,
            }
        except Exception as e:
            logger.error("Failed to get scheduling stats: %s", e)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def suggest_task_configuration(
        self,
        task_type: str = "training",
    ) -> dict[str, Any]:
        """Suggest optimal task configuration based on current state

        Args:
            task_type: Type of task (training, inference, preprocessing, etc.)

        Returns:
            Dictionary with suggested configuration
        """
        metrics = self.monitor.collect()

        # Base configurations by task type
        base_configs = {
            "training": {
                "batch_size": 32,
                "workers": 4,
                "memory_gb": 16.0,
            },
            "inference": {
                "batch_size": 64,
                "workers": 2,
                "memory_gb": 8.0,
            },
            "preprocessing": {
                "batch_size": 128,
                "workers": metrics.cpu_count,
                "memory_gb": 4.0,
            },
        }

        config = base_configs.get(task_type, base_configs["training"]).copy()

        # Adjust based on thermal state
        thermal_multiplier = self._calculate_throttle_level(metrics.thermal_state)
        config["batch_size"] = int(config["batch_size"] * thermal_multiplier)
        config["workers"] = max(1, int(config["workers"] * thermal_multiplier))

        # Adjust based on memory availability
        available_memory = metrics.memory_available_gb * 0.8  # Leave 20% headroom
        config["memory_gb"] = min(config["memory_gb"], available_memory)

        # Add thermal-aware retry configuration
        config["retry_delay"] = self._calculate_retry_delay(metrics.thermal_state)
        config["max_retries"] = 3 if metrics.thermal_state < 2 else 5

        return {
            "config": config,
            "thermal_state": metrics.thermal_state,
            "health_score": metrics.get_health_score(),
            "throttle_level": thermal_multiplier,
        }
