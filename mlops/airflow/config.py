"""Configuration for Thermal-Aware Airflow Scheduling

This module provides configuration classes and defaults for thermal-aware
task scheduling on Apple Silicon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThermalSchedulingConfig:
    """Configuration for thermal-aware scheduling

    Attributes:
        thermal_threshold_nominal: Threshold for nominal operation (0-3)
        thermal_threshold_fair: Threshold for fair operation (0-3)
        thermal_threshold_serious: Threshold for serious operation (0-3)
        thermal_threshold_critical: Threshold for critical operation (0-3)
        memory_threshold_percent: Memory utilization threshold (0-100)
        min_health_score: Minimum health score to run tasks (0-100)
        retry_delays: Retry delays in seconds by thermal state
        throttle_levels: Throttle levels (0.0-1.0) by thermal state
        clearance_timeout_seconds: Default timeout for thermal clearance
        check_interval_seconds: Interval between thermal checks
    """
    thermal_threshold_nominal: int = 0
    thermal_threshold_fair: int = 1
    thermal_threshold_serious: int = 2
    thermal_threshold_critical: int = 3

    memory_threshold_percent: float = 85.0
    min_health_score: float = 40.0

    retry_delays: dict[int, int] = field(default_factory=lambda: {
        0: 10,   # nominal
        1: 30,   # fair
        2: 60,   # serious
        3: 120,  # critical
    })

    throttle_levels: dict[int, float] = field(default_factory=lambda: {
        0: 1.0,   # nominal - full speed
        1: 0.9,   # fair - slight throttle
        2: 0.7,   # serious - moderate throttle
        3: 0.5,   # critical - heavy throttle
    })

    clearance_timeout_seconds: int = 300
    check_interval_seconds: int = 10

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ThermalSchedulingConfig:
        """Create config from dictionary

        Args:
            config: Configuration dictionary

        Returns:
            ThermalSchedulingConfig instance
        """
        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary

        Returns:
            Configuration dictionary
        """
        return {
            "thermal_threshold_nominal": self.thermal_threshold_nominal,
            "thermal_threshold_fair": self.thermal_threshold_fair,
            "thermal_threshold_serious": self.thermal_threshold_serious,
            "thermal_threshold_critical": self.thermal_threshold_critical,
            "memory_threshold_percent": self.memory_threshold_percent,
            "min_health_score": self.min_health_score,
            "retry_delays": self.retry_delays,
            "throttle_levels": self.throttle_levels,
            "clearance_timeout_seconds": self.clearance_timeout_seconds,
            "check_interval_seconds": self.check_interval_seconds,
        }


@dataclass
class TaskConfig:
    """Configuration for individual task thermal requirements

    Attributes:
        task_id: Unique task identifier
        thermal_threshold: Maximum thermal state allowed (0-3)
        memory_required_gb: Memory required for task
        priority: Task priority (low, normal, high, critical)
        retry_on_thermal: Whether to retry on thermal throttle
        wait_for_clearance: Whether to wait for thermal clearance
        max_retries: Maximum number of retries
        retry_delay_seconds: Delay between retries
    """
    task_id: str
    thermal_threshold: int = 2
    memory_required_gb: float | None = None
    priority: str = "normal"
    retry_on_thermal: bool = True
    wait_for_clearance: bool = True
    max_retries: int = 3
    retry_delay_seconds: int = 60

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> TaskConfig:
        """Create task config from dictionary

        Args:
            config: Task configuration dictionary

        Returns:
            TaskConfig instance
        """
        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        """Convert task config to dictionary

        Returns:
            Task configuration dictionary
        """
        return {
            "task_id": self.task_id,
            "thermal_threshold": self.thermal_threshold,
            "memory_required_gb": self.memory_required_gb,
            "priority": self.priority,
            "retry_on_thermal": self.retry_on_thermal,
            "wait_for_clearance": self.wait_for_clearance,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
        }


# Default configurations for different task types
TASK_PROFILES: dict[str, dict[str, Any]] = {
    "training": {
        "thermal_threshold": 2,
        "memory_required_gb": 16.0,
        "priority": "high",
        "retry_on_thermal": True,
        "wait_for_clearance": True,
        "max_retries": 5,
        "retry_delay_seconds": 120,
    },
    "inference": {
        "thermal_threshold": 1,
        "memory_required_gb": 8.0,
        "priority": "normal",
        "retry_on_thermal": True,
        "wait_for_clearance": False,
        "max_retries": 3,
        "retry_delay_seconds": 30,
    },
    "preprocessing": {
        "thermal_threshold": 1,
        "memory_required_gb": 4.0,
        "priority": "normal",
        "retry_on_thermal": False,
        "wait_for_clearance": False,
        "max_retries": 2,
        "retry_delay_seconds": 20,
    },
    "evaluation": {
        "thermal_threshold": 1,
        "memory_required_gb": 8.0,
        "priority": "normal",
        "retry_on_thermal": True,
        "wait_for_clearance": False,
        "max_retries": 3,
        "retry_delay_seconds": 60,
    },
    "deployment": {
        "thermal_threshold": 3,
        "memory_required_gb": 2.0,
        "priority": "critical",
        "retry_on_thermal": True,
        "wait_for_clearance": True,
        "max_retries": 5,
        "retry_delay_seconds": 30,
    },
}


def get_task_profile(profile_name: str) -> dict[str, Any]:
    """Get predefined task profile configuration

    Args:
        profile_name: Profile name (training, inference, preprocessing, etc.)

    Returns:
        Task configuration dictionary

    Raises:
        ValueError: If profile name not found
    """
    if profile_name not in TASK_PROFILES:
        raise ValueError(
            f"Unknown task profile: {profile_name}. "
            f"Available: {list(TASK_PROFILES.keys())}"
        )
    return TASK_PROFILES[profile_name].copy()


def create_task_config(
    task_id: str,
    profile: str | None = None,
    **overrides: Any,
) -> TaskConfig:
    """Create task configuration with optional profile and overrides

    Args:
        task_id: Unique task identifier
        profile: Profile name to use as base (optional)
        **overrides: Configuration overrides

    Returns:
        TaskConfig instance

    Example:
        >>> config = create_task_config(
        ...     "train_model",
        ...     profile="training",
        ...     thermal_threshold=1,  # Override default
        ... )
    """
    config = {"task_id": task_id}

    # Apply profile if specified
    if profile:
        config.update(get_task_profile(profile))

    # Apply overrides
    config.update(overrides)

    return TaskConfig.from_dict(config)
