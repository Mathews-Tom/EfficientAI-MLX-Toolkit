"""Apple Silicon Integration Helpers

Provides backward-compatible wrappers and integration helpers for existing
components to migrate to the centralized Apple Silicon module.
"""

from __future__ import annotations

import logging
from typing import Any

from mlops.silicon.detector import AppleSiliconDetector
from mlops.silicon.metrics import AppleSiliconMetrics
from mlops.silicon.monitor import AppleSiliconMonitor
from mlops.silicon.optimizer import AppleSiliconOptimizer

logger = logging.getLogger(__name__)


def detect_apple_silicon() -> bool:
    """Detect if running on Apple Silicon (backward compatible)

    Returns:
        True if running on Apple Silicon
    """
    detector = AppleSiliconDetector()
    return detector.is_apple_silicon


def get_chip_type() -> str:
    """Get chip type (backward compatible)

    Returns:
        Chip type string (e.g., "M1 Pro", "M2 Max")
    """
    detector = AppleSiliconDetector()
    info = detector.get_hardware_info()
    return f"{info.chip_type} {info.chip_variant}".strip()


def collect_apple_silicon_metrics() -> AppleSiliconMetrics:
    """Collect Apple Silicon metrics (backward compatible)

    Returns:
        AppleSiliconMetrics instance

    Raises:
        RuntimeError: If not running on Apple Silicon
    """
    monitor = AppleSiliconMonitor()
    return monitor.collect()


def get_optimal_config_for_bentoml(
    project_name: str = "default",
) -> dict[str, Any]:
    """Get optimal BentoML configuration for Apple Silicon

    Args:
        project_name: Project identifier

    Returns:
        Configuration dictionary for BentoML
    """
    detector = AppleSiliconDetector()

    if not detector.is_apple_silicon:
        return {
            "workers": 1,
            "max_batch_size": 32,
            "enable_apple_silicon": False,
        }

    info = detector.get_hardware_info()
    optimizer = AppleSiliconOptimizer(info)

    return optimizer.get_deployment_config()


def get_optimal_config_for_training(
    memory_intensive: bool = False,
) -> dict[str, Any]:
    """Get optimal training configuration for Apple Silicon

    Args:
        memory_intensive: Whether training is memory intensive

    Returns:
        Configuration dictionary for training
    """
    detector = AppleSiliconDetector()

    if not detector.is_apple_silicon:
        return {
            "batch_size": 32,
            "num_workers": 1,
            "use_mlx": False,
            "use_mps": False,
        }

    info = detector.get_hardware_info()
    optimizer = AppleSiliconOptimizer(info)

    return optimizer.get_training_config(memory_intensive=memory_intensive)


def check_thermal_state() -> tuple[int, str]:
    """Check thermal state (backward compatible)

    Returns:
        Tuple of (thermal_state_code, thermal_state_name)
    """
    detector = AppleSiliconDetector()
    info = detector.get_hardware_info()

    state_names = {
        0: "nominal",
        1: "fair",
        2: "serious",
        3: "critical",
    }

    return info.thermal_state, state_names.get(info.thermal_state, "unknown")


def get_memory_info() -> dict[str, float]:
    """Get memory information (backward compatible)

    Returns:
        Dictionary with memory metrics in GB
    """
    monitor = AppleSiliconMonitor()
    metrics = monitor.collect()

    return {
        "total_gb": metrics.memory_total_gb,
        "used_gb": metrics.memory_used_gb,
        "available_gb": metrics.memory_available_gb,
        "utilization_percent": metrics.memory_utilization_percent,
    }


class AppleSiliconMetricsCollector:
    """Backward-compatible metrics collector wrapper

    This class provides the same interface as the old monitoring/evidently
    implementation for easy migration.
    """

    def __init__(self, project_name: str = "default") -> None:
        """Initialize collector

        Args:
            project_name: Project name for logging
        """
        self.project_name = project_name
        self._monitor = AppleSiliconMonitor(project_name=project_name)
        self._is_apple_silicon = self._monitor.is_apple_silicon()

    def collect(self) -> AppleSiliconMetrics:
        """Collect current metrics

        Returns:
            AppleSiliconMetrics instance

        Raises:
            RuntimeError: If metrics collection fails
        """
        return self._monitor.collect()

    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon

        Returns:
            True if running on Apple Silicon
        """
        return self._is_apple_silicon


# Compatibility aliases for existing code
def log_metrics_to_mlflow(client: Any) -> None:
    """Log Apple Silicon metrics to MLFlow (backward compatible)

    Args:
        client: MLFlow client with active run
    """
    try:
        monitor = AppleSiliconMonitor()
        metrics = monitor.collect()

        # Log metrics
        client.log_apple_silicon_metrics(metrics.to_mlflow_metrics())

        # Log tags
        client.set_tag("apple_silicon.chip_type", metrics.chip_type)
        client.set_tag("apple_silicon.chip_variant", metrics.chip_variant)
        client.set_tag("apple_silicon.power_mode", metrics.power_mode)

        logger.info("Logged Apple Silicon metrics to MLFlow")

    except Exception as e:
        logger.warning("Failed to log Apple Silicon metrics: %s", e)
