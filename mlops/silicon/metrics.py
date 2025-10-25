"""Apple Silicon Metrics Data Structure

Unified metrics representation for Apple Silicon performance tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class AppleSiliconMetrics:
    """Apple Silicon performance metrics

    This is the canonical metrics structure used across all MLOps components
    for Apple Silicon monitoring and tracking.

    Attributes:
        timestamp: When metrics were collected
        chip_type: Chip type (M1, M2, M3, M4)
        chip_variant: Chip variant (Base, Pro, Max, Ultra)
        memory_total_gb: Total unified memory in GB
        memory_used_gb: Used memory in GB
        memory_available_gb: Available memory in GB
        memory_utilization_percent: Memory usage percentage
        mlx_available: Whether MLX framework is available
        mps_available: Whether MPS backend is available
        ane_available: Whether ANE is available
        thermal_state: Thermal state (0=nominal, 1=fair, 2=serious, 3=critical)
        power_mode: Power mode (low_power, normal, high_performance)
        cpu_percent: CPU utilization percentage
        cpu_count: Total CPU cores
        performance_cores: Number of performance cores
        efficiency_cores: Number of efficiency cores
        cpu_freq_mhz: Current CPU frequency in MHz
        mps_utilization_percent: MPS GPU utilization (if available)
    """

    timestamp: datetime
    chip_type: str
    chip_variant: str
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    memory_utilization_percent: float
    mlx_available: bool
    mps_available: bool
    ane_available: bool
    thermal_state: int
    power_mode: str
    cpu_percent: float
    cpu_count: int
    performance_cores: int
    efficiency_cores: int
    cpu_freq_mhz: float
    mps_utilization_percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary

        Returns:
            Dictionary representation of all metrics
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "chip_type": self.chip_type,
            "chip_variant": self.chip_variant,
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "memory_utilization_percent": self.memory_utilization_percent,
            "mlx_available": self.mlx_available,
            "mps_available": self.mps_available,
            "ane_available": self.ane_available,
            "thermal_state": self.thermal_state,
            "power_mode": self.power_mode,
            "cpu_percent": self.cpu_percent,
            "cpu_count": self.cpu_count,
            "performance_cores": self.performance_cores,
            "efficiency_cores": self.efficiency_cores,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "mps_utilization_percent": self.mps_utilization_percent,
        }

    def to_mlflow_metrics(self) -> dict[str, float | int]:
        """Convert to MLFlow-compatible metrics (numeric values only)

        Returns:
            Dictionary of numeric metrics for MLFlow logging
        """
        metrics = {
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "memory_utilization_percent": self.memory_utilization_percent,
            "mlx_available": 1.0 if self.mlx_available else 0.0,
            "mps_available": 1.0 if self.mps_available else 0.0,
            "ane_available": 1.0 if self.ane_available else 0.0,
            "thermal_state": float(self.thermal_state),
            "cpu_percent": self.cpu_percent,
            "cpu_count": self.cpu_count,
            "performance_cores": self.performance_cores,
            "efficiency_cores": self.efficiency_cores,
            "cpu_freq_mhz": self.cpu_freq_mhz,
        }

        if self.mps_utilization_percent is not None:
            metrics["mps_utilization_percent"] = self.mps_utilization_percent

        return metrics

    def is_thermal_throttling(self) -> bool:
        """Check if system is thermal throttling

        Returns:
            True if thermal state is serious or critical
        """
        return self.thermal_state >= 2

    def is_memory_constrained(self, threshold: float = 80.0) -> bool:
        """Check if memory usage is constrained

        Args:
            threshold: Memory utilization threshold percentage

        Returns:
            True if memory utilization exceeds threshold
        """
        return self.memory_utilization_percent >= threshold

    def get_health_score(self) -> float:
        """Calculate system health score (0-100)

        Returns:
            Health score based on thermal state and memory utilization
        """
        # Start with perfect score
        score = 100.0

        # Deduct for thermal state
        score -= self.thermal_state * 20.0

        # Deduct for memory pressure
        if self.memory_utilization_percent > 90:
            score -= 20.0
        elif self.memory_utilization_percent > 80:
            score -= 10.0

        # Deduct for low power mode
        if self.power_mode == "low_power":
            score -= 10.0

        return max(0.0, score)
