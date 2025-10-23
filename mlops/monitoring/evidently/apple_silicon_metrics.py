"""Apple Silicon Metrics Collector

Collects Apple Silicon-specific performance metrics including MPS utilization,
unified memory usage, thermal state, and ANE availability.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class AppleSiliconMetrics:
    """Apple Silicon hardware metrics

    Attributes:
        timestamp: When metrics were collected
        chip_type: Apple Silicon chip type (M1, M2, M3, etc.)
        memory_total_gb: Total unified memory in GB
        memory_used_gb: Used unified memory in GB
        memory_available_gb: Available unified memory in GB
        memory_percent: Memory usage percentage
        mlx_available: Whether MLX framework is available
        mps_available: Whether MPS (Metal Performance Shaders) is available
        ane_available: Whether ANE (Apple Neural Engine) is available
        thermal_state: System thermal state
        power_mode: Current power mode
        cpu_percent: CPU usage percentage
        cpu_count: Number of CPU cores
        cpu_freq_mhz: Current CPU frequency in MHz
    """

    timestamp: datetime
    chip_type: str
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float
    mlx_available: bool
    mps_available: bool
    ane_available: bool
    thermal_state: str
    power_mode: str
    cpu_percent: float
    cpu_count: int
    cpu_freq_mhz: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary

        Returns:
            Dictionary representation of metrics
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "chip_type": self.chip_type,
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "memory_percent": self.memory_percent,
            "mlx_available": self.mlx_available,
            "mps_available": self.mps_available,
            "ane_available": self.ane_available,
            "thermal_state": self.thermal_state,
            "power_mode": self.power_mode,
            "cpu_percent": self.cpu_percent,
            "cpu_count": self.cpu_count,
            "cpu_freq_mhz": self.cpu_freq_mhz,
        }


class AppleSiliconMetricsCollector:
    """Collects Apple Silicon-specific metrics for monitoring

    This collector gathers hardware metrics specific to Apple Silicon:
    - Unified memory usage
    - MPS and MLX availability
    - ANE (Apple Neural Engine) status
    - Thermal state and power mode
    - CPU metrics optimized for Apple Silicon architecture
    """

    def __init__(self, project_name: str = "default"):
        """Initialize metrics collector

        Args:
            project_name: Name of the project being monitored
        """
        self.project_name = project_name
        self._is_apple_silicon = self._detect_apple_silicon()

        logger.info(
            "Initialized AppleSiliconMetricsCollector for project: %s (Apple Silicon: %s)",
            project_name,
            self._is_apple_silicon,
        )

    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon

        Returns:
            True if running on Apple Silicon, False otherwise
        """
        try:
            return platform.system() == "Darwin" and platform.machine() == "arm64"
        except Exception as e:
            logger.warning("Failed to detect Apple Silicon: %s", e)
            return False

    def _get_chip_type(self) -> str:
        """Get Apple Silicon chip type

        Returns:
            Chip type (M1, M2, M3, etc.) or 'Unknown'
        """
        if not self._is_apple_silicon:
            return "Not Apple Silicon"

        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            brand = result.stdout.strip()

            # Extract chip type (M1, M2, M3, etc.)
            if "M1" in brand:
                return "M1"
            elif "M2" in brand:
                return "M2"
            elif "M3" in brand:
                return "M3"
            elif "M4" in brand:
                return "M4"
            else:
                return "Apple Silicon"

        except Exception as e:
            logger.warning("Failed to get chip type: %s", e)
            return "Unknown"

    def _check_mlx_available(self) -> bool:
        """Check if MLX framework is available

        Returns:
            True if MLX is available, False otherwise
        """
        try:
            import mlx.core as mx

            # Simple test to verify MLX works
            _ = mx.array([1, 2, 3])
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.warning("MLX available but not functional: %s", e)
            return False

    def _check_mps_available(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available

        Returns:
            True if MPS is available, False otherwise
        """
        try:
            import torch

            return torch.backends.mps.is_available()
        except ImportError:
            return False
        except Exception as e:
            logger.warning("Failed to check MPS availability: %s", e)
            return False

    def _check_ane_available(self) -> bool:
        """Check if ANE (Apple Neural Engine) is available

        Returns:
            True if ANE is available, False otherwise
        """
        if not self._is_apple_silicon:
            return False

        try:
            # ANE is available on all Apple Silicon chips
            # More sophisticated detection could be added
            return True
        except Exception as e:
            logger.warning("Failed to check ANE availability: %s", e)
            return False

    def _get_thermal_state(self) -> str:
        """Get system thermal state

        Returns:
            Thermal state (nominal, fair, serious, critical)
        """
        if not self._is_apple_silicon:
            return "not_available"

        try:
            # This is a placeholder - actual thermal state detection
            # would require macOS-specific APIs
            return "nominal"
        except Exception as e:
            logger.warning("Failed to get thermal state: %s", e)
            return "unknown"

    def _get_power_mode(self) -> str:
        """Get current power mode

        Returns:
            Power mode (low_power, automatic, high_performance)
        """
        if not self._is_apple_silicon:
            return "not_available"

        try:
            # This is a placeholder - actual power mode detection
            # would require macOS-specific APIs
            return "automatic"
        except Exception as e:
            logger.warning("Failed to get power mode: %s", e)
            return "unknown"

    def collect(self) -> AppleSiliconMetrics:
        """Collect current Apple Silicon metrics

        Returns:
            AppleSiliconMetrics instance with current metrics

        Raises:
            RuntimeError: If metrics collection fails
        """
        try:
            # Get memory info
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)

            # Get CPU info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0

            # Collect metrics
            metrics = AppleSiliconMetrics(
                timestamp=datetime.now(),
                chip_type=self._get_chip_type(),
                memory_total_gb=round(memory_total_gb, 2),
                memory_used_gb=round(memory_used_gb, 2),
                memory_available_gb=round(memory_available_gb, 2),
                memory_percent=memory.percent,
                mlx_available=self._check_mlx_available(),
                mps_available=self._check_mps_available(),
                ane_available=self._check_ane_available(),
                thermal_state=self._get_thermal_state(),
                power_mode=self._get_power_mode(),
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                cpu_freq_mhz=round(cpu_freq_mhz, 2),
            )

            logger.debug("Collected Apple Silicon metrics: %s", metrics.chip_type)
            return metrics

        except Exception as e:
            logger.error("Failed to collect Apple Silicon metrics: %s", e)
            raise RuntimeError(f"Metrics collection failed: {e}") from e

    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon

        Returns:
            True if running on Apple Silicon, False otherwise
        """
        return self._is_apple_silicon
