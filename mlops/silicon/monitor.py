"""Apple Silicon Real-time Monitor

Collects real-time performance metrics from Apple Silicon hardware with
support for continuous monitoring and alerting.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from typing import Any

import psutil

from mlops.silicon.detector import AppleSiliconDetector, HardwareInfo
from mlops.silicon.metrics import AppleSiliconMetrics

logger = logging.getLogger(__name__)


class AppleSiliconMonitor:
    """Real-time Apple Silicon performance monitor

    Collects continuous performance metrics including memory usage, CPU
    utilization, thermal state, and power mode. Integrates with detector
    for hardware capabilities.

    Example:
        >>> from mlops.silicon import AppleSiliconMonitor
        >>>
        >>> monitor = AppleSiliconMonitor()
        >>> metrics = monitor.collect()
        >>> print(f"Memory used: {metrics.memory_used_gb}GB")
        >>> print(f"CPU: {metrics.cpu_percent}%")
        >>> print(f"Health: {metrics.get_health_score()}/100")
    """

    def __init__(
        self,
        detector: AppleSiliconDetector | None = None,
        project_name: str = "default",
    ) -> None:
        """Initialize monitor

        Args:
            detector: Optional hardware detector instance
            project_name: Project name for logging
        """
        self.project_name = project_name
        self.detector = detector or AppleSiliconDetector()
        self.hardware_info = self.detector.get_hardware_info()

        logger.info(
            "Monitor initialized for project: %s (Apple Silicon: %s)",
            project_name,
            self.hardware_info.is_apple_silicon,
        )

    def collect(self) -> AppleSiliconMetrics:
        """Collect current metrics

        Returns:
            AppleSiliconMetrics with current system state

        Raises:
            RuntimeError: If metrics collection fails
        """
        try:
            # Refresh hardware info for dynamic properties
            self.detector.refresh()
            self.hardware_info = self.detector.get_hardware_info()

            # Collect memory metrics
            memory_info = self._collect_memory()

            # Collect CPU metrics
            cpu_info = self._collect_cpu()

            # Collect MPS utilization if available
            mps_util = self._collect_mps_utilization()

            metrics = AppleSiliconMetrics(
                timestamp=datetime.now(),
                chip_type=self.hardware_info.chip_type,
                chip_variant=self.hardware_info.chip_variant,
                memory_total_gb=self.hardware_info.memory_total_gb,
                memory_used_gb=memory_info["used_gb"],
                memory_available_gb=memory_info["available_gb"],
                memory_utilization_percent=memory_info["utilization_percent"],
                mlx_available=self.hardware_info.mlx_available,
                mps_available=self.hardware_info.mps_available,
                ane_available=self.hardware_info.ane_available,
                thermal_state=self.hardware_info.thermal_state,
                power_mode=self.hardware_info.power_mode,
                cpu_percent=cpu_info["percent"],
                cpu_count=self.hardware_info.core_count,
                performance_cores=self.hardware_info.performance_cores,
                efficiency_cores=self.hardware_info.efficiency_cores,
                cpu_freq_mhz=cpu_info["freq_mhz"],
                mps_utilization_percent=mps_util,
            )

            logger.debug(
                "Collected metrics: %.1f%% CPU, %.1fGB memory, health=%.0f",
                metrics.cpu_percent,
                metrics.memory_used_gb,
                metrics.get_health_score(),
            )

            return metrics

        except Exception as e:
            logger.error("Failed to collect metrics: %s", e)
            raise RuntimeError(f"Metrics collection failed: {e}") from e

    def _collect_memory(self) -> dict[str, float]:
        """Collect memory metrics

        Returns:
            Dictionary with memory metrics
        """
        try:
            if self.hardware_info.is_apple_silicon:
                # Use vm_stat for accurate unified memory on macOS
                return self._collect_memory_vmstat()
            else:
                # Fallback to psutil
                return self._collect_memory_psutil()

        except Exception as e:
            logger.warning("Failed to collect memory via vm_stat, using psutil: %s", e)
            return self._collect_memory_psutil()

    def _collect_memory_vmstat(self) -> dict[str, float]:
        """Collect memory using vm_stat (macOS specific)

        Returns:
            Dictionary with memory metrics in GB
        """
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        output = result.stdout
        page_size = 4096  # Default page size

        # Parse vm_stat output
        pages_free = 0
        pages_active = 0
        pages_inactive = 0
        pages_wired = 0

        for line in output.split("\n"):
            if "page size of" in line:
                parts = line.split()
                if len(parts) >= 8:
                    page_size = int(parts[7])
            elif "Pages free:" in line:
                pages_free = int(line.split(":")[1].strip().replace(".", ""))
            elif "Pages active:" in line:
                pages_active = int(line.split(":")[1].strip().replace(".", ""))
            elif "Pages inactive:" in line:
                pages_inactive = int(line.split(":")[1].strip().replace(".", ""))
            elif "Pages wired down:" in line:
                pages_wired = int(line.split(":")[1].strip().replace(".", ""))

        # Calculate in GB
        bytes_per_gb = 1024**3
        free_gb = (pages_free * page_size) / bytes_per_gb
        active_gb = (pages_active * page_size) / bytes_per_gb
        inactive_gb = (pages_inactive * page_size) / bytes_per_gb
        wired_gb = (pages_wired * page_size) / bytes_per_gb

        used_gb = active_gb + wired_gb
        available_gb = free_gb + inactive_gb
        total_gb = self.hardware_info.memory_total_gb
        utilization = (used_gb / total_gb * 100) if total_gb > 0 else 0.0

        return {
            "used_gb": round(used_gb, 2),
            "available_gb": round(available_gb, 2),
            "utilization_percent": round(utilization, 2),
        }

    def _collect_memory_psutil(self) -> dict[str, float]:
        """Collect memory using psutil (fallback)

        Returns:
            Dictionary with memory metrics in GB
        """
        memory = psutil.virtual_memory()
        return {
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "utilization_percent": memory.percent,
        }

    def _collect_cpu(self) -> dict[str, float]:
        """Collect CPU metrics

        Returns:
            Dictionary with CPU metrics
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0

        return {
            "percent": cpu_percent,
            "freq_mhz": round(cpu_freq_mhz, 2),
        }

    def _collect_mps_utilization(self) -> float | None:
        """Collect MPS GPU utilization

        Returns:
            MPS utilization percentage or None if not available
        """
        # MPS utilization is not directly accessible
        # Would require Metal profiling APIs
        return None

    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon

        Returns:
            True if running on Apple Silicon
        """
        return self.hardware_info.is_apple_silicon

    def get_hardware_summary(self) -> dict[str, Any]:
        """Get hardware summary

        Returns:
            Dictionary with hardware information
        """
        return self.hardware_info.to_dict()

    def check_health(self) -> dict[str, Any]:
        """Check system health

        Returns:
            Dictionary with health status and recommendations
        """
        metrics = self.collect()

        health = {
            "score": metrics.get_health_score(),
            "thermal_throttling": metrics.is_thermal_throttling(),
            "memory_constrained": metrics.is_memory_constrained(),
            "thermal_state": metrics.thermal_state,
            "memory_utilization": metrics.memory_utilization_percent,
            "power_mode": metrics.power_mode,
            "recommendations": [],
        }

        # Generate recommendations
        if metrics.is_thermal_throttling():
            health["recommendations"].append(
                "System is thermal throttling. Reduce workload or improve cooling."
            )

        if metrics.is_memory_constrained():
            health["recommendations"].append(
                f"Memory usage at {metrics.memory_utilization_percent}%. "
                "Consider reducing batch size or worker count."
            )

        if metrics.power_mode == "low_power":
            health["recommendations"].append(
                "System in low power mode. Connect to power for best performance."
            )

        if not health["recommendations"]:
            health["recommendations"].append("System health is good.")

        return health
