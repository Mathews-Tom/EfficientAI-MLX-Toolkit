"""
Apple Silicon metrics collection for MLFlow tracking.

This module provides utilities for collecting Apple Silicon-specific performance metrics
including MPS utilization, thermal state, memory usage, and ANE status.
"""

import logging
import platform
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class AppleSiliconMetricsError(Exception):
    """Raised when Apple Silicon metrics collection fails."""

    def __init__(
        self,
        message: str,
        metric: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.metric = metric
        self.details = dict(details or {})


@dataclass
class AppleSiliconMetrics:
    """
    Apple Silicon performance metrics.

    Attributes:
        chip_type: Apple Silicon chip type (M1, M2, M3, etc.)
        memory_total_gb: Total unified memory in GB
        memory_used_gb: Used memory in GB
        memory_available_gb: Available memory in GB
        memory_utilization_percent: Memory utilization percentage
        mps_available: Whether Metal Performance Shaders is available
        mps_utilization_percent: MPS GPU utilization (if available)
        ane_available: Whether Apple Neural Engine is available
        thermal_state: Thermal state (0=nominal, 1=fair, 2=serious, 3=critical)
        power_mode: Current power mode (low_power, normal, high_performance)
        core_count: Number of CPU cores
        performance_core_count: Number of performance cores
        efficiency_core_count: Number of efficiency cores
    """

    chip_type: str
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    memory_utilization_percent: float
    mps_available: bool
    mps_utilization_percent: float | None = None
    ane_available: bool = False
    thermal_state: int = 0
    power_mode: str = "normal"
    core_count: int = 0
    performance_core_count: int = 0
    efficiency_core_count: int = 0

    def to_dict(self) -> dict[str, str | float | int | bool | None]:
        """
        Convert metrics to dictionary.

        Returns:
            Dictionary representation of metrics
        """
        return {
            "chip_type": self.chip_type,
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "memory_utilization_percent": self.memory_utilization_percent,
            "mps_available": self.mps_available,
            "mps_utilization_percent": self.mps_utilization_percent,
            "ane_available": self.ane_available,
            "thermal_state": self.thermal_state,
            "power_mode": self.power_mode,
            "core_count": self.core_count,
            "performance_core_count": self.performance_core_count,
            "efficiency_core_count": self.efficiency_core_count,
        }

    def to_mlflow_metrics(self) -> dict[str, float | int]:
        """
        Convert metrics to MLFlow-compatible format (numeric values only).

        Returns:
            Dictionary of numeric metrics for MLFlow logging
        """
        metrics = {
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "memory_utilization_percent": self.memory_utilization_percent,
            "mps_available": 1.0 if self.mps_available else 0.0,
            "ane_available": 1.0 if self.ane_available else 0.0,
            "thermal_state": float(self.thermal_state),
            "core_count": self.core_count,
            "performance_core_count": self.performance_core_count,
            "efficiency_core_count": self.efficiency_core_count,
        }

        if self.mps_utilization_percent is not None:
            metrics["mps_utilization_percent"] = self.mps_utilization_percent

        return metrics


def detect_apple_silicon() -> bool:
    """
    Detect if running on Apple Silicon.

    Returns:
        True if running on Apple Silicon hardware
    """
    try:
        system = platform.system()
        machine = platform.machine()
        processor = platform.processor()

        return system == "Darwin" and machine == "arm64" and "arm" in processor.lower()

    except Exception as e:
        logger.warning("Failed to detect Apple Silicon: %s", e)
        return False


def get_chip_type() -> str:
    """
    Get Apple Silicon chip type.

    Returns:
        Chip type string (e.g., "M1", "M2", "M3")
    """
    try:
        # Try to get chip info from sysctl
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )

        brand_string = result.stdout.strip()

        # Extract chip type (M1, M2, M3, etc.)
        if "M1" in brand_string:
            if "Pro" in brand_string:
                return "M1 Pro"
            elif "Max" in brand_string:
                return "M1 Max"
            elif "Ultra" in brand_string:
                return "M1 Ultra"
            else:
                return "M1"
        elif "M2" in brand_string:
            if "Pro" in brand_string:
                return "M2 Pro"
            elif "Max" in brand_string:
                return "M2 Max"
            elif "Ultra" in brand_string:
                return "M2 Ultra"
            else:
                return "M2"
        elif "M3" in brand_string:
            if "Pro" in brand_string:
                return "M3 Pro"
            elif "Max" in brand_string:
                return "M3 Max"
            else:
                return "M3"
        else:
            return "Unknown Apple Silicon"

    except Exception as e:
        logger.warning("Failed to get chip type: %s", e)
        return "Unknown"


def get_memory_info() -> dict[str, float]:
    """
    Get memory information in GB.

    Returns:
        Dictionary with memory metrics in GB
    """
    try:
        # Get memory info from vm_stat
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=True, timeout=5
        )

        output = result.stdout

        # Parse vm_stat output
        page_size = 4096  # Default page size on macOS

        # Extract values
        pages_free = 0
        pages_active = 0
        pages_inactive = 0
        pages_wired = 0

        for line in output.split("\n"):
            if "page size of" in line:
                # Extract page size if present
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

        # Calculate memory in GB
        bytes_per_gb = 1024**3

        free_gb = (pages_free * page_size) / bytes_per_gb
        active_gb = (pages_active * page_size) / bytes_per_gb
        inactive_gb = (pages_inactive * page_size) / bytes_per_gb
        wired_gb = (pages_wired * page_size) / bytes_per_gb

        used_gb = active_gb + wired_gb
        available_gb = free_gb + inactive_gb
        total_gb = used_gb + available_gb

        utilization = (used_gb / total_gb * 100) if total_gb > 0 else 0.0

        return {
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "available_gb": round(available_gb, 2),
            "utilization_percent": round(utilization, 2),
        }

    except Exception as e:
        logger.warning("Failed to get memory info: %s", e)
        return {
            "total_gb": 0.0,
            "used_gb": 0.0,
            "available_gb": 0.0,
            "utilization_percent": 0.0,
        }


def get_mps_info() -> dict[str, bool | float | None]:
    """
    Get Metal Performance Shaders information.

    Returns:
        Dictionary with MPS availability and utilization
    """
    try:
        import torch

        mps_available = torch.backends.mps.is_available()

        # MPS utilization is not directly available, return None
        return {"available": mps_available, "utilization_percent": None}

    except ImportError:
        logger.debug("PyTorch not available for MPS detection")
        return {"available": False, "utilization_percent": None}
    except Exception as e:
        logger.warning("Failed to get MPS info: %s", e)
        return {"available": False, "utilization_percent": None}


def get_ane_info() -> bool:
    """
    Get Apple Neural Engine information.

    Returns:
        True if ANE is available
    """
    try:
        # ANE availability detection is not straightforward
        # For now, assume available on all Apple Silicon
        if detect_apple_silicon():
            return True
        return False

    except Exception as e:
        logger.warning("Failed to get ANE info: %s", e)
        return False


def get_thermal_state() -> int:
    """
    Get thermal state of the system.

    Returns:
        Thermal state: 0=nominal, 1=fair, 2=serious, 3=critical
    """
    try:
        # Thermal state is not easily accessible on macOS
        # This is a simplified implementation
        # In production, you might use IOKit or other system APIs
        return 0  # Assume nominal

    except Exception as e:
        logger.warning("Failed to get thermal state: %s", e)
        return 0


def get_power_mode() -> str:
    """
    Get current power mode.

    Returns:
        Power mode string (low_power, normal, high_performance)
    """
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=True, timeout=5
        )

        output = result.stdout.lower()

        if "low power" in output:
            return "low_power"
        elif "ac power" in output:
            return "high_performance"
        else:
            return "normal"

    except Exception as e:
        logger.debug("Failed to get power mode: %s", e)
        return "normal"


def get_core_info() -> dict[str, int]:
    """
    Get CPU core information.

    Returns:
        Dictionary with core counts
    """
    try:
        # Get total core count
        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True, check=True, timeout=5
        )
        total_cores = int(result.stdout.strip())

        # Try to get performance/efficiency core split
        # This is a simplified approach
        try:
            result_perf = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            perf_cores = int(result_perf.stdout.strip())

            result_eff = subprocess.run(
                ["sysctl", "-n", "hw.perflevel1.logicalcpu"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            eff_cores = int(result_eff.stdout.strip())

        except Exception:
            # If split not available, assume equal distribution
            perf_cores = total_cores // 2
            eff_cores = total_cores - perf_cores

        return {
            "total": total_cores,
            "performance": perf_cores,
            "efficiency": eff_cores,
        }

    except Exception as e:
        logger.warning("Failed to get core info: %s", e)
        return {"total": 0, "performance": 0, "efficiency": 0}


def collect_metrics() -> AppleSiliconMetrics:
    """
    Collect all Apple Silicon metrics.

    Returns:
        AppleSiliconMetrics object with current metrics

    Raises:
        AppleSiliconMetricsError: If not running on Apple Silicon

    Example:
        >>> metrics = collect_metrics()
        >>> print(f"Chip: {metrics.chip_type}")
        >>> print(f"Memory used: {metrics.memory_used_gb} GB")
    """
    if not detect_apple_silicon():
        raise AppleSiliconMetricsError(
            "Not running on Apple Silicon hardware",
            details={"system": platform.system(), "machine": platform.machine()},
        )

    try:
        chip_type = get_chip_type()
        memory_info = get_memory_info()
        mps_info = get_mps_info()
        ane_available = get_ane_info()
        thermal_state = get_thermal_state()
        power_mode = get_power_mode()
        core_info = get_core_info()

        metrics = AppleSiliconMetrics(
            chip_type=chip_type,
            memory_total_gb=memory_info["total_gb"],
            memory_used_gb=memory_info["used_gb"],
            memory_available_gb=memory_info["available_gb"],
            memory_utilization_percent=memory_info["utilization_percent"],
            mps_available=mps_info["available"],
            mps_utilization_percent=mps_info["utilization_percent"],
            ane_available=ane_available,
            thermal_state=thermal_state,
            power_mode=power_mode,
            core_count=core_info["total"],
            performance_core_count=core_info["performance"],
            efficiency_core_count=core_info["efficiency"],
        )

        logger.info("Collected Apple Silicon metrics: %s", chip_type)

        return metrics

    except Exception as e:
        raise AppleSiliconMetricsError(
            f"Failed to collect Apple Silicon metrics: {e}",
            details={"error": str(e)},
        ) from e


def log_metrics_to_mlflow(client: Any) -> None:
    """
    Collect and log Apple Silicon metrics to MLFlow.

    Args:
        client: MLFlow client instance with active run

    Raises:
        AppleSiliconMetricsError: If collection or logging fails

    Example:
        >>> from mlops.client import MLFlowClient
        >>> client = MLFlowClient()
        >>> with client.run(run_name="experiment"):
        ...     log_metrics_to_mlflow(client)
    """
    try:
        metrics = collect_metrics()

        # Log as MLFlow metrics
        client.log_apple_silicon_metrics(metrics.to_mlflow_metrics())

        # Also log chip type as tag
        client.set_tag("apple_silicon.chip_type", metrics.chip_type)
        client.set_tag("apple_silicon.power_mode", metrics.power_mode)

        logger.info("Logged Apple Silicon metrics to MLFlow")

    except Exception as e:
        raise AppleSiliconMetricsError(
            f"Failed to log metrics to MLFlow: {e}",
            details={"error": str(e)},
        ) from e
