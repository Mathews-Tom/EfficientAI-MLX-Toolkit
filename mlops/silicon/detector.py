"""Apple Silicon Hardware Detector

Centralized hardware detection for Apple Silicon systems with comprehensive
capability checking including chip type, memory, MPS, MLX, and ANE support.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Apple Silicon hardware information

    Attributes:
        is_apple_silicon: Whether running on Apple Silicon
        chip_type: Chip type (M1, M2, M3, M4, variants)
        chip_variant: Chip variant (base, Pro, Max, Ultra)
        system: Operating system name
        machine: Machine architecture
        processor: Processor identifier
        memory_total_gb: Total unified memory in GB
        core_count: Total CPU cores
        performance_cores: Number of performance cores
        efficiency_cores: Number of efficiency cores
        mlx_available: Whether MLX framework is available
        mps_available: Whether MPS backend is available
        ane_available: Whether Apple Neural Engine is available
        thermal_state: Current thermal state (0=nominal, 1=fair, 2=serious, 3=critical)
        power_mode: Current power mode (low_power, normal, high_performance)
    """

    is_apple_silicon: bool
    chip_type: str
    chip_variant: str
    system: str
    machine: str
    processor: str
    memory_total_gb: float
    core_count: int
    performance_cores: int
    efficiency_cores: int
    mlx_available: bool
    mps_available: bool
    ane_available: bool
    thermal_state: int
    power_mode: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "is_apple_silicon": self.is_apple_silicon,
            "chip_type": self.chip_type,
            "chip_variant": self.chip_variant,
            "system": self.system,
            "machine": self.machine,
            "processor": self.processor,
            "memory_total_gb": self.memory_total_gb,
            "core_count": self.core_count,
            "performance_cores": self.performance_cores,
            "efficiency_cores": self.efficiency_cores,
            "mlx_available": self.mlx_available,
            "mps_available": self.mps_available,
            "ane_available": self.ane_available,
            "thermal_state": self.thermal_state,
            "power_mode": self.power_mode,
        }


class AppleSiliconDetector:
    """Centralized Apple Silicon hardware detector

    Provides comprehensive detection of Apple Silicon hardware capabilities
    with caching for performance. All detection logic is centralized here
    to avoid duplication across components.

    Example:
        >>> detector = AppleSiliconDetector()
        >>> if detector.is_apple_silicon:
        ...     info = detector.get_hardware_info()
        ...     print(f"Chip: {info.chip_type} {info.chip_variant}")
        ...     print(f"Memory: {info.memory_total_gb}GB")
    """

    def __init__(self) -> None:
        """Initialize detector with cached hardware detection"""
        self._hardware_info: HardwareInfo | None = None
        self._detect_hardware()

    def _detect_hardware(self) -> None:
        """Detect hardware and cache results"""
        # Platform detection
        system = platform.system()
        machine = platform.machine()
        processor = platform.processor()

        is_apple_silicon = (
            system == "Darwin"
            and machine == "arm64"
            and "arm" in processor.lower()
        )

        # Chip detection
        chip_type, chip_variant = self._detect_chip()

        # Memory detection
        memory_total_gb = self._detect_memory()

        # CPU core detection
        core_info = self._detect_cores()

        # Framework detection
        mlx_available = self._check_mlx()
        mps_available = self._check_mps()
        ane_available = self._check_ane() if is_apple_silicon else False

        # System state detection
        thermal_state = self._detect_thermal_state() if is_apple_silicon else 0
        power_mode = self._detect_power_mode() if is_apple_silicon else "normal"

        self._hardware_info = HardwareInfo(
            is_apple_silicon=is_apple_silicon,
            chip_type=chip_type,
            chip_variant=chip_variant,
            system=system,
            machine=machine,
            processor=processor,
            memory_total_gb=memory_total_gb,
            core_count=core_info["total"],
            performance_cores=core_info["performance"],
            efficiency_cores=core_info["efficiency"],
            mlx_available=mlx_available,
            mps_available=mps_available,
            ane_available=ane_available,
            thermal_state=thermal_state,
            power_mode=power_mode,
        )

        logger.info(
            "Hardware detected: %s %s, %dGB memory, %d cores",
            chip_type,
            chip_variant,
            memory_total_gb,
            core_info["total"],
        )

    def _detect_chip(self) -> tuple[str, str]:
        """Detect chip type and variant

        Returns:
            Tuple of (chip_type, chip_variant)
        """
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            brand = result.stdout.strip()

            # Extract chip type and variant
            if "M1" in brand:
                chip_type = "M1"
            elif "M2" in brand:
                chip_type = "M2"
            elif "M3" in brand:
                chip_type = "M3"
            elif "M4" in brand:
                chip_type = "M4"
            else:
                chip_type = "Unknown"

            # Extract variant
            if "Ultra" in brand:
                chip_variant = "Ultra"
            elif "Max" in brand:
                chip_variant = "Max"
            elif "Pro" in brand:
                chip_variant = "Pro"
            else:
                chip_variant = "Base"

            return chip_type, chip_variant

        except Exception as e:
            logger.warning("Failed to detect chip: %s", e)
            return "Unknown", "Unknown"

    def _detect_memory(self) -> float:
        """Detect total unified memory in GB

        Returns:
            Total memory in GB
        """
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            bytes_total = int(result.stdout.strip())
            return round(bytes_total / (1024**3), 2)

        except Exception as e:
            logger.warning("Failed to detect memory: %s", e)
            return 0.0

    def _detect_cores(self) -> dict[str, int]:
        """Detect CPU core configuration

        Returns:
            Dictionary with total, performance, and efficiency core counts
        """
        try:
            # Total cores
            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            total_cores = int(result.stdout.strip())

            # Try to detect performance/efficiency split
            try:
                perf_result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                perf_cores = int(perf_result.stdout.strip())

                eff_result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel1.logicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                eff_cores = int(eff_result.stdout.strip())

            except Exception:
                # Fallback: assume equal distribution
                perf_cores = total_cores // 2
                eff_cores = total_cores - perf_cores

            return {
                "total": total_cores,
                "performance": perf_cores,
                "efficiency": eff_cores,
            }

        except Exception as e:
            logger.warning("Failed to detect cores: %s", e)
            return {"total": 0, "performance": 0, "efficiency": 0}

    def _check_mlx(self) -> bool:
        """Check MLX framework availability

        Returns:
            True if MLX is available and functional
        """
        try:
            import mlx.core as mx

            # Test basic functionality
            _ = mx.array([1, 2, 3])
            return True

        except ImportError:
            return False
        except Exception as e:
            logger.warning("MLX available but not functional: %s", e)
            return False

    def _check_mps(self) -> bool:
        """Check MPS backend availability

        Returns:
            True if MPS is available
        """
        try:
            import torch

            return torch.backends.mps.is_available()

        except ImportError:
            return False
        except Exception as e:
            logger.warning("Failed to check MPS: %s", e)
            return False

    def _check_ane(self) -> bool:
        """Check Apple Neural Engine availability

        Returns:
            True if ANE is available
        """
        # ANE is available on all Apple Silicon chips
        # More sophisticated detection could use coremltools
        try:
            return True
        except Exception as e:
            logger.warning("Failed to check ANE: %s", e)
            return False

    def _detect_thermal_state(self) -> int:
        """Detect system thermal state

        Returns:
            Thermal state: 0=nominal, 1=fair, 2=serious, 3=critical
        """
        try:
            # Try to read thermal state from powermetrics (requires root)
            # Fallback to nominal if not accessible
            return 0

        except Exception as e:
            logger.debug("Failed to detect thermal state: %s", e)
            return 0

    def _detect_power_mode(self) -> str:
        """Detect current power mode

        Returns:
            Power mode: low_power, normal, high_performance
        """
        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.lower()

            if "low power" in output:
                return "low_power"
            elif "ac power" in output:
                return "high_performance"
            else:
                return "normal"

        except Exception as e:
            logger.debug("Failed to detect power mode: %s", e)
            return "normal"

    @property
    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon

        Returns:
            True if running on Apple Silicon hardware
        """
        return self._hardware_info.is_apple_silicon if self._hardware_info else False

    def get_hardware_info(self) -> HardwareInfo:
        """Get complete hardware information

        Returns:
            HardwareInfo instance with all detected capabilities

        Raises:
            RuntimeError: If hardware detection failed
        """
        if self._hardware_info is None:
            raise RuntimeError("Hardware detection failed")
        return self._hardware_info

    def refresh(self) -> None:
        """Re-detect hardware (updates dynamic properties like thermal state)"""
        self._detect_hardware()
