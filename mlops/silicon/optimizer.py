"""Apple Silicon Configuration Optimizer

Provides optimization recommendations based on hardware capabilities and
current system state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mlops.silicon.detector import HardwareInfo

logger = logging.getLogger(__name__)


@dataclass
class OptimalConfig:
    """Optimal configuration recommendations

    Attributes:
        workers: Recommended number of worker processes
        batch_size: Recommended batch size
        memory_limit_gb: Recommended memory limit in GB
        use_mlx: Whether to use MLX framework
        use_mps: Whether to use MPS backend
        use_ane: Whether to use Apple Neural Engine
        prefetch_batches: Number of batches to prefetch
        cpu_threads: Recommended number of CPU threads per worker
        recommendations: List of optimization recommendations
    """

    workers: int
    batch_size: int
    memory_limit_gb: float
    use_mlx: bool
    use_mps: bool
    use_ane: bool
    prefetch_batches: int
    cpu_threads: int
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "workers": self.workers,
            "batch_size": self.batch_size,
            "memory_limit_gb": self.memory_limit_gb,
            "use_mlx": self.use_mlx,
            "use_mps": self.use_mps,
            "use_ane": self.use_ane,
            "prefetch_batches": self.prefetch_batches,
            "cpu_threads": self.cpu_threads,
            "recommendations": self.recommendations,
        }


class AppleSiliconOptimizer:
    """Apple Silicon configuration optimizer

    Analyzes hardware capabilities and provides optimization recommendations
    for MLOps workloads on Apple Silicon.

    Example:
        >>> from mlops.silicon import AppleSiliconDetector, AppleSiliconOptimizer
        >>>
        >>> detector = AppleSiliconDetector()
        >>> info = detector.get_hardware_info()
        >>> optimizer = AppleSiliconOptimizer(info)
        >>> config = optimizer.get_optimal_config()
        >>> print(f"Recommended workers: {config.workers}")
    """

    def __init__(self, hardware_info: HardwareInfo) -> None:
        """Initialize optimizer with hardware information

        Args:
            hardware_info: Hardware information from detector
        """
        self.hardware_info = hardware_info
        logger.info(
            "Optimizer initialized for %s %s with %dGB memory",
            hardware_info.chip_type,
            hardware_info.chip_variant,
            hardware_info.memory_total_gb,
        )

    def get_optimal_config(
        self,
        workload_type: str = "inference",
        memory_intensive: bool = False,
    ) -> OptimalConfig:
        """Get optimal configuration for workload

        Args:
            workload_type: Type of workload (inference, training, serving)
            memory_intensive: Whether workload is memory intensive

        Returns:
            OptimalConfig with recommendations
        """
        if not self.hardware_info.is_apple_silicon:
            return self._get_fallback_config()

        recommendations = []

        # Determine worker count
        workers = self._calculate_workers(workload_type)
        recommendations.append(
            f"Use {workers} workers for {workload_type} workload"
        )

        # Determine batch size
        batch_size = self._calculate_batch_size(memory_intensive)
        recommendations.append(
            f"Use batch size {batch_size} based on {self.hardware_info.memory_total_gb}GB memory"
        )

        # Memory limit (80% of total for safety)
        memory_limit_gb = round(self.hardware_info.memory_total_gb * 0.8, 2)
        recommendations.append(
            f"Set memory limit to {memory_limit_gb}GB (80% of total)"
        )

        # Framework selection
        use_mlx = self.hardware_info.mlx_available
        use_mps = self.hardware_info.mps_available and not use_mlx
        use_ane = self.hardware_info.ane_available

        if use_mlx:
            recommendations.append("Use MLX framework for optimal performance")
        elif use_mps:
            recommendations.append("Use MPS backend for GPU acceleration")

        if use_ane:
            recommendations.append("ANE available for CoreML workloads")

        # Prefetch configuration
        prefetch_batches = self._calculate_prefetch(workload_type)

        # CPU thread configuration
        cpu_threads = self._calculate_cpu_threads()

        # Thermal recommendations
        if self.hardware_info.thermal_state >= 2:
            recommendations.append(
                "WARNING: System in thermal throttling, reduce worker count"
            )
            workers = max(1, workers // 2)

        # Power mode recommendations
        if self.hardware_info.power_mode == "low_power":
            recommendations.append(
                "System in low power mode, consider reducing workload"
            )
            workers = max(1, workers // 2)
            batch_size = max(1, batch_size // 2)

        return OptimalConfig(
            workers=workers,
            batch_size=batch_size,
            memory_limit_gb=memory_limit_gb,
            use_mlx=use_mlx,
            use_mps=use_mps,
            use_ane=use_ane,
            prefetch_batches=prefetch_batches,
            cpu_threads=cpu_threads,
            recommendations=recommendations,
        )

    def _calculate_workers(self, workload_type: str) -> int:
        """Calculate optimal worker count

        Args:
            workload_type: Type of workload

        Returns:
            Recommended number of workers
        """
        # Base on performance cores
        perf_cores = self.hardware_info.performance_cores

        if workload_type == "training":
            # Training: conservative to avoid thermal throttling
            return min(2, max(1, perf_cores // 2))
        elif workload_type == "serving":
            # Serving: more workers for throughput
            return min(4, max(2, perf_cores))
        else:  # inference
            # Inference: balanced approach
            return min(2, max(1, perf_cores // 2))

    def _calculate_batch_size(self, memory_intensive: bool) -> int:
        """Calculate optimal batch size

        Args:
            memory_intensive: Whether workload is memory intensive

        Returns:
            Recommended batch size
        """
        memory_gb = self.hardware_info.memory_total_gb

        if memory_intensive:
            # Conservative for memory-intensive workloads
            if memory_gb >= 64:
                return 64
            elif memory_gb >= 32:
                return 32
            elif memory_gb >= 16:
                return 16
            else:
                return 8
        else:
            # Standard workloads
            if memory_gb >= 64:
                return 128
            elif memory_gb >= 32:
                return 64
            elif memory_gb >= 16:
                return 32
            else:
                return 16

    def _calculate_prefetch(self, workload_type: str) -> int:
        """Calculate prefetch batch count

        Args:
            workload_type: Type of workload

        Returns:
            Number of batches to prefetch
        """
        if workload_type == "training":
            return 3  # More prefetch for training
        else:
            return 2  # Standard prefetch

    def _calculate_cpu_threads(self) -> int:
        """Calculate CPU threads per worker

        Returns:
            Recommended CPU threads per worker
        """
        return max(1, self.hardware_info.performance_cores // 2)

    def _get_fallback_config(self) -> OptimalConfig:
        """Get fallback configuration for non-Apple Silicon

        Returns:
            Conservative default configuration
        """
        return OptimalConfig(
            workers=1,
            batch_size=32,
            memory_limit_gb=8.0,
            use_mlx=False,
            use_mps=False,
            use_ane=False,
            prefetch_batches=2,
            cpu_threads=4,
            recommendations=["Not running on Apple Silicon, using default configuration"],
        )

    def get_deployment_config(self) -> dict[str, Any]:
        """Get configuration for BentoML/Ray Serve deployment

        Returns:
            Deployment configuration dictionary
        """
        config = self.get_optimal_config(workload_type="serving")

        return {
            "workers": config.workers,
            "max_batch_size": config.batch_size,
            "max_latency_ms": 10000,
            "memory_per_worker_mb": int(config.memory_limit_gb * 1024),
            "cpu_per_worker": float(config.cpu_threads),
            "apple_silicon": {
                "enabled": self.hardware_info.is_apple_silicon,
                "mlx": config.use_mlx,
                "mps": config.use_mps,
                "ane": config.use_ane,
                "thermal_aware": True,
            },
        }

    def get_training_config(self, memory_intensive: bool = False) -> dict[str, Any]:
        """Get configuration for training workloads

        Args:
            memory_intensive: Whether training is memory intensive

        Returns:
            Training configuration dictionary
        """
        config = self.get_optimal_config(
            workload_type="training",
            memory_intensive=memory_intensive,
        )

        return {
            "batch_size": config.batch_size,
            "num_workers": config.workers,
            "prefetch_factor": config.prefetch_batches,
            "pin_memory": False,  # Not needed on Apple Silicon
            "persistent_workers": True,
            "use_mlx": config.use_mlx,
            "use_mps": config.use_mps,
            "memory_limit_gb": config.memory_limit_gb,
        }
