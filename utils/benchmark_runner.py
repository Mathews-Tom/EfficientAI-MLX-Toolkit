"""
Standardized benchmarking framework with Apple Silicon optimization support.

This module provides comprehensive benchmarking capabilities for the EfficientAI-MLX-Toolkit,
with specialized support for Apple Silicon hardware detection and performance measurement.
"""

import logging
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Optional Apple Silicon dependencies (truly optional for hardware-specific features)
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

# System monitoring (declared in pyproject.toml)
import psutil

logger = logging.getLogger(__name__)


class BenchmarkError(Exception):
    """Raised when benchmark operations fail."""

    def __init__(
        self,
        message: str,
        benchmark_name: str | None = None,
        details: Mapping[str, str | int | float] | None = None,
    ) -> None:
        super().__init__(message)
        self.benchmark_name = benchmark_name
        self.details = dict(details or {})


@dataclass
class HardwareInfo:
    """Hardware information for benchmarking context."""

    platform: str
    processor: str
    machine: str
    python_version: str
    mlx_available: bool
    mps_available: bool
    memory_total: float | None = None
    cpu_count: int | None = None


@dataclass
class BenchmarkResult:
    """Results from a benchmark execution."""

    name: str
    execution_time: float
    memory_usage: dict[str, float] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    hardware_info: HardwareInfo | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    success: bool = True
    error_message: str | None = None


class BenchmarkRunner:
    """
    Standardized benchmark runner with Apple Silicon optimization support.

    This class provides comprehensive benchmarking capabilities including
    performance measurement, memory usage tracking, and hardware-specific
    optimizations for Apple Silicon.

    Attributes:
        results_dir: Directory for storing benchmark results
        hardware_info: Detected hardware information
    """

    def __init__(self, results_dir: Path | None = None) -> None:
        """
        Initialize the benchmark runner.

        Args:
            results_dir: Directory for storing benchmark results
        """
        self.results_dir = results_dir or Path("benchmark_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.hardware_info = self._detect_hardware()
        self._results: list[BenchmarkResult] = []

        logger.info("Initialized benchmark runner with hardware: %s", self.hardware_info.platform)

    def _detect_hardware(self) -> HardwareInfo:
        """Detect and return hardware information."""
        try:
            # Check for MPS availability (Apple Silicon GPU)
            mps_available = False
            try:
                import torch

                mps_available = torch.backends.mps.is_available()
            except ImportError:
                pass

            # Get memory information
            memory_total = psutil.virtual_memory().total / (1024**3)  # GB
            cpu_count = psutil.cpu_count()

            hardware_info = HardwareInfo(
                platform=platform.system(),
                processor=platform.processor(),
                machine=platform.machine(),
                python_version=platform.python_version(),
                mlx_available=MLX_AVAILABLE,
                mps_available=mps_available,
                memory_total=memory_total,
                cpu_count=cpu_count,
            )

            logger.info(
                "Hardware detection completed: MLX=%s, MPS=%s",
                hardware_info.mlx_available,
                hardware_info.mps_available,
            )

            return hardware_info

        except Exception as e:
            logger.warning("Hardware detection failed: %s", e)
            return HardwareInfo(
                platform="unknown",
                processor="unknown",
                machine="unknown",
                python_version=platform.python_version(),
                mlx_available=False,
                mps_available=False,
            )

    def run_benchmark(
        self,
        name: str,
        benchmark_func: Callable[[], dict[str, float] | None],
        iterations: int = 1,
        warmup_iterations: int = 0,
    ) -> BenchmarkResult:
        """
        Run a benchmark function with performance measurement.

        Args:
            name: Benchmark name
            benchmark_func: Function to benchmark (should return metrics dict or None)
            iterations: Number of iterations to run
            warmup_iterations: Number of warmup iterations

        Returns:
            Benchmark results

        Raises:
            BenchmarkError: If benchmark execution fails

        Example:
            >>> def my_benchmark():
            ...     # Benchmark code here
            ...     return {"accuracy": 0.95}
            >>> runner = BenchmarkRunner()
            >>> result = runner.run_benchmark("my_test", my_benchmark, iterations=5)
        """
        logger.info(
            "Starting benchmark: %s (iterations=%d, warmup=%d)", name, iterations, warmup_iterations
        )

        try:
            # Warmup iterations
            for i in range(warmup_iterations):
                logger.debug("Warmup iteration %d/%d", i + 1, warmup_iterations)
                benchmark_func()

            # Actual benchmark iterations
            execution_times: list[float] = []
            all_metrics: list[dict[str, float]] = []
            memory_usage = self._measure_memory_usage()

            for i in range(iterations):
                logger.debug("Benchmark iteration %d/%d", i + 1, iterations)

                start_time = time.perf_counter()
                metrics = benchmark_func()
                end_time = time.perf_counter()

                execution_time = end_time - start_time
                execution_times.append(execution_time)

                if metrics:
                    all_metrics.append(metrics)

            # Calculate average execution time
            avg_execution_time = sum(execution_times) / len(execution_times)

            # Aggregate metrics
            aggregated_metrics = self._aggregate_metrics(all_metrics)

            # Create result
            result = BenchmarkResult(
                name=name,
                execution_time=avg_execution_time,
                memory_usage=memory_usage,
                performance_metrics=aggregated_metrics,
                hardware_info=self.hardware_info,
                success=True,
            )

            self._results.append(result)
            logger.info("Benchmark completed: %s (avg_time=%.4fs)", name, avg_execution_time)

            return result

        except Exception as e:
            error_result = BenchmarkResult(
                name=name,
                execution_time=0.0,
                hardware_info=self.hardware_info,
                success=False,
                error_message=str(e),
            )

            self._results.append(error_result)

            raise BenchmarkError(
                f"Benchmark failed: {name}", benchmark_name=name, details={"error": str(e)}
            ) from e

    def _measure_memory_usage(self) -> dict[str, float]:
        """Measure current memory usage."""
        memory_stats: dict[str, float] = {}

        # Process memory information
        process = psutil.Process()
        memory_info = process.memory_info()

        memory_stats.update(
            {
                "rss_mb": memory_info.rss / (1024**2),  # MB
                "vms_mb": memory_info.vms / (1024**2),  # MB
            }
        )

        # System memory
        system_memory = psutil.virtual_memory()
        memory_stats.update(
            {
                "system_used_mb": system_memory.used / (1024**2),
                "system_available_mb": system_memory.available / (1024**2),
                "system_percent": system_memory.percent,
            }
        )

        # MLX memory if available
        if MLX_AVAILABLE and mx is not None:
            try:
                mlx_memory = mx.metal.get_active_memory() / (1024**2)  # MB
                memory_stats["mlx_active_mb"] = mlx_memory
            except Exception:
                pass

        return memory_stats

    def _aggregate_metrics(self, metrics_list: Sequence[dict[str, float]]) -> dict[str, float]:
        """Aggregate metrics from multiple iterations."""
        if not metrics_list:
            return {}

        aggregated: dict[str, float] = {}

        # Get all unique metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        # Calculate averages
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in metrics_list if key in metrics]
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated

    def compare_benchmarks(
        self, baseline_name: str, comparison_names: Sequence[str]
    ) -> dict[str, dict[str, float]]:
        """
        Compare benchmark results against a baseline.

        Args:
            baseline_name: Name of baseline benchmark
            comparison_names: Names of benchmarks to compare

        Returns:
            Comparison results with improvement ratios

        Raises:
            BenchmarkError: If baseline or comparison benchmarks not found
        """
        # Find baseline result
        baseline_result = None
        for result in self._results:
            if result.name == baseline_name and result.success:
                baseline_result = result
                break

        if baseline_result is None:
            raise BenchmarkError(
                f"Baseline benchmark not found: {baseline_name}", benchmark_name=baseline_name
            )

        comparisons: dict[str, dict[str, float]] = {}

        for comp_name in comparison_names:
            # Find comparison result
            comp_result = None
            for result in self._results:
                if result.name == comp_name and result.success:
                    comp_result = result
                    break

            if comp_result is None:
                logger.warning("Comparison benchmark not found: %s", comp_name)
                continue

            # Calculate improvements
            comparison = {
                "execution_time_ratio": baseline_result.execution_time / comp_result.execution_time,
                "execution_time_improvement": (
                    (baseline_result.execution_time - comp_result.execution_time)
                    / baseline_result.execution_time
                    * 100
                ),
            }

            # Compare performance metrics
            for metric_name, baseline_value in baseline_result.performance_metrics.items():
                if metric_name in comp_result.performance_metrics:
                    comp_value = comp_result.performance_metrics[metric_name]
                    if baseline_value != 0:
                        comparison[f"{metric_name}_ratio"] = comp_value / baseline_value
                        comparison[f"{metric_name}_improvement"] = (
                            (comp_value - baseline_value) / baseline_value * 100
                        )

            comparisons[comp_name] = comparison

        logger.info("Benchmark comparison completed for %d benchmarks", len(comparisons))
        return comparisons

    def get_results(self) -> list[BenchmarkResult]:
        """Get all benchmark results."""
        return self._results.copy()

    def clear_results(self) -> None:
        """Clear all benchmark results."""
        self._results.clear()
        logger.info("Cleared all benchmark results")

    def save_results(self, output_file: Path | None = None) -> Path:
        """
        Save benchmark results to file.

        Args:
            output_file: Optional output file path

        Returns:
            Path to saved results file
        """
        if output_file is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"benchmark_results_{timestamp}.json"

        # Convert results to serializable format
        results_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "hardware_info": {
                "platform": self.hardware_info.platform,
                "processor": self.hardware_info.processor,
                "machine": self.hardware_info.machine,
                "python_version": self.hardware_info.python_version,
                "mlx_available": self.hardware_info.mlx_available,
                "mps_available": self.hardware_info.mps_available,
                "memory_total": self.hardware_info.memory_total,
                "cpu_count": self.hardware_info.cpu_count,
            },
            "results": [
                {
                    "name": result.name,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage,
                    "performance_metrics": result.performance_metrics,
                    "timestamp": result.timestamp.isoformat(),
                    "success": result.success,
                    "error_message": result.error_message,
                }
                for result in self._results
            ],
        }

        import json

        output_file.write_text(json.dumps(results_data, indent=2), encoding="utf-8")

        logger.info("Saved benchmark results to %s", output_file)
        return output_file
