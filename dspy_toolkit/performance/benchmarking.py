"""
Apple Silicon benchmarking system for DSPy Integration Framework.
"""

# Standard library imports
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Optional third-party imports
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

# Local imports
from ..exceptions import DSPyIntegrationError
from ..types import DSPyConfig, HardwareInfo

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from Apple Silicon benchmarking."""

    test_name: str
    hardware_type: str
    memory_gb: int
    inference_time_ms: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    peak_memory_mb: float
    optimization_level: int
    model_size: str
    batch_size: int
    timestamp: str
    metadata: dict[str, str | int | float | bool]


class AppleSiliconBenchmark:
    """Comprehensive benchmarking system for Apple Silicon performance."""

    def __init__(self, config: DSPyConfig):
        """Initialize Apple Silicon benchmark system."""
        self.config = config
        self.hardware_info = self._detect_hardware()
        self.benchmark_results: list[BenchmarkResult] = []

        if not MLX_AVAILABLE:
            logger.warning("MLX not available, benchmarks will use fallback implementations")

    def _detect_hardware(self) -> HardwareInfo:
        """Detect Apple Silicon hardware capabilities."""
        if not MLX_AVAILABLE:
            return HardwareInfo(
                device_type="cpu",
                total_memory=8,
                available_memory=6,
                metal_available=False,
                mps_available=False,
                optimization_level=1,
            )

        try:
            # Get memory info
            total_memory = mx.metal.get_peak_memory() // (1024**3)  # Convert to GB
            available_memory = (mx.metal.get_peak_memory() - mx.metal.get_active_memory()) // (
                1024**3
            )

            # Detect chip type based on memory and performance characteristics
            if total_memory >= 64:
                device_type = "m2_ultra"
            elif total_memory >= 32:
                device_type = "m2_max"
            elif total_memory >= 16:
                device_type = "m2_pro"
            else:
                device_type = "m2"

            return HardwareInfo(
                device_type=device_type,
                total_memory=total_memory,
                available_memory=available_memory,
                metal_available=mx.metal.is_available(),
                mps_available=True,  # Assume MPS is available on Apple Silicon
                optimization_level=3,  # Maximum optimization for Apple Silicon
            )

        except Exception as e:
            logger.warning("Hardware detection failed: %s", e)
            return HardwareInfo(
                device_type="apple_silicon",
                total_memory=16,
                available_memory=12,
                metal_available=True,
                mps_available=True,
                optimization_level=2,
            )

    def benchmark_inference_speed(
        self,
        module,  # Any callable module
        test_inputs: list[dict[str, str | int | float]],
        batch_sizes: list[int] = None,
    ) -> list[BenchmarkResult]:
        """Benchmark inference speed across different batch sizes."""
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        results = []

        for batch_size in batch_sizes:
            try:
                # Prepare batch
                batch_inputs = (test_inputs * batch_size)[:batch_size]

                # Warm up
                for _ in range(3):
                    try:
                        _ = module(**batch_inputs[0])
                    except Exception:
                        pass  # Ignore warm-up errors

                # Benchmark
                start_memory = self._get_memory_usage()
                start_time = time.perf_counter()

                predictions = []
                for inputs in batch_inputs:
                    prediction = module(**inputs)
                    predictions.append(prediction)

                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                peak_memory = self._get_peak_memory()

                # Calculate metrics
                total_time_ms = (end_time - start_time) * 1000
                avg_time_ms = total_time_ms / batch_size
                throughput = batch_size / (total_time_ms / 1000)
                memory_usage_mb = end_memory - start_memory

                result = BenchmarkResult(
                    test_name="inference_speed",
                    hardware_type=self.hardware_info.device_type,
                    memory_gb=self.hardware_info.total_memory,
                    inference_time_ms=avg_time_ms,
                    throughput_tokens_per_sec=throughput,
                    memory_usage_mb=memory_usage_mb,
                    peak_memory_mb=peak_memory,
                    optimization_level=self.config.optimization_level,
                    model_size=self.config.model_name,
                    batch_size=batch_size,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "total_time_ms": total_time_ms,
                        "num_predictions": len(predictions),
                        "metal_available": self.hardware_info.metal_available,
                    },
                )

                results.append(result)
                self.benchmark_results.append(result)

                logger.info(
                    "Benchmark completed - Batch size: %d, Avg time: %.2fms, Throughput: %.2f/sec",
                    batch_size,
                    avg_time_ms,
                    throughput,
                )

            except Exception as e:
                logger.error("Benchmark failed for batch size %d: %s", batch_size, e)
                continue

        return results

    def benchmark_memory_efficiency(
        self, module, input_sizes: list[int] = None  # Any callable module
    ) -> list[BenchmarkResult]:
        """Benchmark memory efficiency with different input sizes."""
        if input_sizes is None:
            input_sizes = [100, 500, 1000, 2000, 5000]

        results = []

        for input_size in input_sizes:
            try:
                # Create test input of specified size
                test_input = {
                    "input": "test " * (input_size // 5),  # Approximate token count
                    "context": "context " * (input_size // 10),
                }

                # Clear memory
                if MLX_AVAILABLE:
                    mx.metal.clear_cache()

                # Measure memory before
                start_memory = self._get_memory_usage()

                # Run inference
                start_time = time.perf_counter()
                _ = module(**test_input)
                end_time = time.perf_counter()

                # Measure memory after
                end_memory = self._get_memory_usage()
                peak_memory = self._get_peak_memory()

                # Calculate metrics
                inference_time_ms = (end_time - start_time) * 1000
                memory_usage_mb = end_memory - start_memory
                memory_efficiency = input_size / max(memory_usage_mb, 1)  # tokens per MB

                result = BenchmarkResult(
                    test_name="memory_efficiency",
                    hardware_type=self.hardware_info.device_type,
                    memory_gb=self.hardware_info.total_memory,
                    inference_time_ms=inference_time_ms,
                    throughput_tokens_per_sec=input_size / (inference_time_ms / 1000),
                    memory_usage_mb=memory_usage_mb,
                    peak_memory_mb=peak_memory,
                    optimization_level=self.config.optimization_level,
                    model_size=self.config.model_name,
                    batch_size=1,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "input_size": input_size,
                        "memory_efficiency": memory_efficiency,
                        "prediction_length": 0,  # Not tracking prediction length
                    },
                )

                results.append(result)
                self.benchmark_results.append(result)

                logger.info(
                    "Memory benchmark - Input size: %d, Memory: %.2fMB, Efficiency: %.2f tokens/MB",
                    input_size,
                    memory_usage_mb,
                    memory_efficiency,
                )

            except Exception as e:
                logger.error("Memory benchmark failed for input size %d: %s", input_size, e)
                continue

        return results

    def benchmark_optimization_levels(
        self, module, test_input: dict[str, str | int | float]  # Any callable module
    ) -> list[BenchmarkResult]:
        """Benchmark performance across different optimization levels."""
        results = []

        for opt_level in [1, 2, 3]:
            try:
                # Configure optimization level
                if MLX_AVAILABLE:
                    if opt_level == 1:
                        # Basic optimization
                        mx.metal.set_memory_limit(4 * 1024**3)  # 4GB
                    elif opt_level == 2:
                        # Moderate optimization
                        mx.metal.set_memory_limit(8 * 1024**3)  # 8GB
                    else:
                        # Maximum optimization
                        mx.metal.set_memory_limit(16 * 1024**3)  # 16GB

                # Warm up
                for _ in range(5):
                    try:
                        _ = module(**test_input)
                    except Exception:
                        pass

                # Benchmark multiple runs
                times = []
                memory_usages = []

                for _ in range(10):
                    start_memory = self._get_memory_usage()
                    start_time = time.perf_counter()

                    _ = module(**test_input)

                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()

                    times.append((end_time - start_time) * 1000)
                    memory_usages.append(end_memory - start_memory)

                # Calculate statistics
                avg_time_ms = sum(times) / len(times)
                min_time_ms = min(times)
                max_time_ms = max(times)
                avg_memory_mb = sum(memory_usages) / len(memory_usages)

                result = BenchmarkResult(
                    test_name="optimization_levels",
                    hardware_type=self.hardware_info.device_type,
                    memory_gb=self.hardware_info.total_memory,
                    inference_time_ms=avg_time_ms,
                    throughput_tokens_per_sec=1000 / avg_time_ms,
                    memory_usage_mb=avg_memory_mb,
                    peak_memory_mb=self._get_peak_memory(),
                    optimization_level=opt_level,
                    model_size=self.config.model_name,
                    batch_size=1,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "min_time_ms": min_time_ms,
                        "max_time_ms": max_time_ms,
                        "std_dev_ms": (sum((t - avg_time_ms) ** 2 for t in times) / len(times))
                        ** 0.5,
                        "num_runs": len(times),
                    },
                )

                results.append(result)
                self.benchmark_results.append(result)

                logger.info(
                    "Optimization level %d - Avg time: %.2fms, Memory: %.2fMB",
                    opt_level,
                    avg_time_ms,
                    avg_memory_mb,
                )

            except Exception as e:
                logger.error("Optimization benchmark failed for level %d: %s", opt_level, e)
                continue

        return results

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if MLX_AVAILABLE:
            return mx.metal.get_active_memory() / (1024**2)  # Convert to MB

        # Fallback to system memory estimation
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024**2)
        except ImportError:
            return 0.0

    def _get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if MLX_AVAILABLE:
            return mx.metal.get_peak_memory() / (1024**2)  # Convert to MB

        # Fallback estimation
        return self._get_memory_usage() * 1.2

    def get_benchmark_summary(self) -> dict[str, str | int | float | bool]:
        """Get summary of all benchmark results."""
        if not self.benchmark_results:
            return {"error": "No benchmark results available"}

        # Group results by test type
        by_test = {}
        for result in self.benchmark_results:
            if result.test_name not in by_test:
                by_test[result.test_name] = []
            by_test[result.test_name].append(result)

        summary = {
            "hardware_type": self.hardware_info.device_type,
            "total_memory_gb": self.hardware_info.total_memory,
            "metal_available": self.hardware_info.metal_available,
            "total_benchmarks": len(self.benchmark_results),
            "test_types": list(by_test.keys()),
            "timestamp": datetime.now().isoformat(),
        }

        # Add statistics for each test type
        for test_name, results in by_test.items():
            avg_time = sum(r.inference_time_ms for r in results) / len(results)
            avg_throughput = sum(r.throughput_tokens_per_sec for r in results) / len(results)
            avg_memory = sum(r.memory_usage_mb for r in results) / len(results)

            summary[f"{test_name}_avg_time_ms"] = avg_time
            summary[f"{test_name}_avg_throughput"] = avg_throughput
            summary[f"{test_name}_avg_memory_mb"] = avg_memory
            summary[f"{test_name}_count"] = len(results)

        return summary

    def export_results(self, export_path: Path) -> None:
        """Export benchmark results to file."""
        try:
            import json

            export_data = {
                "hardware_info": {
                    "device_type": self.hardware_info.device_type,
                    "total_memory": self.hardware_info.total_memory,
                    "available_memory": self.hardware_info.available_memory,
                    "metal_available": self.hardware_info.metal_available,
                    "mps_available": self.hardware_info.mps_available,
                    "optimization_level": self.hardware_info.optimization_level,
                },
                "config": {
                    "model_provider": self.config.model_provider,
                    "model_name": self.config.model_name,
                    "optimization_level": self.config.optimization_level,
                },
                "results": [
                    {
                        "test_name": r.test_name,
                        "hardware_type": r.hardware_type,
                        "memory_gb": r.memory_gb,
                        "inference_time_ms": r.inference_time_ms,
                        "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                        "memory_usage_mb": r.memory_usage_mb,
                        "peak_memory_mb": r.peak_memory_mb,
                        "optimization_level": r.optimization_level,
                        "model_size": r.model_size,
                        "batch_size": r.batch_size,
                        "timestamp": r.timestamp,
                        "metadata": r.metadata,
                    }
                    for r in self.benchmark_results
                ],
                "summary": self.get_benchmark_summary(),
                "export_timestamp": datetime.now().isoformat(),
            }

            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)

            logger.info("Benchmark results exported to %s", export_path)

        except Exception as e:
            logger.error("Failed to export benchmark results: %s", e)
            raise DSPyIntegrationError("Benchmark export failed") from e
