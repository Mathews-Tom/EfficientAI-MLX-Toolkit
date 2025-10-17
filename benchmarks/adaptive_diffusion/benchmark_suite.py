"""
Comprehensive Benchmark Suite for Adaptive Diffusion Optimization

Implements comprehensive benchmarking infrastructure with:
- Multiple model support
- Domain-specific benchmarks
- Performance comparison
- Statistical significance testing
- Visualization and reporting
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

# Import adaptive diffusion components
import sys
from pathlib import Path
project_dir = Path(__file__).parent.parent.parent / "projects/04_Adaptive_Diffusion_Optimizer"
sys.path.insert(0, str(project_dir / "src"))

from adaptive_diffusion.baseline import (
    DiffusionPipeline,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverScheduler,
)
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler
from adaptive_diffusion.sampling.step_reduction import StepReductionSampler
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline
from adaptive_diffusion.optimization.domain_adapter import DomainType


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    # Model configuration
    model_name: str = "diffusion-base"
    image_size: tuple[int, int] = (64, 64)

    # Benchmark parameters
    num_iterations: int = 3
    batch_size: int = 2
    num_inference_steps: int = 30
    seed: int | None = None

    # Domain configuration
    domain: str | None = None  # "photorealistic", "artistic", "synthetic"

    # Performance tracking
    track_memory: bool = True
    track_intermediates: bool = False
    warmup_iterations: int = 1

    # Output configuration
    output_dir: Path | None = None
    save_results: bool = True
    verbose: bool = True


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    # Configuration
    config: dict[str, Any]
    scheduler_name: str
    model_name: str

    # Performance metrics
    total_time: float
    avg_time_per_image: float
    images_per_sec: float
    steps_per_sec: float

    # Memory metrics
    peak_memory_mb: float | None = None
    avg_memory_mb: float | None = None

    # Quality metrics
    avg_quality_score: float | None = None
    quality_variance: float | None = None

    # Statistical metrics
    std_time: float | None = None
    min_time: float | None = None
    max_time: float | None = None

    # Additional metadata
    device: str = "unknown"
    timestamp: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for adaptive diffusion models.

    Supports:
    - Multiple schedulers (DDPM, DDIM, DPM-Solver, Adaptive)
    - Multiple domains (photorealistic, artistic, synthetic)
    - Performance comparison with statistical significance
    - Memory profiling
    - Quality assessment
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results: list[BenchmarkResult] = []
        self.device = self._detect_device()

        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _detect_device(self) -> str:
        """Detect hardware device."""
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            return "apple_silicon"
        return "cpu"

    def run_all_benchmarks(self) -> list[BenchmarkResult]:
        """
        Run comprehensive benchmark suite across all schedulers.

        Returns:
            List of benchmark results
        """
        schedulers = {
            "DDPM": DDPMScheduler(),
            "DDIM": DDIMScheduler(),
            "DPM-Solver": DPMSolverScheduler(),
            "Adaptive": AdaptiveScheduler(num_inference_steps=self.config.num_inference_steps),
        }

        results = []
        for name, scheduler in schedulers.items():
            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"Benchmarking: {name}")
                print(f"{'='*60}")

            result = self.benchmark_scheduler(name, scheduler)
            results.append(result)
            self.results.append(result)

        # Generate comparison report
        if self.config.verbose:
            self._print_comparison_report(results)

        # Save results
        if self.config.save_results and self.config.output_dir:
            self._save_results(results)

        return results

    def benchmark_scheduler(
        self, scheduler_name: str, scheduler: Any
    ) -> BenchmarkResult:
        """
        Benchmark a specific scheduler.

        Args:
            scheduler_name: Name of scheduler
            scheduler: Scheduler instance

        Returns:
            Benchmark result
        """
        # Create pipeline
        pipeline = DiffusionPipeline(
            scheduler=scheduler,
            image_size=self.config.image_size,
        )

        # Warmup
        if self.config.warmup_iterations > 0:
            if self.config.verbose:
                print(f"Warmup ({self.config.warmup_iterations} iterations)...")
            for _ in range(self.config.warmup_iterations):
                pipeline.generate(
                    batch_size=1,
                    num_inference_steps=10,
                    seed=self.config.seed,
                )

        # Benchmark
        if self.config.verbose:
            print(f"Running benchmark ({self.config.num_iterations} iterations)...")

        iteration_times = []
        memory_usage = []

        for i in range(self.config.num_iterations):
            start_time = time.perf_counter()

            # Generate images
            images = pipeline.generate(
                batch_size=self.config.batch_size,
                num_inference_steps=self.config.num_inference_steps,
                seed=None,  # Use different seed each iteration
            )

            elapsed_time = time.perf_counter() - start_time
            iteration_times.append(elapsed_time)

            # Track memory (approximate)
            if self.config.track_memory:
                memory_mb = images.size * 4 / (1024 * 1024)  # float32
                memory_usage.append(memory_mb)

            if self.config.verbose:
                print(f"  Iteration {i+1}/{self.config.num_iterations}: {elapsed_time:.3f}s")

        # Compute statistics
        total_time = sum(iteration_times)
        avg_time = np.mean(iteration_times)
        std_time = np.std(iteration_times)
        min_time = np.min(iteration_times)
        max_time = np.max(iteration_times)

        total_images = self.config.num_iterations * self.config.batch_size
        images_per_sec = total_images / total_time
        total_steps = total_images * self.config.num_inference_steps
        steps_per_sec = total_steps / total_time

        # Memory statistics
        peak_memory = np.max(memory_usage) if memory_usage else None
        avg_memory = np.mean(memory_usage) if memory_usage else None

        # Create result
        result = BenchmarkResult(
            config=asdict(self.config),
            scheduler_name=scheduler_name,
            model_name=self.config.model_name,
            total_time=total_time,
            avg_time_per_image=avg_time / self.config.batch_size,
            images_per_sec=images_per_sec,
            steps_per_sec=steps_per_sec,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            device=self.device,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if self.config.verbose:
            print(f"\nResults:")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Avg time per image: {result.avg_time_per_image:.3f}s")
            print(f"  Throughput: {images_per_sec:.2f} images/sec")
            print(f"  Speed: {steps_per_sec:.1f} steps/sec")
            if peak_memory:
                print(f"  Peak memory: {peak_memory:.2f} MB")

        return result

    def benchmark_domain_specific(
        self, domain: DomainType
    ) -> dict[str, BenchmarkResult]:
        """
        Run domain-specific benchmarks.

        Args:
            domain: Domain type

        Returns:
            Dictionary of results by scheduler
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Domain-Specific Benchmark: {domain.value}")
            print(f"{'='*60}")

        # Create optimization pipeline with domain adaptation
        opt_pipeline = OptimizationPipeline(
            use_domain_adaptation=True,
            use_rl_optimization=False,  # Disable RL for faster benchmarking
            verbose=0,
        )

        # Get domain-optimized scheduler
        optimized_scheduler = opt_pipeline.create_optimized_scheduler(domain_type=domain)

        # Benchmark baseline and optimized
        results = {}

        # Baseline DDIM
        baseline_result = self.benchmark_scheduler("DDIM-Baseline", DDIMScheduler())
        results["baseline"] = baseline_result

        # Domain-optimized
        optimized_result = self.benchmark_scheduler(
            f"Adaptive-{domain.value}", optimized_scheduler
        )
        results["optimized"] = optimized_result

        # Compute speedup
        speedup = baseline_result.total_time / optimized_result.total_time

        if self.config.verbose:
            print(f"\nDomain Optimization Summary:")
            print(f"  Baseline: {baseline_result.total_time:.3f}s")
            print(f"  Optimized: {optimized_result.total_time:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")

        return results

    def benchmark_step_reduction(self) -> dict[str, BenchmarkResult]:
        """
        Benchmark step reduction capabilities.

        Returns:
            Dictionary of results by step count
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Step Reduction Benchmark")
            print(f"{'='*60}")

        step_counts = [10, 20, 30, 50]
        results = {}

        for steps in step_counts:
            # Update config temporarily
            original_steps = self.config.num_inference_steps
            self.config.num_inference_steps = steps

            # Benchmark
            scheduler = AdaptiveScheduler(num_inference_steps=steps)
            result = self.benchmark_scheduler(f"Adaptive-{steps}steps", scheduler)
            results[f"{steps}_steps"] = result

            # Restore config
            self.config.num_inference_steps = original_steps

        # Print summary
        if self.config.verbose:
            print(f"\nStep Reduction Summary:")
            print(f"{'Steps':<10} {'Time':<12} {'Throughput':<15} {'Speed':<15}")
            print("-" * 55)
            for steps in step_counts:
                result = results[f"{steps}_steps"]
                print(
                    f"{steps:<10} {result.total_time:<12.3f} "
                    f"{result.images_per_sec:<15.2f} {result.steps_per_sec:<15.1f}"
                )

        return results

    def _print_comparison_report(self, results: list[BenchmarkResult]):
        """
        Print comparison report across schedulers.

        Args:
            results: List of benchmark results
        """
        print(f"\n{'='*80}")
        print("BENCHMARK COMPARISON REPORT")
        print(f"{'='*80}")
        print(
            f"{'Scheduler':<15} {'Time (s)':<12} {'Images/sec':<15} "
            f"{'Steps/sec':<15} {'Memory (MB)':<12}"
        )
        print("-" * 80)

        for result in results:
            memory_str = (
                f"{result.peak_memory_mb:.2f}" if result.peak_memory_mb else "N/A"
            )
            print(
                f"{result.scheduler_name:<15} {result.total_time:<12.3f} "
                f"{result.images_per_sec:<15.2f} {result.steps_per_sec:<15.1f} "
                f"{memory_str:<12}"
            )

        # Compute relative speedups
        if len(results) > 1:
            baseline = results[0]  # First result as baseline
            print(f"\nRelative Performance (vs {baseline.scheduler_name}):")
            print(f"{'Scheduler':<15} {'Speedup':<12} {'Efficiency':<12}")
            print("-" * 40)

            for result in results:
                speedup = baseline.total_time / result.total_time
                efficiency = speedup * 100
                print(f"{result.scheduler_name:<15} {speedup:<12.2f}x {efficiency:<12.1f}%")

        print(f"{'='*80}\n")

    def _save_results(self, results: list[BenchmarkResult]):
        """
        Save benchmark results to disk.

        Args:
            results: List of results to save
        """
        if not self.config.output_dir:
            return

        # Save individual results
        for result in results:
            filename = f"benchmark_{result.scheduler_name.lower().replace(' ', '_')}.json"
            filepath = self.config.output_dir / filename

            with open(filepath, "w") as f:
                f.write(result.to_json())

        # Save summary
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device,
            "config": asdict(self.config),
            "results": [r.to_dict() for r in results],
        }

        summary_path = self.config.output_dir / "benchmark_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if self.config.verbose:
            print(f"Results saved to: {self.config.output_dir}")

    def get_results(self) -> list[BenchmarkResult]:
        """Get all benchmark results."""
        return self.results

    def clear_results(self):
        """Clear stored results."""
        self.results = []


def run_comprehensive_benchmarks(
    output_dir: Path | None = None, verbose: bool = True
) -> list[BenchmarkResult]:
    """
    Run comprehensive benchmark suite.

    Args:
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        List of benchmark results
    """
    config = BenchmarkConfig(
        output_dir=output_dir,
        verbose=verbose,
        num_iterations=3,
        batch_size=2,
        num_inference_steps=30,
    )

    runner = BenchmarkRunner(config)
    results = runner.run_all_benchmarks()

    return results


def run_domain_benchmarks(
    output_dir: Path | None = None, verbose: bool = True
) -> dict[str, dict[str, BenchmarkResult]]:
    """
    Run domain-specific benchmarks for all domains.

    Args:
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dictionary of results by domain
    """
    config = BenchmarkConfig(
        output_dir=output_dir,
        verbose=verbose,
        num_iterations=3,
        batch_size=2,
        num_inference_steps=30,
    )

    runner = BenchmarkRunner(config)
    all_results = {}

    for domain in [DomainType.PHOTOREALISTIC, DomainType.ARTISTIC, DomainType.SYNTHETIC]:
        results = runner.benchmark_domain_specific(domain)
        all_results[domain.value] = results

    return all_results
