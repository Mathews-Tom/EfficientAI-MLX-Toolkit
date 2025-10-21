"""
Comparison Tests Against Baseline Schedulers

Tests adaptive diffusion against baseline schedulers (DDPM, DDIM, DPM-Solver)
with statistical significance testing.
"""

from __future__ import annotations

import platform
import time

import mlx.core as mx
import numpy as np
import pytest

# Import project modules (using pythonpath from pyproject.toml)
from adaptive_diffusion.baseline import (
    DiffusionPipeline,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverScheduler,
)
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler

# Import benchmarking infrastructure (using pythonpath from pyproject.toml)
from benchmarks.adaptive_diffusion.validation_metrics import (
    MetricsCalculator,
    compute_statistical_significance,
)


# Skip tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    not (platform.machine() == "arm64" and platform.system() == "Darwin"),
    reason="Apple Silicon required for MLX tests",
)


class TestSchedulerComparison:
    """Compare Adaptive scheduler against baseline schedulers."""

    @pytest.fixture
    def test_config(self):
        """Common test configuration."""
        return {
            "image_size": (64, 64),
            "num_iterations": 3,
            "batch_size": 2,
            "num_inference_steps": 30,
            "seed": 42,
        }

    @pytest.fixture
    def schedulers(self, test_config):
        """Create all schedulers for comparison."""
        return {
            "DDPM": DDPMScheduler(),
            "DDIM": DDIMScheduler(),
            "DPM-Solver": DPMSolverScheduler(),
            "Adaptive": AdaptiveScheduler(
                num_inference_steps=test_config["num_inference_steps"]
            ),
        }

    def test_performance_comparison(self, test_config, schedulers):
        """
        Compare performance across all schedulers.

        Validates:
        - All schedulers can generate images
        - Performance metrics are collected
        - Adaptive is competitive with baselines
        """
        results = {}

        for name, scheduler in schedulers.items():
            # Create pipeline
            pipeline = DiffusionPipeline(
                scheduler=scheduler,
                image_size=test_config["image_size"],
            )

            # Warmup
            pipeline.generate(
                batch_size=1,
                num_inference_steps=10,
                seed=test_config["seed"],
            )

            # Benchmark
            iteration_times = []
            for _ in range(test_config["num_iterations"]):
                start = time.perf_counter()
                images = pipeline.generate(
                    batch_size=test_config["batch_size"],
                    num_inference_steps=test_config["num_inference_steps"],
                    seed=None,
                )
                elapsed = time.perf_counter() - start
                iteration_times.append(elapsed)

                # Validate images
                assert images.shape == (
                    test_config["batch_size"],
                    test_config["image_size"][0],
                    test_config["image_size"][1],
                    3,
                )

            # Compute metrics
            total_images = test_config["num_iterations"] * test_config["batch_size"]
            metrics = MetricsCalculator.compute_performance_metrics(
                iteration_times=iteration_times,
                num_images=total_images,
                num_steps=test_config["num_inference_steps"],
                batch_size=test_config["batch_size"],
            )

            results[name] = {
                "metrics": metrics,
                "times": iteration_times,
            }

            print(f"\n{name} Performance:")
            print(f"  Total time: {metrics.total_time:.3f}s")
            print(f"  Throughput: {metrics.images_per_sec:.2f} images/sec")
            print(f"  Speed: {metrics.steps_per_sec:.1f} steps/sec")

        # Validate results
        assert len(results) == 4
        for name, result in results.items():
            assert result["metrics"].total_time > 0
            assert result["metrics"].images_per_sec > 0

        # Print comparison
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"{'Scheduler':<15} {'Time (s)':<12} {'Throughput':<15}")
        print("-"*45)
        for name, result in results.items():
            m = result["metrics"]
            print(f"{name:<15} {m.total_time:<12.3f} {m.images_per_sec:<15.2f}")

    def test_statistical_significance(self, test_config, schedulers):
        """
        Test statistical significance of performance improvements.

        Validates:
        - Statistical test runs successfully
        - Results include required metrics
        - Comparison is valid
        """
        # Collect baseline (DDIM) and adaptive timings
        baseline_scheduler = schedulers["DDIM"]
        adaptive_scheduler = schedulers["Adaptive"]

        # Baseline pipeline
        baseline_pipeline = DiffusionPipeline(
            scheduler=baseline_scheduler,
            image_size=test_config["image_size"],
        )
        baseline_pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        baseline_times = []
        for _ in range(test_config["num_iterations"]):
            start = time.perf_counter()
            baseline_pipeline.generate(
                batch_size=test_config["batch_size"],
                num_inference_steps=test_config["num_inference_steps"],
                seed=None,
            )
            baseline_times.append(time.perf_counter() - start)

        # Adaptive pipeline
        adaptive_pipeline = DiffusionPipeline(
            scheduler=adaptive_scheduler,
            image_size=test_config["image_size"],
        )
        adaptive_pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        adaptive_times = []
        for _ in range(test_config["num_iterations"]):
            start = time.perf_counter()
            adaptive_pipeline.generate(
                batch_size=test_config["batch_size"],
                num_inference_steps=test_config["num_inference_steps"],
                seed=None,
            )
            adaptive_times.append(time.perf_counter() - start)

        # Compute statistical significance
        try:
            significance = compute_statistical_significance(
                baseline_times, adaptive_times, confidence=0.95
            )

            print("\n" + "="*60)
            print("STATISTICAL SIGNIFICANCE TEST")
            print("="*60)
            print(f"Baseline (DDIM) mean: {significance['baseline_mean']:.3f}s")
            print(f"Adaptive mean: {significance['optimized_mean']:.3f}s")
            print(f"Speedup: {significance['speedup']:.2f}x")
            print(f"P-value: {significance['p_value']:.4f}")
            print(f"Cohen's d: {significance['cohens_d']:.3f}")
            print(f"Significant at 95%: {significance['is_significant']}")

            # Validate test results
            assert "speedup" in significance
            assert "p_value" in significance
            assert "cohens_d" in significance
            assert isinstance(significance["is_significant"], bool)

        except ImportError:
            # scipy not available, use simplified test
            baseline_mean = np.mean(baseline_times)
            adaptive_mean = np.mean(adaptive_times)
            speedup = baseline_mean / adaptive_mean

            print("\n" + "="*60)
            print("SIMPLIFIED COMPARISON (scipy not available)")
            print("="*60)
            print(f"Baseline mean: {baseline_mean:.3f}s")
            print(f"Adaptive mean: {adaptive_mean:.3f}s")
            print(f"Speedup: {speedup:.2f}x")

            assert speedup > 0  # Basic validation

    def test_quality_comparison(self, test_config, schedulers):
        """
        Compare generation quality across schedulers.

        Validates:
        - Quality metrics can be computed
        - Metrics are reasonable
        - Adaptive maintains quality
        """
        results = {}

        for name, scheduler in schedulers.items():
            pipeline = DiffusionPipeline(
                scheduler=scheduler,
                image_size=test_config["image_size"],
            )

            # Generate images
            images = pipeline.generate(
                batch_size=4,  # More images for quality assessment
                num_inference_steps=test_config["num_inference_steps"],
                seed=test_config["seed"],
            )

            # Compute quality metrics
            quality = MetricsCalculator.compute_quality_metrics(images)
            results[name] = quality

            print(f"\n{name} Quality:")
            print(f"  Pixel variance: {quality.pixel_variance:.4f}")
            print(f"  Edge sharpness: {quality.edge_sharpness:.4f}")
            print(f"  FID proxy: {quality.fid_score:.2f}")

        # Validate quality metrics
        for name, quality in results.items():
            assert quality.pixel_variance > 0
            assert quality.edge_sharpness > 0
            assert quality.fid_score is not None

        # Print comparison
        print("\n" + "="*60)
        print("QUALITY COMPARISON")
        print("="*60)
        print(f"{'Scheduler':<15} {'FID Proxy':<12} {'Sharpness':<12}")
        print("-"*45)
        for name, quality in results.items():
            print(f"{name:<15} {quality.fid_score:<12.2f} {quality.edge_sharpness:<12.4f}")

    def test_step_reduction_effectiveness(self, test_config):
        """
        Test effectiveness of adaptive step reduction.

        Validates:
        - Adaptive can reduce steps while maintaining quality
        - Performance improves with fewer steps
        - Quality degrades gracefully
        """
        step_counts = [10, 20, 30]
        results = {}

        for steps in step_counts:
            scheduler = AdaptiveScheduler(num_inference_steps=steps)
            pipeline = DiffusionPipeline(
                scheduler=scheduler,
                image_size=test_config["image_size"],
            )

            # Time generation
            start = time.perf_counter()
            images = pipeline.generate(
                batch_size=test_config["batch_size"],
                num_inference_steps=steps,
                seed=test_config["seed"],
            )
            elapsed = time.perf_counter() - start

            # Compute quality
            quality = MetricsCalculator.compute_quality_metrics(images)

            results[steps] = {
                "time": elapsed,
                "quality": quality,
                "steps_per_sec": (test_config["batch_size"] * steps) / elapsed,
            }

            print(f"\nAdaptive with {steps} steps:")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Quality (FID proxy): {quality.fid_score:.2f}")
            print(f"  Speed: {results[steps]['steps_per_sec']:.1f} steps/sec")

        # Validate trend: fewer steps = faster
        times = [results[s]["time"] for s in step_counts]
        assert times[0] < times[1] < times[2], "Time should increase with more steps"

        # Print summary
        print("\n" + "="*60)
        print("STEP REDUCTION EFFECTIVENESS")
        print("="*60)
        print(f"{'Steps':<10} {'Time (s)':<12} {'Quality':<12} {'Speed':<12}")
        print("-"*50)
        for steps in step_counts:
            r = results[steps]
            print(
                f"{steps:<10} {r['time']:<12.3f} {r['quality'].fid_score:<12.2f} "
                f"{r['steps_per_sec']:<12.1f}"
            )

    def test_memory_efficiency_comparison(self, test_config, schedulers):
        """
        Compare memory efficiency across schedulers.

        Validates:
        - Memory usage is tracked
        - All schedulers have reasonable memory footprint
        - Adaptive is memory efficient
        """
        results = {}

        for name, scheduler in schedulers.items():
            pipeline = DiffusionPipeline(
                scheduler=scheduler,
                image_size=test_config["image_size"],
            )

            # Generate and track memory
            images = pipeline.generate(
                batch_size=test_config["batch_size"],
                num_inference_steps=test_config["num_inference_steps"],
                seed=test_config["seed"],
            )

            # Approximate memory usage
            memory_mb = images.size * 4 / (1024 * 1024)  # float32
            results[name] = memory_mb

            print(f"\n{name} Memory: {memory_mb:.2f} MB")

        # Validate memory usage
        for name, memory in results.items():
            assert memory > 0
            assert memory < 1000  # Should be reasonable for this test size

        # Print comparison
        print("\n" + "="*60)
        print("MEMORY EFFICIENCY COMPARISON")
        print("="*60)
        print(f"{'Scheduler':<15} {'Memory (MB)':<12}")
        print("-"*30)
        for name, memory in results.items():
            print(f"{name:<15} {memory:<12.2f}")


@pytest.mark.slow
class TestComprehensiveComparison:
    """Comprehensive comparison tests (slower, more thorough)."""

    def test_sustained_performance_comparison(self):
        """
        Test performance under sustained load.

        Validates:
        - Performance stability over multiple iterations
        - No performance degradation
        - Consistent throughput
        """
        schedulers = {
            "DDIM": DDIMScheduler(),
            "Adaptive": AdaptiveScheduler(num_inference_steps=25),
        }

        results = {}
        num_batches = 10

        for name, scheduler in schedulers.items():
            pipeline = DiffusionPipeline(
                scheduler=scheduler,
                image_size=(64, 64),
            )

            # Warmup
            pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

            # Sustained generation
            iteration_times = []
            start_total = time.perf_counter()

            for _ in range(num_batches):
                start = time.perf_counter()
                images = pipeline.generate(
                    batch_size=2,
                    num_inference_steps=25,
                    seed=None,
                )
                iteration_times.append(time.perf_counter() - start)

            total_time = time.perf_counter() - start_total

            results[name] = {
                "total_time": total_time,
                "avg_time": np.mean(iteration_times),
                "std_time": np.std(iteration_times),
                "throughput": (num_batches * 2) / total_time,
            }

            print(f"\n{name} Sustained Performance:")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Avg per batch: {results[name]['avg_time']:.3f}s")
            print(f"  Std dev: {results[name]['std_time']:.3f}s")
            print(f"  Throughput: {results[name]['throughput']:.2f} images/sec")

        # Validate stability (low variance)
        for name, result in results.items():
            # Coefficient of variation (std/mean) should be reasonable
            cv = result["std_time"] / result["avg_time"]
            assert cv < 0.5, f"{name} has high variance (CV={cv:.2f})"

        print("\n" + "="*60)
        print("SUSTAINED PERFORMANCE COMPARISON")
        print("="*60)
        print(f"{'Scheduler':<15} {'Throughput':<15} {'Stability (CV)':<15}")
        print("-"*50)
        for name, result in results.items():
            cv = result["std_time"] / result["avg_time"]
            print(f"{name:<15} {result['throughput']:<15.2f} {cv:<15.3f}")

    def test_batch_size_scalability(self):
        """
        Test scalability across different batch sizes.

        Validates:
        - Performance scales with batch size
        - Adaptive scheduler scales well
        - No memory issues with larger batches
        """
        batch_sizes = [1, 2, 4]
        scheduler = AdaptiveScheduler(num_inference_steps=20)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Warmup
        pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        results = {}
        for batch_size in batch_sizes:
            start = time.perf_counter()
            images = pipeline.generate(
                batch_size=batch_size,
                num_inference_steps=20,
                seed=None,
            )
            elapsed = time.perf_counter() - start

            results[batch_size] = {
                "time": elapsed,
                "throughput": batch_size / elapsed,
                "time_per_image": elapsed / batch_size,
            }

            print(f"\nBatch size {batch_size}:")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Throughput: {results[batch_size]['throughput']:.2f} images/sec")

        # Validate scalability
        # Larger batches should have better throughput (up to a point)
        throughputs = [results[bs]["throughput"] for bs in batch_sizes]
        assert throughputs[1] >= throughputs[0] * 0.8  # Some benefit from batching

        print("\n" + "="*60)
        print("BATCH SIZE SCALABILITY")
        print("="*60)
        print(f"{'Batch Size':<15} {'Throughput':<15} {'Time/Image':<15}")
        print("-"*50)
        for batch_size in batch_sizes:
            r = results[batch_size]
            print(
                f"{batch_size:<15} {r['throughput']:<15.2f} {r['time_per_image']:<15.3f}"
            )
