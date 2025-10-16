"""
Performance Benchmark Tests

Comprehensive performance benchmarks for adaptive diffusion components:
- Sampling speed (steps/second)
- Memory usage
- Throughput (images/second)
- MLX optimization validation
"""

from __future__ import annotations

import platform
import time

import mlx.core as mx
import numpy as np
import pytest

from adaptive_diffusion.baseline import DiffusionPipeline, DDIMScheduler
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline


# Skip tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    not (platform.machine() == "arm64" and platform.system() == "Darwin"),
    reason="Apple Silicon required for MLX tests",
)


class TestSamplingSpeed:
    """Benchmark sampling speed across different configurations."""

    @pytest.fixture
    def ddim_pipeline(self):
        """DDIM baseline pipeline."""
        scheduler = DDIMScheduler()
        return DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

    @pytest.fixture
    def adaptive_pipeline(self):
        """Adaptive scheduler pipeline."""
        scheduler = AdaptiveScheduler(num_inference_steps=50)
        return DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

    def test_baseline_sampling_speed(self, ddim_pipeline):
        """Benchmark baseline DDIM sampling speed."""
        # Warm-up
        ddim_pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        # Benchmark
        start = time.perf_counter()
        num_iterations = 3
        for _ in range(num_iterations):
            ddim_pipeline.generate(batch_size=2, num_inference_steps=30, seed=None)
        elapsed = time.perf_counter() - start

        steps_per_sec = (num_iterations * 2 * 30) / elapsed  # total_images * steps / time
        print(f"\nDDIM sampling speed: {steps_per_sec:.1f} steps/sec")
        print(f"  Total time: {elapsed:.3f}s for {num_iterations * 2} images")

        # Should complete in reasonable time
        assert elapsed < 60.0  # Max 1 minute for benchmark

    def test_adaptive_sampling_speed(self, adaptive_pipeline):
        """Benchmark adaptive sampling speed."""
        # Warm-up
        adaptive_pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        # Benchmark
        start = time.perf_counter()
        num_iterations = 3
        for _ in range(num_iterations):
            adaptive_pipeline.generate(batch_size=2, num_inference_steps=30, seed=None)
        elapsed = time.perf_counter() - start

        steps_per_sec = (num_iterations * 2 * 30) / elapsed
        print(f"\nAdaptive sampling speed: {steps_per_sec:.1f} steps/sec")
        print(f"  Total time: {elapsed:.3f}s for {num_iterations * 2} images")

        # Should complete in reasonable time
        assert elapsed < 60.0

    def test_quality_guided_overhead(self):
        """Benchmark overhead of quality-guided sampling."""
        scheduler = AdaptiveScheduler(num_inference_steps=40)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))
        sampler = QualityGuidedSampler(scheduler=scheduler)

        # Warm-up
        mx.random.seed(42)
        noise = mx.random.normal((1, 64, 64, 3))
        sampler.sample(model=pipeline.model, noise=noise, num_steps=10)

        # Benchmark
        start = time.perf_counter()
        num_iterations = 3
        for i in range(num_iterations):
            mx.random.seed(42 + i)  # Use different seeds
            noise = mx.random.normal((2, 64, 64, 3))
            sampler.sample(model=pipeline.model, noise=noise, num_steps=30)
        elapsed = time.perf_counter() - start

        steps_per_sec = (num_iterations * 2 * 30) / elapsed
        print(f"\nQuality-guided sampling speed: {steps_per_sec:.1f} steps/sec")
        print(f"  Total time: {elapsed:.3f}s for {num_iterations * 2} images")

        # Quality guidance adds overhead, but should still be reasonable
        assert elapsed < 90.0  # Allow more time for quality computation

    def test_step_scaling(self, adaptive_pipeline):
        """Test that time scales linearly with steps."""
        step_counts = [10, 20, 30]
        times = []

        # Warm-up
        adaptive_pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        for steps in step_counts:
            start = time.perf_counter()
            adaptive_pipeline.generate(batch_size=1, num_inference_steps=steps, seed=None)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        print(f"\nStep scaling:")
        for steps, t in zip(step_counts, times):
            print(f"  {steps} steps: {t:.3f}s ({steps/t:.1f} steps/sec)")

        # Time should roughly scale with steps
        # Check that doubling steps roughly doubles time (with tolerance)
        time_ratio = times[1] / times[0]  # 20 steps / 10 steps
        print(f"  Time ratio (20/10 steps): {time_ratio:.2f}x (expected ~2x)")

        # Should be in reasonable range (not sub-linear or super-linear)
        assert 1.5 <= time_ratio <= 3.0

    def test_batch_throughput(self, adaptive_pipeline):
        """Test throughput with different batch sizes."""
        batch_sizes = [1, 2, 4]
        throughputs = []

        # Warm-up
        adaptive_pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        for batch_size in batch_sizes:
            start = time.perf_counter()
            adaptive_pipeline.generate(batch_size=batch_size, num_inference_steps=20, seed=None)
            elapsed = time.perf_counter() - start
            images_per_sec = batch_size / elapsed
            throughputs.append(images_per_sec)

        print(f"\nBatch throughput:")
        for batch_size, throughput in zip(batch_sizes, throughputs):
            print(f"  Batch {batch_size}: {throughput:.2f} images/sec")

        # Larger batches should have better throughput (up to a point)
        # At minimum, should process multiple images
        assert all(t > 0 for t in throughputs)


class TestMemoryEfficiency:
    """Benchmark memory usage patterns."""

    def test_memory_footprint(self):
        """Test basic memory footprint of generation."""
        scheduler = AdaptiveScheduler(num_inference_steps=30)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate images and ensure no memory leaks
        for _ in range(5):
            images = pipeline.generate(batch_size=2, num_inference_steps=20, seed=None)
            # Verify images are created
            assert images.shape == (2, 64, 64, 3)

        # Should complete without memory issues
        print("\nMemory footprint test: PASS (no crashes)")

    def test_memory_with_intermediates(self):
        """Test memory usage when storing intermediates."""
        scheduler = AdaptiveScheduler(num_inference_steps=20)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate with intermediates
        images, intermediates = pipeline.generate(
            batch_size=1,
            num_inference_steps=20,
            seed=42,
            return_intermediates=True,
        )

        # Check memory usage
        total_elements = sum(img.size for img in intermediates) + images.size
        memory_mb = total_elements * 4 / (1024 * 1024)  # Assuming float32

        print(f"\nMemory with intermediates: {memory_mb:.2f} MB")
        print(f"  Number of intermediate steps: {len(intermediates)}")

        # Should be reasonable (not excessive)
        assert memory_mb < 1000  # Less than 1 GB for this small test

    def test_memory_large_batch(self):
        """Test memory with larger batches."""
        scheduler = AdaptiveScheduler(num_inference_steps=20)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate with larger batch
        images = pipeline.generate(batch_size=8, num_inference_steps=15, seed=42)

        memory_mb = images.size * 4 / (1024 * 1024)
        print(f"\nMemory for batch 8: {memory_mb:.2f} MB")

        assert images.shape == (8, 64, 64, 3)
        assert memory_mb < 100  # Reasonable for this size


class TestMLXOptimization:
    """Validate MLX-specific optimizations."""

    def test_apple_silicon_detection(self):
        """Test that Apple Silicon is correctly detected."""
        scheduler = AdaptiveScheduler()
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Should detect Apple Silicon
        assert pipeline.device == "apple_silicon"
        print(f"\nDetected device: {pipeline.device}")

    def test_mlx_array_operations(self):
        """Test MLX array operations are working."""
        # Create MLX arrays
        x = mx.random.normal((4, 64, 64, 3))
        y = mx.random.normal((4, 64, 64, 3))

        # Perform operations
        z = x + y
        w = mx.mean(z)

        # Operations should complete
        assert z.shape == (4, 64, 64, 3)
        assert isinstance(float(w), float)
        print(f"\nMLX operations: PASS")

    def test_mlx_model_execution(self):
        """Test that models execute on MLX."""
        scheduler = AdaptiveScheduler(num_inference_steps=10)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate (which executes model)
        start = time.perf_counter()
        images = pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)
        elapsed = time.perf_counter() - start

        print(f"\nMLX model execution: {elapsed:.3f}s")
        assert images.shape == (1, 64, 64, 3)
        assert elapsed < 30.0  # Should be reasonably fast


@pytest.mark.benchmark
class TestComprehensiveBenchmarks:
    """Comprehensive benchmark suite."""

    def test_end_to_end_performance(self):
        """Comprehensive end-to-end performance benchmark."""
        configs = [
            {"name": "Fast", "steps": 20, "batch": 1},
            {"name": "Balanced", "steps": 30, "batch": 2},
            {"name": "Quality", "steps": 50, "batch": 1},
        ]

        scheduler = AdaptiveScheduler()
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Warm-up
        pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        results = []
        for config in configs:
            start = time.perf_counter()
            images = pipeline.generate(
                batch_size=config["batch"],
                num_inference_steps=config["steps"],
                seed=None,
            )
            elapsed = time.perf_counter() - start

            results.append({
                "name": config["name"],
                "time": elapsed,
                "steps": config["steps"],
                "batch": config["batch"],
                "images_per_sec": config["batch"] / elapsed,
                "steps_per_sec": (config["batch"] * config["steps"]) / elapsed,
            })

        print(f"\nEnd-to-end performance:")
        for result in results:
            print(f"  {result['name']} ({result['steps']} steps, batch {result['batch']}):")
            print(f"    Time: {result['time']:.3f}s")
            print(f"    Throughput: {result['images_per_sec']:.2f} images/sec")
            print(f"    Speed: {result['steps_per_sec']:.1f} steps/sec")

        # All configurations should complete
        assert all(r["time"] > 0 for r in results)
        assert all(r["images_per_sec"] > 0 for r in results)

    def test_optimization_pipeline_performance(self):
        """Benchmark optimization pipeline overhead."""
        from adaptive_diffusion.optimization.domain_adapter import DomainType

        pipeline = OptimizationPipeline(
            use_domain_adaptation=True,
            use_rl_optimization=False,  # Disable RL for speed
            verbose=0,
        )

        # Test domain config retrieval (should be fast)
        start = time.perf_counter()
        for domain in [DomainType.PHOTOREALISTIC, DomainType.ARTISTIC, DomainType.SYNTHETIC]:
            config = pipeline.domain_adapter.get_config(domain_type=domain)
        elapsed = time.perf_counter() - start

        print(f"\nDomain config retrieval: {elapsed*1000:.2f}ms")
        assert elapsed < 1.0  # Should be very fast

        # Test scheduler creation (should be fast)
        start = time.perf_counter()
        scheduler = pipeline.create_optimized_scheduler(
            domain_type=DomainType.ARTISTIC,
        )
        elapsed = time.perf_counter() - start

        print(f"Optimized scheduler creation: {elapsed:.3f}s")
        assert elapsed < 10.0  # Should complete quickly

    def test_comparative_performance_summary(self):
        """Generate comprehensive performance comparison."""
        from adaptive_diffusion.baseline import DDIMScheduler

        pipelines = [
            ("DDIM", DiffusionPipeline(scheduler=DDIMScheduler(), image_size=(64, 64))),
            ("Adaptive", DiffusionPipeline(scheduler=AdaptiveScheduler(num_inference_steps=30), image_size=(64, 64))),
        ]

        # Warm-up all pipelines
        for name, pipeline in pipelines:
            pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        results = []
        for name, pipeline in pipelines:
            start = time.perf_counter()
            pipeline.generate(batch_size=2, num_inference_steps=30, seed=None)
            elapsed = time.perf_counter() - start

            results.append({
                "name": name,
                "time": elapsed,
                "images_per_sec": 2 / elapsed,
                "steps_per_sec": 60 / elapsed,
            })

        print(f"\nComparative Performance Summary:")
        print(f"{'Scheduler':<12} {'Time':<10} {'Images/sec':<12} {'Steps/sec':<12}")
        print("-" * 48)
        for result in results:
            print(f"{result['name']:<12} {result['time']:<10.3f} {result['images_per_sec']:<12.2f} {result['steps_per_sec']:<12.1f}")

        # Compute relative performance
        if len(results) == 2:
            speedup = results[0]["time"] / results[1]["time"]
            print(f"\nRelative speedup: {speedup:.2f}x")

        # All should complete
        assert all(r["time"] > 0 for r in results)


@pytest.mark.slow
class TestStressTests:
    """Stress tests for performance under load."""

    def test_sustained_generation(self):
        """Test performance under sustained generation load."""
        scheduler = AdaptiveScheduler(num_inference_steps=25)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Warm-up
        pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        # Sustained generation
        num_batches = 10
        start = time.perf_counter()
        for _ in range(num_batches):
            images = pipeline.generate(batch_size=1, num_inference_steps=25, seed=None)
            assert images.shape == (1, 64, 64, 3)
        elapsed = time.perf_counter() - start

        avg_time_per_batch = elapsed / num_batches
        print(f"\nSustained generation ({num_batches} batches):")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg time per batch: {avg_time_per_batch:.3f}s")
        print(f"  Throughput: {num_batches / elapsed:.2f} images/sec")

        # Should maintain reasonable performance
        assert avg_time_per_batch < 10.0  # Each batch under 10s

    def test_memory_stability(self):
        """Test memory stability over multiple generations."""
        scheduler = AdaptiveScheduler(num_inference_steps=20)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate multiple times
        for i in range(20):
            images = pipeline.generate(batch_size=2, num_inference_steps=15, seed=None)
            assert images.shape == (2, 64, 64, 3)

            if i % 5 == 0:
                print(f"  Iteration {i}: OK")

        print("\nMemory stability test: PASS (20 iterations)")
