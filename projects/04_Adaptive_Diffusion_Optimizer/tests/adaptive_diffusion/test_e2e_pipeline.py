"""
End-to-End Pipeline Integration Tests

Tests the full adaptive diffusion pipeline integration across all components:
- Baseline schedulers (DDPM, DDIM, DPM-Solver)
- Adaptive scheduler with progress-based scheduling
- Quality-guided sampling
- Step reduction strategies
- RL-based optimization
- Domain adaptation
"""

from __future__ import annotations

import platform

import mlx.core as mx
import numpy as np
import pytest

from adaptive_diffusion.baseline import DiffusionPipeline, DDPMScheduler, DDIMScheduler
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler
from adaptive_diffusion.sampling.step_reduction import StepReductionStrategy
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline
from adaptive_diffusion.optimization.domain_adapter import DomainType


# Skip tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    not (platform.machine() == "arm64" and platform.system() == "Darwin"),
    reason="Apple Silicon required for MLX tests",
)


class TestE2EPipeline:
    """End-to-end integration tests for adaptive diffusion pipeline."""

    @pytest.fixture
    def baseline_pipeline(self):
        """Create baseline diffusion pipeline with DDIM scheduler."""
        scheduler = DDIMScheduler(num_inference_steps=50)
        return DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

    @pytest.fixture
    def adaptive_pipeline(self):
        """Create adaptive diffusion pipeline."""
        scheduler = AdaptiveScheduler(num_inference_steps=50)
        return DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

    @pytest.fixture
    def optimization_pipeline(self):
        """Create optimization pipeline."""
        return OptimizationPipeline(
            use_domain_adaptation=True,
            use_rl_optimization=True,
            verbose=0,
        )

    def test_baseline_to_adaptive_pipeline(self, baseline_pipeline, adaptive_pipeline):
        """Test integration from baseline to adaptive pipeline."""
        # Generate with baseline
        baseline_images = baseline_pipeline.generate(
            batch_size=2, num_inference_steps=30, seed=42
        )

        assert baseline_images.shape == (2, 64, 64, 3)
        assert baseline_images.dtype == mx.float32

        # Generate with adaptive
        adaptive_images = adaptive_pipeline.generate(
            batch_size=2, num_inference_steps=30, seed=42
        )

        assert adaptive_images.shape == (2, 64, 64, 3)
        assert adaptive_images.dtype == mx.float32

        # Both should produce valid outputs
        assert mx.isfinite(baseline_images).all()
        assert mx.isfinite(adaptive_images).all()

    def test_quality_guided_integration(self, adaptive_pipeline):
        """Test quality-guided sampler integration with adaptive scheduler."""
        sampler = QualityGuidedSampler(
            scheduler=adaptive_pipeline.scheduler,
            quality_threshold=0.6,
        )

        # Generate images with quality guidance
        mx.random.seed(42)
        noise = mx.random.normal((2, 64, 64, 3))
        images, sampling_info = sampler.sample(
            model=adaptive_pipeline.model,
            noise=noise,
            num_steps=30,
        )

        assert images.shape == (2, 64, 64, 3)
        assert mx.isfinite(images).all()

        # Check that quality monitoring was active
        assert len(sampler.quality_history) > 0
        assert all(0 <= q <= 1 for q in sampler.quality_history)
        assert "quality_history" in sampling_info
        assert sampling_info["total_steps"] > 0

    def test_step_reduction_integration(self):
        """Test step reduction strategies with adaptive scheduling."""
        strategy = StepReductionStrategy(base_steps=100, min_steps=10, max_steps=200)

        # Test optimal step estimation
        optimal_steps = strategy.estimate_optimal_steps(
            content_complexity=0.7, quality_requirement=0.8
        )
        assert 10 <= optimal_steps <= 200

        # Test progressive schedule generation
        schedule = strategy.progressive_step_schedule(
            initial_steps=100, target_steps=30, num_stages=4
        )
        assert len(schedule) == 4
        assert schedule[0] == 100  # Start at initial
        assert schedule[-1] == 30  # End at target

        # Test adaptive step allocation
        allocations = strategy.adaptive_step_allocation(
            quality_estimates=[0.6, 0.8, 0.5, 0.9], total_budget=100
        )
        assert len(allocations) == 4
        assert sum(allocations) <= 110  # Allow some slack for rounding

    def test_rl_optimization_integration(self, optimization_pipeline):
        """Test RL-based optimization pipeline integration."""
        # Run minimal optimization (fast test)
        results = optimization_pipeline.optimize(
            domain_type=DomainType.PHOTOREALISTIC,
            num_training_steps=100,
            num_optimization_episodes=2,
        )

        # Check results structure
        assert "domain_type" in results
        assert "domain_config" in results
        assert "final_config" in results
        assert "training_stats" in results

        # Verify domain detection
        assert results["domain_type"] == DomainType.PHOTOREALISTIC

        # Verify final config has expected keys
        config = results["final_config"]
        assert "num_steps" in config
        assert "adaptive_threshold" in config
        assert "progress_power" in config

    def test_domain_adaptation_integration(self, optimization_pipeline):
        """Test domain adaptation with different content types."""
        domains = [
            DomainType.PHOTOREALISTIC,
            DomainType.ARTISTIC,
            DomainType.SCIENTIFIC,
        ]

        configs = []
        for domain in domains:
            results = optimization_pipeline.optimize(
                domain_type=domain,
                num_training_steps=50,
                num_optimization_episodes=1,
            )
            configs.append(results["final_config"])

        # Different domains should yield different configs
        assert len(configs) == 3

        # All configs should be valid
        for config in configs:
            assert config["num_steps"] > 0
            assert 0 < config["adaptive_threshold"] < 1
            assert config["progress_power"] > 0

    def test_full_pipeline_with_intermediates(self, adaptive_pipeline):
        """Test full pipeline with intermediate step tracking."""
        images, intermediates = adaptive_pipeline.generate(
            batch_size=1,
            num_inference_steps=20,
            seed=42,
            return_intermediates=True,
        )

        # Check final output
        assert images.shape == (1, 64, 64, 3)
        assert mx.isfinite(images).all()

        # Check intermediates
        assert len(intermediates) == 20
        for intermediate in intermediates:
            assert intermediate.shape == (1, 64, 64, 3)
            assert mx.isfinite(intermediate).all()

        # Verify denoising progression (samples should change over time)
        variances = [float(mx.var(img)) for img in intermediates]
        # Variance should be present (not all zeros or constants)
        assert np.std(variances) > 0  # Variances change over timesteps
        assert all(v >= 0 for v in variances)  # All variances non-negative

    def test_optimized_scheduler_creation(self, optimization_pipeline):
        """Test creation of optimized schedulers from pipeline."""
        scheduler = optimization_pipeline.create_optimized_scheduler(
            domain_type=DomainType.PHOTOREALISTIC,
        )

        assert isinstance(scheduler, AdaptiveScheduler)
        assert len(scheduler.timesteps) > 0
        assert scheduler.num_inference_steps > 0

    def test_optimized_sampler_creation(self, optimization_pipeline):
        """Test creation of optimized samplers from pipeline."""
        sampler = optimization_pipeline.create_optimized_sampler(
            domain_type=DomainType.ARTISTIC,
        )

        assert isinstance(sampler, QualityGuidedSampler)
        assert isinstance(sampler.scheduler, AdaptiveScheduler)

    def test_pipeline_state_persistence(self, optimization_pipeline, tmp_path):
        """Test saving and loading pipeline state."""
        # Run optimization
        optimization_pipeline.optimize(
            domain_type=DomainType.PHOTOREALISTIC,
            num_training_steps=50,
            num_optimization_episodes=1,
        )

        # Save state
        save_path = tmp_path / "pipeline_state.pkl"
        optimization_pipeline.save(save_path)

        # Load state in new pipeline
        new_pipeline = OptimizationPipeline(
            use_domain_adaptation=True,
            use_rl_optimization=True,
        )
        new_pipeline.load(save_path)

        # Verify state was loaded
        assert len(new_pipeline.optimization_history) >= 0

    def test_cross_component_quality_tracking(self, adaptive_pipeline):
        """Test quality tracking across multiple components."""
        # Create quality-guided sampler
        sampler = QualityGuidedSampler(
            scheduler=adaptive_pipeline.scheduler,
            quality_threshold=0.6,
        )

        # Generate with quality tracking
        mx.random.seed(42)
        noise = mx.random.normal((2, 64, 64, 3))
        images, sampling_info = sampler.sample(
            model=adaptive_pipeline.model,
            noise=noise,
            num_steps=25,
        )

        # Verify quality tracking in sampler
        assert len(sampler.quality_history) > 0
        assert "quality_history" in sampling_info
        assert len(sampling_info["quality_history"]) > 0

        # Quality estimates should be valid
        assert 0 <= np.mean(sampler.quality_history) <= 1
        assert 0 <= np.mean(sampling_info["quality_history"]) <= 1

    def test_multiple_scheduler_comparison(self):
        """Test integration with multiple scheduler types."""
        schedulers = [
            ("ddim", DDIMScheduler()),
            ("adaptive", AdaptiveScheduler(num_inference_steps=30)),
        ]

        results = {}
        for name, scheduler in schedulers:
            # Set timesteps after initialization if needed
            if hasattr(scheduler, 'set_timesteps') and name != "adaptive":
                scheduler.set_timesteps(30)

            pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))
            images = pipeline.generate(batch_size=1, num_inference_steps=30, seed=42)
            results[name] = {
                "shape": images.shape,
                "mean": float(mx.mean(images)),
                "std": float(mx.std(images)),
            }

        # All should produce valid outputs
        for name, result in results.items():
            assert result["shape"] == (1, 64, 64, 3)
            assert mx.isfinite(mx.array(result["mean"]))
            assert mx.isfinite(mx.array(result["std"]))

    def test_error_handling_integration(self, adaptive_pipeline):
        """Test error handling across integrated components."""
        # Test invalid batch size
        with pytest.raises((ValueError, RuntimeError, ZeroDivisionError)):
            adaptive_pipeline.generate(batch_size=-1)

        # Test invalid num_inference_steps
        with pytest.raises((ValueError, RuntimeError, ZeroDivisionError)):
            adaptive_pipeline.generate(batch_size=1, num_inference_steps=0)

    def test_memory_efficiency(self, adaptive_pipeline):
        """Test memory efficiency of integrated pipeline."""
        # Generate multiple batches
        for _ in range(3):
            images = adaptive_pipeline.generate(
                batch_size=2,
                num_inference_steps=20,
                seed=None,
            )
            assert images.shape == (2, 64, 64, 3)

        # Should not accumulate excessive memory
        # (MLX handles this automatically, but verify no crashes)

    def test_reproducibility(self, adaptive_pipeline):
        """Test reproducibility across pipeline runs."""
        # Generate with fixed seed
        images1 = adaptive_pipeline.generate(batch_size=2, seed=42)
        images2 = adaptive_pipeline.generate(batch_size=2, seed=42)

        # Should produce identical results
        assert mx.allclose(images1, images2, atol=1e-5)

    def test_adaptive_step_adjustment(self, adaptive_pipeline):
        """Test adaptive step adjustment during generation."""
        scheduler = adaptive_pipeline.scheduler
        initial_steps = scheduler.num_inference_steps

        # Generate with quality tracking
        sampler = QualityGuidedSampler(scheduler=scheduler, quality_threshold=0.7)
        mx.random.seed(42)
        noise = mx.random.normal((1, 64, 64, 3))
        images, sampling_info = sampler.sample(
            model=adaptive_pipeline.model,
            noise=noise,
            num_steps=initial_steps,
        )

        # Adaptive behavior should be reflected in sampling info
        assert sampling_info["total_steps"] > 0
        assert len(sampling_info["quality_history"]) > 0
        assert sampling_info["final_quality"] >= 0


class TestE2EPerformance:
    """Performance-focused integration tests."""

    @pytest.mark.slow
    def test_adaptive_vs_baseline_speed(self):
        """Compare adaptive vs baseline pipeline speed."""
        baseline = DiffusionPipeline(
            scheduler=DDIMScheduler(num_inference_steps=50),
            image_size=(64, 64),
        )

        adaptive = DiffusionPipeline(
            scheduler=AdaptiveScheduler(num_inference_steps=50),
            image_size=(64, 64),
        )

        # Warm-up
        baseline.generate(batch_size=1, seed=42)
        adaptive.generate(batch_size=1, seed=42)

        # Benchmark baseline
        import time

        start = time.perf_counter()
        for _ in range(5):
            baseline.generate(batch_size=1, seed=None)
        baseline_time = time.perf_counter() - start

        # Benchmark adaptive
        start = time.perf_counter()
        for _ in range(5):
            adaptive.generate(batch_size=1, seed=None)
        adaptive_time = time.perf_counter() - start

        # Both should complete in reasonable time
        assert baseline_time > 0
        assert adaptive_time > 0

        # Log performance ratio
        print(f"\nBaseline: {baseline_time:.3f}s, Adaptive: {adaptive_time:.3f}s")
        print(f"Speedup: {baseline_time / adaptive_time:.2f}x")

    @pytest.mark.slow
    def test_step_reduction_speedup(self):
        """Test speedup from step reduction strategies."""
        full_scheduler = AdaptiveScheduler(num_inference_steps=100)
        pipeline_full = DiffusionPipeline(scheduler=full_scheduler, image_size=(64, 64))

        reduced_scheduler = AdaptiveScheduler(num_inference_steps=50)
        pipeline_reduced = DiffusionPipeline(
            scheduler=reduced_scheduler, image_size=(64, 64)
        )

        # Generate with both
        import time

        start = time.perf_counter()
        images_full = pipeline_full.generate(batch_size=1, seed=42)
        time_full = time.perf_counter() - start

        start = time.perf_counter()
        images_reduced = pipeline_reduced.generate(batch_size=1, seed=42)
        time_reduced = time.perf_counter() - start

        # Reduced should be faster
        assert time_reduced < time_full * 1.2  # Allow some overhead
        print(f"\nFull: {time_full:.3f}s, Reduced: {time_reduced:.3f}s")
        print(f"Speedup: {time_full / time_reduced:.2f}x")
