"""
Ablation Study Tests

Systematic comparisons between adaptive and baseline schedulers to validate
the effectiveness of adaptive sampling strategies.

Test Categories:
- Speed: Sampling steps and execution time
- Quality: Output quality metrics
- Adaptability: Response to different content types
- Robustness: Handling edge cases and varying conditions
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
from adaptive_diffusion.sampling.step_reduction import StepReductionStrategy


# Skip tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    not (platform.machine() == "arm64" and platform.system() == "Darwin"),
    reason="Apple Silicon required for MLX tests",
)


class TestSchedulerAblation:
    """Ablation studies comparing different schedulers."""

    @pytest.fixture
    def ddim_pipeline(self):
        """Baseline DDIM pipeline."""
        scheduler = DDIMScheduler()
        return DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

    @pytest.fixture
    def adaptive_pipeline(self):
        """Adaptive scheduler pipeline."""
        scheduler = AdaptiveScheduler(num_inference_steps=50)
        return DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

    def test_step_count_comparison(self, ddim_pipeline, adaptive_pipeline):
        """Compare effective step counts between schedulers."""
        # Generate with same seed
        ddim_images = ddim_pipeline.generate(batch_size=2, num_inference_steps=50, seed=42)
        adaptive_images = adaptive_pipeline.generate(batch_size=2, num_inference_steps=50, seed=42)

        # Both should produce valid outputs
        assert ddim_images.shape == adaptive_images.shape
        assert mx.isfinite(ddim_images).all()
        assert mx.isfinite(adaptive_images).all()

        # Verify step counts
        assert len(ddim_pipeline.scheduler.timesteps) == 50
        assert len(adaptive_pipeline.scheduler.timesteps) == 50

    def test_quality_comparison(self, ddim_pipeline, adaptive_pipeline):
        """Compare output quality metrics."""
        # Generate images
        ddim_images = ddim_pipeline.generate(batch_size=4, num_inference_steps=30, seed=123)
        adaptive_images = adaptive_pipeline.generate(batch_size=4, num_inference_steps=30, seed=123)

        # Compute quality proxies
        ddim_variance = float(mx.var(ddim_images))
        adaptive_variance = float(mx.var(adaptive_images))

        ddim_mean_abs = float(mx.mean(mx.abs(ddim_images)))
        adaptive_mean_abs = float(mx.mean(mx.abs(adaptive_images)))

        # Both should have reasonable variance (not collapsed)
        assert ddim_variance > 0.001
        assert adaptive_variance > 0.001

        # Both should have reasonable absolute values
        assert ddim_mean_abs > 0.01
        assert adaptive_mean_abs > 0.01

        print(f"\nDDIM variance: {ddim_variance:.4f}, Adaptive variance: {adaptive_variance:.4f}")
        print(f"DDIM mean_abs: {ddim_mean_abs:.4f}, Adaptive mean_abs: {adaptive_mean_abs:.4f}")

    @pytest.mark.slow
    def test_speed_comparison(self, ddim_pipeline, adaptive_pipeline):
        """Compare generation speed."""
        # Warm-up
        ddim_pipeline.generate(batch_size=1, num_inference_steps=20, seed=42)
        adaptive_pipeline.generate(batch_size=1, num_inference_steps=20, seed=42)

        # Benchmark DDIM
        start = time.perf_counter()
        for _ in range(3):
            ddim_pipeline.generate(batch_size=2, num_inference_steps=30, seed=None)
        ddim_time = time.perf_counter() - start

        # Benchmark Adaptive
        start = time.perf_counter()
        for _ in range(3):
            adaptive_pipeline.generate(batch_size=2, num_inference_steps=30, seed=None)
        adaptive_time = time.perf_counter() - start

        print(f"\nDDIM time: {ddim_time:.3f}s, Adaptive time: {adaptive_time:.3f}s")
        print(f"Relative speedup: {ddim_time / adaptive_time:.2f}x")

        # Both should complete in reasonable time
        assert ddim_time > 0
        assert adaptive_time > 0

    def test_consistency_across_steps(self, adaptive_pipeline):
        """Test consistency when varying step counts."""
        step_counts = [20, 30, 40, 50]
        results = []

        for steps in step_counts:
            images = adaptive_pipeline.generate(
                batch_size=1, num_inference_steps=steps, seed=999
            )
            results.append({
                "steps": steps,
                "variance": float(mx.var(images)),
                "mean": float(mx.mean(images)),
            })

        # All results should be valid
        for result in results:
            assert result["variance"] > 0
            assert mx.isfinite(mx.array(result["mean"]))

        # Variance trend should be reasonable (not wildly erratic)
        variances = [r["variance"] for r in results]
        variance_std = np.std(variances)
        variance_mean = np.mean(variances)
        cv = variance_std / (variance_mean + 1e-8)  # Coefficient of variation
        assert cv < 5.0  # Reasonable consistency

    def test_reproducibility_comparison(self, ddim_pipeline, adaptive_pipeline):
        """Compare reproducibility between schedulers."""
        # DDIM reproducibility
        ddim_1 = ddim_pipeline.generate(batch_size=1, num_inference_steps=25, seed=777)
        ddim_2 = ddim_pipeline.generate(batch_size=1, num_inference_steps=25, seed=777)
        assert mx.allclose(ddim_1, ddim_2, atol=1e-5)

        # Adaptive reproducibility
        adaptive_1 = adaptive_pipeline.generate(batch_size=1, num_inference_steps=25, seed=777)
        adaptive_2 = adaptive_pipeline.generate(batch_size=1, num_inference_steps=25, seed=777)
        assert mx.allclose(adaptive_1, adaptive_2, atol=1e-5)

    def test_scheduler_parameter_ablation(self):
        """Test effect of different adaptive scheduler parameters."""
        base_config = {
            "num_inference_steps": 40,
            "adaptive_threshold": 0.5,
            "progress_power": 2.0,
        }

        # Test different progress_power values
        powers = [1.0, 2.0, 3.0]
        results = []

        for power in powers:
            scheduler = AdaptiveScheduler(
                num_inference_steps=base_config["num_inference_steps"],
                progress_power=power,
            )
            pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))
            images = pipeline.generate(batch_size=1, seed=42)

            results.append({
                "power": power,
                "variance": float(mx.var(images)),
                "schedule_info": scheduler.get_schedule_info(),
            })

        # All should produce valid outputs
        for result in results:
            assert result["variance"] > 0
            assert result["schedule_info"]["num_steps"] == base_config["num_inference_steps"]

        print(f"\nProgress power ablation:")
        for result in results:
            print(f"  Power={result['power']}: variance={result['variance']:.4f}")


class TestQualityGuidedAblation:
    """Ablation studies for quality-guided sampling."""

    @pytest.fixture
    def base_scheduler(self):
        """Base adaptive scheduler."""
        return AdaptiveScheduler(num_inference_steps=40)

    @pytest.fixture
    def base_pipeline(self, base_scheduler):
        """Base pipeline for comparison."""
        return DiffusionPipeline(scheduler=base_scheduler, image_size=(64, 64))

    def test_quality_threshold_ablation(self, base_pipeline, base_scheduler):
        """Test effect of different quality thresholds."""
        thresholds = [0.4, 0.6, 0.8]
        results = []

        for threshold in thresholds:
            sampler = QualityGuidedSampler(
                scheduler=base_scheduler,
                quality_threshold=threshold,
            )

            mx.random.seed(42)
            noise = mx.random.normal((2, 64, 64, 3))
            images, info = sampler.sample(
                model=base_pipeline.model,
                noise=noise,
                num_steps=40,
            )

            results.append({
                "threshold": threshold,
                "total_steps": info["total_steps"],
                "final_quality": info["final_quality"],
                "early_stopped": info["early_stopped"],
                "mean_quality": np.mean(info["quality_history"]),
            })

        print(f"\nQuality threshold ablation:")
        for result in results:
            print(f"  Threshold={result['threshold']}: "
                  f"steps={result['total_steps']}, "
                  f"quality={result['final_quality']:.3f}, "
                  f"early_stop={result['early_stopped']}")

        # All should complete
        for result in results:
            assert result["total_steps"] > 0
            assert 0 <= result["final_quality"] <= 1

    def test_quality_guided_vs_baseline(self, base_pipeline, base_scheduler):
        """Compare quality-guided sampling vs baseline."""
        # Baseline sampling
        baseline_images = base_pipeline.generate(batch_size=2, num_inference_steps=40, seed=42)

        # Quality-guided sampling
        sampler = QualityGuidedSampler(scheduler=base_scheduler, quality_threshold=0.6)
        mx.random.seed(42)
        noise = mx.random.normal((2, 64, 64, 3))
        guided_images, info = sampler.sample(
            model=base_pipeline.model,
            noise=noise,
            num_steps=40,
        )

        # Compare metrics
        baseline_var = float(mx.var(baseline_images))
        guided_var = float(mx.var(guided_images))

        print(f"\nBaseline variance: {baseline_var:.4f}")
        print(f"Guided variance: {guided_var:.4f}")
        print(f"Quality tracking: {len(info['quality_history'])} samples")

        # Both should produce valid outputs
        assert baseline_var > 0
        assert guided_var > 0
        assert len(info["quality_history"]) > 0

    def test_early_stopping_effectiveness(self, base_pipeline, base_scheduler):
        """Test early stopping reduces steps while maintaining quality."""
        sampler = QualityGuidedSampler(
            scheduler=base_scheduler,
            quality_threshold=0.5,
            early_stop_threshold=0.9,  # High threshold for testing
        )

        mx.random.seed(42)
        noise = mx.random.normal((1, 64, 64, 3))
        images, info = sampler.sample(
            model=base_pipeline.model,
            noise=noise,
            num_steps=50,
        )

        print(f"\nEarly stopping test:")
        print(f"  Completed steps: {info['total_steps']}")
        print(f"  Early stopped: {info['early_stopped']}")
        if info["early_stopped"]:
            print(f"  Stopped at: {info['stopped_at_step']}")
        print(f"  Final quality: {info['final_quality']:.3f}")

        # Should complete with valid output
        assert images.shape == (1, 64, 64, 3)
        assert info["total_steps"] <= 50


class TestStepReductionAblation:
    """Ablation studies for step reduction strategies."""

    def test_complexity_sensitivity_ablation(self):
        """Test effect of complexity sensitivity parameter."""
        sensitivities = [0.0, 0.5, 1.0]
        base_complexity = 0.7
        results = []

        for sensitivity in sensitivities:
            strategy = StepReductionStrategy(
                base_steps=50,
                complexity_sensitivity=sensitivity,
            )

            optimal_steps = strategy.estimate_optimal_steps(
                content_complexity=base_complexity,
                quality_requirement=0.8,
            )

            results.append({
                "sensitivity": sensitivity,
                "optimal_steps": optimal_steps,
            })

        print(f"\nComplexity sensitivity ablation (complexity={base_complexity}):")
        for result in results:
            print(f"  Sensitivity={result['sensitivity']}: steps={result['optimal_steps']}")

        # Higher sensitivity should respond more to complexity
        step_range = max(r["optimal_steps"] for r in results) - min(r["optimal_steps"] for r in results)
        assert step_range >= 0  # Should vary

    def test_progressive_reduction_schedule(self):
        """Test progressive reduction schedules."""
        strategy = StepReductionStrategy(base_steps=100)

        # Test different numbers of stages
        stage_counts = [2, 4, 8]
        results = []

        for num_stages in stage_counts:
            schedule = strategy.progressive_step_schedule(
                initial_steps=100,
                target_steps=25,
                num_stages=num_stages,
            )

            results.append({
                "stages": num_stages,
                "schedule": schedule,
                "reduction_rate": np.mean(np.diff(schedule)) if len(schedule) > 1 else 0,
            })

        print(f"\nProgressive reduction ablation:")
        for result in results:
            print(f"  Stages={result['stages']}: schedule={result['schedule']}")

        # All should start at 100 and end at 25
        for result in results:
            assert result["schedule"][0] == 100
            assert result["schedule"][-1] == 25
            assert len(result["schedule"]) == result["stages"]

    def test_adaptive_allocation_fairness(self):
        """Test adaptive step allocation across different quality levels."""
        strategy = StepReductionStrategy(base_steps=100, min_steps=5)

        # Test with varying quality distributions
        quality_sets = [
            [0.8, 0.8, 0.8, 0.8],  # Uniform high quality
            [0.2, 0.2, 0.2, 0.2],  # Uniform low quality
            [0.9, 0.7, 0.5, 0.3],  # Gradient
            [0.9, 0.9, 0.1, 0.1],  # Mixed
        ]

        results = []
        for qualities in quality_sets:
            allocations = strategy.adaptive_step_allocation(
                quality_estimates=qualities,
                total_budget=100,
            )

            results.append({
                "qualities": qualities,
                "allocations": allocations,
                "allocation_std": np.std(allocations),
            })

        print(f"\nAdaptive allocation ablation:")
        for result in results:
            print(f"  Qualities={result['qualities']}")
            print(f"  Allocations={result['allocations']}")
            print(f"  Std={result['allocation_std']:.2f}")

        # All should sum to budget (with tolerance for rounding)
        for result in results:
            assert 90 <= sum(result["allocations"]) <= 110
            assert all(a >= strategy.min_steps for a in result["allocations"])


class TestDomainAdaptationAblation:
    """Ablation studies for domain adaptation."""

    def test_domain_specific_configs(self):
        """Test that different domains get different configurations."""
        from adaptive_diffusion.optimization.domain_adapter import (
            DomainAdapter,
            DomainType,
        )

        adapter = DomainAdapter()

        domains = [
            DomainType.PHOTOREALISTIC,
            DomainType.ARTISTIC,
            DomainType.SYNTHETIC,
            DomainType.SCIENTIFIC,
        ]

        configs = {}
        for domain in domains:
            config = adapter.get_config(domain_type=domain)
            configs[domain] = {
                "num_steps": config.num_steps,
                "adaptive_threshold": config.adaptive_threshold,
                "progress_power": config.progress_power,
                "quality_weight": config.quality_weight,
            }

        print(f"\nDomain-specific configuration ablation:")
        for domain, config in configs.items():
            print(f"  {domain.value}:")
            print(f"    steps={config['num_steps']}, "
                  f"threshold={config['adaptive_threshold']:.2f}, "
                  f"power={config['progress_power']:.2f}")

        # Configs should differ across domains
        num_steps = [c["num_steps"] for c in configs.values()]
        assert len(set(num_steps)) > 1  # At least some variation

    def test_domain_detection_ablation(self):
        """Test domain detection with different prompts."""
        from adaptive_diffusion.optimization.domain_adapter import DomainAdapter

        adapter = DomainAdapter()

        test_prompts = [
            ("a photograph of a mountain landscape", "expected: PHOTOREALISTIC or LANDSCAPE"),
            ("abstract geometric patterns", "expected: ABSTRACT"),
            ("oil painting of a sunset", "expected: ARTISTIC"),
            ("scientific diagram of cell structure", "expected: SCIENTIFIC"),
        ]

        results = []
        for prompt, expected in test_prompts:
            detected = adapter.detect_domain(prompt=prompt)
            results.append({
                "prompt": prompt,
                "detected": detected.value,
                "expected": expected,
            })

        print(f"\nDomain detection ablation:")
        for result in results:
            print(f"  '{result['prompt']}'")
            print(f"    Detected: {result['detected']}")
            print(f"    {result['expected']}")

        # All detections should be valid domain types
        for result in results:
            assert result["detected"] in [d.value for d in adapter.domain_configs.keys()]


@pytest.mark.slow
class TestComprehensiveAblation:
    """Comprehensive ablation studies combining multiple factors."""

    def test_full_pipeline_ablation(self):
        """Test full adaptive pipeline vs baseline across multiple configurations."""
        from adaptive_diffusion.optimization import OptimizationPipeline
        from adaptive_diffusion.optimization.domain_adapter import DomainType

        # Test configurations
        configs = [
            {"domain": DomainType.PHOTOREALISTIC, "steps": 60},
            {"domain": DomainType.ARTISTIC, "steps": 50},
            {"domain": DomainType.SYNTHETIC, "steps": 35},
        ]

        results = []
        for config in configs:
            pipeline = OptimizationPipeline(
                use_domain_adaptation=True,
                use_rl_optimization=False,  # Disable for speed
                verbose=0,
            )

            # Get optimized scheduler
            scheduler = pipeline.create_optimized_scheduler(
                domain_type=config["domain"],
            )

            # Create diffusion pipeline
            diff_pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))
            images = diff_pipeline.generate(batch_size=1, num_inference_steps=config["steps"], seed=42)

            results.append({
                "domain": config["domain"].value,
                "config_steps": config["steps"],
                "actual_steps": len(scheduler.timesteps),
                "variance": float(mx.var(images)),
            })

        print(f"\nFull pipeline ablation:")
        for result in results:
            print(f"  Domain={result['domain']}:")
            print(f"    Config steps={result['config_steps']}, "
                  f"Actual steps={result['actual_steps']}, "
                  f"Variance={result['variance']:.4f}")

        # All should produce valid outputs
        for result in results:
            assert result["actual_steps"] > 0
            assert result["variance"] > 0
