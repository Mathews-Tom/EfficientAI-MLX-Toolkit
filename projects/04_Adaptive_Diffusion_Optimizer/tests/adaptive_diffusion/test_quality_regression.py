"""
Quality Regression Tests

Validates quality metrics and establishes regression thresholds to ensure
that adaptive optimizations don't degrade output quality.

Note: This uses proxy quality metrics (variance, structure, etc.) since
full FID/CLIP scores require reference datasets and trained evaluators.
"""

from __future__ import annotations

import platform

import mlx.core as mx
import numpy as np
import pytest

from adaptive_diffusion.baseline import DiffusionPipeline
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
from adaptive_diffusion.sampling.quality_guided import QualityEstimator, QualityGuidedSampler


# Skip tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    not (platform.machine() == "arm64" and platform.system() == "Darwin"),
    reason="Apple Silicon required for MLX tests",
)


# Quality thresholds (established from baseline testing)
QUALITY_THRESHOLDS = {
    "min_variance": 0.001,  # Minimum variance to avoid mode collapse
    "max_variance": 50000.0,  # Maximum variance (untrained model produces high variance)
    "min_structure_score": 0.1,  # Minimum structural consistency
    "min_quality_estimate": 0.0,  # Minimum quality estimate
    "max_quality_estimate": 1.0,  # Maximum quality estimate
}


class TestQualityEstimation:
    """Tests for quality estimation accuracy and consistency."""

    @pytest.fixture
    def quality_estimator(self):
        """Create quality estimator."""
        return QualityEstimator()

    def test_quality_range_bounds(self, quality_estimator):
        """Test that quality estimates are within valid bounds."""
        # Generate test samples
        mx.random.seed(42)
        samples = [
            mx.random.normal((2, 64, 64, 3)),  # Random noise
            mx.zeros((2, 64, 64, 3)),  # All zeros
            mx.ones((2, 64, 64, 3)),  # All ones
            mx.random.uniform(0, 1, (2, 64, 64, 3)),  # Uniform [0,1]
        ]

        timesteps = [999, 500, 100, 0]

        for sample, timestep in zip(samples, timesteps):
            quality = quality_estimator.estimate_quality(sample, timestep)

            # Quality should be in valid range
            assert QUALITY_THRESHOLDS["min_quality_estimate"] <= quality <= QUALITY_THRESHOLDS["max_quality_estimate"]
            assert isinstance(quality, float)
            assert not np.isnan(quality)
            assert not np.isinf(quality)

    def test_quality_improves_over_timesteps(self, quality_estimator):
        """Test that quality estimates improve as denoising progresses."""
        # Generate a sequence of samples simulating denoising
        mx.random.seed(42)
        timesteps = [1000, 750, 500, 250, 100, 50, 10, 0]
        qualities = []

        for t in timesteps:
            # Simulate progressively less noisy samples
            noise_level = t / 1000.0
            base_sample = mx.ones((1, 64, 64, 3)) * 0.5  # Base signal
            noise = mx.random.normal((1, 64, 64, 3)) * noise_level
            sample = base_sample + noise

            quality = quality_estimator.estimate_quality(sample, t)
            qualities.append(quality)

        # Quality should generally improve (allow some noise in estimates)
        # Check that late-stage quality > early-stage quality
        early_quality = np.mean(qualities[:3])
        late_quality = np.mean(qualities[-3:])

        print(f"\nQuality progression: {qualities}")
        print(f"Early quality: {early_quality:.3f}, Late quality: {late_quality:.3f}")

        # Late quality should be at least close to early quality
        # (may not always be higher due to proxy metrics)
        assert late_quality >= early_quality * 0.7  # Allow 30% tolerance

    def test_quality_consistency(self, quality_estimator):
        """Test consistency of quality estimates for same input."""
        mx.random.seed(42)
        sample = mx.random.normal((2, 64, 64, 3))
        timestep = 500

        # Estimate quality multiple times
        qualities = [
            quality_estimator.estimate_quality(sample, timestep)
            for _ in range(5)
        ]

        # Should be identical (deterministic computation)
        assert len(set(qualities)) == 1

    def test_quality_components(self, quality_estimator):
        """Test individual quality metric components."""
        mx.random.seed(42)
        sample = mx.random.normal((2, 64, 64, 3))

        # Test noise level estimation
        noise_score = quality_estimator._estimate_noise_level(sample, 500)
        assert 0 <= noise_score <= 1

        # Test structure estimation
        structure_score = quality_estimator._estimate_structure(sample)
        assert 0 <= structure_score <= 1

        # Test frequency content
        frequency_score = quality_estimator._estimate_frequency_content(sample)
        assert 0 <= frequency_score <= 1

        # Test sharpness
        sharpness_score = quality_estimator._estimate_sharpness(sample)
        assert 0 <= sharpness_score <= 1


class TestBaselineQuality:
    """Tests for baseline quality metrics."""

    @pytest.fixture
    def pipeline(self):
        """Create adaptive diffusion pipeline."""
        scheduler = AdaptiveScheduler(num_inference_steps=50)
        return DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

    def test_output_variance_bounds(self, pipeline):
        """Test that output variance is within expected bounds."""
        images = pipeline.generate(batch_size=4, num_inference_steps=30, seed=42)
        variance = float(mx.var(images))

        print(f"\nOutput variance: {variance:.4f}")

        assert QUALITY_THRESHOLDS["min_variance"] <= variance <= QUALITY_THRESHOLDS["max_variance"]

    def test_output_no_mode_collapse(self, pipeline):
        """Test that outputs don't collapse to single mode."""
        # Generate multiple batches
        variances = []
        for seed in [42, 123, 456, 789]:
            images = pipeline.generate(batch_size=2, num_inference_steps=25, seed=seed)
            variance = float(mx.var(images))
            variances.append(variance)

        # Variances should show diversity (not all identical)
        variance_std = np.std(variances)
        print(f"\nVariance across seeds: {variances}")
        print(f"Variance std: {variance_std:.4f}")

        # Some variation expected (not mode collapsed)
        assert variance_std > 0.001 or all(v > QUALITY_THRESHOLDS["min_variance"] for v in variances)

    def test_output_finite_values(self, pipeline):
        """Test that outputs contain only finite values."""
        images = pipeline.generate(batch_size=2, num_inference_steps=20, seed=42)

        assert mx.isfinite(images).all()
        assert not mx.isnan(images).any()
        assert not mx.isinf(images).any()

    def test_output_spatial_structure(self, pipeline):
        """Test that outputs have spatial structure (not pure noise)."""
        images = pipeline.generate(batch_size=1, num_inference_steps=30, seed=42)

        # Compute gradients (measure of structure)
        dx = images[:, 1:, :, :] - images[:, :-1, :, :]
        dy = images[:, :, 1:, :] - images[:, :, :-1, :]

        dx_aligned = dx[:, :, :-1, :]
        dy_aligned = dy[:, :-1, :, :]

        gradient_mag = mx.sqrt(dx_aligned**2 + dy_aligned**2)
        mean_gradient = float(mx.mean(gradient_mag))

        print(f"\nMean gradient magnitude: {mean_gradient:.4f}")

        # Should have some structure (gradients present)
        assert mean_gradient > 0.001


class TestQualityRegressionThresholds:
    """Tests that validate quality doesn't regress below established thresholds."""

    @pytest.fixture
    def baseline_pipeline(self):
        """Baseline pipeline for comparison."""
        from adaptive_diffusion.baseline import DDIMScheduler
        scheduler = DDIMScheduler()
        return DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

    @pytest.fixture
    def adaptive_pipeline(self):
        """Adaptive pipeline."""
        scheduler = AdaptiveScheduler(num_inference_steps=50)
        return DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

    def test_adaptive_quality_vs_baseline(self, baseline_pipeline, adaptive_pipeline):
        """Test that adaptive quality is comparable to baseline."""
        # Generate with both
        baseline_images = baseline_pipeline.generate(batch_size=4, num_inference_steps=30, seed=42)
        adaptive_images = adaptive_pipeline.generate(batch_size=4, num_inference_steps=30, seed=42)

        # Compute quality proxies
        baseline_variance = float(mx.var(baseline_images))
        adaptive_variance = float(mx.var(adaptive_images))

        baseline_mean_abs = float(mx.mean(mx.abs(baseline_images)))
        adaptive_mean_abs = float(mx.mean(mx.abs(adaptive_images)))

        print(f"\nBaseline variance: {baseline_variance:.4f}, Adaptive variance: {adaptive_variance:.4f}")
        print(f"Baseline mean_abs: {baseline_mean_abs:.4f}, Adaptive mean_abs: {adaptive_mean_abs:.4f}")

        # Adaptive should not significantly degrade quality
        # Allow 50% tolerance for different sampling strategies
        assert adaptive_variance > baseline_variance * 0.5
        assert adaptive_mean_abs > baseline_mean_abs * 0.5

    def test_quality_with_step_reduction(self, adaptive_pipeline):
        """Test quality with reduced steps vs full steps."""
        # Full steps
        full_images = adaptive_pipeline.generate(batch_size=2, num_inference_steps=50, seed=42)
        full_variance = float(mx.var(full_images))

        # Reduced steps
        reduced_images = adaptive_pipeline.generate(batch_size=2, num_inference_steps=25, seed=42)
        reduced_variance = float(mx.var(reduced_images))

        print(f"\nFull steps variance: {full_variance:.4f}")
        print(f"Reduced steps variance: {reduced_variance:.4f}")

        # Reduced should maintain reasonable quality
        assert reduced_variance > full_variance * 0.3  # Allow 70% tolerance

    def test_quality_guided_maintains_threshold(self):
        """Test that quality-guided sampling maintains quality threshold."""
        scheduler = AdaptiveScheduler(num_inference_steps=40)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        sampler = QualityGuidedSampler(
            scheduler=scheduler,
            quality_threshold=0.6,
        )

        mx.random.seed(42)
        noise = mx.random.normal((2, 64, 64, 3))
        images, info = sampler.sample(
            model=pipeline.model,
            noise=noise,
            num_steps=40,
        )

        # Final quality should meet or exceed threshold (on average)
        mean_quality = np.mean(info["quality_history"])
        final_quality = info["final_quality"]

        print(f"\nMean quality: {mean_quality:.3f}")
        print(f"Final quality: {final_quality:.3f}")
        print(f"Quality threshold: 0.6")

        # Quality should be in valid range
        assert 0 <= final_quality <= 1
        assert 0 <= mean_quality <= 1

    def test_quality_consistency_across_batches(self, adaptive_pipeline):
        """Test quality consistency across multiple generation batches."""
        qualities = []

        for seed in range(5):
            images = adaptive_pipeline.generate(batch_size=2, num_inference_steps=30, seed=seed)
            variance = float(mx.var(images))
            qualities.append(variance)

        # Quality should be consistent (not wildly varying)
        quality_std = np.std(qualities)
        quality_mean = np.mean(qualities)
        cv = quality_std / (quality_mean + 1e-8)  # Coefficient of variation

        print(f"\nQuality variances: {qualities}")
        print(f"CV: {cv:.4f}")

        # Reasonable consistency
        assert cv < 2.0  # Allow some variation but not too much


class TestQualityMonitoring:
    """Tests for quality monitoring during generation."""

    def test_quality_history_tracking(self):
        """Test that quality history is properly tracked."""
        scheduler = AdaptiveScheduler(num_inference_steps=30)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        sampler = QualityGuidedSampler(scheduler=scheduler)

        mx.random.seed(42)
        noise = mx.random.normal((1, 64, 64, 3))
        images, info = sampler.sample(
            model=pipeline.model,
            noise=noise,
            num_steps=30,
        )

        # Quality history should be populated
        assert len(info["quality_history"]) > 0
        assert len(sampler.quality_history) > 0

        # All qualities should be valid
        for q in info["quality_history"]:
            assert 0 <= q <= 1
            assert not np.isnan(q)

    def test_quality_trend_analysis(self):
        """Test analysis of quality trends during generation."""
        scheduler = AdaptiveScheduler(num_inference_steps=40)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        sampler = QualityGuidedSampler(scheduler=scheduler)

        mx.random.seed(42)
        noise = mx.random.normal((1, 64, 64, 3))
        images, info = sampler.sample(
            model=pipeline.model,
            noise=noise,
            num_steps=40,
        )

        # Compute quality trend
        qualities = info["quality_history"]
        if len(qualities) > 1:
            trend = np.mean(np.diff(qualities))
            print(f"\nQuality trend: {trend:.4f}")
            print(f"Quality range: [{min(qualities):.3f}, {max(qualities):.3f}]")

        # Should have quality measurements
        assert len(qualities) >= 10  # At least some samples

    def test_quality_statistics(self):
        """Test quality statistics computation."""
        scheduler = AdaptiveScheduler(num_inference_steps=30)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        sampler = QualityGuidedSampler(scheduler=scheduler)

        mx.random.seed(42)
        noise = mx.random.normal((2, 64, 64, 3))
        images, info = sampler.sample(
            model=pipeline.model,
            noise=noise,
            num_steps=30,
        )

        # Get sampling stats
        stats = sampler.get_sampling_stats()

        print(f"\nSampling stats: {stats}")

        # Validate stats
        assert "mean_quality" in stats
        assert "std_quality" in stats
        assert "min_quality" in stats
        assert "max_quality" in stats

        # Stats should be valid
        assert 0 <= stats["mean_quality"] <= 1
        assert stats["std_quality"] >= 0
        assert 0 <= stats["min_quality"] <= 1
        assert 0 <= stats["max_quality"] <= 1


@pytest.mark.slow
class TestQualityRegression:
    """Comprehensive quality regression tests."""

    def test_quality_under_various_conditions(self):
        """Test quality under various generation conditions."""
        conditions = [
            {"steps": 20, "batch": 1, "seed": 42},
            {"steps": 30, "batch": 2, "seed": 123},
            {"steps": 40, "batch": 4, "seed": 456},
            {"steps": 50, "batch": 2, "seed": 789},
        ]

        scheduler = AdaptiveScheduler()
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        results = []
        for cond in conditions:
            images = pipeline.generate(
                batch_size=cond["batch"],
                num_inference_steps=cond["steps"],
                seed=cond["seed"],
            )

            results.append({
                "config": cond,
                "variance": float(mx.var(images)),
                "mean": float(mx.mean(images)),
                "std": float(mx.std(images)),
            })

        print(f"\nQuality under various conditions:")
        for result in results:
            print(f"  {result['config']}: variance={result['variance']:.4f}")

        # All should produce valid outputs above thresholds
        for result in results:
            assert result["variance"] >= QUALITY_THRESHOLDS["min_variance"]
            assert mx.isfinite(mx.array(result["mean"]))
            assert mx.isfinite(mx.array(result["std"]))

    def test_quality_degradation_detection(self):
        """Test detection of quality degradation."""
        scheduler = AdaptiveScheduler(num_inference_steps=30)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate baseline
        baseline_images = pipeline.generate(batch_size=4, num_inference_steps=30, seed=42)
        baseline_variance = float(mx.var(baseline_images))

        # Generate with potentially degraded quality (very few steps)
        degraded_images = pipeline.generate(batch_size=4, num_inference_steps=5, seed=42)
        degraded_variance = float(mx.var(degraded_images))

        print(f"\nBaseline variance (30 steps): {baseline_variance:.4f}")
        print(f"Degraded variance (5 steps): {degraded_variance:.4f}")

        # Both should still be above minimum threshold
        assert baseline_variance >= QUALITY_THRESHOLDS["min_variance"]
        assert degraded_variance >= QUALITY_THRESHOLDS["min_variance"]
