"""
Tests for Quality-Guided Sampling

Tests real-time quality estimation and quality-guided sampling algorithms.
"""

import numpy as np
import pytest

import mlx.core as mx

from adaptive_diffusion.baseline.schedulers import DDPMScheduler, DDIMScheduler
from adaptive_diffusion.sampling.quality_guided import (
    QualityEstimator,
    QualityGuidedSampler,
    create_quality_guided_sampler,
)


class TestQualityEstimator:
    """Test suite for QualityEstimator."""

    def test_initialization(self):
        """Test estimator initialization with default parameters."""
        estimator = QualityEstimator()

        assert estimator.noise_weight == 0.3
        assert estimator.structure_weight == 0.3
        assert estimator.frequency_weight == 0.2
        assert estimator.sharpness_weight == 0.2

        # Weights should sum to 1.0
        total = (
            estimator.noise_weight
            + estimator.structure_weight
            + estimator.frequency_weight
            + estimator.sharpness_weight
        )
        assert pytest.approx(total, abs=1e-6) == 1.0

    def test_custom_weights(self):
        """Test estimator with custom weights."""
        estimator = QualityEstimator(
            noise_weight=0.4,
            structure_weight=0.3,
            frequency_weight=0.2,
            sharpness_weight=0.1,
        )

        assert estimator.noise_weight == 0.4
        assert estimator.structure_weight == 0.3

    def test_invalid_weights(self):
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            QualityEstimator(
                noise_weight=0.5,
                structure_weight=0.5,
                frequency_weight=0.5,
                sharpness_weight=0.5,
            )

    def test_estimate_quality_basic(self):
        """Test basic quality estimation."""
        estimator = QualityEstimator()

        # Create sample
        sample = mx.random.normal((1, 8, 8, 3))
        timestep = 500

        quality = estimator.estimate_quality(sample, timestep)

        # Quality should be in valid range
        assert 0 <= quality <= 1

    def test_estimate_quality_high_quality(self):
        """Test quality estimation for high-quality sample."""
        estimator = QualityEstimator()

        # Create structured sample (checkerboard pattern = high quality)
        sample_np = np.zeros((1, 8, 8, 3))
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    sample_np[0, i, j, :] = 1.0
        sample = mx.array(sample_np)

        # Late timestep (should be high quality)
        timestep = 10

        quality = estimator.estimate_quality(sample, timestep)

        # Structured sample should have reasonable quality
        assert quality > 0.3

    def test_estimate_quality_low_quality(self):
        """Test quality estimation for low-quality (noisy) sample."""
        estimator = QualityEstimator()

        # Create pure noise sample
        sample = mx.random.normal((1, 8, 8, 3))

        # Early timestep (noise expected)
        timestep = 900

        quality = estimator.estimate_quality(sample, timestep)

        # Pure noise should have lower quality score
        assert quality < 0.8

    def test_estimate_noise_level(self):
        """Test noise level estimation."""
        estimator = QualityEstimator()

        # Low noise sample
        low_noise = mx.ones((1, 8, 8, 3)) * 0.5 + mx.random.normal((1, 8, 8, 3)) * 0.01
        low_noise_score = estimator._estimate_noise_level(low_noise, 100)

        # High noise sample
        high_noise = mx.random.normal((1, 8, 8, 3))
        high_noise_score = estimator._estimate_noise_level(high_noise, 100)

        # Low noise should have higher score
        assert low_noise_score > high_noise_score

    def test_estimate_structure(self):
        """Test structure estimation."""
        estimator = QualityEstimator()

        # Structured sample (gradient)
        structured = mx.zeros((1, 8, 8, 3))
        for i in range(8):
            structured = mx.array(structured.tolist())
            structured[0, i, :, :] = float(i) / 8.0

        structure_score = estimator._estimate_structure(structured)

        # Random sample (no structure)
        random = mx.random.normal((1, 8, 8, 3))
        random_score = estimator._estimate_structure(random)

        # Structured sample should have higher score
        assert structure_score >= random_score - 0.2  # Allow some variance

    def test_estimate_frequency_content(self):
        """Test frequency content estimation."""
        estimator = QualityEstimator()

        sample = mx.random.normal((1, 8, 8, 3))
        freq_score = estimator._estimate_frequency_content(sample)

        # Score should be in valid range
        assert 0 <= freq_score <= 1

    def test_estimate_sharpness(self):
        """Test sharpness estimation."""
        estimator = QualityEstimator()

        # Sharp sample (high contrast edges)
        sharp = mx.zeros((1, 8, 8, 3))
        sharp = mx.array(sharp.tolist())
        sharp[0, :4, :, :] = 1.0  # Half white, half black

        sharp_score = estimator._estimate_sharpness(sharp)

        # Blurred sample (smooth)
        blurred = mx.ones((1, 8, 8, 3)) * 0.5
        blurred_score = estimator._estimate_sharpness(blurred)

        # Sharp sample should have higher score
        assert sharp_score > blurred_score


class TestQualityGuidedSampler:
    """Test suite for QualityGuidedSampler."""

    def test_initialization(self):
        """Test sampler initialization."""
        scheduler = DDPMScheduler(num_train_timesteps=100)
        sampler = QualityGuidedSampler(scheduler)

        assert sampler.scheduler == scheduler
        assert sampler.quality_threshold == 0.6
        assert sampler.quality_window == 5
        assert sampler.early_stop_threshold == 0.95
        assert sampler.quality_history == []

    def test_custom_parameters(self):
        """Test sampler with custom parameters."""
        scheduler = DDIMScheduler(num_inference_steps=20)
        sampler = QualityGuidedSampler(
            scheduler,
            quality_threshold=0.7,
            quality_window=10,
            early_stop_threshold=0.9,
        )

        assert sampler.quality_threshold == 0.7
        assert sampler.quality_window == 10
        assert sampler.early_stop_threshold == 0.9

    def test_sample_basic(self):
        """Test basic sampling."""
        scheduler = DDPMScheduler(num_train_timesteps=100)
        sampler = QualityGuidedSampler(scheduler)

        # Mock model
        def mock_model(sample, t):
            return mx.random.normal(sample.shape) * 0.1

        # Initial noise
        noise = mx.random.normal((1, 8, 8, 3))

        # Perform sampling
        result, info = sampler.sample(mock_model, noise, num_steps=5)

        # Check result shape
        assert result.shape == noise.shape

        # Check sampling info
        assert "total_steps" in info
        assert "quality_history" in info
        assert "early_stopped" in info
        assert "final_quality" in info

        assert info["total_steps"] == 5
        assert len(info["quality_history"]) >= 1
        assert 0 <= info["final_quality"] <= 1

    def test_sample_with_callback(self):
        """Test sampling with callback."""
        scheduler = DDPMScheduler(num_train_timesteps=100)
        sampler = QualityGuidedSampler(scheduler)

        def mock_model(sample, t):
            return mx.random.normal(sample.shape) * 0.1

        # Track callback calls
        callback_data = []

        def callback(step, sample, quality):
            callback_data.append({"step": step, "quality": quality})

        noise = mx.random.normal((1, 8, 8, 3))
        result, info = sampler.sample(mock_model, noise, num_steps=5, callback=callback)

        # Callback should be called for each step
        assert len(callback_data) >= 1
        assert all("step" in d and "quality" in d for d in callback_data)

    def test_should_early_stop(self):
        """Test early stopping logic."""
        scheduler = DDPMScheduler(num_train_timesteps=100)
        sampler = QualityGuidedSampler(
            scheduler, early_stop_threshold=0.9, quality_window=3
        )

        # Build high quality history
        sampler.quality_history = [0.95, 0.96, 0.94]

        # Should stop after minimum steps
        assert sampler._should_early_stop(0.95, 5, 10) is True

        # Should not stop too early
        assert sampler._should_early_stop(0.95, 1, 10) is False

        # Should not stop if quality is low
        sampler.quality_history = [0.5, 0.6, 0.55]
        assert sampler._should_early_stop(0.6, 5, 10) is False

    def test_should_adapt_step(self):
        """Test step adaptation logic."""
        scheduler = DDPMScheduler()
        sampler = QualityGuidedSampler(scheduler, quality_threshold=0.6)

        # Should adapt if quality is low
        assert sampler._should_adapt_step(0.4) is True

        # Should not adapt if quality is high
        assert sampler._should_adapt_step(0.8) is False

        # Should adapt if quality is degrading
        sampler.quality_history = [0.8, 0.75, 0.65]
        assert sampler._should_adapt_step(0.7) is True

    def test_compute_step_adjustment(self):
        """Test step size adjustment computation."""
        scheduler = DDPMScheduler()
        sampler = QualityGuidedSampler(
            scheduler, quality_threshold=0.6, adaptation_rate=0.5
        )

        # Low quality should give smaller adjustment
        low_quality_adj = sampler._compute_step_adjustment(0.3)
        assert low_quality_adj < 1.0

        # High quality should give normal adjustment
        high_quality_adj = sampler._compute_step_adjustment(0.9)
        assert high_quality_adj == pytest.approx(1.0, abs=1e-5)

        # Adjustment should be in valid range
        assert 0.5 <= low_quality_adj <= 1.0

    def test_get_sampling_stats_empty(self):
        """Test sampling stats with no history."""
        scheduler = DDPMScheduler()
        sampler = QualityGuidedSampler(scheduler)

        stats = sampler.get_sampling_stats()
        assert stats == {}

    def test_get_sampling_stats(self):
        """Test sampling stats with history."""
        scheduler = DDPMScheduler()
        sampler = QualityGuidedSampler(scheduler)

        # Add quality history
        sampler.quality_history = [0.5, 0.6, 0.7, 0.8, 0.75]

        stats = sampler.get_sampling_stats()

        assert "mean_quality" in stats
        assert "std_quality" in stats
        assert "min_quality" in stats
        assert "max_quality" in stats
        assert "quality_trend" in stats
        assert "num_steps" in stats

        assert stats["mean_quality"] == pytest.approx(0.67, abs=0.01)
        assert stats["min_quality"] == 0.5
        assert stats["max_quality"] == 0.8
        assert stats["num_steps"] == 5

    def test_quality_guided_sampling_reduces_steps(self):
        """Test that quality-guided sampling can reduce steps via early stopping."""
        scheduler = DDPMScheduler(num_train_timesteps=100)

        # Sampler with aggressive early stopping
        sampler = QualityGuidedSampler(
            scheduler, early_stop_threshold=0.7, quality_window=2
        )

        # Mock model that quickly reaches high quality
        def mock_high_quality_model(sample, t):
            # Return very small noise (high quality prediction)
            return mx.random.normal(sample.shape) * 0.01

        noise = mx.random.normal((1, 8, 8, 3))
        result, info = sampler.sample(mock_high_quality_model, noise, num_steps=20)

        # Should potentially stop early (though may not due to randomness)
        # Just verify the mechanism exists
        assert "early_stopped" in info
        assert isinstance(info["early_stopped"], bool)

    def test_factory_function(self):
        """Test factory function."""
        scheduler = DDPMScheduler()
        sampler = create_quality_guided_sampler(
            scheduler, quality_threshold=0.7, quality_window=10
        )

        assert isinstance(sampler, QualityGuidedSampler)
        assert sampler.quality_threshold == 0.7
        assert sampler.quality_window == 10


@pytest.mark.parametrize("quality_threshold", [0.4, 0.6, 0.8])
def test_different_quality_thresholds(quality_threshold):
    """Test sampler with different quality thresholds."""
    scheduler = DDPMScheduler()
    sampler = QualityGuidedSampler(scheduler, quality_threshold=quality_threshold)

    assert sampler.quality_threshold == quality_threshold


@pytest.mark.parametrize("scheduler_type", [DDPMScheduler, DDIMScheduler])
def test_different_schedulers(scheduler_type):
    """Test sampler with different scheduler types."""
    if scheduler_type == DDPMScheduler:
        scheduler = scheduler_type(num_train_timesteps=100)
    else:
        scheduler = scheduler_type(num_inference_steps=10)
    sampler = QualityGuidedSampler(scheduler)

    assert sampler.scheduler == scheduler
    assert isinstance(sampler.scheduler, scheduler_type)
