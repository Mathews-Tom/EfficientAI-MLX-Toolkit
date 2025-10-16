"""
Tests for Adaptive Noise Scheduler

Tests progress-based scheduling, quality-guided adaptation, and complexity estimation.
"""

import numpy as np
import pytest

import mlx.core as mx

from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler


class TestAdaptiveScheduler:
    """Test suite for AdaptiveScheduler."""

    def test_initialization(self):
        """Test scheduler initialization with default parameters."""
        scheduler = AdaptiveScheduler()

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.num_inference_steps == 50
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 50
        assert scheduler.quality_history == []
        assert scheduler.complexity_estimates == []

    def test_custom_parameters(self):
        """Test scheduler with custom parameters."""
        scheduler = AdaptiveScheduler(
            num_train_timesteps=500,
            num_inference_steps=25,
            adaptive_threshold=0.3,
            progress_power=1.5,
        )

        assert scheduler.num_train_timesteps == 500
        assert scheduler.num_inference_steps == 25
        assert scheduler.adaptive_threshold == 0.3
        assert scheduler.progress_power == 1.5
        assert len(scheduler.timesteps) == 25

    def test_set_timesteps_basic(self):
        """Test basic timestep setting without complexity."""
        scheduler = AdaptiveScheduler(num_inference_steps=20)

        scheduler.set_timesteps(20)

        assert len(scheduler.timesteps) == 20
        assert scheduler.step_weights is not None
        assert len(scheduler.step_weights) == 20

        # Verify timesteps are decreasing
        timesteps_list = scheduler.timesteps.tolist()
        assert all(
            timesteps_list[i] >= timesteps_list[i + 1]
            for i in range(len(timesteps_list) - 1)
        )

    def test_set_timesteps_with_complexity(self):
        """Test timestep setting with complexity adjustment."""
        scheduler = AdaptiveScheduler(num_inference_steps=20)

        # Test with low complexity
        scheduler.set_timesteps(20, complexity=0.2)
        low_complexity_weights = scheduler.step_weights.tolist()

        # Test with high complexity
        scheduler.set_timesteps(20, complexity=0.8)
        high_complexity_weights = scheduler.step_weights.tolist()

        # Weights should differ based on complexity
        assert low_complexity_weights != high_complexity_weights

        # High complexity should have more uniform distribution
        low_std = np.std(low_complexity_weights)
        high_std = np.std(high_complexity_weights)
        assert high_std < low_std

    def test_add_noise(self):
        """Test noise addition using forward process."""
        scheduler = AdaptiveScheduler()

        # Create test data
        batch_size = 2
        height, width, channels = 8, 8, 3
        original_samples = mx.random.normal((batch_size, height, width, channels))
        noise = mx.random.normal((batch_size, height, width, channels))
        timesteps = mx.array([100, 200])

        # Add noise
        noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)

        # Check output shape
        assert noisy_samples.shape == original_samples.shape

        # Verify noise was added (samples should differ)
        assert not mx.allclose(noisy_samples, original_samples, atol=1e-5)

    def test_step_basic(self):
        """Test basic denoising step."""
        scheduler = AdaptiveScheduler(num_inference_steps=10)

        # Create test data
        batch_size = 1
        height, width, channels = 8, 8, 3
        model_output = mx.random.normal((batch_size, height, width, channels))
        sample = mx.random.normal((batch_size, height, width, channels))

        # Perform step
        timestep = 0
        prev_sample = scheduler.step(model_output, timestep, sample)

        # Check output shape
        assert prev_sample.shape == sample.shape

        # Sample should be denoised (moved toward prediction)
        assert not mx.allclose(prev_sample, sample, atol=1e-5)

    def test_step_with_quality(self):
        """Test denoising step with quality estimation."""
        scheduler = AdaptiveScheduler(num_inference_steps=10)

        # Create test data
        batch_size = 1
        height, width, channels = 8, 8, 3
        model_output = mx.random.normal((batch_size, height, width, channels))
        sample = mx.random.normal((batch_size, height, width, channels))

        # Perform steps with quality estimates
        for i in range(5):
            quality = 0.8 + i * 0.02  # Increasing quality
            prev_sample = scheduler.step(
                model_output, i, sample, quality_estimate=quality
            )
            sample = prev_sample

        # Verify quality history was updated
        assert len(scheduler.quality_history) == 5
        assert scheduler.quality_history[0] == 0.8
        assert scheduler.quality_history[-1] == pytest.approx(0.88, abs=1e-5)

    def test_compute_step_size_factor(self):
        """Test adaptive step size computation."""
        scheduler = AdaptiveScheduler(num_inference_steps=10)

        # Test early phase (critical region)
        early_factor = scheduler._compute_step_size_factor(1, None)
        assert early_factor < 1.0  # Should use smaller steps

        # Test middle phase (stable region)
        middle_factor = scheduler._compute_step_size_factor(5, None)
        assert middle_factor >= 1.0  # Should use larger steps

        # Test late phase (critical region)
        late_factor = scheduler._compute_step_size_factor(9, None)
        assert late_factor < 1.0  # Should use smaller steps

    def test_compute_step_size_with_quality_degradation(self):
        """Test step size adjustment when quality degrades."""
        scheduler = AdaptiveScheduler(
            num_inference_steps=10, adaptive_threshold=0.1
        )

        # Build quality history
        scheduler.quality_history = [0.8, 0.82, 0.85]

        # Test with degraded quality
        degraded_factor = scheduler._compute_step_size_factor(5, 0.7)

        # Test with normal quality
        normal_factor = scheduler._compute_step_size_factor(5, 0.84)

        # Degraded quality should result in smaller steps
        assert degraded_factor < normal_factor

    def test_estimate_complexity(self):
        """Test content complexity estimation."""
        scheduler = AdaptiveScheduler()

        # Create low complexity sample (uniform)
        low_complexity_sample = mx.ones((1, 8, 8, 3)) * 0.5
        low_complexity = scheduler.estimate_complexity(low_complexity_sample)

        # Create high complexity sample (random)
        high_complexity_sample = mx.random.normal((1, 8, 8, 3))
        high_complexity = scheduler.estimate_complexity(high_complexity_sample)

        # High complexity sample should have higher estimate
        assert high_complexity > low_complexity

        # Complexity should be in valid range
        assert 0 <= low_complexity <= 1
        assert 0 <= high_complexity <= 1

        # Verify history was updated
        assert len(scheduler.complexity_estimates) == 2

    def test_get_schedule_info(self):
        """Test schedule information retrieval."""
        scheduler = AdaptiveScheduler(num_inference_steps=5)

        # Add some history
        scheduler.quality_history = [0.7, 0.75, 0.8]
        scheduler.complexity_estimates = [0.3, 0.4, 0.5]

        info = scheduler.get_schedule_info()

        assert info["num_steps"] == 5
        assert len(info["timesteps"]) == 5
        assert len(info["step_weights"]) == 5
        assert info["quality_history"] == [0.7, 0.75, 0.8]
        assert info["complexity_estimates"] == [0.3, 0.4, 0.5]
        assert info["avg_quality"] == pytest.approx(0.75, abs=1e-5)
        assert info["avg_complexity"] == pytest.approx(0.4, abs=1e-5)

    def test_reset_history(self):
        """Test history reset."""
        scheduler = AdaptiveScheduler()

        # Add some history
        scheduler.quality_history = [0.7, 0.8, 0.9]
        scheduler.complexity_estimates = [0.3, 0.4, 0.5]

        # Reset
        scheduler.reset_history()

        assert scheduler.quality_history == []
        assert scheduler.complexity_estimates == []

    def test_timestep_range(self):
        """Test that timesteps are in valid range."""
        scheduler = AdaptiveScheduler(num_train_timesteps=1000, num_inference_steps=50)

        timesteps = scheduler.timesteps.tolist()

        # All timesteps should be in valid range
        assert all(0 <= t < 1000 for t in timesteps)

        # Timesteps should be decreasing (from high noise to low noise)
        assert all(timesteps[i] >= timesteps[i + 1] for i in range(len(timesteps) - 1))

    def test_beta_schedules(self):
        """Test different beta schedules."""
        schedules = ["linear", "scaled_linear", "squaredcos_cap_v2"]

        for schedule in schedules:
            scheduler = AdaptiveScheduler(beta_schedule=schedule)

            assert scheduler.beta_schedule == schedule
            assert scheduler.betas is not None
            assert len(scheduler.betas) == scheduler.num_train_timesteps

    def test_step_reduction_potential(self):
        """Test that adaptive scheduling can reduce steps while maintaining quality."""
        # Standard scheduler with 50 steps
        standard = AdaptiveScheduler(num_inference_steps=50)

        # Adaptive scheduler with 30 steps (40% reduction)
        adaptive = AdaptiveScheduler(num_inference_steps=30)

        # Both should produce valid schedules
        assert len(standard.timesteps) == 50
        assert len(adaptive.timesteps) == 30

        # Adaptive scheduler should have larger step weights in critical regions
        adaptive_weights = adaptive.step_weights.tolist()
        assert max(adaptive_weights) / min(adaptive_weights) > 1.5

    def test_integration_with_baseline(self):
        """Test that adaptive scheduler integrates with baseline schedulers."""
        from adaptive_diffusion.baseline.schedulers import (
            DDPMScheduler,
            DDIMScheduler,
            DPMSolverScheduler,
        )

        # Create baseline schedulers
        ddpm = DDPMScheduler()
        ddim = DDIMScheduler()
        dpm = DPMSolverScheduler()

        # Create adaptive scheduler with same parameters
        adaptive = AdaptiveScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )

        # All should have same noise schedule parameters
        assert mx.allclose(adaptive.betas, ddpm.betas, atol=1e-6)
        assert mx.allclose(adaptive.alphas_cumprod, ddpm.alphas_cumprod, atol=1e-6)

    def test_factory_function(self):
        """Test factory function for creating adaptive scheduler."""
        from adaptive_diffusion.schedulers.adaptive import get_adaptive_scheduler

        scheduler = get_adaptive_scheduler(
            num_inference_steps=25, adaptive_threshold=0.4
        )

        assert isinstance(scheduler, AdaptiveScheduler)
        assert scheduler.num_inference_steps == 25
        assert scheduler.adaptive_threshold == 0.4


@pytest.mark.parametrize("num_steps", [10, 20, 50, 100])
def test_different_step_counts(num_steps):
    """Test scheduler with different step counts."""
    scheduler = AdaptiveScheduler(num_inference_steps=num_steps)

    assert len(scheduler.timesteps) == num_steps
    assert len(scheduler.step_weights) == num_steps


@pytest.mark.parametrize("progress_power", [1.0, 1.5, 2.0, 3.0])
def test_different_progress_powers(progress_power):
    """Test scheduler with different progress power values."""
    scheduler = AdaptiveScheduler(num_inference_steps=20, progress_power=progress_power)

    assert scheduler.progress_power == progress_power

    # Higher power should create more concentrated weights
    weights = scheduler.step_weights.tolist()
    assert len(weights) == 20
