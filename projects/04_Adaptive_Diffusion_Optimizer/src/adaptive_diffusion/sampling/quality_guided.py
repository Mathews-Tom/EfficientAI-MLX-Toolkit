"""
Quality-Guided Sampling Algorithm

Implements real-time quality estimation and quality-guided sampling for diffusion models.
Dynamically adjusts sampling strategy based on estimated quality metrics.

Based on research:
- Quality-Diversity Sampling: Dhariwal & Nichol (2021)
- Progressive Quality Estimation: Karras et al. (2022)
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import numpy as np

from adaptive_diffusion.baseline.schedulers import BaseScheduler


class QualityEstimator:
    """
    Real-time quality estimator for diffusion samples.

    Estimates quality using multiple metrics:
    - Noise level estimation
    - Structural consistency
    - Frequency domain analysis
    - Perceptual sharpness
    """

    def __init__(
        self,
        noise_weight: float = 0.3,
        structure_weight: float = 0.3,
        frequency_weight: float = 0.2,
        sharpness_weight: float = 0.2,
    ):
        """
        Initialize quality estimator.

        Args:
            noise_weight: Weight for noise level metric
            structure_weight: Weight for structural consistency
            frequency_weight: Weight for frequency domain metric
            sharpness_weight: Weight for sharpness metric
        """
        self.noise_weight = noise_weight
        self.structure_weight = structure_weight
        self.frequency_weight = frequency_weight
        self.sharpness_weight = sharpness_weight

        # Validate weights sum to 1.0
        total_weight = (
            noise_weight + structure_weight + frequency_weight + sharpness_weight
        )
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def estimate_quality(self, sample: mx.array, timestep: int) -> float:
        """
        Estimate quality of current sample.

        Args:
            sample: Current sample
            timestep: Current timestep

        Returns:
            Quality estimate (0-1 scale, higher is better)
        """
        # Compute individual metrics
        noise_score = self._estimate_noise_level(sample, timestep)
        structure_score = self._estimate_structure(sample)
        frequency_score = self._estimate_frequency_content(sample)
        sharpness_score = self._estimate_sharpness(sample)

        # Weighted combination
        quality = (
            self.noise_weight * noise_score
            + self.structure_weight * structure_score
            + self.frequency_weight * frequency_score
            + self.sharpness_weight * sharpness_score
        )

        return float(np.clip(quality, 0, 1))

    def _estimate_noise_level(self, sample: mx.array, timestep: int) -> float:
        """
        Estimate noise level in sample.

        Lower noise = higher quality at later timesteps.

        Args:
            sample: Current sample
            timestep: Current timestep

        Returns:
            Noise quality score (0-1)
        """
        # Compute local variance as proxy for noise
        if len(sample.shape) == 4:
            # (B, H, W, C) - compute local variance
            padded = mx.pad(sample, [(0, 0), (1, 1), (1, 1), (0, 0)], mode="edge")
            local_mean = (
                padded[:, :-2, 1:-1, :]
                + padded[:, 2:, 1:-1, :]
                + padded[:, 1:-1, :-2, :]
                + padded[:, 1:-1, 2:, :]
            ) / 4.0

            local_var = mx.mean((sample - local_mean) ** 2)
            noise_estimate = float(mx.sqrt(local_var))
        else:
            # Fallback: use standard deviation
            noise_estimate = float(mx.std(sample))

        # Normalize: lower noise = higher score
        # Expected noise decreases with timestep
        expected_noise = 1.0 - (timestep / 1000.0)
        noise_score = 1.0 - min(noise_estimate / (expected_noise + 0.1), 1.0)

        return noise_score

    def _estimate_structure(self, sample: mx.array) -> float:
        """
        Estimate structural consistency.

        Measures presence of coherent structures vs random noise.

        Args:
            sample: Current sample

        Returns:
            Structure quality score (0-1)
        """
        if len(sample.shape) != 4:
            return 0.5  # Neutral score for non-image data

        # Compute gradients
        dx = sample[:, 1:, :, :] - sample[:, :-1, :, :]
        dy = sample[:, :, 1:, :] - sample[:, :, :-1, :]

        # Align shapes
        dx_aligned = dx[:, :, :-1, :]
        dy_aligned = dy[:, :-1, :, :]

        # Compute gradient magnitude
        gradient_mag = mx.sqrt(dx_aligned**2 + dy_aligned**2)

        # Compute gradient direction consistency
        # Structured content has consistent gradient directions
        dx_norm = dx_aligned / (gradient_mag + 1e-8)
        dy_norm = dy_aligned / (gradient_mag + 1e-8)

        # Measure local gradient consistency
        dx_consistency = mx.std(dx_norm)
        dy_consistency = mx.std(dy_norm)

        # Higher consistency = more structure
        structure_score = 1.0 - float((dx_consistency + dy_consistency) / 2.0)

        return float(np.clip(structure_score, 0, 1))

    def _estimate_frequency_content(self, sample: mx.array) -> float:
        """
        Estimate frequency domain content.

        Natural images have characteristic frequency distributions.

        Args:
            sample: Current sample

        Returns:
            Frequency quality score (0-1)
        """
        if len(sample.shape) != 4:
            return 0.5

        # Use gradient magnitudes as proxy for frequency content
        dx = sample[:, 1:, :, :] - sample[:, :-1, :, :]
        dy = sample[:, :, 1:, :] - sample[:, :, :-1, :]

        dx_aligned = dx[:, :, :-1, :]
        dy_aligned = dy[:, :-1, :, :]

        # High frequency content (edges)
        high_freq = mx.mean(mx.abs(dx_aligned) + mx.abs(dy_aligned))

        # Low frequency content (smooth regions)
        # Use downsampled version
        if sample.shape[1] > 4 and sample.shape[2] > 4:
            low_freq_sample = sample[:, ::2, ::2, :]
            low_freq = mx.mean(mx.abs(low_freq_sample))
        else:
            low_freq = mx.mean(mx.abs(sample))

        # Balance between high and low frequency (natural images have both)
        freq_balance = float(
            1.0 - abs(high_freq - low_freq) / (high_freq + low_freq + 1e-8)
        )

        return float(np.clip(freq_balance, 0, 1))

    def _estimate_sharpness(self, sample: mx.array) -> float:
        """
        Estimate image sharpness.

        Sharper images indicate better quality in later stages.

        Args:
            sample: Current sample

        Returns:
            Sharpness score (0-1)
        """
        if len(sample.shape) != 4:
            return 0.5

        # Compute Laplacian (measure of sharpness)
        # Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        padded = mx.pad(sample, [(0, 0), (1, 1), (1, 1), (0, 0)], mode="edge")

        laplacian = (
            padded[:, :-2, 1:-1, :]
            + padded[:, 2:, 1:-1, :]
            + padded[:, 1:-1, :-2, :]
            + padded[:, 1:-1, 2:, :]
            - 4 * sample
        )

        # Variance of Laplacian as sharpness measure
        sharpness = float(mx.var(laplacian))

        # Normalize to 0-1 range (empirically determined scale)
        sharpness_score = min(sharpness / 0.1, 1.0)

        return sharpness_score


class QualityGuidedSampler:
    """
    Quality-guided sampling algorithm for diffusion models.

    Dynamically adjusts sampling strategy based on real-time quality estimation.
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        quality_threshold: float = 0.6,
        quality_window: int = 5,
        early_stop_threshold: float = 0.95,
        adaptation_rate: float = 0.3,
    ):
        """
        Initialize quality-guided sampler.

        Args:
            scheduler: Base scheduler for diffusion sampling
            quality_threshold: Minimum quality threshold for normal sampling
            quality_window: Window size for quality tracking
            early_stop_threshold: Quality threshold for early stopping
            adaptation_rate: Rate of adaptation to quality changes
        """
        self.scheduler = scheduler
        self.quality_threshold = quality_threshold
        self.quality_window = quality_window
        self.early_stop_threshold = early_stop_threshold
        self.adaptation_rate = adaptation_rate

        self.quality_estimator = QualityEstimator()
        self.quality_history = []

    def sample(
        self,
        model: Callable,
        noise: mx.array,
        num_steps: int | None = None,
        guidance_scale: float = 7.5,
        callback: Callable[[int, mx.array, float], None] | None = None,
    ) -> tuple[mx.array, dict[str, Any]]:
        """
        Perform quality-guided sampling.

        Args:
            model: Denoising model
            noise: Initial noise
            num_steps: Number of sampling steps (uses scheduler default if None)
            guidance_scale: Guidance scale for conditional generation
            callback: Optional callback(step, sample, quality)

        Returns:
            Tuple of (final_sample, sampling_info)
        """
        # Reset quality history
        self.quality_history = []

        # Get timesteps from scheduler
        if num_steps is not None and hasattr(self.scheduler, "set_timesteps"):
            self.scheduler.set_timesteps(num_steps)

        # Get timesteps - handle different scheduler types
        if hasattr(self.scheduler, "timesteps"):
            timesteps = self.scheduler.timesteps
        else:
            # Fallback: create linear timesteps
            import numpy as np
            steps = num_steps if num_steps is not None else 50
            timesteps = mx.array(
                np.linspace(
                    self.scheduler.num_train_timesteps - 1, 0, steps
                ).astype(np.int64)
            )

        sample = noise

        # Track sampling info
        sampling_info = {
            "total_steps": len(timesteps),
            "quality_history": [],
            "early_stopped": False,
            "final_quality": 0.0,
        }

        for i, t in enumerate(timesteps):
            # Get model prediction
            model_output = model(sample, int(t))

            # Estimate quality before step
            quality = self.quality_estimator.estimate_quality(sample, int(t))
            self.quality_history.append(quality)
            sampling_info["quality_history"].append(quality)

            # Call callback if provided
            if callback is not None:
                callback(i, sample, quality)

            # Check for early stopping
            if self._should_early_stop(quality, i, len(timesteps)):
                sampling_info["early_stopped"] = True
                sampling_info["stopped_at_step"] = i
                break

            # Adaptive step based on quality
            if self._should_adapt_step(quality):
                # Use smaller step size if quality is low
                step_adjustment = self._compute_step_adjustment(quality)
                adjusted_output = model_output * step_adjustment
            else:
                adjusted_output = model_output

            # Perform denoising step
            sample = self.scheduler.step(adjusted_output, i, sample)

        # Final quality estimate
        final_quality = self.quality_estimator.estimate_quality(sample, 0)
        sampling_info["final_quality"] = final_quality

        return sample, sampling_info

    def _should_early_stop(
        self, quality: float, step: int, total_steps: int
    ) -> bool:
        """
        Determine if sampling should stop early.

        Args:
            quality: Current quality estimate
            step: Current step
            total_steps: Total number of steps

        Returns:
            True if should stop early
        """
        # Don't stop in first 20% of steps
        if step < total_steps * 0.2:
            return False

        # Check if quality is consistently high
        if len(self.quality_history) >= self.quality_window:
            recent_qualities = self.quality_history[-self.quality_window :]
            avg_quality = np.mean(recent_qualities)

            if avg_quality >= self.early_stop_threshold:
                return True

        return False

    def _should_adapt_step(self, quality: float) -> bool:
        """
        Determine if step size should be adapted.

        Args:
            quality: Current quality estimate

        Returns:
            True if should adapt step size
        """
        # Adapt if quality is below threshold
        if quality < self.quality_threshold:
            return True

        # Adapt if quality is degrading
        if len(self.quality_history) >= 3:
            recent_trend = np.diff(self.quality_history[-3:])
            if np.mean(recent_trend) < -0.05:  # Degrading trend
                return True

        return False

    def _compute_step_adjustment(self, quality: float) -> float:
        """
        Compute step size adjustment based on quality.

        Args:
            quality: Current quality estimate

        Returns:
            Step adjustment factor (< 1.0 for smaller steps)
        """
        # Lower quality = smaller steps
        if quality < self.quality_threshold:
            adjustment = 0.5 + 0.5 * (quality / self.quality_threshold)
        else:
            adjustment = 1.0

        # Smooth adjustment
        adjustment = 1.0 - self.adaptation_rate * (1.0 - adjustment)

        return float(np.clip(adjustment, 0.5, 1.0))

    def get_sampling_stats(self) -> dict[str, Any]:
        """
        Get statistics about sampling quality.

        Returns:
            Dictionary with sampling statistics
        """
        if not self.quality_history:
            return {}

        return {
            "mean_quality": float(np.mean(self.quality_history)),
            "std_quality": float(np.std(self.quality_history)),
            "min_quality": float(np.min(self.quality_history)),
            "max_quality": float(np.max(self.quality_history)),
            "quality_trend": float(
                np.mean(np.diff(self.quality_history))
                if len(self.quality_history) > 1
                else 0.0
            ),
            "num_steps": len(self.quality_history),
        }


def create_quality_guided_sampler(
    scheduler: BaseScheduler, **kwargs
) -> QualityGuidedSampler:
    """
    Factory function to create quality-guided sampler.

    Args:
        scheduler: Base scheduler
        **kwargs: Additional arguments for QualityGuidedSampler

    Returns:
        Initialized QualityGuidedSampler
    """
    return QualityGuidedSampler(scheduler, **kwargs)
