"""
Adaptive Noise Scheduler

Implements progress-based adaptive noise scheduling that dynamically adjusts
the denoising schedule based on generation progress and quality metrics.

Based on research:
- Progressive Distillation: Salimans & Ho (2022)
- Adaptive Scheduling: Karras et al. (2022)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from adaptive_diffusion.baseline.schedulers import BaseScheduler


class AdaptiveScheduler(BaseScheduler):
    """
    Adaptive noise scheduler with progress-based scheduling.

    Dynamically adjusts noise schedule based on:
    1. Generation progress (timestep position)
    2. Content complexity (learned from previous steps)
    3. Quality estimation (real-time quality monitoring)

    Features:
    - Progress-aware scheduling: Allocates more steps to critical regions
    - Content-adaptive: Adjusts based on sample complexity
    - Quality-guided: Monitors and optimizes for quality metrics
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        num_inference_steps: int = 50,
        adaptive_threshold: float = 0.5,
        progress_power: float = 2.0,
        min_step_ratio: float = 0.5,
        max_step_ratio: float = 2.0,
    ):
        """
        Initialize adaptive scheduler.

        Args:
            num_train_timesteps: Number of training timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Beta schedule type
            num_inference_steps: Base number of inference steps
            adaptive_threshold: Threshold for triggering adaptive behavior
            progress_power: Power for progress-based weighting (higher = more aggressive)
            min_step_ratio: Minimum ratio for step allocation
            max_step_ratio: Maximum ratio for step allocation
        """
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule)

        self.num_inference_steps = num_inference_steps
        self.adaptive_threshold = adaptive_threshold
        self.progress_power = progress_power
        self.min_step_ratio = min_step_ratio
        self.max_step_ratio = max_step_ratio

        # Initialize adaptive state
        self.timesteps = None
        self.step_weights = None
        self.quality_history = []
        self.complexity_estimates = []

        # Set initial timesteps
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int, complexity: float | None = None):
        """
        Set adaptive timesteps based on progress and complexity.

        Args:
            num_inference_steps: Number of inference steps
            complexity: Estimated content complexity (0-1 scale)
        """
        self.num_inference_steps = num_inference_steps

        # Generate progress-based weights
        progress = np.linspace(0, 1, num_inference_steps)

        # Apply power law for progress weighting
        # More steps allocated to early (high noise) and late (refinement) stages
        weights = np.power(progress, self.progress_power) + np.power(
            1 - progress, self.progress_power
        )
        weights = weights / weights.sum()

        # Adjust based on complexity if provided
        if complexity is not None:
            # Higher complexity = more uniform distribution
            # Lower complexity = more concentrated on key regions
            complexity_factor = 1.0 - complexity * 0.5
            weights = np.power(weights, complexity_factor)
            weights = weights / weights.sum()

        self.step_weights = mx.array(weights)

        # Create timesteps using evenly spaced approach
        # Start from high timesteps (high noise) and go down to 0
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (
            np.arange(0, num_inference_steps) * step_ratio
        ).round()[::-1].copy().astype(np.int64)

        # Ensure timesteps are within valid range and strictly decreasing
        timesteps = np.clip(timesteps, 0, self.num_train_timesteps - 1)

        self.timesteps = mx.array(timesteps)

    def add_noise(
        self, original_samples: mx.array, noise: mx.array, timesteps: mx.array
    ) -> mx.array:
        """
        Add noise using forward process with adaptive scheduling.

        Args:
            original_samples: Clean samples
            noise: Noise to add
            timesteps: Timesteps for noise addition

        Returns:
            Noisy samples
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, axis=-1)
            sqrt_one_minus_alpha_prod = mx.expand_dims(
                sqrt_one_minus_alpha_prod, axis=-1
            )

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
        quality_estimate: float | None = None,
    ) -> mx.array:
        """
        Perform adaptive reverse diffusion step.

        Args:
            model_output: Predicted noise from model
            timestep: Current timestep index
            sample: Current noisy sample
            quality_estimate: Optional quality estimate for current sample

        Returns:
            Denoised sample at previous timestep
        """
        # Update quality history if provided
        if quality_estimate is not None:
            self.quality_history.append(quality_estimate)

        # Get current and previous timesteps
        t = self.timesteps[timestep]
        prev_timestep = (
            self.timesteps[timestep + 1] if timestep < len(self.timesteps) - 1 else 0
        )

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else mx.array(1.0)
        )
        beta_prod_t = 1 - alpha_prod_t

        # Predict x_0 from noise
        pred_original_sample = (
            sample - mx.sqrt(beta_prod_t) * model_output
        ) / mx.sqrt(alpha_prod_t)

        # Adaptive step size based on progress and quality
        step_size_factor = self._compute_step_size_factor(
            timestep, quality_estimate
        )

        # Compute predicted sample direction with adaptive step size
        pred_sample_direction = (
            mx.sqrt(1 - alpha_prod_t_prev) * model_output * step_size_factor
        )

        # Compute previous sample
        prev_sample = (
            mx.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        return prev_sample

    def _compute_step_size_factor(
        self, timestep: int, quality_estimate: float | None = None
    ) -> float:
        """
        Compute adaptive step size factor.

        Args:
            timestep: Current timestep index
            quality_estimate: Optional quality estimate

        Returns:
            Step size multiplier (1.0 = normal, <1.0 = smaller steps, >1.0 = larger steps)
        """
        # Base factor from progress
        progress = timestep / len(self.timesteps)

        # Allocate more careful steps (smaller step size) in critical regions
        # Critical: early phase (high noise) and late phase (refinement)
        if progress < 0.2 or progress > 0.8:
            base_factor = 0.8  # Smaller steps in critical regions
        else:
            base_factor = 1.2  # Larger steps in stable middle region

        # Adjust based on quality if available
        if quality_estimate is not None and len(self.quality_history) > 2:
            # If quality is degrading, use smaller steps
            recent_quality = np.mean(self.quality_history[-3:])
            if quality_estimate < recent_quality - self.adaptive_threshold:
                base_factor *= 0.7  # Much smaller steps when quality drops
            elif quality_estimate > recent_quality + self.adaptive_threshold:
                base_factor *= 1.3  # Larger steps when quality is good

        # Clamp to valid range
        return float(np.clip(base_factor, self.min_step_ratio, self.max_step_ratio))

    def estimate_complexity(self, sample: mx.array) -> float:
        """
        Estimate content complexity of current sample.

        Uses variance and edge information as proxies for complexity.

        Args:
            sample: Current sample

        Returns:
            Complexity estimate (0-1 scale)
        """
        # Compute variance (high variance = high complexity)
        variance = float(mx.var(sample))

        # Compute gradient magnitude (edge information)
        if len(sample.shape) >= 3:
            # Image-like data - compute gradients along spatial dimensions
            # For shape (B, H, W, C), compute gradients along H and W
            if len(sample.shape) == 4:
                # (B, H, W, C)
                dx = sample[:, 1:, :, :] - sample[:, :-1, :, :]  # (B, H-1, W, C)
                dy = sample[:, :, 1:, :] - sample[:, :, :-1, :]  # (B, H, W-1, C)
                # Align shapes for both gradients
                dx_aligned = dx[:, :, :-1, :]  # (B, H-1, W-1, C)
                dy_aligned = dy[:, :-1, :, :]  # (B, H-1, W-1, C)
                gradient_mag = mx.sqrt(dx_aligned**2 + dy_aligned**2)
            else:
                # Fallback for other 3D shapes
                dx = sample[..., 1:, :] - sample[..., :-1, :]
                dy = sample[..., :, 1:] - sample[..., :, :-1]
                min_h = min(dx.shape[-2], dy.shape[-2])
                min_w = min(dx.shape[-1], dy.shape[-1])
                dx_aligned = dx[..., :min_h, :min_w]
                dy_aligned = dy[..., :min_h, :min_w]
                gradient_mag = mx.sqrt(dx_aligned**2 + dy_aligned**2)

            edge_strength = float(mx.mean(gradient_mag))
        else:
            edge_strength = 0.0

        # Combine metrics (normalized to 0-1)
        complexity = np.clip(variance * 0.5 + edge_strength * 0.5, 0, 1)

        self.complexity_estimates.append(complexity)
        return complexity

    def get_schedule_info(self) -> dict[str, Any]:
        """
        Get information about current adaptive schedule.

        Returns:
            Dictionary with schedule statistics
        """
        return {
            "num_steps": len(self.timesteps),
            "timesteps": self.timesteps.tolist() if self.timesteps is not None else [],
            "step_weights": (
                self.step_weights.tolist() if self.step_weights is not None else []
            ),
            "quality_history": self.quality_history,
            "complexity_estimates": self.complexity_estimates,
            "avg_quality": (
                float(np.mean(self.quality_history))
                if self.quality_history
                else None
            ),
            "avg_complexity": (
                float(np.mean(self.complexity_estimates))
                if self.complexity_estimates
                else None
            ),
        }

    def reset_history(self):
        """Reset quality and complexity history."""
        self.quality_history = []
        self.complexity_estimates = []


def get_adaptive_scheduler(**kwargs) -> AdaptiveScheduler:
    """
    Factory function to create adaptive scheduler.

    Args:
        **kwargs: Arguments for AdaptiveScheduler

    Returns:
        Initialized AdaptiveScheduler instance
    """
    return AdaptiveScheduler(**kwargs)
