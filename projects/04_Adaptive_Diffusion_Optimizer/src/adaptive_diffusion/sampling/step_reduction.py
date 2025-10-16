"""
Step Reduction Algorithm

Implements intelligent step reduction for efficient diffusion sampling.
Dynamically determines optimal number of steps based on content and quality requirements.

Based on research:
- Progressive Distillation: Salimans & Ho (2022)
- DDIM Fast Sampling: Song et al. (2021)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from adaptive_diffusion.baseline.schedulers import BaseScheduler


class StepReductionStrategy:
    """
    Intelligent step reduction for diffusion sampling.

    Analyzes content complexity and quality requirements to determine
    optimal number of sampling steps, enabling 2-3x speedup.
    """

    def __init__(
        self,
        base_steps: int = 50,
        min_steps: int = 10,
        max_steps: int = 100,
        quality_target: float = 0.8,
        complexity_sensitivity: float = 0.5,
    ):
        """
        Initialize step reduction strategy.

        Args:
            base_steps: Baseline number of steps
            min_steps: Minimum number of steps
            max_steps: Maximum number of steps
            quality_target: Target quality (0-1)
            complexity_sensitivity: How much complexity affects step count (0-1)
        """
        self.base_steps = base_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.quality_target = quality_target
        self.complexity_sensitivity = complexity_sensitivity

        # History tracking
        self.complexity_history = []
        self.step_history = []
        self.quality_history = []

    def estimate_optimal_steps(
        self,
        content_complexity: float,
        quality_requirement: float | None = None,
    ) -> int:
        """
        Estimate optimal number of steps based on content and quality requirements.

        Args:
            content_complexity: Estimated complexity (0-1 scale)
            quality_requirement: Optional quality requirement override

        Returns:
            Optimal number of steps
        """
        quality_req = quality_requirement if quality_requirement is not None else self.quality_target

        # Base calculation: complexity and quality both increase step count
        complexity_factor = 1.0 + (content_complexity - 0.5) * self.complexity_sensitivity
        quality_factor = quality_req / self.quality_target

        # Combine factors
        step_multiplier = complexity_factor * quality_factor

        # Calculate optimal steps
        optimal_steps = int(self.base_steps * step_multiplier)

        # Clamp to valid range
        optimal_steps = max(self.min_steps, min(optimal_steps, self.max_steps))

        # Track history
        self.complexity_history.append(content_complexity)
        self.step_history.append(optimal_steps)

        return optimal_steps

    def analyze_content_complexity(self, sample: mx.array) -> float:
        """
        Analyze content complexity to guide step allocation.

        Args:
            sample: Input sample

        Returns:
            Complexity estimate (0-1 scale)
        """
        # Compute variance (high variance = high complexity)
        variance = float(mx.var(sample))

        # Compute gradient magnitude for edge information
        if len(sample.shape) == 4:
            # (B, H, W, C)
            dx = sample[:, 1:, :, :] - sample[:, :-1, :, :]
            dy = sample[:, :, 1:, :] - sample[:, :, :-1, :]

            # Align shapes
            dx_aligned = dx[:, :, :-1, :]
            dy_aligned = dy[:, :-1, :, :]

            gradient_mag = mx.sqrt(dx_aligned**2 + dy_aligned**2)
            edge_density = float(mx.mean(gradient_mag))
        else:
            # Fallback: use standard deviation
            edge_density = float(mx.std(sample))

        # Frequency analysis: compute high/low frequency ratio
        if len(sample.shape) == 4 and sample.shape[1] > 4:
            # High frequency (detailed)
            high_freq = mx.mean(mx.abs(dx_aligned) + mx.abs(dy_aligned))

            # Low frequency (smooth, downsampled)
            low_freq_sample = sample[:, ::2, ::2, :]
            low_freq = mx.mean(mx.abs(low_freq_sample))

            freq_ratio = float(high_freq / (low_freq + 1e-8))
        else:
            freq_ratio = 1.0

        # Combine metrics (normalized to 0-1)
        complexity = np.clip(
            variance * 0.3 + edge_density * 0.4 + np.clip(freq_ratio / 10.0, 0, 1) * 0.3,
            0,
            1,
        )

        return float(complexity)

    def progressive_step_schedule(
        self, initial_steps: int, target_steps: int, num_stages: int = 4
    ) -> list[int]:
        """
        Generate progressive step reduction schedule.

        Gradually reduces steps over multiple stages to maintain quality.

        Args:
            initial_steps: Starting number of steps
            target_steps: Final target steps
            num_stages: Number of reduction stages

        Returns:
            List of step counts for each stage
        """
        if initial_steps <= target_steps:
            return [initial_steps] * num_stages

        # Exponential decay schedule
        stages = []
        for i in range(num_stages):
            progress = i / (num_stages - 1)
            # Exponential interpolation
            steps = int(initial_steps * (target_steps / initial_steps) ** progress)
            stages.append(max(target_steps, steps))

        return stages

    def adaptive_step_allocation(
        self, quality_estimates: list[float], total_budget: int
    ) -> list[int]:
        """
        Allocate steps adaptively based on quality estimates.

        More steps allocated to regions where quality is low.

        Args:
            quality_estimates: List of quality estimates for different regions
            total_budget: Total step budget

        Returns:
            List of step allocations per region
        """
        if not quality_estimates:
            return []

        # Invert quality (low quality = more steps needed)
        step_needs = [1.0 - q for q in quality_estimates]

        # Normalize to sum to 1.0
        total_need = sum(step_needs)
        if total_need == 0:
            # Equal allocation if all quality is perfect
            return [total_budget // len(quality_estimates)] * len(quality_estimates)

        step_weights = [need / total_need for need in step_needs]

        # Allocate steps proportionally
        allocations = [int(weight * total_budget) for weight in step_weights]

        # Ensure at least min_steps per region and sum equals budget
        allocations = [max(self.min_steps, alloc) for alloc in allocations]

        # Adjust to match exact budget
        current_sum = sum(allocations)
        if current_sum != total_budget:
            diff = total_budget - current_sum
            # Distribute difference to highest need regions
            sorted_indices = sorted(
                range(len(step_needs)), key=lambda i: step_needs[i], reverse=True
            )
            for i in range(abs(diff)):
                idx = sorted_indices[i % len(sorted_indices)]
                allocations[idx] += 1 if diff > 0 else -1

        return allocations

    def get_reduction_stats(self) -> dict[str, Any]:
        """
        Get statistics about step reduction performance.

        Returns:
            Dictionary with reduction statistics
        """
        if not self.step_history:
            return {}

        return {
            "mean_steps": float(np.mean(self.step_history)),
            "min_steps": int(np.min(self.step_history)),
            "max_steps": int(np.max(self.step_history)),
            "std_steps": float(np.std(self.step_history)),
            "mean_complexity": (
                float(np.mean(self.complexity_history))
                if self.complexity_history
                else None
            ),
            "total_samples": len(self.step_history),
            "base_steps": self.base_steps,
            "reduction_ratio": float(np.mean(self.step_history) / self.base_steps),
        }

    def reset_history(self):
        """Reset tracking history."""
        self.complexity_history = []
        self.step_history = []
        self.quality_history = []


def create_step_reduction_strategy(**kwargs) -> StepReductionStrategy:
    """
    Factory function to create step reduction strategy.

    Args:
        **kwargs: Arguments for StepReductionStrategy

    Returns:
        Initialized StepReductionStrategy
    """
    return StepReductionStrategy(**kwargs)
