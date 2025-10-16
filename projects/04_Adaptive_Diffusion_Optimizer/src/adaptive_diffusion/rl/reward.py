"""
Reward Functions for RL-Based Hyperparameter Optimization

Provides various reward formulations for optimizing quality-speed trade-offs
in diffusion model sampling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RewardFunction(ABC):
    """
    Base class for reward functions.

    Reward functions compute scalar rewards from quality and speed metrics
    to guide RL optimization.
    """

    @abstractmethod
    def compute_reward(self, quality: float, speed: float) -> float:
        """
        Compute reward from quality and speed metrics.

        Args:
            quality: Quality metric (0-1 scale, higher is better)
            speed: Speed metric (0-1 scale, higher is faster)

        Returns:
            Scalar reward value
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Get reward function configuration.

        Returns:
            Configuration dictionary
        """
        return {"type": self.__class__.__name__}


class QualitySpeedReward(RewardFunction):
    """
    Quality-speed trade-off reward function.

    Balances quality and speed with configurable weights.
    R = quality_weight * quality + speed_weight * speed
    """

    def __init__(
        self,
        quality_weight: float = 0.7,
        speed_weight: float = 0.3,
        quality_threshold: float = 0.6,
        speed_threshold: float = 0.5,
        penalty_factor: float = 0.5,
    ):
        """
        Initialize reward function.

        Args:
            quality_weight: Weight for quality term
            speed_weight: Weight for speed term
            quality_threshold: Minimum quality threshold (penalty below)
            speed_threshold: Target speed threshold
            penalty_factor: Penalty multiplier for threshold violations
        """
        self.quality_weight = quality_weight
        self.speed_weight = speed_weight
        self.quality_threshold = quality_threshold
        self.speed_threshold = speed_threshold
        self.penalty_factor = penalty_factor

        # Validate weights
        if not np.isclose(quality_weight + speed_weight, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {quality_weight + speed_weight}"
            )

    def compute_reward(self, quality: float, speed: float) -> float:
        """
        Compute weighted quality-speed reward.

        Args:
            quality: Quality metric (0-1)
            speed: Speed metric (0-1)

        Returns:
            Scalar reward
        """
        # Base reward: weighted combination
        reward = self.quality_weight * quality + self.speed_weight * speed

        # Apply penalties for threshold violations
        if quality < self.quality_threshold:
            quality_penalty = (self.quality_threshold - quality) * self.penalty_factor
            reward -= quality_penalty

        # Bonus for exceeding both thresholds
        if quality >= self.quality_threshold and speed >= self.speed_threshold:
            reward += 0.1  # Small bonus for balanced performance

        return float(reward)

    def get_config(self) -> dict[str, Any]:
        """Get configuration."""
        return {
            "type": "QualitySpeedReward",
            "quality_weight": self.quality_weight,
            "speed_weight": self.speed_weight,
            "quality_threshold": self.quality_threshold,
            "speed_threshold": self.speed_threshold,
            "penalty_factor": self.penalty_factor,
        }


class MultiObjectiveReward(RewardFunction):
    """
    Multi-objective reward with configurable objectives.

    Supports:
    - Quality maximization
    - Speed maximization
    - Quality variance minimization (stability)
    - Pareto frontier optimization
    """

    def __init__(
        self,
        quality_weight: float = 0.5,
        speed_weight: float = 0.3,
        stability_weight: float = 0.2,
        pareto_bonus: float = 0.15,
        quality_history_size: int = 10,
    ):
        """
        Initialize multi-objective reward.

        Args:
            quality_weight: Weight for quality objective
            speed_weight: Weight for speed objective
            stability_weight: Weight for stability objective
            pareto_bonus: Bonus for Pareto-optimal solutions
            quality_history_size: Window size for stability computation
        """
        self.quality_weight = quality_weight
        self.speed_weight = speed_weight
        self.stability_weight = stability_weight
        self.pareto_bonus = pareto_bonus
        self.quality_history_size = quality_history_size

        # Validate weights
        total_weight = quality_weight + speed_weight + stability_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Track history for stability
        self.quality_history = []
        self.pareto_front = []  # Track (quality, speed) pairs

    def compute_reward(self, quality: float, speed: float) -> float:
        """
        Compute multi-objective reward.

        Args:
            quality: Quality metric (0-1)
            speed: Speed metric (0-1)

        Returns:
            Scalar reward
        """
        # Update quality history
        self.quality_history.append(quality)
        if len(self.quality_history) > self.quality_history_size:
            self.quality_history.pop(0)

        # Compute stability (inverse of variance)
        if len(self.quality_history) > 1:
            quality_variance = float(np.var(self.quality_history))
            stability = 1.0 - min(quality_variance, 1.0)
        else:
            stability = 0.5  # Neutral stability for first steps

        # Base reward: weighted combination
        reward = (
            self.quality_weight * quality
            + self.speed_weight * speed
            + self.stability_weight * stability
        )

        # Check if current solution is Pareto-optimal
        is_pareto_optimal = self._is_pareto_optimal(quality, speed)
        if is_pareto_optimal:
            reward += self.pareto_bonus

        # Update Pareto front
        self._update_pareto_front(quality, speed)

        return float(reward)

    def _is_pareto_optimal(self, quality: float, speed: float) -> bool:
        """
        Check if current solution is Pareto-optimal.

        Args:
            quality: Quality metric
            speed: Speed metric

        Returns:
            True if solution is not dominated by any in Pareto front
        """
        if not self.pareto_front:
            return True

        # Check if any existing solution dominates this one
        for pq, ps in self.pareto_front:
            if pq >= quality and ps >= speed and (pq > quality or ps > speed):
                return False  # Dominated

        return True

    def _update_pareto_front(self, quality: float, speed: float):
        """
        Update Pareto front with new solution.

        Args:
            quality: Quality metric
            speed: Speed metric
        """
        # Remove dominated solutions
        self.pareto_front = [
            (pq, ps)
            for pq, ps in self.pareto_front
            if not (quality >= pq and speed >= ps and (quality > pq or speed > ps))
        ]

        # Add new solution if not dominated
        if self._is_pareto_optimal(quality, speed):
            self.pareto_front.append((quality, speed))

        # Keep front size manageable
        if len(self.pareto_front) > 50:
            # Keep only top solutions by sum of objectives
            self.pareto_front.sort(key=lambda x: x[0] + x[1], reverse=True)
            self.pareto_front = self.pareto_front[:50]

    def reset_history(self):
        """Reset quality history and Pareto front."""
        self.quality_history = []
        self.pareto_front = []

    def get_config(self) -> dict[str, Any]:
        """Get configuration."""
        return {
            "type": "MultiObjectiveReward",
            "quality_weight": self.quality_weight,
            "speed_weight": self.speed_weight,
            "stability_weight": self.stability_weight,
            "pareto_bonus": self.pareto_bonus,
            "quality_history_size": self.quality_history_size,
        }

    def get_pareto_front(self) -> list[tuple[float, float]]:
        """
        Get current Pareto front.

        Returns:
            List of (quality, speed) tuples
        """
        return self.pareto_front.copy()


class AdaptiveReward(RewardFunction):
    """
    Adaptive reward that adjusts weights based on training progress.

    Starts with emphasis on quality, gradually shifts to speed optimization.
    """

    def __init__(
        self,
        initial_quality_weight: float = 0.9,
        final_quality_weight: float = 0.6,
        transition_steps: int = 1000,
    ):
        """
        Initialize adaptive reward.

        Args:
            initial_quality_weight: Starting quality weight
            final_quality_weight: Ending quality weight
            transition_steps: Steps over which to transition weights
        """
        self.initial_quality_weight = initial_quality_weight
        self.final_quality_weight = final_quality_weight
        self.transition_steps = transition_steps
        self.current_step = 0

    def compute_reward(self, quality: float, speed: float) -> float:
        """
        Compute adaptive reward with dynamic weights.

        Args:
            quality: Quality metric (0-1)
            speed: Speed metric (0-1)

        Returns:
            Scalar reward
        """
        # Compute current weights based on progress
        progress = min(self.current_step / self.transition_steps, 1.0)

        quality_weight = self.initial_quality_weight + progress * (
            self.final_quality_weight - self.initial_quality_weight
        )
        speed_weight = 1.0 - quality_weight

        # Compute reward
        reward = quality_weight * quality + speed_weight * speed

        self.current_step += 1

        return float(reward)

    def reset(self):
        """Reset step counter."""
        self.current_step = 0

    def get_config(self) -> dict[str, Any]:
        """Get configuration."""
        return {
            "type": "AdaptiveReward",
            "initial_quality_weight": self.initial_quality_weight,
            "final_quality_weight": self.final_quality_weight,
            "transition_steps": self.transition_steps,
            "current_step": self.current_step,
        }


def create_reward_function(
    reward_type: str = "quality_speed", **kwargs
) -> RewardFunction:
    """
    Factory function to create reward functions.

    Args:
        reward_type: Type of reward function ("quality_speed", "multi_objective", "adaptive")
        **kwargs: Additional parameters for reward function

    Returns:
        Initialized reward function

    Raises:
        ValueError: If reward_type is unknown
    """
    if reward_type == "quality_speed":
        return QualitySpeedReward(**kwargs)
    elif reward_type == "multi_objective":
        return MultiObjectiveReward(**kwargs)
    elif reward_type == "adaptive":
        return AdaptiveReward(**kwargs)
    else:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Choose from: quality_speed, multi_objective, adaptive"
        )
