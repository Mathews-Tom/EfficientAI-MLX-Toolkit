"""
RL Environment for Diffusion Hyperparameter Optimization

Provides a Gym-compatible environment for training RL agents to optimize
diffusion model hyperparameters (scheduler settings, sampling steps, etc.).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mlx.core as mx
import numpy as np
from gymnasium import spaces

from adaptive_diffusion.rl.reward import QualitySpeedReward, RewardFunction


class DiffusionHyperparameterEnv(gym.Env):
    """
    Gym environment for diffusion hyperparameter optimization.

    State space: [
        num_steps (normalized),
        adaptive_threshold,
        progress_power,
        quality_estimate,
        complexity_estimate,
        current_timestep_ratio
    ]

    Action space: [
        num_steps_adjustment (-1 to 1),
        adaptive_threshold_adjustment (-1 to 1),
        progress_power_adjustment (-1 to 1)
    ]

    The agent learns to adjust hyperparameters to maximize quality while
    minimizing sampling steps.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_function: RewardFunction | None = None,
        min_steps: int = 10,
        max_steps: int = 100,
        max_episode_length: int = 50,
        initial_quality: float = 0.5,
        initial_complexity: float = 0.5,
        quality_noise: float = 0.05,
        complexity_noise: float = 0.05,
    ):
        """
        Initialize environment.

        Args:
            reward_function: Custom reward function (uses QualitySpeedReward if None)
            min_steps: Minimum number of sampling steps
            max_steps: Maximum number of sampling steps
            max_episode_length: Maximum steps per episode
            initial_quality: Starting quality estimate
            initial_complexity: Starting complexity estimate
            quality_noise: Noise level for quality simulation
            complexity_noise: Noise level for complexity simulation
        """
        super().__init__()

        self.reward_function = reward_function or QualitySpeedReward()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.max_episode_length = max_episode_length
        self.initial_quality = initial_quality
        self.initial_complexity = initial_complexity
        self.quality_noise = quality_noise
        self.complexity_noise = complexity_noise

        # Define observation space
        # [num_steps, adaptive_threshold, progress_power, quality, complexity, timestep_ratio]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Define action space
        # [num_steps_adj, adaptive_threshold_adj, progress_power_adj]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode state
        self.current_step = 0
        self.current_num_steps = (min_steps + max_steps) // 2
        self.current_adaptive_threshold = 0.5
        self.current_progress_power = 2.0
        self.current_quality = initial_quality
        self.current_complexity = initial_complexity
        self.episode_history = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = 0
        self.current_num_steps = (self.min_steps + self.max_steps) // 2
        self.current_adaptive_threshold = 0.5
        self.current_progress_power = 2.0

        # Add some randomness to initial state
        if self.np_random is not None:
            self.current_quality = self.initial_quality + self.np_random.normal(
                0, self.quality_noise
            )
            self.current_complexity = self.initial_complexity + self.np_random.normal(
                0, self.complexity_noise
            )
        else:
            self.current_quality = self.initial_quality
            self.current_complexity = self.initial_complexity

        self.current_quality = np.clip(self.current_quality, 0, 1)
        self.current_complexity = np.clip(self.current_complexity, 0, 1)

        self.episode_history = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Take environment step with given action.

        Args:
            action: Action array [num_steps_adj, threshold_adj, power_adj]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Apply action to adjust hyperparameters
        num_steps_adj, threshold_adj, power_adj = action

        # Update hyperparameters with clipping
        steps_change = int(num_steps_adj * 10)  # Scale to reasonable range
        self.current_num_steps = np.clip(
            self.current_num_steps + steps_change, self.min_steps, self.max_steps
        )

        self.current_adaptive_threshold = np.clip(
            self.current_adaptive_threshold + threshold_adj * 0.1, 0.0, 1.0
        )

        self.current_progress_power = np.clip(
            self.current_progress_power + power_adj * 0.5, 0.5, 5.0
        )

        # Simulate diffusion generation with current hyperparameters
        quality, speed = self._simulate_diffusion()

        # Compute reward
        reward = self.reward_function.compute_reward(quality, speed)

        # Update state
        self.current_step += 1
        timestep_ratio = self.current_step / self.max_episode_length

        # Record history
        self.episode_history.append(
            {
                "step": self.current_step,
                "num_steps": self.current_num_steps,
                "adaptive_threshold": self.current_adaptive_threshold,
                "progress_power": self.current_progress_power,
                "quality": quality,
                "speed": speed,
                "reward": reward,
            }
        )

        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_episode_length

        observation = self._get_observation()
        info = self._get_info()
        info["quality"] = quality
        info["speed"] = speed

        return observation, float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Observation array
        """
        # Normalize num_steps to [0, 1]
        normalized_steps = (self.current_num_steps - self.min_steps) / (
            self.max_steps - self.min_steps
        )

        timestep_ratio = self.current_step / self.max_episode_length

        return np.array(
            [
                normalized_steps,
                self.current_adaptive_threshold,
                self.current_progress_power / 5.0,  # Normalize to [0, 1]
                self.current_quality,
                self.current_complexity,
                timestep_ratio,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        """
        Get environment info.

        Returns:
            Info dictionary
        """
        return {
            "step": self.current_step,
            "num_steps": self.current_num_steps,
            "adaptive_threshold": self.current_adaptive_threshold,
            "progress_power": self.current_progress_power,
            "episode_history": self.episode_history,
        }

    def _simulate_diffusion(self) -> tuple[float, float]:
        """
        Simulate diffusion generation with current hyperparameters.

        This is a simplified simulation for training. In practice, this would
        run actual diffusion sampling.

        Returns:
            Tuple of (quality, speed) metrics
        """
        # Simulate quality based on hyperparameters
        # More steps generally improve quality, but with diminishing returns
        step_quality_factor = np.log1p(self.current_num_steps) / np.log1p(
            self.max_steps
        )

        # Adaptive threshold affects quality-speed trade-off
        threshold_factor = 1.0 - abs(self.current_adaptive_threshold - 0.5) * 0.2

        # Progress power affects convergence
        power_factor = 1.0 - abs(self.current_progress_power - 2.0) * 0.1

        # Base quality from current state
        base_quality = (
            step_quality_factor * 0.5
            + threshold_factor * 0.3
            + power_factor * 0.2
            + self.current_quality * 0.3
        )

        # Add complexity influence
        complexity_penalty = self.current_complexity * 0.1
        quality = base_quality - complexity_penalty

        # Add noise
        if self.np_random is not None:
            quality += self.np_random.normal(0, self.quality_noise)

        quality = float(np.clip(quality, 0, 1))

        # Simulate speed (inverse of steps, normalized)
        speed = 1.0 - (self.current_num_steps - self.min_steps) / (
            self.max_steps - self.min_steps
        )
        speed = float(np.clip(speed, 0, 1))

        # Update current quality (with momentum)
        self.current_quality = 0.7 * self.current_quality + 0.3 * quality

        return quality, speed

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Clean up environment resources."""
        pass


def create_diffusion_env(
    reward_function: RewardFunction | None = None, **kwargs
) -> DiffusionHyperparameterEnv:
    """
    Factory function to create diffusion hyperparameter environment.

    Args:
        reward_function: Custom reward function
        **kwargs: Additional environment parameters

    Returns:
        Initialized environment
    """
    return DiffusionHyperparameterEnv(reward_function=reward_function, **kwargs)
