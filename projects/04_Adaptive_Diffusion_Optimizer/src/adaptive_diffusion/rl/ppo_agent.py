"""
PPO-Based Hyperparameter Tuning Agent

Implements Proximal Policy Optimization (PPO) agent for learning optimal
diffusion model hyperparameters through reinforcement learning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from adaptive_diffusion.rl.environment import DiffusionHyperparameterEnv

logger = logging.getLogger(__name__)


class HyperparameterTuningAgent:
    """
    PPO-based agent for hyperparameter optimization.

    Uses Proximal Policy Optimization to learn optimal diffusion sampling
    hyperparameters that maximize quality while minimizing sampling steps.
    """

    def __init__(
        self,
        env: gym.Env | None = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_env: bool = True,
        device: str = "auto",
        verbose: int = 1,
    ):
        """
        Initialize PPO agent.

        Args:
            env: Gymnasium environment (creates default if None)
            learning_rate: Learning rate for optimizer
            n_steps: Steps per update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            normalize_env: Whether to normalize observations/rewards
            device: Device for training ("auto", "cpu", "cuda")
            verbose: Verbosity level
        """
        # Create environment if not provided
        if env is None:
            env = DiffusionHyperparameterEnv()

        # Wrap in vectorized environment
        if not isinstance(env, (DummyVecEnv, VecNormalize)):
            env = DummyVecEnv([lambda: env])

        # Normalize if requested
        if normalize_env and not isinstance(env, VecNormalize):
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        self.env = env
        self.normalize_env = normalize_env

        # Create PPO agent
        self.agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            verbose=verbose,
        )

        # Training stats
        self.training_stats = {
            "total_timesteps": 0,
            "episodes": 0,
            "best_reward": float("-inf"),
            "convergence_achieved": False,
        }

    def train(
        self,
        total_timesteps: int,
        callback: BaseCallback | list[BaseCallback] | None = None,
        eval_env: gym.Env | None = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        convergence_threshold: float = 0.01,
        convergence_window: int = 5,
    ) -> dict[str, Any]:
        """
        Train the PPO agent.

        Args:
            total_timesteps: Total training timesteps
            callback: Optional callback(s)
            eval_env: Optional evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            convergence_threshold: Reward std threshold for convergence
            convergence_window: Window for convergence check

        Returns:
            Training statistics dictionary
        """
        callbacks = []

        # Add evaluation callback if eval_env provided
        if eval_env is not None:
            # Wrap eval env
            if not isinstance(eval_env, (DummyVecEnv, VecNormalize)):
                eval_env = DummyVecEnv([lambda: eval_env])

            if self.normalize_env and not isinstance(eval_env, VecNormalize):
                eval_env = VecNormalize(
                    eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0
                )

            eval_callback = EvalCallback(
                eval_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        # Add convergence callback
        convergence_callback = ConvergenceCallback(
            threshold=convergence_threshold, window=convergence_window
        )
        callbacks.append(convergence_callback)

        # Add custom callback if provided
        if callback is not None:
            if isinstance(callback, list):
                callbacks.extend(callback)
            else:
                callbacks.append(callback)

        # Train agent
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        self.agent.learn(total_timesteps=total_timesteps, callback=callbacks)

        # Update stats
        self.training_stats["total_timesteps"] += total_timesteps
        self.training_stats["convergence_achieved"] = (
            convergence_callback.convergence_achieved
        )

        if callbacks and isinstance(callbacks[0], EvalCallback):
            self.training_stats["best_reward"] = callbacks[0].best_mean_reward

        logger.info(f"Training completed. Total timesteps: {self.training_stats['total_timesteps']}")
        logger.info(
            f"Convergence achieved: {self.training_stats['convergence_achieved']}"
        )

        return self.training_stats

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Predict action for given observation.

        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, value_estimate)
        """
        action, state = self.agent.predict(observation, deterministic=deterministic)
        return action, state

    def optimize_hyperparameters(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters using trained agent.

        Args:
            num_episodes: Number of optimization episodes
            deterministic: Whether to use deterministic policy

        Returns:
            Dictionary with optimized hyperparameters and results
        """
        results = {
            "episodes": [],
            "best_quality": 0.0,
            "best_speed": 0.0,
            "best_hyperparameters": {},
            "mean_quality": 0.0,
            "mean_speed": 0.0,
        }

        qualities = []
        speeds = []

        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_quality = []
            episode_speed = []
            final_hyperparams = {}

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)

                # Extract quality and speed from info
                if isinstance(info, list):
                    info = info[0]

                if "quality" in info:
                    episode_quality.append(info["quality"])
                if "speed" in info:
                    episode_speed.append(info["speed"])

                # Store final hyperparameters
                if done:
                    final_hyperparams = {
                        "num_steps": info.get("num_steps", 50),
                        "adaptive_threshold": info.get("adaptive_threshold", 0.5),
                        "progress_power": info.get("progress_power", 2.0),
                    }

            # Calculate episode metrics
            avg_quality = np.mean(episode_quality) if episode_quality else 0.0
            avg_speed = np.mean(episode_speed) if episode_speed else 0.0

            qualities.append(avg_quality)
            speeds.append(avg_speed)

            results["episodes"].append(
                {
                    "episode": episode,
                    "quality": avg_quality,
                    "speed": avg_speed,
                    "hyperparameters": final_hyperparams,
                }
            )

            # Update best if this is better
            if avg_quality > results["best_quality"]:
                results["best_quality"] = avg_quality
                results["best_speed"] = avg_speed
                results["best_hyperparameters"] = final_hyperparams

        results["mean_quality"] = np.mean(qualities)
        results["mean_speed"] = np.mean(speeds)

        logger.info(f"Optimization complete. Best quality: {results['best_quality']:.4f}")
        logger.info(f"Best hyperparameters: {results['best_hyperparameters']}")

        return results

    def save(self, path: str | Path):
        """
        Save agent model.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.agent.save(str(path))

        # Save normalization stats if using VecNormalize
        if isinstance(self.env, VecNormalize):
            stats_path = path.parent / f"{path.stem}_vecnormalize.pkl"
            self.env.save(str(stats_path))

        logger.info(f"Agent saved to {path}")

    def load(self, path: str | Path):
        """
        Load agent model.

        Args:
            path: Path to load model from
        """
        path = Path(path)

        self.agent = PPO.load(str(path), env=self.env)

        # Load normalization stats if they exist
        if isinstance(self.env, VecNormalize):
            stats_path = path.parent / f"{path.stem}_vecnormalize.pkl"
            if stats_path.exists():
                self.env = VecNormalize.load(str(stats_path), self.env)

        logger.info(f"Agent loaded from {path}")

    def get_config(self) -> dict[str, Any]:
        """
        Get agent configuration.

        Returns:
            Configuration dictionary
        """
        return {
            "learning_rate": self.agent.learning_rate,
            "n_steps": self.agent.n_steps,
            "batch_size": self.agent.batch_size,
            "n_epochs": self.agent.n_epochs,
            "gamma": self.agent.gamma,
            "gae_lambda": self.agent.gae_lambda,
            "clip_range": self.agent.clip_range,
            "ent_coef": self.agent.ent_coef,
            "vf_coef": self.agent.vf_coef,
            "max_grad_norm": self.agent.max_grad_norm,
            "normalize_env": self.normalize_env,
        }


class ConvergenceCallback(BaseCallback):
    """
    Callback to monitor training convergence.

    Checks if reward variance falls below threshold over a window.
    """

    def __init__(
        self, threshold: float = 0.01, window: int = 5, verbose: int = 0
    ):
        """
        Initialize convergence callback.

        Args:
            threshold: Variance threshold for convergence
            window: Window size for variance computation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.threshold = threshold
        self.window = window
        self.reward_history = []
        self.convergence_achieved = False

    def _on_step(self) -> bool:
        """
        Called at each step.

        Returns:
            Whether training should continue
        """
        # Get episode reward if available
        if len(self.model.ep_info_buffer) > 0:
            reward = self.model.ep_info_buffer[-1]["r"]
            self.reward_history.append(reward)

            # Keep only recent history
            if len(self.reward_history) > self.window * 2:
                self.reward_history = self.reward_history[-self.window * 2 :]

            # Check convergence
            if len(self.reward_history) >= self.window:
                recent_rewards = self.reward_history[-self.window :]
                reward_std = np.std(recent_rewards)

                if reward_std < self.threshold:
                    self.convergence_achieved = True
                    if self.verbose > 0:
                        logger.info(
                            f"Convergence achieved at step {self.num_timesteps}. "
                            f"Reward std: {reward_std:.6f}"
                        )

        return True  # Continue training


def create_ppo_agent(env: gym.Env | None = None, **kwargs) -> HyperparameterTuningAgent:
    """
    Factory function to create PPO agent.

    Args:
        env: Gymnasium environment
        **kwargs: Additional agent parameters

    Returns:
        Initialized HyperparameterTuningAgent
    """
    return HyperparameterTuningAgent(env=env, **kwargs)
