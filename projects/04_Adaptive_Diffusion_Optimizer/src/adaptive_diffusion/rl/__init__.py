"""
Reinforcement Learning Module for Hyperparameter Optimization

This module provides RL-based hyperparameter optimization for diffusion models.
Includes environment, reward functions, and RL agents for adaptive tuning.
"""

from adaptive_diffusion.rl.environment import (
    DiffusionHyperparameterEnv,
    create_diffusion_env,
)
from adaptive_diffusion.rl.ppo_agent import (
    HyperparameterTuningAgent,
    create_ppo_agent,
)
from adaptive_diffusion.rl.reward import (
    AdaptiveReward,
    MultiObjectiveReward,
    QualitySpeedReward,
    RewardFunction,
    create_reward_function,
)

__all__ = [
    "DiffusionHyperparameterEnv",
    "create_diffusion_env",
    "HyperparameterTuningAgent",
    "create_ppo_agent",
    "QualitySpeedReward",
    "MultiObjectiveReward",
    "AdaptiveReward",
    "RewardFunction",
    "create_reward_function",
]
