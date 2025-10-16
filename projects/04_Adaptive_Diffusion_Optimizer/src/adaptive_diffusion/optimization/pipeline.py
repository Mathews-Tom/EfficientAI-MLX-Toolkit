"""
Optimization Pipeline for Adaptive Diffusion

Integrates RL-based hyperparameter tuning, domain adaptation, and adaptive
scheduling into a unified optimization pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from adaptive_diffusion.optimization.domain_adapter import (
    DomainAdapter,
    DomainType,
)
from adaptive_diffusion.rl.environment import DiffusionHyperparameterEnv
from adaptive_diffusion.rl.ppo_agent import HyperparameterTuningAgent
from adaptive_diffusion.rl.reward import RewardFunction, create_reward_function
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler

logger = logging.getLogger(__name__)


class OptimizationPipeline:
    """
    Unified optimization pipeline for diffusion hyperparameter tuning.

    Combines:
    - Domain adaptation for content-specific optimization
    - RL-based hyperparameter tuning
    - Adaptive scheduling
    - Quality-guided sampling
    """

    def __init__(
        self,
        domain_adapter: DomainAdapter | None = None,
        rl_agent: HyperparameterTuningAgent | None = None,
        reward_function: RewardFunction | None = None,
        use_domain_adaptation: bool = True,
        use_rl_optimization: bool = True,
        verbose: int = 1,
    ):
        """
        Initialize optimization pipeline.

        Args:
            domain_adapter: Domain adapter (creates default if None)
            rl_agent: RL agent for hyperparameter tuning (creates default if None)
            reward_function: Custom reward function
            use_domain_adaptation: Whether to use domain adaptation
            use_rl_optimization: Whether to use RL optimization
            verbose: Verbosity level
        """
        self.domain_adapter = domain_adapter or DomainAdapter()
        self.use_domain_adaptation = use_domain_adaptation
        self.use_rl_optimization = use_rl_optimization
        self.verbose = verbose

        # Create reward function if not provided
        if reward_function is None:
            reward_function = create_reward_function("quality_speed")

        # Create RL agent if needed and not provided
        if use_rl_optimization and rl_agent is None:
            env = DiffusionHyperparameterEnv(reward_function=reward_function)
            self.rl_agent = HyperparameterTuningAgent(
                env=env, verbose=max(0, verbose - 1)
            )
        else:
            self.rl_agent = rl_agent

        # Optimization history
        self.optimization_history = []

    def optimize(
        self,
        domain_type: DomainType | None = None,
        prompt: str | None = None,
        sample: mx.array | None = None,
        num_training_steps: int = 1000,
        num_optimization_episodes: int = 10,
    ) -> dict[str, Any]:
        """
        Run full optimization pipeline.

        Args:
            domain_type: Explicit domain type (auto-detected if None)
            prompt: Optional prompt for domain detection
            sample: Optional sample for domain detection
            num_training_steps: RL training steps
            num_optimization_episodes: Optimization episodes

        Returns:
            Optimization results dictionary
        """
        results = {
            "domain_type": None,
            "domain_config": None,
            "rl_optimized_config": None,
            "final_config": None,
            "training_stats": None,
            "optimization_results": None,
        }

        # Step 1: Domain detection and configuration
        if self.use_domain_adaptation:
            domain_config = self.domain_adapter.get_config(
                domain_type=domain_type, sample=sample, prompt=prompt
            )
            results["domain_type"] = domain_config.domain_type
            results["domain_config"] = domain_config.to_dict()

            logger.info(
                f"Using domain-adapted config for {domain_config.domain_type.value}"
            )
        else:
            domain_config = None
            results["domain_type"] = DomainType.GENERAL

        # Step 2: RL-based hyperparameter optimization
        if self.use_rl_optimization and self.rl_agent is not None:
            logger.info(f"Training RL agent for {num_training_steps} steps")

            # Train RL agent
            training_stats = self.rl_agent.train(total_timesteps=num_training_steps)
            results["training_stats"] = training_stats

            # Run optimization
            logger.info(f"Running optimization for {num_optimization_episodes} episodes")
            optimization_results = self.rl_agent.optimize_hyperparameters(
                num_episodes=num_optimization_episodes
            )
            results["optimization_results"] = optimization_results
            results["rl_optimized_config"] = optimization_results["best_hyperparameters"]

        # Step 3: Combine domain config and RL results
        final_config = self._combine_configs(
            domain_config=domain_config,
            rl_config=results.get("rl_optimized_config"),
        )
        results["final_config"] = final_config

        # Step 4: Learn from results if using domain adaptation
        if self.use_domain_adaptation and domain_config is not None:
            if results.get("optimization_results"):
                opt_results = results["optimization_results"]
                self.domain_adapter.learn_from_results(
                    domain_type=domain_config.domain_type,
                    quality=opt_results.get("best_quality", 0.0),
                    speed=opt_results.get("best_speed", 0.0),
                    hyperparameters=final_config,
                )

        # Record in history
        self.optimization_history.append(results)

        logger.info(f"Optimization complete. Final config: {final_config}")

        return results

    def _combine_configs(
        self,
        domain_config: Any = None,
        rl_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Combine domain and RL configurations.

        Args:
            domain_config: Domain-specific config
            rl_config: RL-optimized config

        Returns:
            Combined configuration
        """
        # Start with domain config if available
        if domain_config is not None:
            final = {
                "num_steps": domain_config.num_steps,
                "adaptive_threshold": domain_config.adaptive_threshold,
                "progress_power": domain_config.progress_power,
                "quality_weight": domain_config.quality_weight,
                "speed_weight": domain_config.speed_weight,
            }
        else:
            # Defaults
            final = {
                "num_steps": 50,
                "adaptive_threshold": 0.5,
                "progress_power": 2.0,
                "quality_weight": 0.7,
                "speed_weight": 0.3,
            }

        # Blend with RL config if available (weighted average)
        if rl_config is not None:
            blend_weight = 0.6  # Favor RL optimization

            final["num_steps"] = int(
                final["num_steps"] * (1 - blend_weight)
                + rl_config.get("num_steps", final["num_steps"]) * blend_weight
            )

            final["adaptive_threshold"] = (
                final["adaptive_threshold"] * (1 - blend_weight)
                + rl_config.get("adaptive_threshold", final["adaptive_threshold"])
                * blend_weight
            )

            final["progress_power"] = (
                final["progress_power"] * (1 - blend_weight)
                + rl_config.get("progress_power", final["progress_power"]) * blend_weight
            )

        return final

    def create_optimized_scheduler(
        self,
        config: dict[str, Any] | None = None,
        domain_type: DomainType | None = None,
        prompt: str | None = None,
    ) -> AdaptiveScheduler:
        """
        Create adaptive scheduler with optimized configuration.

        Args:
            config: Explicit config (auto-optimized if None)
            domain_type: Domain type for optimization
            prompt: Prompt for domain detection

        Returns:
            Configured AdaptiveScheduler
        """
        if config is None:
            # Run quick optimization
            results = self.optimize(
                domain_type=domain_type,
                prompt=prompt,
                num_training_steps=500,
                num_optimization_episodes=5,
            )
            config = results["final_config"]

        # Create scheduler with optimized params
        scheduler = AdaptiveScheduler(
            num_inference_steps=config.get("num_steps", 50),
            adaptive_threshold=config.get("adaptive_threshold", 0.5),
            progress_power=config.get("progress_power", 2.0),
        )

        logger.info(f"Created optimized scheduler with config: {config}")

        return scheduler

    def create_optimized_sampler(
        self,
        scheduler: AdaptiveScheduler | None = None,
        config: dict[str, Any] | None = None,
        domain_type: DomainType | None = None,
        prompt: str | None = None,
    ) -> QualityGuidedSampler:
        """
        Create quality-guided sampler with optimized configuration.

        Args:
            scheduler: Base scheduler (creates optimized if None)
            config: Explicit config (auto-optimized if None)
            domain_type: Domain type for optimization
            prompt: Prompt for domain detection

        Returns:
            Configured QualityGuidedSampler
        """
        if scheduler is None:
            scheduler = self.create_optimized_scheduler(
                config=config, domain_type=domain_type, prompt=prompt
            )

        # Create quality-guided sampler
        sampler = QualityGuidedSampler(
            scheduler=scheduler,
            quality_threshold=config.get("adaptive_threshold", 0.6) if config else 0.6,
        )

        logger.info("Created optimized quality-guided sampler")

        return sampler

    def save(self, path: str | Path):
        """
        Save pipeline state.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save RL agent if available
        if self.rl_agent is not None:
            agent_path = path.parent / f"{path.stem}_rl_agent.zip"
            self.rl_agent.save(agent_path)

        # TODO: Save domain adapter configs
        logger.info(f"Pipeline saved to {path}")

    def load(self, path: str | Path):
        """
        Load pipeline state.

        Args:
            path: Path to load from
        """
        path = Path(path)

        # Load RL agent if available
        if self.rl_agent is not None:
            agent_path = path.parent / f"{path.stem}_rl_agent.zip"
            if agent_path.exists():
                self.rl_agent.load(agent_path)

        logger.info(f"Pipeline loaded from {path}")

    def get_history(self) -> list[dict[str, Any]]:
        """
        Get optimization history.

        Returns:
            List of optimization results
        """
        return self.optimization_history.copy()


def create_optimization_pipeline(
    use_domain_adaptation: bool = True,
    use_rl_optimization: bool = True,
    **kwargs,
) -> OptimizationPipeline:
    """
    Factory function to create optimization pipeline.

    Args:
        use_domain_adaptation: Whether to use domain adaptation
        use_rl_optimization: Whether to use RL optimization
        **kwargs: Additional pipeline parameters

    Returns:
        Initialized OptimizationPipeline
    """
    return OptimizationPipeline(
        use_domain_adaptation=use_domain_adaptation,
        use_rl_optimization=use_rl_optimization,
        **kwargs,
    )
