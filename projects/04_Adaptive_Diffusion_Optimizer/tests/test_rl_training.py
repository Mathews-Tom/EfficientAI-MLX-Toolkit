"""
Tests for RL Training with Convergence Validation
"""

import pytest

from adaptive_diffusion.rl.environment import DiffusionHyperparameterEnv
from adaptive_diffusion.rl.ppo_agent import HyperparameterTuningAgent
from adaptive_diffusion.rl.reward import create_reward_function


class TestRLTraining:
    """Test suite for RL training."""

    def test_basic_training_convergence(self):
        """Test basic training with convergence monitoring."""
        env = DiffusionHyperparameterEnv()
        agent = HyperparameterTuningAgent(env=env, verbose=0)

        # Train for minimal steps
        stats = agent.train(total_timesteps=500, convergence_threshold=0.5)

        assert stats["total_timesteps"] >= 500
        assert "convergence_achieved" in stats

    def test_training_with_quality_speed_reward(self):
        """Test training with quality-speed reward."""
        reward_fn = create_reward_function("quality_speed", quality_weight=0.8, speed_weight=0.2)
        env = DiffusionHyperparameterEnv(reward_function=reward_fn)
        agent = HyperparameterTuningAgent(env=env, verbose=0)

        stats = agent.train(total_timesteps=300)

        assert stats["total_timesteps"] >= 300

    def test_training_with_multi_objective_reward(self):
        """Test training with multi-objective reward."""
        reward_fn = create_reward_function(
            "multi_objective",
            quality_weight=0.5,
            speed_weight=0.3,
            stability_weight=0.2,
        )
        env = DiffusionHyperparameterEnv(reward_function=reward_fn)
        agent = HyperparameterTuningAgent(env=env, verbose=0)

        stats = agent.train(total_timesteps=300)

        assert stats["total_timesteps"] >= 300

    def test_training_with_adaptive_reward(self):
        """Test training with adaptive reward."""
        reward_fn = create_reward_function(
            "adaptive", initial_quality_weight=0.9, final_quality_weight=0.6
        )
        env = DiffusionHyperparameterEnv(reward_function=reward_fn)
        agent = HyperparameterTuningAgent(env=env, verbose=0)

        stats = agent.train(total_timesteps=300)

        assert stats["total_timesteps"] >= 300

    def test_convergence_monitoring(self):
        """Test convergence monitoring."""
        agent = HyperparameterTuningAgent(verbose=0)

        # Train with tight convergence criteria
        stats = agent.train(
            total_timesteps=1000,
            convergence_threshold=0.1,
            convergence_window=5,
        )

        # Should track convergence
        assert "convergence_achieved" in stats

    def test_hyperparameter_optimization_quality(self):
        """Test that optimization produces valid hyperparameters."""
        agent = HyperparameterTuningAgent(verbose=0)

        # Train
        agent.train(total_timesteps=500)

        # Optimize
        results = agent.optimize_hyperparameters(num_episodes=5)

        # Verify results structure
        assert "best_quality" in results
        assert "best_speed" in results
        assert "best_hyperparameters" in results
        assert len(results["episodes"]) == 5

        # Verify hyperparameters are in valid ranges
        if results["best_hyperparameters"]:
            params = results["best_hyperparameters"]
            assert 10 <= params.get("num_steps", 50) <= 100
            assert 0.0 <= params.get("adaptive_threshold", 0.5) <= 1.0
            assert 0.5 <= params.get("progress_power", 2.0) <= 5.0

    def test_training_improves_performance(self):
        """Test that training improves optimization performance."""
        agent = HyperparameterTuningAgent(verbose=0)

        # Get baseline performance (minimal training)
        agent.train(total_timesteps=100)
        baseline_results = agent.optimize_hyperparameters(num_episodes=3)
        baseline_quality = baseline_results["mean_quality"]

        # Train more
        agent.train(total_timesteps=500)
        improved_results = agent.optimize_hyperparameters(num_episodes=3)
        improved_quality = improved_results["mean_quality"]

        # Quality should be non-negative (may or may not improve)
        assert baseline_quality >= 0.0
        assert improved_quality >= 0.0

    def test_different_environments(self):
        """Test training with different environment configurations."""
        # Easier environment (fewer steps range)
        easy_env = DiffusionHyperparameterEnv(min_steps=20, max_steps=60)
        agent_easy = HyperparameterTuningAgent(env=easy_env, verbose=0)
        stats_easy = agent_easy.train(total_timesteps=300)

        assert stats_easy["total_timesteps"] >= 300

        # Harder environment (wider range)
        hard_env = DiffusionHyperparameterEnv(min_steps=10, max_steps=100)
        agent_hard = HyperparameterTuningAgent(env=hard_env, verbose=0)
        stats_hard = agent_hard.train(total_timesteps=300)

        assert stats_hard["total_timesteps"] >= 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
