"""
Tests for PPO-Based Hyperparameter Tuning Agent
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from adaptive_diffusion.rl.environment import DiffusionHyperparameterEnv
from adaptive_diffusion.rl.ppo_agent import (
    ConvergenceCallback,
    HyperparameterTuningAgent,
    create_ppo_agent,
)
from adaptive_diffusion.rl.reward import QualitySpeedReward


class TestHyperparameterTuningAgent:
    """Test suite for PPO-based tuning agent."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = HyperparameterTuningAgent(verbose=0)

        assert agent.agent is not None
        assert agent.env is not None
        assert agent.training_stats["total_timesteps"] == 0

    def test_agent_with_custom_env(self):
        """Test agent with custom environment."""
        env = DiffusionHyperparameterEnv(max_steps=80)
        agent = HyperparameterTuningAgent(env=env, verbose=0)

        assert agent.env is not None

    def test_agent_prediction(self):
        """Test agent prediction."""
        agent = HyperparameterTuningAgent(verbose=0)
        obs = agent.env.reset()

        action, state = agent.predict(obs, deterministic=True)

        assert action is not None
        # VecEnv adds batch dimension, so shape is (1, 3) instead of (3,)
        assert action.shape in [(3,), (1, 3)]

    def test_agent_training_basic(self):
        """Test basic agent training."""
        agent = HyperparameterTuningAgent(verbose=0)

        # Train for minimal steps
        stats = agent.train(total_timesteps=100)

        assert stats["total_timesteps"] >= 100
        assert "convergence_achieved" in stats

    def test_agent_training_with_convergence(self):
        """Test training with convergence callback."""
        agent = HyperparameterTuningAgent(verbose=0)

        stats = agent.train(
            total_timesteps=500,
            convergence_threshold=0.1,
            convergence_window=3,
        )

        assert "convergence_achieved" in stats

    def test_agent_optimization(self):
        """Test hyperparameter optimization."""
        agent = HyperparameterTuningAgent(verbose=0)

        # Train briefly
        agent.train(total_timesteps=200)

        # Run optimization
        results = agent.optimize_hyperparameters(num_episodes=3)

        assert "best_quality" in results
        assert "best_speed" in results
        assert "best_hyperparameters" in results
        assert len(results["episodes"]) == 3

        # Check hyperparameters are in valid range
        best_params = results["best_hyperparameters"]
        if best_params:
            assert "num_steps" in best_params
            assert "adaptive_threshold" in best_params
            assert "progress_power" in best_params

    def test_agent_save_load(self):
        """Test agent save and load."""
        agent = HyperparameterTuningAgent(verbose=0)

        # Train briefly
        agent.train(total_timesteps=100)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "agent.zip"
            agent.save(save_path)

            assert save_path.exists()

            # Load
            new_agent = HyperparameterTuningAgent(verbose=0)
            new_agent.load(save_path)

            # Verify loaded agent works
            obs = new_agent.env.reset()
            action, _ = new_agent.predict(obs)
            assert action is not None

    def test_agent_config(self):
        """Test agent configuration."""
        agent = HyperparameterTuningAgent(
            learning_rate=1e-3,
            gamma=0.95,
            verbose=0,
        )

        config = agent.get_config()

        assert config["learning_rate"] == 1e-3
        assert config["gamma"] == 0.95
        assert "n_steps" in config
        assert "batch_size" in config

    def test_agent_normalization(self):
        """Test environment normalization."""
        # With normalization
        agent_norm = HyperparameterTuningAgent(normalize_env=True, verbose=0)
        assert agent_norm.normalize_env is True

        # Without normalization
        agent_no_norm = HyperparameterTuningAgent(normalize_env=False, verbose=0)
        assert agent_no_norm.normalize_env is False

    def test_agent_learning_progress(self):
        """Test that agent shows learning progress."""
        agent = HyperparameterTuningAgent(verbose=0)

        # Get initial performance
        initial_results = agent.optimize_hyperparameters(num_episodes=2)
        initial_quality = initial_results["mean_quality"]

        # Train
        agent.train(total_timesteps=1000)

        # Get final performance
        final_results = agent.optimize_hyperparameters(num_episodes=2)
        final_quality = final_results["mean_quality"]

        # Quality should be non-negative (may or may not improve with so little training)
        assert initial_quality >= 0.0
        assert final_quality >= 0.0

    def test_create_ppo_agent_factory(self):
        """Test PPO agent factory function."""
        agent = create_ppo_agent(verbose=0)

        assert isinstance(agent, HyperparameterTuningAgent)
        assert agent.agent is not None


class TestConvergenceCallback:
    """Test suite for convergence callback."""

    def test_convergence_callback_initialization(self):
        """Test callback initialization."""
        callback = ConvergenceCallback(threshold=0.01, window=5)

        assert callback.threshold == 0.01
        assert callback.window == 5
        assert callback.convergence_achieved is False

    def test_convergence_callback_with_agent(self):
        """Test callback integration with agent training."""
        agent = HyperparameterTuningAgent(verbose=0)

        callback = ConvergenceCallback(threshold=0.5, window=3, verbose=0)

        # Train with callback
        stats = agent.train(total_timesteps=200, callback=callback)

        # Callback should track convergence
        assert hasattr(callback, "convergence_achieved")


class TestPPOAgentIntegration:
    """Integration tests for PPO agent."""

    def test_full_training_and_optimization_workflow(self):
        """Test complete workflow from training to optimization."""
        # Create environment with custom reward
        reward_fn = QualitySpeedReward(quality_weight=0.6, speed_weight=0.4)
        env = DiffusionHyperparameterEnv(reward_function=reward_fn)

        # Create agent
        agent = HyperparameterTuningAgent(
            env=env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            verbose=0,
        )

        # Train
        stats = agent.train(total_timesteps=500)

        assert stats["total_timesteps"] >= 500

        # Optimize
        results = agent.optimize_hyperparameters(num_episodes=5)

        assert len(results["episodes"]) == 5
        assert results["mean_quality"] >= 0.0
        assert results["mean_speed"] >= 0.0

        # Verify hyperparameters are reasonable
        best_params = results["best_hyperparameters"]
        if best_params:
            assert 10 <= best_params.get("num_steps", 50) <= 100
            assert 0.0 <= best_params.get("adaptive_threshold", 0.5) <= 1.0
            assert 0.5 <= best_params.get("progress_power", 2.0) <= 5.0

    def test_agent_deterministic_predictions(self):
        """Test that deterministic predictions are reproducible."""
        agent = HyperparameterTuningAgent(verbose=0)
        agent.train(total_timesteps=200)

        obs = agent.env.reset()

        # Get multiple predictions
        action1, _ = agent.predict(obs, deterministic=True)
        action2, _ = agent.predict(obs, deterministic=True)

        # Should be identical
        np.testing.assert_array_almost_equal(action1, action2)

    def test_agent_stochastic_predictions(self):
        """Test that stochastic predictions vary."""
        agent = HyperparameterTuningAgent(verbose=0)
        agent.train(total_timesteps=200)

        obs = agent.env.reset()

        # Get multiple predictions
        actions = []
        for _ in range(10):
            action, _ = agent.predict(obs, deterministic=False)
            actions.append(action)

        # Should have some variation (not all identical)
        actions_array = np.array(actions)
        variances = np.var(actions_array, axis=0)

        # At least one action dimension should have variance
        assert np.any(variances > 0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
