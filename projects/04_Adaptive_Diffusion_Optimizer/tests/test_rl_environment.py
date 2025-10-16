"""
Tests for RL Environment for Hyperparameter Optimization
"""

import numpy as np
import pytest

from adaptive_diffusion.rl.environment import (
    DiffusionHyperparameterEnv,
    create_diffusion_env,
)
from adaptive_diffusion.rl.reward import (
    AdaptiveReward,
    MultiObjectiveReward,
    QualitySpeedReward,
    create_reward_function,
)


class TestRewardFunctions:
    """Test suite for reward functions."""

    def test_quality_speed_reward_basic(self):
        """Test basic quality-speed reward computation."""
        reward_fn = QualitySpeedReward(quality_weight=0.7, speed_weight=0.3)

        # Test with good quality and speed
        reward = reward_fn.compute_reward(quality=0.8, speed=0.6)
        assert 0.0 <= reward <= 2.0  # Should be positive

        # Test with low quality (below threshold)
        reward_low = reward_fn.compute_reward(quality=0.4, speed=0.8)
        assert reward_low < reward  # Should have penalty

    def test_quality_speed_reward_threshold_bonus(self):
        """Test threshold bonus in quality-speed reward."""
        reward_fn = QualitySpeedReward(
            quality_weight=0.7,
            speed_weight=0.3,
            quality_threshold=0.6,
            speed_threshold=0.5,
        )

        # Both above threshold - should get bonus
        reward_bonus = reward_fn.compute_reward(quality=0.8, speed=0.7)

        # Only one above threshold - no bonus
        reward_no_bonus = reward_fn.compute_reward(quality=0.8, speed=0.4)

        assert reward_bonus > reward_no_bonus

    def test_quality_speed_reward_weights_validation(self):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            QualitySpeedReward(quality_weight=0.5, speed_weight=0.3)

    def test_multi_objective_reward_basic(self):
        """Test basic multi-objective reward computation."""
        reward_fn = MultiObjectiveReward(
            quality_weight=0.5, speed_weight=0.3, stability_weight=0.2
        )

        reward = reward_fn.compute_reward(quality=0.8, speed=0.6)
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 2.0

    def test_multi_objective_reward_stability(self):
        """Test stability component of multi-objective reward."""
        reward_fn = MultiObjectiveReward(
            quality_weight=0.5, speed_weight=0.3, stability_weight=0.2
        )

        # Stable quality over time
        for _ in range(5):
            reward_fn.compute_reward(quality=0.8, speed=0.6)

        stable_reward = reward_fn.compute_reward(quality=0.8, speed=0.6)

        # Reset and create unstable quality
        reward_fn.reset_history()
        qualities = [0.5, 0.9, 0.4, 0.85, 0.45]
        for q in qualities:
            reward_fn.compute_reward(quality=q, speed=0.6)

        unstable_reward = reward_fn.compute_reward(quality=0.8, speed=0.6)

        # Stable should have higher reward
        assert stable_reward > unstable_reward

    def test_multi_objective_reward_pareto_optimal(self):
        """Test Pareto optimality detection."""
        reward_fn = MultiObjectiveReward()

        # First solution is always Pareto-optimal
        reward1 = reward_fn.compute_reward(quality=0.7, speed=0.5)

        # Dominated solution (lower quality and speed)
        reward2 = reward_fn.compute_reward(quality=0.6, speed=0.4)

        # Pareto-optimal solution (better in one dimension)
        reward3 = reward_fn.compute_reward(quality=0.8, speed=0.5)

        # Pareto-optimal solutions should get bonus
        assert reward3 > reward2

    def test_multi_objective_reward_pareto_front(self):
        """Test Pareto front tracking."""
        reward_fn = MultiObjectiveReward()

        solutions = [
            (0.7, 0.5),
            (0.6, 0.7),  # Different trade-off
            (0.8, 0.6),  # Dominates first
            (0.5, 0.4),  # Dominated
        ]

        for q, s in solutions:
            reward_fn.compute_reward(quality=q, speed=s)

        pareto_front = reward_fn.get_pareto_front()

        # Should contain non-dominated solutions
        assert len(pareto_front) >= 2
        assert (0.8, 0.6) in pareto_front  # Best overall
        assert (0.5, 0.4) not in pareto_front  # Dominated

    def test_adaptive_reward_weight_transition(self):
        """Test adaptive reward weight transition."""
        reward_fn = AdaptiveReward(
            initial_quality_weight=0.9,
            final_quality_weight=0.6,
            transition_steps=100,
        )

        # Early in training - high quality weight
        early_rewards = []
        for _ in range(10):
            r = reward_fn.compute_reward(quality=0.5, speed=0.9)
            early_rewards.append(r)

        # Late in training - lower quality weight, higher speed weight
        for _ in range(90):
            reward_fn.compute_reward(quality=0.7, speed=0.7)

        late_rewards = []
        for _ in range(10):
            r = reward_fn.compute_reward(quality=0.5, speed=0.9)
            late_rewards.append(r)

        # With high speed, late rewards should be higher (speed weight increased)
        assert np.mean(late_rewards) > np.mean(early_rewards)

    def test_reward_function_factory(self):
        """Test reward function factory."""
        # Test quality_speed type
        reward1 = create_reward_function("quality_speed")
        assert isinstance(reward1, QualitySpeedReward)

        # Test multi_objective type
        reward2 = create_reward_function("multi_objective")
        assert isinstance(reward2, MultiObjectiveReward)

        # Test adaptive type
        reward3 = create_reward_function("adaptive")
        assert isinstance(reward3, AdaptiveReward)

        # Test unknown type
        with pytest.raises(ValueError, match="Unknown reward type"):
            create_reward_function("unknown")


class TestDiffusionHyperparameterEnv:
    """Test suite for diffusion hyperparameter environment."""

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = DiffusionHyperparameterEnv()

        assert env.observation_space.shape == (6,)
        assert env.action_space.shape == (3,)
        assert env.min_steps == 10
        assert env.max_steps == 100

    def test_environment_reset(self):
        """Test environment reset."""
        env = DiffusionHyperparameterEnv()
        obs, info = env.reset()

        # Check observation shape and values
        assert obs.shape == (6,)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 5.0)  # progress_power can be up to 5.0

        # Check info
        assert "step" in info
        assert info["step"] == 0

    def test_environment_step(self):
        """Test environment step."""
        env = DiffusionHyperparameterEnv()
        env.reset()

        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Check outputs
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "quality" in info
        assert "speed" in info

    def test_environment_episode_completion(self):
        """Test full episode completion."""
        env = DiffusionHyperparameterEnv(max_episode_length=10)
        env.reset()

        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(20):  # More than episode length
            if terminated or truncated:
                break

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Episode should be truncated
        assert truncated
        assert info["step"] == 10

    def test_environment_hyperparameter_bounds(self):
        """Test that hyperparameters stay within valid bounds."""
        env = DiffusionHyperparameterEnv()
        env.reset()

        # Take extreme actions
        for _ in range(10):
            # Maximum adjustment
            action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            # Check bounds
            assert env.min_steps <= info["num_steps"] <= env.max_steps
            assert 0.0 <= info["adaptive_threshold"] <= 1.0
            assert 0.5 <= info["progress_power"] <= 5.0

        env.reset()

        for _ in range(10):
            # Minimum adjustment
            action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            # Check bounds
            assert env.min_steps <= info["num_steps"] <= env.max_steps
            assert 0.0 <= info["adaptive_threshold"] <= 1.0
            assert 0.5 <= info["progress_power"] <= 5.0

    def test_environment_quality_speed_relationship(self):
        """Test that quality and speed have expected relationship."""
        env = DiffusionHyperparameterEnv()
        env.reset()

        # More steps should generally improve quality but reduce speed
        # Set many steps
        env.current_num_steps = 90
        quality_high_steps, speed_high_steps = env._simulate_diffusion()

        env.reset()
        # Set few steps
        env.current_num_steps = 15
        quality_low_steps, speed_low_steps = env._simulate_diffusion()

        # More steps should give better quality but lower speed
        # (allowing some tolerance due to noise)
        assert quality_high_steps >= quality_low_steps - 0.2
        assert speed_low_steps >= speed_high_steps - 0.1

    def test_environment_with_custom_reward(self):
        """Test environment with custom reward function."""
        custom_reward = MultiObjectiveReward()
        env = DiffusionHyperparameterEnv(reward_function=custom_reward)

        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)

    def test_environment_history_tracking(self):
        """Test episode history tracking."""
        env = DiffusionHyperparameterEnv()
        env.reset()

        num_steps = 5
        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        # Check history
        assert len(info["episode_history"]) == num_steps
        assert all("quality" in entry for entry in info["episode_history"])
        assert all("speed" in entry for entry in info["episode_history"])
        assert all("reward" in entry for entry in info["episode_history"])

    def test_create_diffusion_env_factory(self):
        """Test environment factory function."""
        env = create_diffusion_env(min_steps=20, max_steps=80)

        assert isinstance(env, DiffusionHyperparameterEnv)
        assert env.min_steps == 20
        assert env.max_steps == 80

    def test_environment_observation_normalization(self):
        """Test that observations are properly normalized."""
        env = DiffusionHyperparameterEnv()
        obs, _ = env.reset()

        # Most values should be in [0, 1] except progress_power which can be up to 5.0
        assert 0.0 <= obs[0] <= 1.0  # normalized_steps
        assert 0.0 <= obs[1] <= 1.0  # adaptive_threshold
        assert 0.0 <= obs[2] <= 1.0  # progress_power (normalized)
        assert 0.0 <= obs[3] <= 1.0  # quality
        assert 0.0 <= obs[4] <= 1.0  # complexity
        assert 0.0 <= obs[5] <= 1.0  # timestep_ratio

    def test_environment_deterministic_with_seed(self):
        """Test environment is deterministic with fixed seed."""
        # Create two environments with same seed
        env1 = DiffusionHyperparameterEnv()
        env2 = DiffusionHyperparameterEnv()

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Should produce same initial observations
        np.testing.assert_array_almost_equal(obs1, obs2)

        # Take same actions
        action = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        for _ in range(5):
            obs1, reward1, _, _, _ = env1.step(action)
            obs2, reward2, _, _, _ = env2.step(action)

            # Should produce same results
            np.testing.assert_array_almost_equal(obs1, obs2, decimal=5)
            assert abs(reward1 - reward2) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
