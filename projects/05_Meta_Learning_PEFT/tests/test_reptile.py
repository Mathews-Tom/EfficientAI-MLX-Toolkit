"""Tests for Reptile meta-learning algorithm."""

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meta_learning.models import SimpleClassifier, cross_entropy_loss
from meta_learning.reptile import ReptileLearner
from task_embedding.task_distribution import TaskConfig, TaskDistribution


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestReptileLearner:
    """Test Reptile meta-learning implementation."""

    def test_reptile_initialization(self) -> None:
        """Test ReptileLearner initialization."""
        model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
        learner = ReptileLearner(
            model=model,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5,
            meta_batch_size=4,
        )

        assert learner.inner_lr == 0.01
        assert learner.outer_lr == 0.001
        assert learner.num_inner_steps == 5
        assert learner.meta_batch_size == 4
        assert learner.meta_params is not None

    def test_inner_loop(self) -> None:
        """Test inner loop task adaptation."""
        model = SimpleClassifier(input_dim=2, hidden_dim=32, num_classes=2)
        learner = ReptileLearner(model=model, inner_lr=0.1, num_inner_steps=3)

        # Create synthetic support set
        support_x = mx.random.normal((10, 2))
        support_y = mx.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # Run inner loop
        task_params = learner.inner_loop(support_x, support_y, cross_entropy_loss)

        assert isinstance(task_params, dict)
        assert len(task_params) > 0

    def test_outer_loop_update(self) -> None:
        """Test outer loop meta-update."""
        model = SimpleClassifier(input_dim=2, hidden_dim=32, num_classes=2)
        learner = ReptileLearner(model=model, outer_lr=0.1)

        # Get initial parameters
        initial_params = {k: v.copy() for k, v in learner.meta_params.items()}

        # Create mock task parameters (slightly different from initial)
        task_params_list = []
        for _ in range(4):
            task_params = {}
            for key, value in learner.meta_params.items():
                # Add small perturbation
                task_params[key] = value + mx.random.normal(value.shape) * 0.01
            task_params_list.append(task_params)

        # Perform outer loop update
        learner.outer_loop_update(task_params_list)

        # Check that parameters changed
        params_changed = False
        for key in learner.meta_params.keys():
            if not mx.allclose(learner.meta_params[key], initial_params[key]):
                params_changed = True
                break

        assert params_changed, "Meta-parameters should be updated"

    def test_meta_train_step(self) -> None:
        """Test complete meta-training step."""
        model = SimpleClassifier(input_dim=2, hidden_dim=32, num_classes=2)
        learner = ReptileLearner(
            model=model, inner_lr=0.1, outer_lr=0.01, num_inner_steps=3
        )

        # Create synthetic episodes
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((5, 2))
            support_y = mx.array([0, 1, 0, 1, 0])
            query_x = mx.random.normal((20, 2))
            query_y = mx.random.randint(0, 2, (20,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Run meta-training step
        metrics = learner.meta_train_step(episodes, cross_entropy_loss)

        assert "inner_loss" in metrics
        assert "query_loss" in metrics
        assert isinstance(metrics["inner_loss"], float)
        assert isinstance(metrics["query_loss"], float)

    def test_evaluate(self) -> None:
        """Test meta-learning evaluation."""
        model = SimpleClassifier(input_dim=2, hidden_dim=32, num_classes=2)
        learner = ReptileLearner(model=model, inner_lr=0.1, num_inner_steps=5)

        # Create evaluation episodes
        episodes = []
        for _ in range(5):
            support_x = mx.random.normal((5, 2))
            support_y = mx.array([0, 1, 0, 1, 0])
            query_x = mx.random.normal((20, 2))
            query_y = mx.random.randint(0, 2, (20,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Evaluate
        results = learner.evaluate(
            episodes, cross_entropy_loss, num_adaptation_steps=[0, 1, 5]
        )

        # Check results structure
        assert "step_0_loss" in results
        assert "step_0_acc" in results
        assert "step_1_loss" in results
        assert "step_1_acc" in results
        assert "step_5_loss" in results
        assert "step_5_acc" in results

    def test_reptile_with_task_distribution(self) -> None:
        """Test Reptile with TaskDistribution."""
        # Create task distribution
        configs = [
            TaskConfig(
                task_id=f"task_{i}",
                task_family="linear_classification",
                num_classes=2,
                input_dim=2,
                support_size=5,
                query_size=20,
            )
            for i in range(10)
        ]
        dist = TaskDistribution(configs, seed=42)

        # Create model and learner
        model = SimpleClassifier(input_dim=2, hidden_dim=32, num_classes=2)
        learner = ReptileLearner(
            model=model, inner_lr=0.1, outer_lr=0.01, num_inner_steps=5
        )

        # Sample episodes and train
        episodes = []
        for _ in range(4):
            _, support_x, support_y, query_x, query_y = dist.sample_episode()
            episodes.append((support_x, support_y, query_x, query_y))

        metrics = learner.meta_train_step(episodes, cross_entropy_loss)

        assert "inner_loss" in metrics
        assert "query_loss" in metrics

    @pytest.mark.slow
    def test_reptile_learning_curve(self) -> None:
        """Test that Reptile improves over iterations."""
        # Create simple task distribution
        configs = [
            TaskConfig(
                task_id=f"task_{i}",
                task_family="linear_classification",
                num_classes=2,
                input_dim=2,
                support_size=10,
                query_size=20,
            )
            for i in range(20)
        ]
        dist = TaskDistribution(configs, seed=42)

        # Create model and learner
        model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
        learner = ReptileLearner(
            model=model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5
        )

        # Initial evaluation
        eval_episodes = []
        for _ in range(10):
            _, support_x, support_y, query_x, query_y = dist.sample_episode()
            eval_episodes.append((support_x, support_y, query_x, query_y))

        initial_results = learner.evaluate(
            eval_episodes, cross_entropy_loss, num_adaptation_steps=[5]
        )
        initial_acc = initial_results["step_5_acc"]

        # Train for several iterations
        for _ in range(20):
            episodes = []
            for _ in range(4):
                _, support_x, support_y, query_x, query_y = dist.sample_episode()
                episodes.append((support_x, support_y, query_x, query_y))
            learner.meta_train_step(episodes, cross_entropy_loss)

        # Final evaluation
        final_results = learner.evaluate(
            eval_episodes, cross_entropy_loss, num_adaptation_steps=[5]
        )
        final_acc = final_results["step_5_acc"]

        # Check that accuracy improved (or at least didn't degrade significantly)
        assert (
            final_acc >= initial_acc - 0.05
        ), f"Accuracy decreased: {initial_acc} -> {final_acc}"
