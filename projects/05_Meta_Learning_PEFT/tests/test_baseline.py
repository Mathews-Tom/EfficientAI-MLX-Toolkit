"""Tests for baseline evaluation utilities."""

import pytest

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meta_learning.models import SimpleClassifier, cross_entropy_loss
from utils.baseline import BaselineEvaluator, compare_meta_vs_baseline


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestBaselineEvaluator:
    """Test baseline evaluation functionality."""

    def test_baseline_evaluator_initialization(self) -> None:
        """Test BaselineEvaluator initialization."""

        def model_factory():
            return SimpleClassifier(input_dim=2, hidden_dim=32, num_classes=2)

        evaluator = BaselineEvaluator(
            model_factory=model_factory, learning_rate=0.01, num_steps=50
        )

        assert evaluator.learning_rate == 0.01
        assert evaluator.num_steps == 50

    def test_evaluate_task(self) -> None:
        """Test evaluating baseline on single task."""

        def model_factory():
            return SimpleClassifier(input_dim=2, hidden_dim=32, num_classes=2)

        evaluator = BaselineEvaluator(
            model_factory=model_factory, learning_rate=0.1, num_steps=20
        )

        # Create synthetic data
        support_x = mx.random.normal((10, 2))
        support_y = mx.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        query_x = mx.random.normal((50, 2))
        query_y = mx.random.randint(0, 2, (50,))

        results = evaluator.evaluate_task(
            support_x, support_y, query_x, query_y, cross_entropy_loss
        )

        assert "final_query_loss" in results
        assert "final_query_acc" in results
        assert "train_losses" in results
        assert "query_losses" in results
        assert "query_accs" in results
        assert "elapsed_time" in results
        assert "num_steps" in results

        assert len(results["train_losses"]) == 20
        assert results["elapsed_time"] > 0

    def test_evaluate_tasks(self) -> None:
        """Test evaluating baseline on multiple tasks."""

        def model_factory():
            return SimpleClassifier(input_dim=2, hidden_dim=32, num_classes=2)

        evaluator = BaselineEvaluator(
            model_factory=model_factory, learning_rate=0.1, num_steps=10
        )

        # Create multiple episodes
        episodes = []
        for _ in range(5):
            support_x = mx.random.normal((10, 2))
            support_y = mx.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            query_x = mx.random.normal((20, 2))
            query_y = mx.random.randint(0, 2, (20,))
            episodes.append((support_x, support_y, query_x, query_y))

        results = evaluator.evaluate_tasks(episodes, cross_entropy_loss)

        assert "mean_query_loss" in results
        assert "mean_query_acc" in results
        assert "mean_elapsed_time" in results
        assert "std_query_acc" in results
        assert "num_tasks" in results

        assert results["num_tasks"] == 5
        assert results["std_query_acc"] >= 0

    def test_baseline_training_convergence(self) -> None:
        """Test that baseline training reduces loss."""

        def model_factory():
            return SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)

        evaluator = BaselineEvaluator(
            model_factory=model_factory, learning_rate=0.1, num_steps=50
        )

        # Create simple separable data
        support_x = mx.concatenate(
            [mx.random.normal((5, 2)) - 2, mx.random.normal((5, 2)) + 2]
        )
        support_y = mx.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        query_x = mx.concatenate(
            [mx.random.normal((25, 2)) - 2, mx.random.normal((25, 2)) + 2]
        )
        query_y = mx.concatenate([mx.zeros((25,)), mx.ones((25,))]).astype(mx.int32)

        results = evaluator.evaluate_task(
            support_x, support_y, query_x, query_y, cross_entropy_loss
        )

        # Check that training loss decreased
        initial_loss = results["train_losses"][0]
        final_loss = results["train_losses"][-1]

        assert (
            final_loss < initial_loss
        ), f"Loss should decrease: {initial_loss} -> {final_loss}"


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestCompareMetaVsBaseline:
    """Test comparison utilities."""

    def test_compare_meta_vs_baseline(self) -> None:
        """Test comparing meta-learning vs baseline results."""
        meta_results = {
            "step_5_acc": 0.85,
            "step_5_loss": 0.3,
            "adaptation_time": 0.5,
        }

        baseline_results = {
            "mean_query_acc": 0.70,
            "mean_query_loss": 0.5,
            "mean_elapsed_time": 2.0,
        }

        comparison = compare_meta_vs_baseline(meta_results, baseline_results)

        assert "meta_acc" in comparison
        assert "baseline_acc" in comparison
        assert "meta_loss" in comparison
        assert "baseline_loss" in comparison
        assert "acc_improvement_pct" in comparison
        assert "speedup" in comparison

        assert comparison["meta_acc"] == 0.85
        assert comparison["baseline_acc"] == 0.70
        assert comparison["acc_improvement_pct"] > 0  # Meta should be better
        assert comparison["speedup"] == 4.0  # 2.0 / 0.5

    def test_compare_zero_baseline(self) -> None:
        """Test comparison handles zero baseline accuracy."""
        meta_results = {"step_5_acc": 0.80, "step_5_loss": 0.4}

        baseline_results = {"mean_query_acc": 0.0, "mean_query_loss": 1.0}

        comparison = compare_meta_vs_baseline(meta_results, baseline_results)

        assert comparison["acc_improvement_pct"] == 0.0
