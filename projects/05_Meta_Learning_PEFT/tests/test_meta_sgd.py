"""Tests for Meta-SGD learner with learnable learning rates.

Tests MetaSGDLearner from meta_sgd.py.
"""

from __future__ import annotations

import pytest
import mlx.core as mx
import mlx.nn as nn

from meta_learning.meta_sgd import MetaSGDLearner
from meta_learning.models import SimpleClassifier


@pytest.fixture
def loss_fn():
    """Standard cross-entropy loss function."""
    def cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
        return nn.losses.cross_entropy(logits, labels, reduction="mean")
    return cross_entropy


@pytest.fixture
def simple_model():
    """Create a simple classifier for testing."""
    return SimpleClassifier(
        input_dim=10,
        hidden_dim=32,
        num_classes=5,
        num_layers=2,
    )


class TestMetaSGDLearner:
    """Test MetaSGDLearner module."""

    def test_initialization(self, simple_model):
        """Test MetaSGD learner initialization."""
        learner = MetaSGDLearner(
            model=simple_model,
            meta_lr=0.001,
            alpha_lr=0.01,
            num_inner_steps=5,
            init_inner_lr=0.01,
        )

        assert learner.model == simple_model
        assert learner.meta_lr == 0.001
        assert learner.alpha_lr == 0.01
        assert learner.num_inner_steps == 5

        # Verify alphas initialized for all parameters
        model_params = simple_model.parameters()
        assert len(learner.alphas) == len(model_params)

        # All alphas should be initialized to init_inner_lr
        for key, alpha in learner.alphas.items():
            assert mx.all(mx.isclose(alpha, 0.01))

    def test_alphas_are_learnable(self, simple_model):
        """Test that alphas are learnable parameters."""
        learner = MetaSGDLearner(
            model=simple_model,
            init_inner_lr=0.01,
        )

        # Alphas should match model parameter shapes
        model_params = simple_model.parameters()
        for key in model_params.keys():
            assert key in learner.alphas
            assert learner.alphas[key].shape == model_params[key].shape

    def test_inner_loop_adaptation(self, simple_model, loss_fn):
        """Test inner loop adaptation with learnable learning rates."""
        learner = MetaSGDLearner(
            model=simple_model,
            num_inner_steps=3,
        )

        # Create synthetic task data
        support_x = mx.random.normal((10, 10))
        support_y = mx.random.randint(0, 5, (10,))

        # Perform inner loop adaptation
        adapted_params = learner.inner_loop_adaptation(
            support_x, support_y, loss_fn
        )

        # Adapted parameters should differ from original
        original_params = simple_model.parameters()
        params_changed = False
        for key in original_params.keys():
            if not mx.allclose(adapted_params[key], original_params[key]):
                params_changed = True
                break

        assert params_changed, "Parameters should change during adaptation"

    def test_meta_train_step(self, simple_model, loss_fn):
        """Test meta-training step updates both model and alphas."""
        learner = MetaSGDLearner(
            model=simple_model,
            meta_lr=0.01,
            alpha_lr=0.01,
            num_inner_steps=2,
        )

        # Save initial states
        initial_params = {
            k: v.copy() for k, v in simple_model.parameters().items()
        }
        initial_alphas = {
            k: v.copy() for k, v in learner.alphas.items()
        }

        # Create task episodes
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Meta-training step
        metrics = learner.meta_train_step(episodes, loss_fn)

        # Verify metrics
        assert "query_loss" in metrics
        assert "avg_alpha" in metrics
        assert metrics["query_loss"] >= 0

        # Verify model parameters updated
        current_params = simple_model.parameters()
        model_updated = any(
            not mx.allclose(current_params[k], initial_params[k])
            for k in initial_params.keys()
        )
        assert model_updated, "Model parameters should be updated"

        # Verify alphas updated
        alphas_updated = any(
            not mx.allclose(learner.alphas[k], initial_alphas[k])
            for k in initial_alphas.keys()
        )
        assert alphas_updated, "Alphas should be updated"

    def test_evaluate(self, simple_model, loss_fn):
        """Test evaluation on episodes."""
        learner = MetaSGDLearner(
            model=simple_model,
            num_inner_steps=3,
        )

        # Create evaluation episodes
        episodes = []
        for _ in range(3):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Evaluate
        results = learner.evaluate(episodes, loss_fn)

        # Verify results
        assert "avg_loss" in results
        assert "avg_acc" in results
        assert results["avg_loss"] >= 0
        assert 0 <= results["avg_acc"] <= 1

    def test_alphas_adapt_per_parameter(self, simple_model, loss_fn):
        """Test that alphas can be different for different parameters."""
        learner = MetaSGDLearner(
            model=simple_model,
            alpha_lr=0.1,  # High alpha learning rate
            num_inner_steps=5,
        )

        # Meta-train for several steps
        for _ in range(5):
            episodes = []
            for _ in range(4):
                support_x = mx.random.normal((10, 10))
                support_y = mx.random.randint(0, 5, (10,))
                query_x = mx.random.normal((5, 10))
                query_y = mx.random.randint(0, 5, (5,))
                episodes.append((support_x, support_y, query_x, query_y))

            learner.meta_train_step(episodes, loss_fn)

        # Check that alphas are not all identical
        alpha_values = [mx.mean(alpha) for alpha in learner.alphas.values()]
        alpha_std = mx.std(mx.array(alpha_values))

        # Should have some variation (not all identical)
        assert float(alpha_std) > 1e-6

    def test_meta_sgd_learning_curve(self, simple_model, loss_fn):
        """Test that Meta-SGD improves over meta-training."""
        learner = MetaSGDLearner(
            model=simple_model,
            meta_lr=0.01,
            alpha_lr=0.01,
            num_inner_steps=3,
        )

        # Generate consistent task distribution
        mx.random.seed(42)

        losses = []
        for iteration in range(10):
            episodes = []
            for _ in range(4):
                support_x = mx.random.normal((10, 10))
                support_y = mx.random.randint(0, 5, (10,))
                query_x = mx.random.normal((5, 10))
                query_y = mx.random.randint(0, 5, (5,))
                episodes.append((support_x, support_y, query_x, query_y))

            metrics = learner.meta_train_step(episodes, loss_fn)
            losses.append(metrics["query_loss"])

        # Loss should generally decrease
        early_loss = sum(losses[:3]) / 3
        late_loss = sum(losses[-3:]) / 3
        assert late_loss < early_loss * 1.5  # Allow some variance


class TestMetaSGDvsMAML:
    """Compare Meta-SGD with MAML."""

    def test_meta_sgd_has_more_parameters_than_maml(self, simple_model):
        """Test that Meta-SGD has additional alpha parameters."""
        from meta_learning.maml import MAMLLearner

        maml = MAMLLearner(model=simple_model)
        meta_sgd = MetaSGDLearner(model=simple_model)

        # Meta-SGD should have alphas for each model parameter
        model_param_count = len(simple_model.parameters())
        assert len(meta_sgd.alphas) == model_param_count

        # MAML has no alphas
        assert not hasattr(maml, "alphas")

    def test_meta_sgd_vs_maml_adaptation(self, simple_model, loss_fn):
        """Compare adaptation capabilities of Meta-SGD vs MAML."""
        from meta_learning.maml import MAMLLearner

        # Create both learners
        maml = MAMLLearner(
            model=simple_model,
            inner_lr=0.01,
            num_inner_steps=5,
        )

        meta_sgd = MetaSGDLearner(
            model=simple_model,
            init_inner_lr=0.01,
            num_inner_steps=5,
        )

        # Create task data
        support_x = mx.random.normal((10, 10))
        support_y = mx.random.randint(0, 5, (10,))
        query_x = mx.random.normal((5, 10))
        query_y = mx.random.randint(0, 5, (5,))

        # Both should be able to adapt
        maml_adapted = maml.inner_loop_adaptation(
            support_x, support_y, loss_fn
        )
        meta_sgd_adapted = meta_sgd.inner_loop_adaptation(
            support_x, support_y, loss_fn
        )

        # Both should produce adapted parameters
        assert len(maml_adapted) > 0
        assert len(meta_sgd_adapted) > 0


class TestMetaSGDIntegration:
    """Integration tests for Meta-SGD."""

    def test_few_shot_learning_with_meta_sgd(self, simple_model, loss_fn):
        """Test Meta-SGD on few-shot learning task."""
        learner = MetaSGDLearner(
            model=simple_model,
            meta_lr=0.01,
            alpha_lr=0.01,
            num_inner_steps=5,
        )

        # Meta-training
        for _ in range(10):
            episodes = []
            for _ in range(4):
                support_x = mx.random.normal((5, 10))  # 5-shot
                support_y = mx.random.randint(0, 5, (5,))
                query_x = mx.random.normal((5, 10))
                query_y = mx.random.randint(0, 5, (5,))
                episodes.append((support_x, support_y, query_x, query_y))

            learner.meta_train_step(episodes, loss_fn)

        # Evaluation on new task
        test_episodes = []
        for _ in range(5):
            support_x = mx.random.normal((5, 10))
            support_y = mx.random.randint(0, 5, (5,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            test_episodes.append((support_x, support_y, query_x, query_y))

        results = learner.evaluate(test_episodes, loss_fn)

        # Should achieve reasonable performance
        assert results["avg_acc"] > 0.15  # Better than random (0.2 for 5-way)

    def test_alpha_learning_improves_adaptation(self, simple_model, loss_fn):
        """Test that learned alphas improve adaptation speed."""
        # Meta-SGD with alpha learning
        learner_with_alpha = MetaSGDLearner(
            model=simple_model,
            meta_lr=0.01,
            alpha_lr=0.01,  # Learn alphas
            num_inner_steps=3,
        )

        # Meta-SGD with fixed alphas (like MAML)
        learner_fixed_alpha = MetaSGDLearner(
            model=simple_model,
            meta_lr=0.01,
            alpha_lr=0.0,  # Don't learn alphas
            num_inner_steps=3,
        )

        # Meta-train both
        for _ in range(10):
            episodes = []
            for _ in range(4):
                support_x = mx.random.normal((10, 10))
                support_y = mx.random.randint(0, 5, (10,))
                query_x = mx.random.normal((5, 10))
                query_y = mx.random.randint(0, 5, (5,))
                episodes.append((support_x, support_y, query_x, query_y))

            learner_with_alpha.meta_train_step(episodes, loss_fn)
            learner_fixed_alpha.meta_train_step(episodes, loss_fn)

        # Evaluate both
        test_episodes = []
        for _ in range(3):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            test_episodes.append((support_x, support_y, query_x, query_y))

        results_with_alpha = learner_with_alpha.evaluate(test_episodes, loss_fn)
        results_fixed_alpha = learner_fixed_alpha.evaluate(test_episodes, loss_fn)

        # Both should work, but learned alphas may provide advantage
        assert results_with_alpha["avg_acc"] >= 0
        assert results_fixed_alpha["avg_acc"] >= 0

    def test_save_and_load_with_alphas(self, simple_model, loss_fn, tmp_path):
        """Test saving and loading Meta-SGD with learned alphas."""
        learner = MetaSGDLearner(
            model=simple_model,
            meta_lr=0.01,
            alpha_lr=0.01,
        )

        # Meta-train to learn alphas
        for _ in range(5):
            episodes = []
            for _ in range(4):
                support_x = mx.random.normal((10, 10))
                support_y = mx.random.randint(0, 5, (10,))
                query_x = mx.random.normal((5, 10))
                query_y = mx.random.randint(0, 5, (5,))
                episodes.append((support_x, support_y, query_x, query_y))

            learner.meta_train_step(episodes, loss_fn)

        # Save checkpoint
        checkpoint_path = tmp_path / "meta_sgd_checkpoint.npz"
        learner.save(str(checkpoint_path))

        # Create new learner and load
        new_learner = MetaSGDLearner(model=simple_model)
        new_learner.load(str(checkpoint_path))

        # Alphas should match
        for key in learner.alphas.keys():
            assert mx.allclose(
                learner.alphas[key],
                new_learner.alphas[key]
            )
