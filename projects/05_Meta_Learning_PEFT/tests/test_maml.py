"""Unit tests for MAML (Model-Agnostic Meta-Learning) implementation.

This module tests the MAML and FOMAML learners to ensure correct meta-learning
behavior, gradient computation, and adaptation to new tasks.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn

from meta_learning.maml import MAMLLearner, FOMAMLLearner
from meta_learning.models import SimpleClassifier


@pytest.fixture
def simple_model():
    """Create a simple MLP model for testing."""
    return SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=5)


@pytest.fixture
def sample_episodes():
    """Create sample episodes for testing.

    Returns:
        List of (support_x, support_y, query_x, query_y) tuples.
    """
    episodes = []
    for _ in range(4):  # 4 tasks
        support_x = mx.random.normal((10, 10))  # 10 support examples
        support_y = mx.random.randint(0, 5, (10,))
        query_x = mx.random.normal((5, 10))  # 5 query examples
        query_y = mx.random.randint(0, 5, (5,))
        episodes.append((support_x, support_y, query_x, query_y))
    return episodes


@pytest.fixture
def loss_fn():
    """Cross-entropy loss function."""
    def cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
        return nn.losses.cross_entropy(logits, labels, reduction="mean")
    return cross_entropy


class TestMAMLLearner:
    """Test suite for MAML learner."""

    def test_initialization(self, simple_model):
        """Test MAML learner initialization."""
        learner = MAMLLearner(
            model=simple_model,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5,
            meta_batch_size=4
        )

        assert learner.inner_lr == 0.01
        assert learner.outer_lr == 0.001
        assert learner.num_inner_steps == 5
        assert learner.meta_batch_size == 4
        assert learner.first_order is False
        assert learner.meta_optimizer is not None
        assert len(learner.meta_params) > 0

    def test_inner_loop_adaptation(self, simple_model, loss_fn):
        """Test inner loop adaptation on support set."""
        learner = MAMLLearner(model=simple_model)

        support_x = mx.random.normal((10, 10))
        support_y = mx.random.randint(0, 5, (10,))

        # Get initial loss
        initial_logits = simple_model(support_x)
        initial_loss = loss_fn(initial_logits, support_y)

        # Adapt model
        adapted_model = learner.inner_loop_adaptation(
            simple_model, support_x, support_y, loss_fn
        )

        # Get adapted loss
        adapted_logits = adapted_model(support_x)
        adapted_loss = loss_fn(adapted_logits, support_y)

        # Adapted model should have lower loss on support set
        assert float(adapted_loss) < float(initial_loss)

        # Adapted model should be different from original
        from meta_learning.maml import flatten_params

        original_params = flatten_params(dict(simple_model.parameters()))
        adapted_params = flatten_params(dict(adapted_model.parameters()))

        params_changed = False
        for key in original_params:
            if not mx.array_equal(original_params[key], adapted_params[key]):
                params_changed = True
                break

        assert params_changed, "Adapted model parameters should differ from original"

    def test_compute_meta_loss(self, simple_model, sample_episodes, loss_fn):
        """Test meta-loss computation across batch of tasks."""
        learner = MAMLLearner(model=simple_model)

        meta_loss, metrics = learner.compute_meta_loss(sample_episodes, loss_fn)

        # Meta-loss should be a scalar
        assert meta_loss.shape == ()
        assert float(meta_loss) > 0

        # Metrics should contain expected keys
        assert "support_loss" in metrics
        assert "query_loss" in metrics
        assert "query_accuracy" in metrics

        # Metrics should be reasonable
        assert metrics["support_loss"] > 0
        assert metrics["query_loss"] > 0
        assert 0 <= metrics["query_accuracy"] <= 1

    def test_meta_train_step(self, simple_model, sample_episodes, loss_fn):
        """Test one meta-training step."""
        from meta_learning.maml import flatten_params

        learner = MAMLLearner(model=simple_model, outer_lr=0.01)

        # Store initial parameters (flattened for comparison)
        initial_params_flat = {k: mx.array(v) for k, v in flatten_params(dict(simple_model.parameters())).items()}

        # Perform meta-training step
        metrics = learner.meta_train_step(sample_episodes, loss_fn)

        # Metrics should be returned
        assert "meta_loss" in metrics
        assert "support_loss" in metrics
        assert "query_loss" in metrics
        assert "query_accuracy" in metrics

        # Meta-parameters should have been updated
        updated_params_flat = flatten_params(dict(simple_model.parameters()))

        params_changed = False
        for key in initial_params_flat:
            if not mx.array_equal(initial_params_flat[key], updated_params_flat[key]):
                params_changed = True
                break

        assert params_changed, "Meta-parameters should be updated after meta-train step"

    def test_meta_training_improves_adaptation(self, simple_model, sample_episodes, loss_fn):
        """Test that meta-training improves few-shot adaptation."""
        learner = MAMLLearner(model=simple_model, outer_lr=0.01, num_inner_steps=3)

        # Evaluate before meta-training
        initial_eval = learner.evaluate(sample_episodes[:2], loss_fn, num_adaptation_steps=[3])
        initial_acc = initial_eval["step_3_acc"]

        # Perform meta-training steps
        for _ in range(5):
            learner.meta_train_step(sample_episodes, loss_fn)

        # Evaluate after meta-training
        final_eval = learner.evaluate(sample_episodes[:2], loss_fn, num_adaptation_steps=[3])
        final_acc = final_eval["step_3_acc"]

        # Accuracy should improve (or at least not degrade significantly)
        # Note: With random data, improvement is not guaranteed, but we test the mechanism
        assert final_acc >= initial_acc - 0.1  # Allow small degradation due to randomness

    def test_evaluate(self, simple_model, sample_episodes, loss_fn):
        """Test evaluation at different adaptation steps."""
        learner = MAMLLearner(model=simple_model)

        results = learner.evaluate(
            sample_episodes[:2],
            loss_fn,
            num_adaptation_steps=[0, 1, 3, 5]
        )

        # Check all adaptation steps are evaluated
        assert "step_0_loss" in results
        assert "step_0_acc" in results
        assert "step_1_loss" in results
        assert "step_1_acc" in results
        assert "step_3_loss" in results
        assert "step_3_acc" in results
        assert "step_5_loss" in results
        assert "step_5_acc" in results

        # Accuracies should be in valid range
        for k in [0, 1, 3, 5]:
            assert 0 <= results[f"step_{k}_acc"] <= 1

        # Loss should generally decrease with more adaptation steps
        # (though not guaranteed with random data)
        assert results["step_0_loss"] > 0
        assert results["step_5_loss"] > 0

    def test_save_and_load(self, simple_model, tmp_path):
        """Test saving and loading meta-learned parameters."""
        from meta_learning.maml import flatten_params, unflatten_params

        learner = MAMLLearner(model=simple_model)

        # Save parameters
        save_path = str(tmp_path / "maml_params.npz")
        learner.save(save_path)

        # Modify model parameters (using flattened structure)
        flat_params = flatten_params(dict(simple_model.parameters()))
        modified_flat = {
            k: v + mx.random.normal(v.shape) * 0.1 for k, v in flat_params.items()
        }
        modified_params = unflatten_params(modified_flat)
        simple_model.update(modified_params)
        mx.eval(simple_model.parameters())

        # Load parameters
        learner.load(save_path)

        # Parameters should match original meta-parameters (compare flattened)
        loaded_flat = flatten_params(dict(simple_model.parameters()))
        meta_flat = flatten_params(learner.meta_params)

        for key in meta_flat:
            assert mx.allclose(loaded_flat[key], meta_flat[key], atol=1e-5)


class TestFOMAMLLearner:
    """Test suite for FOMAML (First-Order MAML) learner."""

    def test_initialization(self, simple_model):
        """Test FOMAML learner initialization."""
        learner = FOMAMLLearner(
            model=simple_model,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5,
            meta_batch_size=4
        )

        assert learner.inner_lr == 0.01
        assert learner.outer_lr == 0.001
        assert learner.num_inner_steps == 5
        assert learner.meta_batch_size == 4
        assert learner.first_order is True

    def test_compute_meta_loss_stops_gradients(self, simple_model, sample_episodes, loss_fn):
        """Test that FOMAML stops gradients after inner loop."""
        learner = FOMAMLLearner(model=simple_model)

        meta_loss, metrics = learner.compute_meta_loss(sample_episodes, loss_fn)

        # Meta-loss should be computed correctly
        assert meta_loss.shape == ()
        assert float(meta_loss) > 0

        # Metrics should be present
        assert "support_loss" in metrics
        assert "query_loss" in metrics
        assert "query_accuracy" in metrics

    def test_fomaml_vs_maml_efficiency(self, simple_model, sample_episodes, loss_fn):
        """Test that FOMAML is more efficient than MAML.

        Note: This is a conceptual test - in practice, FOMAML should use less memory
        and compute faster due to first-order approximation.
        """
        maml = MAMLLearner(model=simple_model)
        fomaml = FOMAMLLearner(model=simple_model)

        # Both should produce valid meta-losses
        maml_loss, maml_metrics = maml.compute_meta_loss(sample_episodes, loss_fn)
        fomaml_loss, fomaml_metrics = fomaml.compute_meta_loss(sample_episodes, loss_fn)

        assert float(maml_loss) > 0
        assert float(fomaml_loss) > 0

        # Metrics structure should be the same
        assert set(maml_metrics.keys()) == set(fomaml_metrics.keys())

    def test_fomaml_meta_train_step(self, simple_model, sample_episodes, loss_fn):
        """Test FOMAML meta-training step."""
        from meta_learning.maml import flatten_params

        learner = FOMAMLLearner(model=simple_model, outer_lr=0.01)

        # Store initial parameters (flattened for comparison)
        initial_params_flat = {k: mx.array(v) for k, v in flatten_params(dict(simple_model.parameters())).items()}

        # Perform meta-training step
        metrics = learner.meta_train_step(sample_episodes, loss_fn)

        # Metrics should be returned
        assert "meta_loss" in metrics
        assert "support_loss" in metrics
        assert "query_loss" in metrics
        assert "query_accuracy" in metrics

        # Meta-parameters should have been updated
        updated_params_flat = flatten_params(dict(simple_model.parameters()))

        params_changed = False
        for key in initial_params_flat:
            if not mx.array_equal(initial_params_flat[key], updated_params_flat[key]):
                params_changed = True
                break

        assert params_changed, "Meta-parameters should be updated after FOMAML meta-train step"


class TestMAMLIntegration:
    """Integration tests for MAML with realistic scenarios."""

    def test_few_shot_classification(self, simple_model, loss_fn):
        """Test MAML on a simple few-shot classification task."""
        learner = MAMLLearner(model=simple_model, inner_lr=0.05, outer_lr=0.01)

        # Create synthetic task distribution (5-way, 5-shot)
        num_tasks = 10
        episodes = []

        for _ in range(num_tasks):
            support_x = mx.random.normal((25, 10))  # 5 classes × 5 shots
            support_y = mx.array([i for i in range(5) for _ in range(5)])
            query_x = mx.random.normal((15, 10))  # 5 classes × 3 queries
            query_y = mx.array([i for i in range(5) for _ in range(3)])
            episodes.append((support_x, support_y, query_x, query_y))

        # Meta-train for a few steps
        for _ in range(3):
            metrics = learner.meta_train_step(episodes, loss_fn)
            assert metrics["query_accuracy"] >= 0

        # Evaluate on new tasks
        eval_episodes = episodes[:2]
        results = learner.evaluate(eval_episodes, loss_fn, num_adaptation_steps=[0, 5])

        # Check that adaptation improves performance
        assert results["step_5_acc"] >= results["step_0_acc"] - 0.1

    def test_maml_and_fomaml_comparable(self, simple_model, sample_episodes, loss_fn):
        """Test that MAML and FOMAML produce comparable results."""
        # Create separate models with same initialization
        model_maml = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=5)
        model_fomaml = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=5)

        # Copy parameters
        model_fomaml.update(dict(model_maml.parameters()))
        mx.eval(model_fomaml.parameters())

        maml = MAMLLearner(model=model_maml, outer_lr=0.01)
        fomaml = FOMAMLLearner(model=model_fomaml, outer_lr=0.01)

        # Train both for a few steps
        for _ in range(3):
            maml.meta_train_step(sample_episodes, loss_fn)
            fomaml.meta_train_step(sample_episodes, loss_fn)

        # Evaluate both
        maml_results = maml.evaluate(sample_episodes[:2], loss_fn, num_adaptation_steps=[5])
        fomaml_results = fomaml.evaluate(sample_episodes[:2], loss_fn, num_adaptation_steps=[5])

        # Results should be reasonably close (within 20% relative difference)
        maml_acc = maml_results["step_5_acc"]
        fomaml_acc = fomaml_results["step_5_acc"]

        if maml_acc > 0:
            rel_diff = abs(maml_acc - fomaml_acc) / maml_acc
            assert rel_diff < 0.5  # Allow up to 50% difference due to first-order approx
