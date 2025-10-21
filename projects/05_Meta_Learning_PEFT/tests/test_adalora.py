"""Tests for AdaLoRA with adaptive rank allocation.

Tests AdaLoRALayer, AdaLoRAModel, and AdaLoRAMetaLearner from adalora.py.
"""

from __future__ import annotations

import pytest
import mlx.core as mx
import mlx.nn as nn

from adapter_generation.adalora import (
    AdaLoRALayer,
    AdaLoRAModel,
    AdaLoRAMetaLearner,
)


@pytest.fixture
def loss_fn():
    """Standard cross-entropy loss function."""
    def cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
        return nn.losses.cross_entropy(logits, labels, reduction="mean")
    return cross_entropy


class TestAdaLoRALayer:
    """Test AdaLoRA layer with SVD parameterization."""

    def test_initialization(self):
        """Test AdaLoRA layer initialization."""
        layer = AdaLoRALayer(
            in_features=64,
            out_features=128,
            rank=8,
            alpha=16.0,
        )

        assert layer.in_features == 64
        assert layer.out_features == 128
        assert layer.rank == 8
        assert layer.current_rank == 8

        # Verify SVD components
        assert layer.lora_U.shape == (128, 8)
        assert layer.lora_S.shape == (8,)
        assert layer.lora_V.shape == (8, 64)

    def test_forward_pass(self):
        """Test forward pass combines base and LoRA."""
        layer = AdaLoRALayer(
            in_features=64,
            out_features=128,
            rank=8,
            alpha=16.0,
        )

        x = mx.random.normal((10, 64))
        output = layer(x)

        assert output.shape == (10, 128)

    def test_compute_importance_scores(self):
        """Test importance score computation."""
        layer = AdaLoRALayer(
            in_features=64,
            out_features=128,
            rank=8,
        )

        importance = layer.compute_importance_scores()

        # Should have one score per rank
        assert importance.shape == (8,)

        # All scores should be non-negative
        assert mx.all(importance >= 0)

    def test_prune_low_rank_components(self):
        """Test pruning low-importance components."""
        layer = AdaLoRALayer(
            in_features=64,
            out_features=128,
            rank=8,
        )

        # Set known importance scores
        layer.lora_S = mx.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

        # Prune to rank 4 (keep top 4)
        layer.prune_low_rank_components(target_rank=4)

        assert layer.current_rank == 4
        assert layer.lora_U.shape == (128, 4)
        assert layer.lora_S.shape == (4,)
        assert layer.lora_V.shape == (4, 64)

        # Should keep highest importance scores
        assert mx.all(layer.lora_S >= 5.0)

    def test_adaptive_forward_after_pruning(self):
        """Test forward pass still works after pruning."""
        layer = AdaLoRALayer(
            in_features=64,
            out_features=128,
            rank=8,
        )

        x = mx.random.normal((10, 64))

        # Forward before pruning
        output_before = layer(x)

        # Prune
        layer.prune_low_rank_components(target_rank=4)

        # Forward after pruning
        output_after = layer(x)

        assert output_before.shape == output_after.shape

    def test_rank_reduction_decreases_parameters(self):
        """Test that pruning reduces parameter count."""
        layer = AdaLoRALayer(
            in_features=64,
            out_features=128,
            rank=8,
        )

        # Initial parameter count
        initial_params = (
            layer.lora_U.size +
            layer.lora_S.size +
            layer.lora_V.size
        )

        # Prune to half rank
        layer.prune_low_rank_components(target_rank=4)

        # New parameter count
        new_params = (
            layer.lora_U.size +
            layer.lora_S.size +
            layer.lora_V.size
        )

        # Should be roughly half
        assert new_params < initial_params


class TestAdaLoRAModel:
    """Test AdaLoRA model."""

    def test_initialization(self):
        """Test AdaLoRA model initialization."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
            lora_alpha=16.0,
        )

        assert model.input_dim == 10
        assert model.hidden_dim == 32
        assert model.output_dim == 5

        # Verify layers exist
        assert isinstance(model.layer1, AdaLoRALayer)
        assert isinstance(model.layer2, AdaLoRALayer)

    def test_forward_pass(self):
        """Test forward pass through model."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        x = mx.random.normal((10, 10))
        logits = model(x)

        assert logits.shape == (10, 5)

    def test_compute_importance_scores(self):
        """Test computing importance scores for all layers."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        importance_dict = model.compute_importance_scores()

        # Should have scores for both layers
        assert "layer1" in importance_dict
        assert "layer2" in importance_dict

        # Each should be a vector of size rank
        assert importance_dict["layer1"].shape == (8,)
        assert importance_dict["layer2"].shape == (8,)

    def test_prune_adapters(self):
        """Test pruning adapters across model."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        # Prune to rank 4
        model.prune_adapters(target_rank=4)

        # All layers should be pruned
        assert model.layer1.current_rank == 4
        assert model.layer2.current_rank == 4

    def test_get_current_ranks(self):
        """Test getting current ranks of all layers."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        ranks = model.get_current_ranks()

        assert ranks == {"layer1": 8, "layer2": 8}

        # After pruning
        model.prune_adapters(target_rank=4)
        ranks = model.get_current_ranks()

        assert ranks == {"layer1": 4, "layer2": 4}


class TestAdaLoRAMetaLearner:
    """Test AdaLoRA meta-learner."""

    def test_initialization(self, loss_fn):
        """Test AdaLoRA meta-learner initialization."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        learner = AdaLoRAMetaLearner(
            model=model,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5,
            target_rank=4,
            prune_every=10,
        )

        assert learner.model == model
        assert learner.inner_lr == 0.01
        assert learner.outer_lr == 0.001
        assert learner.target_rank == 4
        assert learner.prune_every == 10

    def test_inner_loop_adaptation(self, loss_fn):
        """Test inner loop adaptation."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        learner = AdaLoRAMetaLearner(model=model, num_inner_steps=3)

        # Create task data
        support_x = mx.random.normal((10, 10))
        support_y = mx.random.randint(0, 5, (10,))

        # Adapt
        adapted_params = learner.inner_loop_adaptation(
            support_x, support_y, loss_fn
        )

        # Should return adapted parameters
        assert len(adapted_params) > 0

    def test_meta_train_step(self, loss_fn):
        """Test meta-training step."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        learner = AdaLoRAMetaLearner(
            model=model,
            inner_lr=0.01,
            outer_lr=0.01,
            num_inner_steps=2,
        )

        # Create episodes
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Meta-train
        metrics = learner.meta_train_step(episodes, loss_fn)

        # Verify metrics
        assert "query_loss" in metrics
        assert "current_ranks" in metrics
        assert metrics["query_loss"] >= 0

    def test_automatic_pruning(self, loss_fn):
        """Test that automatic pruning happens on schedule."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        learner = AdaLoRAMetaLearner(
            model=model,
            target_rank=4,
            prune_every=3,  # Prune every 3 iterations
            num_inner_steps=2,
        )

        # Initial ranks
        initial_ranks = model.get_current_ranks()
        assert all(r == 8 for r in initial_ranks.values())

        # Create episodes
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Train for iterations without pruning
        for _ in range(2):
            learner.meta_train_step(episodes, loss_fn)

        # Ranks should still be 8
        ranks_before = model.get_current_ranks()
        assert all(r == 8 for r in ranks_before.values())

        # Train one more iteration (should trigger prune)
        learner.meta_train_step(episodes, loss_fn)

        # Ranks should be pruned to 4
        ranks_after = model.get_current_ranks()
        assert all(r == 4 for r in ranks_after.values())

    def test_evaluate(self, loss_fn):
        """Test evaluation on episodes."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        learner = AdaLoRAMetaLearner(model=model, num_inner_steps=3)

        # Create episodes
        episodes = []
        for _ in range(3):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Evaluate
        results = learner.evaluate(episodes, loss_fn)

        assert "avg_loss" in results
        assert "avg_acc" in results
        assert 0 <= results["avg_acc"] <= 1

    def test_importance_scores_updated_during_training(self, loss_fn):
        """Test that importance scores are updated during training."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        learner = AdaLoRAMetaLearner(model=model, num_inner_steps=2)

        # Initial importance scores
        initial_scores = model.compute_importance_scores()

        # Create episodes and train
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        for _ in range(5):
            learner.meta_train_step(episodes, loss_fn)

        # Importance scores should have changed
        current_scores = model.compute_importance_scores()

        # At least one layer should have different scores
        scores_changed = any(
            not mx.allclose(initial_scores[k], current_scores[k])
            for k in initial_scores.keys()
        )
        assert scores_changed


class TestAdaLoRAIntegration:
    """Integration tests for AdaLoRA."""

    def test_few_shot_learning_with_adalora(self, loss_fn):
        """Test AdaLoRA on few-shot learning task."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        learner = AdaLoRAMetaLearner(
            model=model,
            inner_lr=0.01,
            outer_lr=0.01,
            num_inner_steps=5,
            target_rank=4,
            prune_every=5,
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

        # Evaluation
        test_episodes = []
        for _ in range(5):
            support_x = mx.random.normal((5, 10))
            support_y = mx.random.randint(0, 5, (5,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            test_episodes.append((support_x, support_y, query_x, query_y))

        results = learner.evaluate(test_episodes, loss_fn)

        # Should achieve reasonable performance
        assert results["avg_acc"] > 0.15

    def test_adaptive_rank_reduces_parameters(self, loss_fn):
        """Test that adaptive ranking reduces parameter count."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        learner = AdaLoRAMetaLearner(
            model=model,
            target_rank=4,
            prune_every=1,
            num_inner_steps=2,
        )

        # Count initial parameters
        initial_param_count = sum(
            p.size for p in model.parameters().values()
        )

        # Create episodes and train
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Train with pruning
        learner.meta_train_step(episodes, loss_fn)

        # Count parameters after pruning
        final_param_count = sum(
            p.size for p in model.parameters().values()
        )

        # Should have fewer parameters
        assert final_param_count < initial_param_count

    def test_adalora_vs_lora_parameter_efficiency(self):
        """Compare parameter efficiency of AdaLoRA vs LoRA."""
        from adapter_generation.peft_integration import SimpleLoRAModel

        # LoRA with rank 8
        lora_model = SimpleLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        # AdaLoRA with rank 8, pruned to 4
        adalora_model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )
        adalora_model.prune_adapters(target_rank=4)

        # Count parameters
        lora_params = sum(p.size for p in lora_model.parameters().values())
        adalora_params = sum(p.size for p in adalora_model.parameters().values())

        # AdaLoRA should have fewer parameters after pruning
        assert adalora_params < lora_params

    def test_save_and_load_with_pruned_ranks(self, loss_fn, tmp_path):
        """Test saving and loading AdaLoRA with pruned ranks."""
        model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        learner = AdaLoRAMetaLearner(
            model=model,
            target_rank=4,
            prune_every=1,
        )

        # Create episodes and train (with pruning)
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        learner.meta_train_step(episodes, loss_fn)

        # Save ranks
        saved_ranks = model.get_current_ranks()

        # Save checkpoint
        checkpoint_path = tmp_path / "adalora_checkpoint.npz"
        learner.save(str(checkpoint_path))

        # Create new learner and load
        new_model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )
        new_learner = AdaLoRAMetaLearner(model=new_model)
        new_learner.load(str(checkpoint_path))

        # Ranks should match
        loaded_ranks = new_model.get_current_ranks()
        assert loaded_ranks == saved_ranks
