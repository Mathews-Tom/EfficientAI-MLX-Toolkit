"""Integration tests for meta-learning + PEFT system.

This module tests end-to-end workflows combining meta-learning algorithms
(MAML, Reptile) with PEFT methods (LoRA) and task distribution systems.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn

from adapter_generation import (
    AdapterFactory,
    LoRAMetaLearner,
    PEFTConfig,
    PEFTMethod,
)
from meta_learning.maml import MAMLLearner
from meta_learning.reptile import ReptileLearner
from meta_learning.models import SimpleClassifier
from utils.baseline import BaselineEvaluator


@pytest.fixture
def loss_fn():
    """Standard cross-entropy loss function."""
    def cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
        return nn.losses.cross_entropy(logits, labels, reduction="mean")
    return cross_entropy


@pytest.fixture
def regression_loss_fn():
    """Mean squared error for regression tasks."""
    def mse_loss(predictions: mx.array, targets: mx.array) -> mx.array:
        return mx.mean((predictions - targets) ** 2)
    return mse_loss


class TestEndToEndWorkflow:
    """Test complete end-to-end meta-learning workflows."""

    def test_maml_lora_few_shot_learning(self, loss_fn):
        """Test complete MAML + LoRA few-shot learning pipeline."""
        # Create LoRA meta-learner
        learner = AdapterFactory.create_lora_meta_learner(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=4,
            inner_lr=0.05,
            outer_lr=0.01,
        )

        # Generate synthetic tasks
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Meta-training loop
        initial_acc = None
        for step in range(5):
            metrics = learner.meta_train_step(episodes, loss_fn)

            if initial_acc is None:
                initial_acc = metrics["query_accuracy"]

            # Validate metrics structure
            assert "meta_loss" in metrics
            assert "query_accuracy" in metrics
            assert 0 <= metrics["query_accuracy"] <= 1

        # Evaluate few-shot adaptation
        results = learner.evaluate(
            episodes[:2],
            loss_fn,
            num_adaptation_steps=[0, 5],
        )

        # Verify evaluation metrics
        assert "step_0_loss" in results
        assert "step_5_loss" in results
        assert 0 <= results["step_0_acc"] <= 1
        assert 0 <= results["step_5_acc"] <= 1

    def test_maml_vs_reptile_comparison(self, loss_fn):
        """Test and compare MAML vs Reptile on same task distribution."""
        # Shared model architecture
        def create_model():
            return SimpleClassifier(
                input_dim=10,
                hidden_dim=32,
                num_classes=5,
                num_layers=2,
            )

        # Create learners
        maml_learner = MAMLLearner(
            model=create_model(),
            inner_lr=0.05,
            outer_lr=0.01,
            num_inner_steps=3,
        )

        reptile_learner = ReptileLearner(
            model=create_model(),
            inner_lr=0.05,
            outer_lr=0.01,
            num_inner_steps=3,
        )

        # Generate tasks
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Train both learners
        maml_metrics = maml_learner.meta_train_step(episodes, loss_fn)
        reptile_metrics = reptile_learner.meta_train_step(episodes, loss_fn)

        # Both should produce valid metrics
        assert maml_metrics["query_accuracy"] >= 0
        assert reptile_metrics["query_accuracy"] >= 0

        # Evaluate both
        maml_results = maml_learner.evaluate(episodes[:2], loss_fn, [0, 3])
        reptile_results = reptile_learner.evaluate(episodes[:2], loss_fn, [0, 3])

        # Both should adapt (step_3 should be different from step_0)
        assert maml_results["step_0_acc"] != maml_results["step_3_acc"] or \
               maml_results["step_0_loss"] != maml_results["step_3_loss"]
        assert reptile_results["step_0_acc"] != reptile_results["step_3_acc"] or \
               reptile_results["step_0_loss"] != reptile_results["step_3_loss"]

    def test_baseline_vs_metalearning_comparison(self, loss_fn):
        """Test baseline fine-tuning vs meta-learning approaches."""
        # Create baseline evaluator
        def model_factory():
            return SimpleClassifier(10, 32, 5, num_layers=2)

        baseline_eval = BaselineEvaluator(
            model_factory=model_factory,
            learning_rate=0.05,
            num_steps=5,
        )

        # Create meta-learner
        meta_learner = MAMLLearner(
            model=model_factory(),
            inner_lr=0.05,
            outer_lr=0.01,
            num_inner_steps=3,
        )

        # Generate tasks
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Train meta-learner
        for _ in range(3):
            meta_learner.meta_train_step(episodes, loss_fn)

        # Evaluate on new task
        test_episode = episodes[0]
        support_x, support_y, query_x, query_y = test_episode

        # Baseline: direct fine-tuning
        baseline_metrics = baseline_eval.evaluate_task(
            support_x, support_y, query_x, query_y, loss_fn
        )

        # Meta-learner: adapted from meta-learned init
        meta_results = meta_learner.evaluate([test_episode], loss_fn, [5])

        # Both should produce valid results
        assert 0 <= baseline_metrics["final_query_acc"] <= 1
        assert 0 <= meta_results["step_5_acc"] <= 1


class TestTaskDistributionIntegration:
    """Test integration with different task types and data distributions."""

    def test_regression_task_with_maml(self, regression_loss_fn):
        """Test MAML with regression tasks (e.g., sinusoid approximation)."""
        # Create model for regression
        model = SimpleClassifier(
            input_dim=1,
            hidden_dim=64,
            num_classes=1,  # Regression output
            num_layers=3,
        )

        learner = MAMLLearner(
            model=model,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5,
        )

        # Generate synthetic regression episodes
        episodes = []
        for _ in range(4):
            # Random linear function for each task
            support_x = mx.random.normal((10, 1))
            support_y = support_x * 2 + mx.random.normal((10, 1)) * 0.1  # y = 2x + noise
            query_x = mx.random.normal((5, 1))
            query_y = query_x * 2 + mx.random.normal((5, 1)) * 0.1
            episodes.append((support_x, support_y, query_x, query_y))

        # Train
        metrics = learner.meta_train_step(episodes, regression_loss_fn)

        # Verify metrics
        assert "meta_loss" in metrics
        assert "query_accuracy" in metrics
        assert metrics["meta_loss"] > 0

    @pytest.mark.slow
    def test_high_dimensional_task_with_lora(self, loss_fn):
        """Test LoRA meta-learner with high-dimensional image-like tasks."""
        # Create LoRA meta-learner with image dimensions
        learner = AdapterFactory.create_lora_meta_learner(
            input_dim=28 * 28,  # Flattened 28x28 image
            hidden_dim=128,
            output_dim=5,
            lora_rank=8,
            inner_lr=0.01,
            outer_lr=0.001,
        )

        # Generate synthetic image-like episodes (5-way 5-shot)
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((25, 28 * 28))  # 5 classes × 5 shots
            support_y = mx.array([i // 5 for i in range(25)])  # 5 classes
            query_x = mx.random.normal((15, 28 * 28))  # 3 query per class
            query_y = mx.array([i // 3 for i in range(15)])
            episodes.append((support_x, support_y, query_x, query_y))

        # Train
        metrics = learner.meta_train_step(episodes, loss_fn)

        # Verify training works
        assert "meta_loss" in metrics
        assert 0 <= metrics["query_accuracy"] <= 1


class TestSaveLoadIntegration:
    """Test save/load functionality for meta-learned models."""

    def test_maml_save_and_load(self, loss_fn, tmp_path):
        """Test saving and loading MAML meta-learned parameters."""
        # Create and train learner
        learner = MAMLLearner(
            model=SimpleClassifier(10, 32, 5, num_layers=2),
            inner_lr=0.05,
            outer_lr=0.01,
        )

        # Generate and train on episodes
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        # Train for a few steps
        for _ in range(3):
            learner.meta_train_step(episodes, loss_fn)

        # Evaluate before save
        results_before = learner.evaluate(episodes[:2], loss_fn, [0, 1])

        # Save parameters
        save_path = tmp_path / "maml_params.npz"
        learner.save(str(save_path))

        # Create new learner and load
        new_learner = MAMLLearner(
            model=SimpleClassifier(10, 32, 5, num_layers=2),
            inner_lr=0.05,
            outer_lr=0.01,
        )
        new_learner.load(str(save_path))

        # Evaluate after load
        results_after = new_learner.evaluate(episodes[:2], loss_fn, [0, 1])

        # Results should be very similar (allow small numerical differences)
        assert abs(results_before["step_0_loss"] - results_after["step_0_loss"]) < 0.1
        assert abs(results_before["step_1_loss"] - results_after["step_1_loss"]) < 0.1

    def test_lora_metalearner_parameter_persistence(self, loss_fn, tmp_path):
        """Test that LoRA parameters persist correctly through save/load."""
        # Create LoRA meta-learner
        learner = AdapterFactory.create_lora_meta_learner(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=4,
            inner_lr=0.05,
            outer_lr=0.01,
        )

        # Get initial parameter counts
        param_counts_before = learner.count_trainable_parameters()

        # Generate and train
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        for _ in range(3):
            learner.meta_train_step(episodes, loss_fn)

        # Save
        save_path = tmp_path / "lora_params.npz"
        learner.save(str(save_path))

        # Load into new learner
        new_learner = AdapterFactory.create_lora_meta_learner(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=4,
            inner_lr=0.05,
            outer_lr=0.01,
        )
        new_learner.load(str(save_path))

        # Verify parameter counts match
        param_counts_after = new_learner.count_trainable_parameters()
        assert param_counts_before["adapter"] == param_counts_after["adapter"]
        assert param_counts_before["total"] == param_counts_after["total"]


class TestPEFTMethodSelection:
    """Test adapter factory and PEFT method selection."""

    def test_factory_creates_correct_learner_type(self):
        """Test that factory creates appropriate learner for each method."""
        # Test LoRA method
        lora_learner = AdapterFactory.create_meta_learner(
            method=PEFTMethod.LORA,
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=4,
        )
        assert isinstance(lora_learner, LoRAMetaLearner)
        assert lora_learner.peft_config.method == "lora"

        # Test string-based method selection
        lora_learner_str = AdapterFactory.create_meta_learner(
            method="lora",
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=4,
        )
        assert isinstance(lora_learner_str, LoRAMetaLearner)

    def test_factory_validates_unsupported_methods(self):
        """Test that factory raises errors for unsupported methods."""
        with pytest.raises(ValueError, match="Unsupported PEFT method"):
            AdapterFactory.create_meta_learner(
                method="invalid_method",
                input_dim=10,
                hidden_dim=32,
                output_dim=5,
            )

    def test_factory_lists_supported_methods(self):
        """Test that factory correctly lists all supported PEFT methods."""
        methods = AdapterFactory.list_supported_methods()
        assert "lora" in methods
        assert isinstance(methods, list)
        assert len(methods) == 5  # All PEFTMethod enum values

    def test_unimplemented_methods_raise_not_implemented(self):
        """Test that unimplemented PEFT methods raise NotImplementedError."""
        for method in [
            PEFTMethod.ADALORA,
            PEFTMethod.PROMPT_TUNING,
            PEFTMethod.PREFIX_TUNING,
            PEFTMethod.P_TUNING,
        ]:
            with pytest.raises(NotImplementedError):
                AdapterFactory.create_meta_learner(
                    method=method,
                    input_dim=10,
                    hidden_dim=32,
                    output_dim=5,
                )


class TestParameterEfficiency:
    """Test parameter efficiency and reduction ratios."""

    def test_lora_reduces_parameters_significantly(self):
        """Test that LoRA achieves significant parameter reduction."""
        # Create large model to see clear reduction
        learner = AdapterFactory.create_lora_meta_learner(
            input_dim=100,
            hidden_dim=512,
            output_dim=10,
            lora_rank=8,
        )

        param_counts = learner.count_trainable_parameters()

        # LoRA should reduce parameters significantly
        assert param_counts["reduction_ratio"] >= 5
        assert param_counts["trainable"] < param_counts["total"] / 3
        assert param_counts["adapter"] == param_counts["trainable"]

    def test_parameter_counts_consistent_across_operations(self, loss_fn):
        """Test that parameter counts remain consistent during training."""
        learner = AdapterFactory.create_lora_meta_learner(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=4,
        )

        # Get initial counts
        initial_counts = learner.count_trainable_parameters()

        # Train for a few steps
        episodes = []
        for _ in range(4):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            episodes.append((support_x, support_y, query_x, query_y))

        for _ in range(3):
            learner.meta_train_step(episodes, loss_fn)

        # Get counts after training
        final_counts = learner.count_trainable_parameters()

        # Counts should remain the same
        assert initial_counts["total"] == final_counts["total"]
        assert initial_counts["adapter"] == final_counts["adapter"]
        assert initial_counts["trainable"] == final_counts["trainable"]


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for complete training and evaluation pipelines."""

    def test_complete_meta_learning_pipeline(self, loss_fn):
        """Test complete meta-learning pipeline from creation to evaluation."""
        # 1. Create learner
        learner = AdapterFactory.create_lora_meta_learner(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=4,
            inner_lr=0.05,
            outer_lr=0.01,
        )

        # 2. Generate tasks
        train_episodes = []
        for _ in range(8):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            train_episodes.append((support_x, support_y, query_x, query_y))

        val_episodes = train_episodes[:2]

        # 3. Meta-training loop
        training_history = []
        for epoch in range(5):
            metrics = learner.meta_train_step(train_episodes, loss_fn)
            training_history.append(metrics)

        # 4. Evaluation at different adaptation steps
        eval_results = learner.evaluate(
            val_episodes,
            loss_fn,
            num_adaptation_steps=[0, 1, 3, 5],
        )

        # 5. Validate results
        assert len(training_history) == 5
        for metrics in training_history:
            assert "meta_loss" in metrics
            assert "query_accuracy" in metrics

        # Verify evaluation at all adaptation steps
        for k in [0, 1, 3, 5]:
            assert f"step_{k}_loss" in eval_results
            assert f"step_{k}_acc" in eval_results
            assert 0 <= eval_results[f"step_{k}_acc"] <= 1

        # Verify we have some valid loss values
        assert eval_results["step_0_loss"] > 0
        assert eval_results["step_5_loss"] > 0
