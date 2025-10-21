"""Tests for task-conditional adapter generation.

Tests LoRAHyperNetwork, TaskConditionalLoRALayer, TaskConditionalAdapterModel,
AdapterHyperparameterOptimizer, and auto_select_peft_method from task_conditional.py.
"""

from __future__ import annotations

import pytest
import mlx.core as mx
import mlx.nn as nn

from adapter_generation.task_conditional import (
    LoRAHyperNetwork,
    TaskConditionalLoRALayer,
    TaskConditionalAdapterModel,
    AdapterHyperparameterOptimizer,
    auto_select_peft_method,
)


class TestLoRAHyperNetwork:
    """Test HyperNetwork for generating LoRA parameters."""

    def test_initialization(self):
        """Test HyperNetwork initialization."""
        hyper_net = LoRAHyperNetwork(
            task_embedding_dim=64,
            in_features=128,
            out_features=256,
            rank=8,
        )

        assert hyper_net.task_embedding_dim == 64
        assert hyper_net.in_features == 128
        assert hyper_net.out_features == 256
        assert hyper_net.rank == 8

    def test_generate_lora_parameters(self):
        """Test generating LoRA A and B matrices."""
        hyper_net = LoRAHyperNetwork(
            task_embedding_dim=64,
            in_features=128,
            out_features=256,
            rank=8,
        )

        # Task embedding
        task_embedding = mx.random.normal((64,))

        # Generate LoRA parameters
        lora_A, lora_B = hyper_net(task_embedding)

        # Verify shapes
        assert lora_A.shape == (8, 128)
        assert lora_B.shape == (256, 8)

    def test_different_tasks_different_parameters(self):
        """Test different task embeddings produce different LoRA parameters."""
        hyper_net = LoRAHyperNetwork(
            task_embedding_dim=64,
            in_features=128,
            out_features=256,
            rank=8,
        )

        # Two different task embeddings
        task1 = mx.random.normal((64,))
        task2 = mx.random.normal((64,))

        lora_A1, lora_B1 = hyper_net(task1)
        lora_A2, lora_B2 = hyper_net(task2)

        # Should produce different parameters
        assert not mx.allclose(lora_A1, lora_A2)
        assert not mx.allclose(lora_B1, lora_B2)

    def test_same_task_same_parameters(self):
        """Test same task embedding produces same parameters."""
        hyper_net = LoRAHyperNetwork(
            task_embedding_dim=64,
            in_features=128,
            out_features=256,
            rank=8,
        )

        task = mx.random.normal((64,))

        lora_A1, lora_B1 = hyper_net(task)
        lora_A2, lora_B2 = hyper_net(task)

        # Should be identical
        assert mx.allclose(lora_A1, lora_A2)
        assert mx.allclose(lora_B1, lora_B2)


class TestTaskConditionalLoRALayer:
    """Test task-conditional LoRA layer."""

    def test_initialization(self):
        """Test task-conditional LoRA layer initialization."""
        layer = TaskConditionalLoRALayer(
            in_features=128,
            out_features=256,
            rank=8,
            alpha=16.0,
            task_embedding_dim=64,
        )

        assert layer.in_features == 128
        assert layer.out_features == 256
        assert layer.rank == 8
        assert isinstance(layer.hyper_network, LoRAHyperNetwork)

    def test_set_task_embedding(self):
        """Test setting task embedding."""
        layer = TaskConditionalLoRALayer(
            in_features=128,
            out_features=256,
            rank=8,
            task_embedding_dim=64,
        )

        # Initially no cached parameters
        assert layer._cached_lora_A is None
        assert layer._cached_lora_B is None

        # Set task embedding
        task_embedding = mx.random.normal((64,))
        layer.set_task_embedding(task_embedding)

        # Should cache parameters
        assert layer._cached_lora_A is not None
        assert layer._cached_lora_B is not None
        assert layer._cached_lora_A.shape == (8, 128)
        assert layer._cached_lora_B.shape == (256, 8)

    def test_forward_without_task_embedding(self):
        """Test forward pass without task embedding uses base model only."""
        layer = TaskConditionalLoRALayer(
            in_features=128,
            out_features=256,
            rank=8,
        )

        x = mx.random.normal((10, 128))
        output = layer(x)

        # Should produce output using base weight only
        assert output.shape == (10, 256)

    def test_forward_with_task_embedding(self):
        """Test forward pass with task embedding uses LoRA."""
        layer = TaskConditionalLoRALayer(
            in_features=128,
            out_features=256,
            rank=8,
        )

        # Set task
        task_embedding = mx.random.normal((64,))
        layer.set_task_embedding(task_embedding)

        x = mx.random.normal((10, 128))
        output = layer(x)

        # Should produce output using base + LoRA
        assert output.shape == (10, 256)

    def test_different_tasks_different_outputs(self):
        """Test different task embeddings produce different outputs."""
        layer = TaskConditionalLoRALayer(
            in_features=128,
            out_features=256,
            rank=8,
        )

        x = mx.random.normal((10, 128))

        # Task 1
        task1 = mx.random.normal((64,))
        layer.set_task_embedding(task1)
        output1 = layer(x)

        # Task 2
        task2 = mx.random.normal((64,))
        layer.set_task_embedding(task2)
        output2 = layer(x)

        # Should be different
        assert not mx.allclose(output1, output2)


class TestTaskConditionalAdapterModel:
    """Test complete task-conditional adapter model."""

    def test_initialization(self):
        """Test task-conditional adapter model initialization."""
        model = TaskConditionalAdapterModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
            task_embedding_dim=64,
        )

        assert model.input_dim == 10
        assert model.hidden_dim == 32
        assert model.output_dim == 5

        # Should have task-conditional layers
        assert isinstance(model.layer1, TaskConditionalLoRALayer)
        assert isinstance(model.layer2, TaskConditionalLoRALayer)

    def test_set_task_with_features(self):
        """Test setting task using task features."""
        model = TaskConditionalAdapterModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
            task_embedding_dim=64,
        )

        # Set task using features
        task_features = mx.random.normal((128,))
        model.set_task(task_features=task_features)

        # Layers should have cached LoRA parameters
        assert model.layer1._cached_lora_A is not None
        assert model.layer2._cached_lora_A is not None

    def test_set_task_with_embedding(self):
        """Test setting task using pre-computed embedding."""
        model = TaskConditionalAdapterModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
            task_embedding_dim=64,
        )

        # Set task using embedding directly
        task_embedding = mx.random.normal((64,))
        model.set_task(task_embedding=task_embedding)

        # Layers should have cached LoRA parameters
        assert model.layer1._cached_lora_A is not None
        assert model.layer2._cached_lora_A is not None

    def test_forward_pass(self):
        """Test forward pass through task-conditional model."""
        model = TaskConditionalAdapterModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        # Set task
        task_embedding = mx.random.normal((64,))
        model.set_task(task_embedding=task_embedding)

        # Forward pass
        x = mx.random.normal((10, 10))
        logits = model(x)

        assert logits.shape == (10, 5)

    def test_different_tasks_different_predictions(self):
        """Test different tasks produce different predictions."""
        model = TaskConditionalAdapterModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        x = mx.random.normal((10, 10))

        # Task 1
        task1 = mx.random.normal((64,))
        model.set_task(task_embedding=task1)
        logits1 = model(x)

        # Task 2
        task2 = mx.random.normal((64,))
        model.set_task(task_embedding=task2)
        logits2 = model(x)

        # Should produce different predictions
        assert not mx.allclose(logits1, logits2)


class TestAdapterHyperparameterOptimizer:
    """Test Bayesian optimizer for adapter hyperparameters."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = AdapterHyperparameterOptimizer()

        assert optimizer.param_ranges is not None
        assert "rank" in optimizer.param_ranges
        assert "alpha" in optimizer.param_ranges

    def test_initialization_with_custom_ranges(self):
        """Test initialization with custom parameter ranges."""
        custom_ranges = {
            "rank": (2, 16),
            "alpha": (4.0, 64.0),
        }

        optimizer = AdapterHyperparameterOptimizer(param_ranges=custom_ranges)

        assert optimizer.param_ranges == custom_ranges

    def test_suggest_hyperparameters_initial(self):
        """Test initial hyperparameter suggestion."""
        optimizer = AdapterHyperparameterOptimizer()

        params = optimizer.suggest_hyperparameters()

        # Should return default suggestions
        assert "rank" in params
        assert "alpha" in params
        assert isinstance(params["rank"], int)
        assert isinstance(params["alpha"], float)

    def test_record_trial(self):
        """Test recording trial results."""
        optimizer = AdapterHyperparameterOptimizer()

        params = {"rank": 8, "alpha": 16.0}
        score = 0.85
        metrics = {"loss": 0.3, "accuracy": 0.85}

        optimizer.record_trial(params, score, metrics)

        assert len(optimizer.trial_history) == 1
        assert optimizer.trial_history[0]["params"] == params
        assert optimizer.trial_history[0]["score"] == score

    def test_suggest_after_trials(self):
        """Test suggestions improve after trials."""
        optimizer = AdapterHyperparameterOptimizer()

        # Record some trials
        optimizer.record_trial(
            {"rank": 4, "alpha": 8.0}, score=0.70, metrics={}
        )
        optimizer.record_trial(
            {"rank": 8, "alpha": 16.0}, score=0.85, metrics={}
        )

        # Next suggestion should be based on best trial
        new_params = optimizer.suggest_hyperparameters()

        assert "rank" in new_params
        assert "alpha" in new_params

    def test_get_best_params(self):
        """Test getting best parameters."""
        optimizer = AdapterHyperparameterOptimizer()

        # Record trials
        optimizer.record_trial(
            {"rank": 4, "alpha": 8.0}, score=0.70, metrics={}
        )
        optimizer.record_trial(
            {"rank": 8, "alpha": 16.0}, score=0.85, metrics={}
        )
        optimizer.record_trial(
            {"rank": 6, "alpha": 12.0}, score=0.75, metrics={}
        )

        # Best should be rank=8, alpha=16.0
        best_params = optimizer.get_best_params()

        assert best_params["rank"] == 8
        assert best_params["alpha"] == 16.0


class TestAutoSelectPEFTMethod:
    """Test automatic PEFT method selection."""

    def test_select_for_very_small_dataset(self):
        """Test selection for very small dataset."""
        task_features = mx.random.normal((128,))
        dataset_size = 50

        method = auto_select_peft_method(task_features, dataset_size)

        # Should recommend prompt tuning
        assert method == "prompt_tuning"

    def test_select_for_small_dataset(self):
        """Test selection for small dataset."""
        task_features = mx.random.normal((128,))
        dataset_size = 500

        method = auto_select_peft_method(task_features, dataset_size)

        # Should recommend LoRA
        assert method == "lora"

    def test_select_for_large_dataset(self):
        """Test selection for large dataset."""
        task_features = mx.random.normal((128,))
        dataset_size = 10000

        method = auto_select_peft_method(task_features, dataset_size)

        # Should recommend LoRA
        assert method == "lora"

    def test_select_with_low_memory(self):
        """Test selection with memory constraint."""
        task_features = mx.random.normal((128,))
        dataset_size = 500
        available_memory = 4.0  # 4GB

        method = auto_select_peft_method(
            task_features, dataset_size, available_memory
        )

        # Should recommend AdaLoRA for low memory
        assert method == "adalora"

    def test_select_with_high_memory(self):
        """Test selection with high memory."""
        task_features = mx.random.normal((128,))
        dataset_size = 500
        available_memory = 16.0  # 16GB

        method = auto_select_peft_method(
            task_features, dataset_size, available_memory
        )

        # Should recommend LoRA
        assert method == "lora"


class TestTaskConditionalIntegration:
    """Integration tests for task-conditional adapters."""

    def test_end_to_end_task_conditional_adaptation(self):
        """Test complete task-conditional adaptation pipeline."""
        model = TaskConditionalAdapterModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
            task_embedding_dim=64,
        )

        # Task 1
        task1_embedding = mx.random.normal((64,))
        model.set_task(task_embedding=task1_embedding)

        task1_x = mx.random.normal((20, 10))
        task1_logits = model(task1_x)

        # Task 2
        task2_embedding = mx.random.normal((64,))
        model.set_task(task_embedding=task2_embedding)

        task2_x = mx.random.normal((20, 10))
        task2_logits = model(task2_x)

        # Should work for both tasks
        assert task1_logits.shape == (20, 5)
        assert task2_logits.shape == (20, 5)

    def test_hyperparameter_optimization_workflow(self):
        """Test hyperparameter optimization workflow."""
        optimizer = AdapterHyperparameterOptimizer()

        # Simulate multiple optimization trials
        for _ in range(5):
            # Get suggestion
            params = optimizer.suggest_hyperparameters()

            # Simulate training with these params
            # (in reality would train and evaluate)
            simulated_score = mx.random.uniform() * 0.5 + 0.5  # 0.5 to 1.0

            # Record result
            optimizer.record_trial(
                params,
                float(simulated_score),
                {"accuracy": float(simulated_score)},
            )

        # Get best parameters
        best_params = optimizer.get_best_params()

        assert "rank" in best_params
        assert "alpha" in best_params

    def test_automatic_method_selection_workflow(self):
        """Test automatic PEFT method selection workflow."""
        # Different scenarios
        scenarios = [
            (50, None, "prompt_tuning"),
            (500, None, "lora"),
            (10000, None, "lora"),
            (500, 4.0, "adalora"),
        ]

        for dataset_size, memory, expected_method in scenarios:
            task_features = mx.random.normal((128,))
            method = auto_select_peft_method(
                task_features, dataset_size, memory
            )

            assert method == expected_method

    def test_task_conditional_meta_learning(self):
        """Test task-conditional adapters in meta-learning setting."""
        model = TaskConditionalAdapterModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        # Simulate multiple tasks
        num_tasks = 5
        for _ in range(num_tasks):
            # Generate task embedding
            task_embedding = mx.random.normal((64,))
            model.set_task(task_embedding=task_embedding)

            # Task data
            x = mx.random.normal((10, 10))
            logits = model(x)

            # Should produce valid predictions
            assert logits.shape == (10, 5)

    def test_hypernetwork_parameter_sharing(self):
        """Test that HyperNetwork parameters are shared across calls."""
        hyper_net = LoRAHyperNetwork(
            task_embedding_dim=64,
            in_features=128,
            out_features=256,
            rank=8,
        )

        # Get initial parameter count
        initial_params = {
            k: v.copy() for k, v in hyper_net.parameters().items()
        }

        # Generate parameters for multiple tasks
        for _ in range(5):
            task = mx.random.normal((64,))
            lora_A, lora_B = hyper_net(task)

        # HyperNetwork parameters should remain same
        final_params = hyper_net.parameters()

        for key in initial_params.keys():
            assert mx.allclose(initial_params[key], final_params[key])
