"""Tests for Prefix Tuning and Prompt Tuning.

Tests PrefixTuningLayer, PromptTuningLayer, PromptTuningModel,
and PromptTuningMetaLearner from prefix_prompt_tuning.py.
"""

from __future__ import annotations

import pytest
import mlx.core as mx
import mlx.nn as nn

from adapter_generation.prefix_prompt_tuning import (
    PrefixTuningLayer,
    PromptTuningLayer,
    PromptTuningModel,
    PromptTuningMetaLearner,
    compare_peft_methods,
)


@pytest.fixture
def loss_fn():
    """Standard cross-entropy loss function."""
    def cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
        return nn.losses.cross_entropy(logits, labels, reduction="mean")
    return cross_entropy


class TestPrefixTuningLayer:
    """Test Prefix Tuning layer."""

    def test_initialization(self):
        """Test Prefix Tuning initialization."""
        layer = PrefixTuningLayer(
            prefix_length=10,
            hidden_dim=128,
            num_heads=4,
        )

        assert layer.prefix_length == 10
        assert layer.hidden_dim == 128
        assert layer.num_heads == 4

    def test_get_prefix_kv(self):
        """Test getting prefix key and value."""
        layer = PrefixTuningLayer(
            prefix_length=10,
            hidden_dim=128,
        )

        prefix_key, prefix_value = layer.get_prefix_kv()

        assert prefix_key.shape == (10, 128)
        assert prefix_value.shape == (10, 128)

    def test_forward_extends_sequence(self):
        """Test that prefix tuning extends sequence length."""
        layer = PrefixTuningLayer(
            prefix_length=10,
            hidden_dim=128,
        )

        batch_size = 4
        seq_len = 20

        query = mx.random.normal((batch_size, seq_len, 128))
        key = mx.random.normal((batch_size, seq_len, 128))
        value = mx.random.normal((batch_size, seq_len, 128))

        output = layer(query, key, value)

        # Output should have same shape as query
        assert output.shape == (batch_size, seq_len, 128)

    def test_reparameterization(self):
        """Test prefix reparameterization through bottleneck."""
        layer = PrefixTuningLayer(
            prefix_length=10,
            hidden_dim=128,
        )

        # With reparameterization
        assert layer.use_reparameterization

        # Should have bottleneck parameters
        assert hasattr(layer, "prefix_params")
        assert hasattr(layer, "prefix_key_proj")
        assert hasattr(layer, "prefix_value_proj")


class TestPromptTuningLayer:
    """Test Prompt Tuning layer."""

    def test_initialization(self):
        """Test Prompt Tuning initialization."""
        layer = PromptTuningLayer(
            num_prompts=10,
            hidden_dim=128,
        )

        assert layer.num_prompts == 10
        assert layer.hidden_dim == 128
        assert layer.prompt_embeddings.shape == (10, 128)

    def test_forward_prepends_prompts(self):
        """Test that prompts are prepended to input."""
        layer = PromptTuningLayer(
            num_prompts=10,
            hidden_dim=128,
        )

        batch_size = 4
        seq_len = 20
        x = mx.random.normal((batch_size, seq_len, 128))

        output = layer(x)

        # Sequence length should increase by num_prompts
        assert output.shape == (batch_size, seq_len + 10, 128)

    def test_different_inputs_same_prompts(self):
        """Test that same prompts are used for different inputs."""
        layer = PromptTuningLayer(
            num_prompts=10,
            hidden_dim=128,
        )

        x1 = mx.random.normal((4, 20, 128))
        x2 = mx.random.normal((4, 20, 128))

        output1 = layer(x1)
        output2 = layer(x2)

        # Prompts (first 10 tokens) should be identical
        assert mx.allclose(
            output1[:, :10, :],
            output2[:, :10, :]
        )


class TestPromptTuningModel:
    """Test Prompt Tuning model."""

    def test_initialization(self):
        """Test Prompt Tuning model initialization."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        assert model.input_dim == 10
        assert model.hidden_dim == 32
        assert model.output_dim == 5
        assert isinstance(model.prompt_layer, PromptTuningLayer)

    def test_forward_pass_2d_input(self):
        """Test forward pass with 2D input."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        # 2D input (batch_size, input_dim)
        x = mx.random.normal((4, 10))
        logits = model(x)

        assert logits.shape == (4, 5)

    def test_forward_pass_3d_input(self):
        """Test forward pass with 3D input."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        # 3D input (batch_size, seq_len, input_dim)
        x = mx.random.normal((4, 5, 10))
        logits = model(x)

        assert logits.shape == (4, 5)

    def test_get_prompt_parameters(self):
        """Test getting prompt parameters."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        params = model.get_prompt_parameters()

        assert "prompts" in params
        assert params["prompts"].shape == (10, 10)


class TestPromptTuningMetaLearner:
    """Test Prompt Tuning meta-learner."""

    def test_initialization(self):
        """Test Prompt Tuning meta-learner initialization."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        learner = PromptTuningMetaLearner(
            model=model,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5,
        )

        assert learner.model == model
        assert learner.inner_lr == 0.01
        assert learner.outer_lr == 0.001
        assert learner.num_inner_steps == 5

    def test_inner_loop_adaptation(self, loss_fn):
        """Test inner loop adapts only prompts."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        learner = PromptTuningMetaLearner(
            model=model,
            num_inner_steps=3,
        )

        # Create task data
        support_x = mx.random.normal((10, 10))
        support_y = mx.random.randint(0, 5, (10,))

        # Save initial base model parameters
        initial_base_params = {
            k: v.copy() for k, v in model.parameters().items()
            if "prompt" not in k
        }

        # Adapt
        adapted_prompts = learner.inner_loop_adaptation(
            support_x, support_y, loss_fn
        )

        # Prompts should change
        assert "prompts" in adapted_prompts
        assert not mx.allclose(
            adapted_prompts["prompts"],
            model.get_prompt_parameters()["prompts"]
        )

        # Base model should not change
        current_base_params = {
            k: v for k, v in model.parameters().items()
            if "prompt" not in k
        }

        for key in initial_base_params.keys():
            if key in current_base_params:
                assert mx.allclose(
                    initial_base_params[key],
                    current_base_params[key]
                )

    def test_meta_train_step(self, loss_fn):
        """Test meta-training step."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        learner = PromptTuningMetaLearner(
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
        assert "num_prompts" in metrics
        assert metrics["query_loss"] >= 0
        assert metrics["num_prompts"] == 10

    def test_evaluate(self, loss_fn):
        """Test evaluation on episodes."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        learner = PromptTuningMetaLearner(
            model=model,
            num_inner_steps=3,
        )

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

    def test_save_and_load_prompts(self, tmp_path):
        """Test saving and loading prompt parameters."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        learner = PromptTuningMetaLearner(model=model)

        # Modify prompts
        model.prompt_layer.prompt_embeddings = mx.random.normal((10, 10))
        original_prompts = model.prompt_layer.prompt_embeddings.copy()

        # Save
        checkpoint_path = tmp_path / "prompt_checkpoint.npz"
        learner.save(str(checkpoint_path))

        # Create new learner and load
        new_model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )
        new_learner = PromptTuningMetaLearner(model=new_model)
        new_learner.load(str(checkpoint_path))

        # Prompts should match
        assert mx.allclose(
            original_prompts,
            new_model.prompt_layer.prompt_embeddings
        )


class TestPromptTuningIntegration:
    """Integration tests for Prompt Tuning."""

    def test_few_shot_learning_with_prompts(self, loss_fn):
        """Test Prompt Tuning on few-shot learning."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        learner = PromptTuningMetaLearner(
            model=model,
            inner_lr=0.01,
            outer_lr=0.01,
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

    def test_parameter_efficiency_of_prompts(self):
        """Test that prompt tuning is parameter efficient."""
        model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        # Count prompt parameters
        prompt_params = model.prompt_layer.prompt_embeddings.size

        # Count total model parameters
        total_params = sum(p.size for p in model.parameters().values())

        # Prompts should be small fraction of total
        prompt_ratio = prompt_params / total_params
        assert prompt_ratio < 0.3  # Less than 30%

    def test_varying_prompt_lengths(self, loss_fn):
        """Test models with different prompt lengths."""
        for num_prompts in [5, 10, 20]:
            model = PromptTuningModel(
                input_dim=10,
                hidden_dim=32,
                output_dim=5,
                num_prompts=num_prompts,
            )

            # Should work with any prompt length
            x = mx.random.normal((4, 10))
            logits = model(x)
            assert logits.shape == (4, 5)

    def test_prompt_tuning_vs_lora(self, loss_fn):
        """Compare prompt tuning parameter efficiency with LoRA."""
        from adapter_generation.peft_integration import SimpleLoRAModel

        # Prompt tuning
        prompt_model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )

        # LoRA
        lora_model = SimpleLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )

        # Count trainable parameters
        prompt_trainable = prompt_model.prompt_layer.prompt_embeddings.size
        lora_trainable = sum(
            p.size for k, p in lora_model.parameters().items()
            if "lora" in k
        )

        # Both should be efficient
        assert prompt_trainable < 1000
        assert lora_trainable < 5000


class TestComparePEFTMethods:
    """Test PEFT method comparison utility."""

    def test_compare_peft_methods_structure(self, loss_fn):
        """Test compare_peft_methods returns proper structure."""
        from adapter_generation.peft_integration import LoRAMetaLearner, SimpleLoRAModel
        from adapter_generation.adalora import AdaLoRAModel, AdaLoRAMetaLearner

        # Create learners
        lora_model = SimpleLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )
        lora_learner = LoRAMetaLearner(
            model=lora_model,
            peft_config=None,
            num_inner_steps=3,
        )

        adalora_model = AdaLoRAModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            lora_rank=8,
        )
        adalora_learner = AdaLoRAMetaLearner(
            model=adalora_model,
            num_inner_steps=3,
        )

        prompt_model = PromptTuningModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            num_prompts=10,
        )
        prompt_learner = PromptTuningMetaLearner(
            model=prompt_model,
            num_inner_steps=3,
        )

        # Create test episodes
        test_episodes = []
        for _ in range(3):
            support_x = mx.random.normal((10, 10))
            support_y = mx.random.randint(0, 5, (10,))
            query_x = mx.random.normal((5, 10))
            query_y = mx.random.randint(0, 5, (5,))
            test_episodes.append((support_x, support_y, query_x, query_y))

        # Compare methods
        results = compare_peft_methods(
            lora_learner,
            adalora_learner,
            prompt_learner,
            test_episodes,
            loss_fn,
        )

        # Verify structure
        assert "lora" in results
        assert "adalora" in results
        assert "prompt_tuning" in results

        for method in ["lora", "adalora", "prompt_tuning"]:
            assert "accuracy" in results[method]
            assert "loss" in results[method]
            assert 0 <= results[method]["accuracy"] <= 1
            assert results[method]["loss"] >= 0
