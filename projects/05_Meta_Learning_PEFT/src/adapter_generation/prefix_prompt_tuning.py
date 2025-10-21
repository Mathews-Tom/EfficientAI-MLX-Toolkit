"""Prefix Tuning and Prompt Tuning for meta-learning.

This module implements Prefix Tuning and Prompt Tuning, which add learnable
continuous prompts/prefixes to the input instead of modifying model weights.

References:
    - Li & Liang (2021) "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
    - Lester et al. (2021) "The Power of Scale for Parameter-Efficient Prompt Tuning"
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from utils.logging import get_logger

logger = get_logger(__name__)


class PrefixTuningLayer(nn.Module):
    """Prefix Tuning layer.

    Adds learnable prefix vectors to the key and value projections
    in attention mechanisms.

    Attributes:
        prefix_length: Number of prefix tokens
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        prefix_length: int = 10,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        """Initialize Prefix Tuning layer.

        Args:
            prefix_length: Number of prefix tokens (default: 10)
            hidden_dim: Hidden dimension (default: 128)
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout rate (default: 0.0)
        """
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout_p = dropout

        # Learnable prefix parameters (meta-learned)
        # Shape: (prefix_length, hidden_dim)
        self.prefix_key = mx.random.normal((prefix_length, hidden_dim)) * 0.01
        self.prefix_value = mx.random.normal((prefix_length, hidden_dim)) * 0.01

        # Optional reparameterization (for better optimization)
        # Prefix is generated from smaller bottleneck
        self.use_reparameterization = True
        if self.use_reparameterization:
            bottleneck_dim = hidden_dim // 4
            self.prefix_key_proj = nn.Linear(bottleneck_dim, hidden_dim)
            self.prefix_value_proj = nn.Linear(bottleneck_dim, hidden_dim)
            self.prefix_params = mx.random.normal(
                (prefix_length, bottleneck_dim)
            ) * 0.01

    def get_prefix_kv(self) -> tuple[mx.array, mx.array]:
        """Get prefix key and value vectors.

        Returns:
            Tuple of (prefix_key, prefix_value)
        """
        if self.use_reparameterization:
            # Generate from bottleneck
            prefix_key = self.prefix_key_proj(self.prefix_params)
            prefix_value = self.prefix_value_proj(self.prefix_params)
        else:
            prefix_key = self.prefix_key
            prefix_value = self.prefix_value

        return prefix_key, prefix_value

    def __call__(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> mx.array:
        """Apply prefix tuning to attention.

        Args:
            query: Query tensor (batch, seq_len, hidden_dim)
            key: Key tensor (batch, seq_len, hidden_dim)
            value: Value tensor (batch, seq_len, hidden_dim)

        Returns:
            Attention output with prefix
        """
        batch_size, seq_len, _ = query.shape

        # Get prefix key and value
        prefix_key, prefix_value = self.get_prefix_kv()

        # Expand prefix for batch
        # (prefix_length, hidden_dim) -> (batch, prefix_length, hidden_dim)
        prefix_key = mx.expand_dims(prefix_key, 0)
        prefix_key = mx.broadcast_to(prefix_key, (batch_size, self.prefix_length, self.hidden_dim))

        prefix_value = mx.expand_dims(prefix_value, 0)
        prefix_value = mx.broadcast_to(prefix_value, (batch_size, self.prefix_length, self.hidden_dim))

        # Concatenate prefix to key and value
        extended_key = mx.concatenate([prefix_key, key], axis=1)
        extended_value = mx.concatenate([prefix_value, value], axis=1)

        # Compute attention (simplified, no multi-head)
        # Q * K^T / sqrt(d)
        scores = query @ extended_key.transpose(0, 2, 1)
        scores = scores / mx.sqrt(float(self.hidden_dim))

        # Softmax
        attn_weights = mx.softmax(scores, axis=-1)

        # Apply dropout
        if self.dropout_p > 0 and self.training:
            mask = mx.random.bernoulli(1 - self.dropout_p, attn_weights.shape)
            attn_weights = attn_weights * mask / (1 - self.dropout_p)

        # Weighted sum of values
        output = attn_weights @ extended_value

        return output


class PromptTuningLayer(nn.Module):
    """Prompt Tuning layer.

    Prepends learnable soft prompt tokens to the input sequence.

    Attributes:
        num_prompts: Number of prompt tokens
        hidden_dim: Hidden dimension size
    """

    def __init__(
        self,
        num_prompts: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        """Initialize Prompt Tuning layer.

        Args:
            num_prompts: Number of prompt tokens (default: 10)
            hidden_dim: Hidden dimension (default: 128)
            dropout: Dropout rate (default: 0.0)
        """
        super().__init__()
        self.num_prompts = num_prompts
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        # Learnable prompt embeddings (meta-learned)
        # Shape: (num_prompts, hidden_dim)
        self.prompt_embeddings = mx.random.normal(
            (num_prompts, hidden_dim)
        ) * 0.01

    def __call__(self, x: mx.array) -> mx.array:
        """Prepend prompts to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Output with prompts of shape (batch_size, num_prompts + seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Expand prompts for batch
        # (num_prompts, hidden_dim) -> (batch, num_prompts, hidden_dim)
        prompts = mx.expand_dims(self.prompt_embeddings, 0)
        prompts = mx.broadcast_to(prompts, (batch_size, self.num_prompts, hidden_dim))

        # Apply dropout to prompts
        if self.dropout_p > 0 and self.training:
            mask = mx.random.bernoulli(1 - self.dropout_p, prompts.shape)
            prompts = prompts * mask / (1 - self.dropout_p)

        # Concatenate prompts to input
        output = mx.concatenate([prompts, x], axis=1)

        return output


class PromptTuningModel(nn.Module):
    """Simple model with Prompt Tuning.

    Architecture:
        prompts -> input -> linear -> relu -> linear -> output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_prompts: int = 10,
        dropout: float = 0.0,
    ):
        """Initialize Prompt Tuning model.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            num_prompts: Number of prompt tokens
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Prompt tuning layer
        self.prompt_layer = PromptTuningLayer(
            num_prompts=num_prompts, hidden_dim=input_dim, dropout=dropout
        )

        # Frozen base model layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
                or (batch_size, seq_len, input_dim)

        Returns:
            Logits of shape (batch_size, output_dim)
        """
        # Handle 2D input (batch_size, input_dim)
        if len(x.shape) == 2:
            x = mx.expand_dims(x, 1)  # (batch_size, 1, input_dim)

        # Add prompts
        x_with_prompts = self.prompt_layer(x)

        # Average pooling over sequence dimension (including prompts)
        x_pooled = mx.mean(x_with_prompts, axis=1)  # (batch_size, input_dim)

        # Forward through frozen base model
        x = self.layer1(x_pooled)
        x = nn.relu(x)
        x = self.layer2(x)

        return x

    def get_prompt_parameters(self) -> dict[str, mx.array]:
        """Get prompt parameters.

        Returns:
            Dictionary of prompt parameters
        """
        return {"prompts": self.prompt_layer.prompt_embeddings}


class PromptTuningMetaLearner:
    """Meta-learner for Prompt Tuning.

    Meta-learns optimal prompt embeddings that can be quickly adapted
    to new tasks.
    """

    def __init__(
        self,
        model: PromptTuningModel,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
    ):
        """Initialize Prompt Tuning meta-learner.

        Args:
            model: Prompt tuning model
            inner_lr: Inner loop learning rate
            outer_lr: Outer loop (meta) learning rate
            num_inner_steps: Gradient steps per task
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps

    def inner_loop_adaptation(
        self,
        support_x: mx.array,
        support_y: mx.array,
        loss_fn: Callable,
    ) -> dict[str, mx.array]:
        """Adapt prompt parameters to task.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            loss_fn: Loss function

        Returns:
            Adapted prompt parameters
        """
        import copy

        # Only adapt prompts, not base model weights
        adapted_prompts = copy.deepcopy(self.model.get_prompt_parameters())

        for step in range(self.num_inner_steps):

            def compute_loss(params):
                # Update prompts
                self.model.prompt_layer.prompt_embeddings = params["prompts"]

                logits = self.model(support_x)
                return loss_fn(logits, support_y)

            loss, grads = mx.value_and_grad(compute_loss)(adapted_prompts)

            # Update prompts
            adapted_prompts = {
                key: adapted_prompts[key] - self.inner_lr * grads[key]
                for key in adapted_prompts.keys()
            }

        return adapted_prompts

    def meta_train_step(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable,
    ) -> dict[str, Any]:
        """Meta-training step.

        Args:
            episodes: List of (support_x, support_y, query_x, query_y)
            loss_fn: Loss function

        Returns:
            Training metrics
        """
        total_query_loss = 0.0
        prompt_grads = {
            key: mx.zeros_like(val)
            for key, val in self.model.get_prompt_parameters().items()
        }

        # Process each task
        for support_x, support_y, query_x, query_y in episodes:
            # Inner loop: adapt prompts
            adapted_prompts = self.inner_loop_adaptation(
                support_x, support_y, loss_fn
            )

            # Outer loop: compute gradients on query set
            def query_loss_fn(params):
                self.model.prompt_layer.prompt_embeddings = params["prompts"]
                logits = self.model(query_x)
                return loss_fn(logits, query_y)

            query_loss, grads = mx.value_and_grad(query_loss_fn)(
                adapted_prompts
            )
            total_query_loss += float(query_loss)

            # Accumulate gradients
            for key in prompt_grads.keys():
                prompt_grads[key] += grads[key]

        # Average gradients
        num_tasks = len(episodes)
        for key in prompt_grads.keys():
            prompt_grads[key] /= num_tasks

        # Meta-update: update prompt parameters
        current_prompts = self.model.get_prompt_parameters()
        updated_prompts = {
            key: current_prompts[key] - self.outer_lr * prompt_grads[key]
            for key in current_prompts.keys()
        }

        # Update model
        self.model.prompt_layer.prompt_embeddings = updated_prompts["prompts"]

        return {
            "query_loss": total_query_loss / num_tasks,
            "num_prompts": self.model.prompt_layer.num_prompts,
        }

    def evaluate(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable,
    ) -> dict[str, float]:
        """Evaluate on validation tasks.

        Args:
            episodes: Validation episodes
            loss_fn: Loss function

        Returns:
            Evaluation metrics
        """
        total_loss = 0.0
        total_acc = 0.0

        for support_x, support_y, query_x, query_y in episodes:
            # Adapt
            adapted_prompts = self.inner_loop_adaptation(
                support_x, support_y, loss_fn
            )

            # Evaluate
            self.model.prompt_layer.prompt_embeddings = adapted_prompts[
                "prompts"
            ]

            logits = self.model(query_x)
            loss = float(loss_fn(logits, query_y))
            acc = float(mx.mean(mx.argmax(logits, axis=-1) == query_y))

            total_loss += loss
            total_acc += acc

        num_episodes = len(episodes)

        return {
            "avg_loss": total_loss / num_episodes,
            "avg_acc": total_acc / num_episodes,
        }

    def save(self, path: str) -> None:
        """Save prompt parameters.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = self.model.get_prompt_parameters()
        mx.savez(path, **checkpoint)
        logger.info(f"Saved Prompt Tuning checkpoint to {path}")

    def load(self, path: str) -> None:
        """Load prompt parameters.

        Args:
            path: Path to checkpoint
        """
        checkpoint = mx.load(path)
        self.model.prompt_layer.prompt_embeddings = checkpoint["prompts"]
        logger.info(f"Loaded Prompt Tuning checkpoint from {path}")


def compare_peft_methods(
    lora_learner: Any,
    adalora_learner: Any,
    prompt_learner: PromptTuningMetaLearner,
    test_episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
    loss_fn: Callable,
) -> dict[str, dict[str, float]]:
    """Compare different PEFT methods on test tasks.

    Args:
        lora_learner: LoRA meta-learner
        adalora_learner: AdaLoRA meta-learner
        prompt_learner: Prompt Tuning meta-learner
        test_episodes: Test episodes
        loss_fn: Loss function

    Returns:
        Comparison results
    """
    results = {}

    # Evaluate LoRA
    lora_results = lora_learner.evaluate(test_episodes, loss_fn)
    results["lora"] = {
        "accuracy": lora_results.get("avg_acc", 0.0),
        "loss": lora_results.get("avg_loss", 0.0),
    }

    # Evaluate AdaLoRA
    adalora_results = adalora_learner.evaluate(test_episodes, loss_fn)
    results["adalora"] = {
        "accuracy": adalora_results.get("avg_acc", 0.0),
        "loss": adalora_results.get("avg_loss", 0.0),
    }

    # Evaluate Prompt Tuning
    prompt_results = prompt_learner.evaluate(test_episodes, loss_fn)
    results["prompt_tuning"] = {
        "accuracy": prompt_results.get("avg_acc", 0.0),
        "loss": prompt_results.get("avg_loss", 0.0),
    }

    return results
