"""AdaLoRA: Adaptive Low-Rank Adaptation.

This module implements AdaLoRA, which adaptively allocates the parameter budget
among different LoRA modules based on their importance.

Reference:
    Zhang et al. (2023) "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from utils.logging import get_logger

logger = get_logger(__name__)


class AdaLoRALayer(nn.Module):
    """AdaLoRA layer with adaptive rank allocation.

    Unlike standard LoRA which uses fixed rank, AdaLoRA dynamically adjusts
    the rank of each adapter based on its importance during training.

    Attributes:
        in_features: Input dimension
        out_features: Output dimension
        rank: Maximum rank (actual rank may be lower)
        alpha: Scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_rank: int | None = None,
    ):
        """Initialize AdaLoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Maximum rank (default: 8)
            alpha: Scaling factor (default: 16.0)
            dropout: Dropout rate (default: 0.0)
            target_rank: Target rank for pruning (default: rank // 2)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = rank
        self.current_rank = rank
        self.target_rank = target_rank or rank // 2
        self.scaling = alpha / rank
        self.dropout_p = dropout

        # Frozen base weight
        self.weight = mx.random.normal((out_features, in_features)) * 0.01

        # AdaLoRA uses SVD-parameterization: BA = USV^T
        # U: (out_features, rank)
        # S: (rank,) - singular values (importance scores)
        # V: (rank, in_features)
        self.lora_U = mx.random.normal((out_features, rank)) * 0.01
        self.lora_S = mx.ones(rank)  # Initialize with ones
        self.lora_V = mx.random.normal((rank, in_features)) * 0.01

        # Importance scores (updated during training)
        self.importance_scores = mx.ones(rank)

    def compute_importance_scores(self) -> mx.array:
        """Compute importance scores for each singular value.

        Importance is based on the magnitude of singular values.

        Returns:
            Importance scores of shape (rank,)
        """
        # Importance = |singular value|
        importance = mx.abs(self.lora_S)
        return importance

    def prune_low_rank_components(self, target_rank: int | None = None) -> None:
        """Prune low-importance singular values to reduce rank.

        Args:
            target_rank: Target rank after pruning (default: self.target_rank)
        """
        if target_rank is None:
            target_rank = self.target_rank

        if target_rank >= self.current_rank:
            return  # No pruning needed

        # Compute importance scores
        importance = self.compute_importance_scores()

        # Get indices of top-k important components
        top_k_indices = mx.argsort(importance)[-target_rank:]

        # Prune U, S, V to keep only important components
        self.lora_U = self.lora_U[:, top_k_indices]
        self.lora_S = self.lora_S[top_k_indices]
        self.lora_V = self.lora_V[top_k_indices, :]
        self.importance_scores = importance[top_k_indices]

        self.current_rank = target_rank
        logger.info(f"Pruned AdaLoRA rank: {self.max_rank} → {target_rank}")

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with AdaLoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base forward pass (frozen)
        base_output = x @ self.weight.T

        # AdaLoRA forward: (USV^T)x = U(S(V^T x))
        # Apply dropout
        if self.dropout_p > 0 and self.training:
            mask = mx.random.bernoulli(1 - self.dropout_p, x.shape)
            x = x * mask / (1 - self.dropout_p)

        # Forward through SVD components
        v_out = x @ self.lora_V.T  # (batch, rank)
        s_out = v_out * self.lora_S  # Element-wise multiply by singular values
        u_out = s_out @ self.lora_U.T  # (batch, out_features)

        # Scale and add to base output
        ada_output = u_out * self.scaling

        return base_output + ada_output

    def get_current_rank(self) -> int:
        """Get current rank of the adapter.

        Returns:
            Current rank
        """
        return self.current_rank

    def get_parameter_count(self) -> dict[str, int]:
        """Get parameter counts for the adapter.

        Returns:
            Dictionary with base and adapter parameter counts
        """
        base_params = self.out_features * self.in_features
        adapter_params = (
            self.out_features * self.current_rank  # U
            + self.current_rank  # S
            + self.current_rank * self.in_features  # V
        )

        return {
            "base_params": base_params,
            "adapter_params": adapter_params,
            "total_params": base_params + adapter_params,
            "reduction_ratio": base_params / adapter_params if adapter_params > 0 else 0,
        }


class AdaLoRAModel(nn.Module):
    """Simple model with AdaLoRA layers.

    Architecture:
        input -> AdaLoRA -> relu -> AdaLoRA -> output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """Initialize AdaLoRA model.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            lora_rank: Maximum rank for AdaLoRA
            lora_alpha: Scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Create AdaLoRA layers
        self.layer1 = AdaLoRALayer(
            input_dim, hidden_dim, rank=lora_rank, alpha=lora_alpha, dropout=dropout
        )
        self.layer2 = AdaLoRALayer(
            hidden_dim, output_dim, rank=lora_rank, alpha=lora_alpha, dropout=dropout
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, output_dim)
        """
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
        return x

    def prune_adapters(self, target_rank: int) -> None:
        """Prune all AdaLoRA adapters to target rank.

        Args:
            target_rank: Target rank for all adapters
        """
        self.layer1.prune_low_rank_components(target_rank)
        self.layer2.prune_low_rank_components(target_rank)

    def get_adapter_parameters(self) -> dict[str, mx.array]:
        """Get all AdaLoRA adapter parameters.

        Returns:
            Dictionary of adapter parameters
        """
        params = {}

        # Layer 1 adapters
        params["layer1_U"] = self.layer1.lora_U
        params["layer1_S"] = self.layer1.lora_S
        params["layer1_V"] = self.layer1.lora_V

        # Layer 2 adapters
        params["layer2_U"] = self.layer2.lora_U
        params["layer2_S"] = self.layer2.lora_S
        params["layer2_V"] = self.layer2.lora_V

        return params


class AdaLoRAMetaLearner:
    """Meta-learner for AdaLoRA adapters.

    Combines meta-learning with adaptive rank allocation.
    """

    def __init__(
        self,
        model: AdaLoRAModel,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        prune_every: int = 100,
        target_rank_schedule: list[tuple[int, int]] | None = None,
    ):
        """Initialize AdaLoRA meta-learner.

        Args:
            model: AdaLoRA model
            inner_lr: Inner loop learning rate
            outer_lr: Outer loop (meta) learning rate
            num_inner_steps: Gradient steps per task
            prune_every: Prune adapters every N iterations
            target_rank_schedule: List of (iteration, target_rank) tuples
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.prune_every = prune_every
        self.target_rank_schedule = target_rank_schedule or []
        self.iteration = 0

    def inner_loop_adaptation(
        self,
        support_x: mx.array,
        support_y: mx.array,
        loss_fn: Callable,
    ) -> dict[str, mx.array]:
        """Adapt adapter parameters to task.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            loss_fn: Loss function

        Returns:
            Adapted adapter parameters
        """
        import copy

        # Only adapt adapter parameters, not base weights
        adapted_params = copy.deepcopy(self.model.get_adapter_parameters())

        for step in range(self.num_inner_steps):

            def compute_loss(params):
                # Update model with adapted adapter params
                self.model.layer1.lora_U = params["layer1_U"]
                self.model.layer1.lora_S = params["layer1_S"]
                self.model.layer1.lora_V = params["layer1_V"]
                self.model.layer2.lora_U = params["layer2_U"]
                self.model.layer2.lora_S = params["layer2_S"]
                self.model.layer2.lora_V = params["layer2_V"]

                logits = self.model(support_x)
                return loss_fn(logits, support_y)

            loss, grads = mx.value_and_grad(compute_loss)(adapted_params)

            # Update adapter parameters
            adapted_params = {
                key: adapted_params[key] - self.inner_lr * grads[key]
                for key in adapted_params.keys()
            }

        return adapted_params

    def meta_train_step(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable,
    ) -> dict[str, Any]:
        """Meta-training step with adaptive pruning.

        Args:
            episodes: List of (support_x, support_y, query_x, query_y)
            loss_fn: Loss function

        Returns:
            Training metrics
        """
        total_query_loss = 0.0
        adapter_grads = {key: mx.zeros_like(val) for key, val in self.model.get_adapter_parameters().items()}

        # Process each task
        for support_x, support_y, query_x, query_y in episodes:
            # Inner loop: adapt adapters
            adapted_params = self.inner_loop_adaptation(support_x, support_y, loss_fn)

            # Outer loop: compute gradients on query set
            def query_loss_fn(params):
                # Set adapted params
                self.model.layer1.lora_U = params["layer1_U"]
                self.model.layer1.lora_S = params["layer1_S"]
                self.model.layer1.lora_V = params["layer1_V"]
                self.model.layer2.lora_U = params["layer2_U"]
                self.model.layer2.lora_S = params["layer2_S"]
                self.model.layer2.lora_V = params["layer2_V"]

                logits = self.model(query_x)
                return loss_fn(logits, query_y)

            query_loss, grads = mx.value_and_grad(query_loss_fn)(adapted_params)
            total_query_loss += float(query_loss)

            # Accumulate gradients
            for key in adapter_grads.keys():
                adapter_grads[key] += grads[key]

        # Average gradients
        num_tasks = len(episodes)
        for key in adapter_grads.keys():
            adapter_grads[key] /= num_tasks

        # Meta-update: update adapter parameters
        current_params = self.model.get_adapter_parameters()
        updated_params = {
            key: current_params[key] - self.outer_lr * adapter_grads[key]
            for key in current_params.keys()
        }

        # Update model
        self.model.layer1.lora_U = updated_params["layer1_U"]
        self.model.layer1.lora_S = updated_params["layer1_S"]
        self.model.layer1.lora_V = updated_params["layer1_V"]
        self.model.layer2.lora_U = updated_params["layer2_U"]
        self.model.layer2.lora_S = updated_params["layer2_S"]
        self.model.layer2.lora_V = updated_params["layer2_V"]

        # Update importance scores
        self.model.layer1.importance_scores = self.model.layer1.compute_importance_scores()
        self.model.layer2.importance_scores = self.model.layer2.compute_importance_scores()

        # Adaptive pruning
        if self.iteration % self.prune_every == 0 and self.iteration > 0:
            # Check if we should prune based on schedule
            for iter_threshold, target_rank in self.target_rank_schedule:
                if self.iteration == iter_threshold:
                    self.model.prune_adapters(target_rank)
                    break

        self.iteration += 1

        return {
            "query_loss": total_query_loss / num_tasks,
            "layer1_rank": self.model.layer1.get_current_rank(),
            "layer2_rank": self.model.layer2.get_current_rank(),
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
            adapted_params = self.inner_loop_adaptation(support_x, support_y, loss_fn)

            # Evaluate
            self.model.layer1.lora_U = adapted_params["layer1_U"]
            self.model.layer1.lora_S = adapted_params["layer1_S"]
            self.model.layer1.lora_V = adapted_params["layer1_V"]
            self.model.layer2.lora_U = adapted_params["layer2_U"]
            self.model.layer2.lora_S = adapted_params["layer2_S"]
            self.model.layer2.lora_V = adapted_params["layer2_V"]

            logits = self.model(query_x)
            loss = float(loss_fn(logits, query_y))
            acc = float(mx.mean(mx.argmax(logits, axis=-1) == query_y))

            total_loss += loss
            total_acc += acc

        num_episodes = len(episodes)

        return {
            "avg_loss": total_loss / num_episodes,
            "avg_acc": total_acc / num_episodes,
            "layer1_rank": self.model.layer1.get_current_rank(),
            "layer2_rank": self.model.layer2.get_current_rank(),
        }

    def save(self, path: str) -> None:
        """Save model and adapter parameters.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = self.model.get_adapter_parameters()
        mx.savez(path, **checkpoint)
        logger.info(f"Saved AdaLoRA checkpoint to {path}")

    def load(self, path: str) -> None:
        """Load model and adapter parameters.

        Args:
            path: Path to checkpoint
        """
        checkpoint = mx.load(path)

        self.model.layer1.lora_U = checkpoint["layer1_U"]
        self.model.layer1.lora_S = checkpoint["layer1_S"]
        self.model.layer1.lora_V = checkpoint["layer1_V"]
        self.model.layer2.lora_U = checkpoint["layer2_U"]
        self.model.layer2.lora_S = checkpoint["layer2_S"]
        self.model.layer2.lora_V = checkpoint["layer2_V"]

        logger.info(f"Loaded AdaLoRA checkpoint from {path}")
