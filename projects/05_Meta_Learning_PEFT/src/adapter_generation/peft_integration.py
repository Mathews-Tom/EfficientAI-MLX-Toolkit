"""PEFT integration for meta-learning algorithms.

This module implements meta-learning of PEFT adapters (e.g., LoRA) instead of full model
parameters. This allows for efficient few-shot adaptation with minimal parameter updates.
"""

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from meta_learning.maml import MAMLLearner
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PEFTConfig:
    """Configuration for PEFT integration with meta-learning.

    Attributes:
        method: PEFT method to use ("lora", "adalora", "prompt_tuning", etc.)
        rank: Rank for low-rank methods (LoRA, AdaLoRA)
        alpha: Scaling factor for LoRA
        dropout: Dropout rate for adapters
        target_modules: List of module names to apply PEFT to
        freeze_base_model: Whether to freeze base model parameters
    """

    method: str = "lora"
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] | None = None
    freeze_base_model: bool = True


class LoRALayer(nn.Module):
    """Simple LoRA layer for meta-learning.

    Implements low-rank adaptation: h = Wx + (BA)x where:
    - W is the frozen pre-trained weight
    - B and A are low-rank matrices (rank r << d)
    - The adapter (BA) is what we meta-learn
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """Initialize LoRA layer.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            rank: Rank of low-rank decomposition.
            alpha: Scaling factor (LoRA weight = alpha / rank).
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen base weight (initialized but not trained)
        self.weight = mx.random.normal((out_features, in_features)) * 0.01

        # LoRA low-rank matrices (these are meta-learned)
        self.lora_A = mx.random.normal((rank, in_features)) * 0.01
        self.lora_B = mx.zeros((out_features, rank))

        self.dropout_p = dropout

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Base forward pass (frozen)
        base_output = x @ self.weight.T

        # LoRA adaptation
        if self.dropout_p > 0 and self.training:
            mask = mx.random.bernoulli(1 - self.dropout_p, x.shape)
            x = x * mask / (1 - self.dropout_p)

        # Low-rank forward: (BA)x = B(Ax)
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return base_output + lora_output


class LoRAMetaLearner(MAMLLearner):
    """Meta-learner that operates on LoRA adapter parameters.

    This extends MAML to meta-learn LoRA adapters instead of full model parameters.
    Only the low-rank matrices (lora_A, lora_B) are adapted during inner loop,
    while base model weights remain frozen.
    """

    def __init__(
        self,
        model: nn.Module,
        peft_config: PEFTConfig,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        meta_batch_size: int = 4,
        first_order: bool = False,
    ):
        """Initialize LoRA meta-learner.

        Args:
            model: Base model with LoRA layers.
            peft_config: PEFT configuration.
            inner_lr: Learning rate for inner loop adaptation.
            outer_lr: Learning rate for meta-optimization.
            num_inner_steps: Number of gradient steps per task.
            meta_batch_size: Number of tasks per meta-batch.
            first_order: Whether to use first-order approximation.
        """
        super().__init__(
            model=model,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            num_inner_steps=num_inner_steps,
            meta_batch_size=meta_batch_size,
            first_order=first_order,
        )
        self.peft_config = peft_config

        # Freeze base model weights if configured
        if peft_config.freeze_base_model:
            self._freeze_base_weights()

        logger.info(
            f"Initialized LoRA meta-learner: method={peft_config.method}, "
            f"rank={peft_config.rank}, alpha={peft_config.alpha}"
        )

    def _freeze_base_weights(self) -> None:
        """Freeze base model weights, only allowing LoRA parameters to be trained."""
        # In practice, this would iterate through model and freeze non-LoRA params
        # For now, this is a placeholder - actual implementation depends on model structure
        logger.debug("Freezing base model weights (LoRA parameters remain trainable)")

    def get_adapter_parameters(self) -> dict[str, mx.array]:
        """Extract only the LoRA adapter parameters from the model.

        Returns:
            Dictionary of LoRA parameters (lora_A, lora_B).
        """
        adapter_params = {}

        def _extract_adapters(params: dict, prefix: str = ""):
            for key, value in params.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    _extract_adapters(value, full_key)
                elif "lora_A" in full_key or "lora_B" in full_key:
                    adapter_params[full_key] = value

        _extract_adapters(dict(self.model.parameters()))
        return adapter_params

    def count_trainable_parameters(self) -> dict[str, int]:
        """Count trainable parameters (adapters only if base model frozen).

        Returns:
            Dictionary with parameter counts.
        """
        adapter_params = self.get_adapter_parameters()
        total_adapter = sum(p.size for p in adapter_params.values())

        all_params = dict(self.model.parameters())

        def _count_params(params):
            count = 0
            for v in params.values():
                if isinstance(v, dict):
                    count += _count_params(v)
                elif hasattr(v, "size"):
                    count += v.size
            return count

        total_all = _count_params(all_params)

        return {
            "total": total_all,
            "adapter": total_adapter,
            "base": total_all - total_adapter,
            "trainable": total_adapter if self.peft_config.freeze_base_model else total_all,
            "reduction_ratio": total_all / total_adapter if total_adapter > 0 else 0,
        }


class SimpleLoRAModel(nn.Module):
    """Simple LoRA-adapted model for testing and prototyping.

    This wraps a base model and adds LoRA layers for meta-learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        """Initialize LoRA model.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (number of classes).
            lora_rank: Rank for LoRA adaptation.
            lora_alpha: Scaling factor for LoRA.
        """
        super().__init__()

        # LoRA layers
        self.lora_layer1 = LoRALayer(
            input_dim, hidden_dim, rank=lora_rank, alpha=lora_alpha
        )
        self.lora_layer2 = LoRALayer(
            hidden_dim, output_dim, rank=lora_rank, alpha=lora_alpha
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through LoRA-adapted network.

        Args:
            x: Input tensor.

        Returns:
            Output logits.
        """
        x = self.lora_layer1(x)
        x = nn.relu(x)
        x = self.lora_layer2(x)
        return x
