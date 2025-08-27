"""
MLX-optimized LoRA layer implementations.

Provides efficient LoRA layers specifically designed for Apple Silicon using
the MLX framework with unified memory architecture optimizations.
"""

import logging
import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import LoRAConfig

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
except ImportError:
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


class LoRALinear(nn.Module):
    """
    MLX-optimized LoRA linear layer.

    Implements Low-Rank Adaptation for linear layers with Apple Silicon optimizations
    using MLX framework for maximum performance on unified memory architecture.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        fan_in_fan_out: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.fan_in_fan_out = fan_in_fan_out

        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA matrices
        if rank > 0:
            self.lora_A = mx.random.normal((rank, in_features), dtype=mx.float32) * 0.02
            self.lora_B = mx.zeros((out_features, rank), dtype=mx.float32)

            # Dropout for LoRA path
            if dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = None

        # MLX parameters are frozen by default, so no need to explicitly freeze

    def __call__(self, x: mx.array, original_layer=None) -> mx.array:
        """Forward pass with LoRA adaptation."""
        # Use original layer if provided, otherwise use our linear layer
        if original_layer is not None:
            result = original_layer(x)
        else:
            result = self.linear(x)

        if self.rank > 0:
            # LoRA path: x -> A -> dropout -> B -> scale
            lora_x = x

            if self.fan_in_fan_out:
                # For certain models like GPT-2, we need to transpose
                lora_result = mx.matmul(lora_x, self.lora_A.T)
            else:
                lora_result = mx.matmul(lora_x, self.lora_A.T)

            if self.dropout is not None:
                lora_result = self.dropout(lora_result)

            lora_result = mx.matmul(lora_result, self.lora_B.T)
            lora_result = lora_result * self.scaling

            # Add LoRA adaptation to original output
            result = result + lora_result

        return result

    def merge_weights(self) -> None:
        """Merge LoRA weights into the original linear layer."""
        if self.rank > 0:
            # Compute LoRA weight update: scaling * B @ A
            weight_update = self.scaling * mx.matmul(self.lora_B, self.lora_A)

            # Add to original weights
            if self.fan_in_fan_out:
                self.linear.weight = self.linear.weight + weight_update.T
            else:
                self.linear.weight = self.linear.weight + weight_update

            # Clear LoRA matrices to save memory
            self.lora_A = None
            self.lora_B = None
            self.rank = 0

    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights from the original linear layer."""
        if self.rank == 0:
            logger.warning("LoRA weights already unmerged or not initialized")
            return

        if self.lora_A is None or self.lora_B is None:
            raise ValueError("Cannot unmerge weights: LoRA matrices are None")

        # Compute LoRA weight update: scaling * B @ A
        weight_update = self.scaling * mx.matmul(self.lora_B, self.lora_A)

        # Subtract from original weights (reverse of merge operation)
        if self.fan_in_fan_out:
            self.linear.weight = self.linear.weight - weight_update.T
        else:
            self.linear.weight = self.linear.weight - weight_update

        logger.info("LoRA weights successfully unmerged from base layer")


class LoRAAttention(nn.Module):
    """
    MLX-optimized LoRA attention layer.

    Applies LoRA adaptation to attention projections (q, k, v, o) with
    Apple Silicon optimizations for multi-head attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        target_modules: list[str] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # Original attention projections (frozen)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLX parameters are frozen by default

        # LoRA adaptations for selected projections
        self.lora_adapters = {}
        for module_name in target_modules:
            if hasattr(self, module_name):
                self.lora_adapters[module_name] = LoRALinear(
                    hidden_size, hidden_size, rank=rank, alpha=alpha,
                    dropout=dropout, bias=False
                )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, Optional[mx.array]]:
        """Forward pass with LoRA-adapted attention."""
        batch_size, seq_length = hidden_states.shape[:2]

        # Apply LoRA adaptations to projections
        if "q_proj" in self.lora_adapters:
            query_states = self.lora_adapters["q_proj"](hidden_states)
        else:
            query_states = self.q_proj(hidden_states)

        if "k_proj" in self.lora_adapters:
            key_states = self.lora_adapters["k_proj"](hidden_states)
        else:
            key_states = self.k_proj(hidden_states)

        if "v_proj" in self.lora_adapters:
            value_states = self.lora_adapters["v_proj"](hidden_states)
        else:
            value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        key_states = key_states.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        value_states = value_states.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = mx.matmul(query_states, key_states.transpose(0, 1, 3, 2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)

        attn_output = mx.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.hidden_size
        )

        # Apply LoRA adaptation to output projection
        if "o_proj" in self.lora_adapters:
            attn_output = self.lora_adapters["o_proj"](attn_output)
        else:
            attn_output = self.o_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class LoRAEmbedding(nn.Module):
    """
    MLX-optimized LoRA embedding layer.

    Applies LoRA adaptation to embedding layers for token and position embeddings.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int = 16,
        alpha: float = 32.0,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.padding_idx = padding_idx

        # Original embedding (frozen)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # MLX parameters are frozen by default

        # LoRA matrices for embedding
        if rank > 0:
            self.lora_A = mx.random.normal((rank, num_embeddings), dtype=mx.float32) * 0.02
            self.lora_B = mx.zeros((embedding_dim, rank), dtype=mx.float32)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass with LoRA-adapted embedding."""
        # Original embedding (frozen)
        result = self.embedding(input_ids)

        if self.rank > 0:
            # LoRA path for embedding
            # Convert input_ids to one-hot for matrix multiplication
            batch_size, seq_length = input_ids.shape
            one_hot = mx.zeros((batch_size, seq_length, self.num_embeddings))
            one_hot[mx.arange(batch_size)[:, None], mx.arange(seq_length)[None, :], input_ids] = 1

            # Apply LoRA transformation
            lora_result = mx.matmul(one_hot, self.lora_A.T)  # -> (batch, seq, rank)
            lora_result = mx.matmul(lora_result, self.lora_B.T)  # -> (batch, seq, embed_dim)
            lora_result = lora_result * self.scaling

            # Add LoRA adaptation to original output
            result = result + lora_result

        return result


class LoRAConv1D(nn.Module):
    """
    MLX-optimized LoRA 1D convolution layer.

    Specialized for transformer models that use 1D convolutions like GPT-2.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original conv1d (frozen) - implemented as linear for simplicity
        self.conv1d = nn.Linear(in_features, out_features, bias=True)

        # MLX parameters are frozen by default

        # LoRA matrices
        if rank > 0:
            self.lora_A = mx.random.normal((rank, in_features), dtype=mx.float32) * 0.02
            self.lora_B = mx.zeros((out_features, rank), dtype=mx.float32)

            if dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with LoRA adaptation."""
        # Original transformation (frozen)
        result = self.conv1d(x)

        if self.rank > 0:
            # LoRA path
            lora_result = mx.matmul(x, self.lora_A.T)

            if self.dropout is not None:
                lora_result = self.dropout(lora_result)

            lora_result = mx.matmul(lora_result, self.lora_B.T)
            lora_result = lora_result * self.scaling

            # Add LoRA adaptation
            result = result + lora_result

        return result