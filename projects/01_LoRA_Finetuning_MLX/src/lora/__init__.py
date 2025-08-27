"""
MLX-optimized LoRA (Low-Rank Adaptation) implementation.

This module provides Apple Silicon optimized LoRA layers, adapters, and configuration
for efficient fine-tuning of large language models using the MLX framework.
"""

from .adapters import AdapterManager, LoRAAdapter, ModelAdapter
from .config import InferenceConfig, LoRAConfig, TrainingConfig, load_config
from .layers import LoRAAttention, LoRAEmbedding, LoRALinear

__all__ = [
    "LoRAConfig",
    "TrainingConfig",
    "InferenceConfig",
    "load_config",
    "LoRALinear",
    "LoRAAttention",
    "LoRAEmbedding",
    "LoRAAdapter",
    "AdapterManager",
    "ModelAdapter",
]
