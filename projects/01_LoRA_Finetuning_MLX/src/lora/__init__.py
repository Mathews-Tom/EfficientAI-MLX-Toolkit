"""
MLX-optimized LoRA (Low-Rank Adaptation) implementation.

This module provides Apple Silicon optimized LoRA layers, adapters, and configuration
for efficient fine-tuning of large language models using the MLX framework.
"""

from .config import LoRAConfig, TrainingConfig, InferenceConfig, load_config
from .layers import LoRALinear, LoRAAttention, LoRAEmbedding
from .adapters import LoRAAdapter, AdapterManager, ModelAdapter

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