"""
Meta-Learning PEFT System with MLX

Research implementation of meta-learning for Parameter-Efficient Fine-Tuning.
"""

__version__ = "0.1.0"
__author__ = "EfficientAI Toolkit Team"

from . import meta_learning, task_embedding, utils

__all__ = ["meta_learning", "task_embedding", "utils"]
