"""
MLX-Native Model Compression Framework for Apple Silicon.

This package provides comprehensive model compression capabilities including:
- Quantization (4-bit, 8-bit, 16-bit)
- Pruning (structured and unstructured)
- Knowledge Distillation
- Benchmarking and Evaluation Tools

Optimized for Apple Silicon (M1/M2/M3) with MLX acceleration.
"""

__version__ = "0.1.0"
__author__ = "EfficientAI-MLX-Toolkit Team"

from .compression import *
from .quantization import *
from .pruning import *
from .distillation import *
from .evaluation import *
from .benchmarking import *

__all__ = [
    # Core compression classes
    "CompressionConfig",
    "ModelCompressor",

    # Quantization
    "QuantizationConfig",
    "MLXQuantizer",

    # Pruning
    "PruningConfig",
    "MLXPruner",

    # Distillation
    "DistillationConfig",
    "KnowledgeDistiller",

    # Evaluation
    "CompressionEvaluator",
    "CompressionMetrics",

    # Benchmarking
    "CompressionBenchmark",
]