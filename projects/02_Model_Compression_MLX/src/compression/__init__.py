"""
Core compression module for orchestrating different compression techniques.

Provides unified interface for:
- Quantization
- Pruning  
- Knowledge Distillation
- Combined compression strategies
"""

from .config import CompressionConfig
from .compressor import ModelCompressor
from .strategies import CompressionStrategy, SequentialStrategy, ParallelStrategy

__all__ = [
    "CompressionConfig",
    "ModelCompressor", 
    "CompressionStrategy",
    "SequentialStrategy",
    "ParallelStrategy",
]