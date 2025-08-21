"""
Benchmarking module for model compression evaluation.

Provides comprehensive benchmarking capabilities for:
- Compression ratio analysis
- Performance impact measurement
- Accuracy evaluation
- Hardware-specific optimizations
"""

from .benchmark import CompressionBenchmark
from .metrics import CompressionMetrics

__all__ = [
    "CompressionBenchmark",
    "CompressionMetrics",
]