"""
Benchmarking framework for the EfficientAI-MLX-Toolkit.

This module provides standardized benchmarking capabilities including
performance measurement, memory usage tracking, and comparative analysis.
"""

# Try to import existing benchmarks, but don't fail if they don't exist
try:
    from .comparative_analysis import BenchmarkComparator
except ImportError:
    BenchmarkComparator = None

try:
    from .memory_benchmarks import MemoryBenchmark
except ImportError:
    MemoryBenchmark = None

try:
    from .performance_benchmarks import MLXBenchmark, PerformanceBenchmark
except ImportError:
    MLXBenchmark = None
    PerformanceBenchmark = None

__all__ = []
if BenchmarkComparator:
    __all__.append("BenchmarkComparator")
if MemoryBenchmark:
    __all__.append("MemoryBenchmark")
if MLXBenchmark:
    __all__.append("MLXBenchmark")
if PerformanceBenchmark:
    __all__.append("PerformanceBenchmark")
