"""
Adaptive Diffusion Benchmarking Suite

Comprehensive benchmarking infrastructure for adaptive diffusion optimization.
"""

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ValidationMetrics",
    "QualityMetrics",
    "PerformanceMetrics",
]

def __getattr__(name):
    """Lazy import to avoid import errors at module level."""
    if name in ["BenchmarkRunner", "BenchmarkConfig", "BenchmarkResult"]:
        from benchmarks.adaptive_diffusion.benchmark_suite import (
            BenchmarkRunner,
            BenchmarkConfig,
            BenchmarkResult,
        )
        return locals()[name]
    elif name in ["ValidationMetrics", "QualityMetrics", "PerformanceMetrics"]:
        from benchmarks.adaptive_diffusion.validation_metrics import (
            ValidationMetrics,
            QualityMetrics,
            PerformanceMetrics,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
