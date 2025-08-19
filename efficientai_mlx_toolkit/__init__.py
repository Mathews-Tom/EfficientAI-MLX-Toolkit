"""
EfficientAI-MLX-Toolkit: Apple Silicon optimized AI toolkit.

This package provides a comprehensive suite of tools for machine learning
optimization, training, and deployment on Apple Silicon hardware.
"""

__version__ = "0.1.0"

from environment import EnvironmentSetup, detect_apple_silicon, setup_mlx_optimization
from utils import (
    BenchmarkRunner,
    ConfigManager,
    create_comparison_plot,
    create_memory_usage_plot,
    create_performance_plot,
    get_logger,
    save_benchmark_results,
    setup_logging,
)

__all__ = [
    "__version__",
    "setup_logging",
    "get_logger",
    "ConfigManager",
    "BenchmarkRunner",
    "create_performance_plot",
    "create_comparison_plot",
    "create_memory_usage_plot",
    "save_benchmark_results",
    "EnvironmentSetup",
    "detect_apple_silicon",
    "setup_mlx_optimization",
]
