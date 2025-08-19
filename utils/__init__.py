"""
Shared utilities for the EfficientAI-MLX-Toolkit.

This module provides common functionality across all projects including:
- Logging utilities with pathlib support
- Configuration management
- Benchmarking framework
- Visualization utilities
- Apple Silicon optimization helpers
"""

from .benchmark_runner import BenchmarkRunner
from .config_manager import ConfigManager
from .file_operations import (
    CrossPlatformUtils,
    FileValidator,
    SafeFileHandler,
    find_files,
    read_json_file,
    write_json_file,
)
from .logging_utils import AppleSiliconLogger, LogManager, get_logger, setup_logging
from .plotting_utils import (
    create_comparison_plot,
    create_memory_usage_plot,
    create_performance_plot,
    save_benchmark_results,
)

__all__ = [
    # Logging utilities
    "setup_logging",
    "get_logger",
    "AppleSiliconLogger",
    "LogManager",
    # Configuration management
    "ConfigManager",
    # Benchmarking
    "BenchmarkRunner",
    # File operations
    "SafeFileHandler",
    "FileValidator",
    "CrossPlatformUtils",
    "read_json_file",
    "write_json_file",
    "find_files",
    # Plotting and visualization
    "create_performance_plot",
    "create_comparison_plot",
    "create_memory_usage_plot",
    "save_benchmark_results",
]
