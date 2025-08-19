"""
Environment management for the EfficientAI-MLX-Toolkit.

This module provides environment setup, dependency management, and
Apple Silicon optimization configuration.
"""

from .setup_manager import EnvironmentSetup, detect_apple_silicon, setup_mlx_optimization

__all__ = [
    "EnvironmentSetup",
    "detect_apple_silicon",
    "setup_mlx_optimization",
]
