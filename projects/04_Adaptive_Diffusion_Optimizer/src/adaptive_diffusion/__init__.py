"""
Adaptive Diffusion Optimizer with MLX Integration

An intelligent system for optimizing diffusion models during training using MLX
for Apple Silicon. Incorporates progressive distillation, efficient sampling,
and hardware-aware optimization.
"""

__version__ = "0.1.0"
__author__ = "EfficientAI MLX Toolkit Team"

from adaptive_diffusion.baseline import (
    DiffusionPipeline,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverScheduler,
)

__all__ = [
    "DiffusionPipeline",
    "DDPMScheduler",
    "DDIMScheduler",
    "DPMSolverScheduler",
]
