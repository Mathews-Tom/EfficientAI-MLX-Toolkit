"""
Core ML Stable Diffusion Style Transfer Framework.

A comprehensive framework for artistic style transfer using Stable Diffusion
models optimized for Apple Silicon via Core ML and MLX.
"""

__version__ = "0.1.0"
__author__ = "EfficientAI-MLX-Toolkit"

from .coreml import CoreMLConverter, CoreMLOptimizer

# Core exports
from .diffusion import DiffusionConfig, StableDiffusionMLX
from .inference import InferenceConfig, InferenceEngine
from .style_transfer import StyleTransferConfig, StyleTransferPipeline

__all__ = [
    "StableDiffusionMLX",
    "DiffusionConfig",
    "StyleTransferPipeline",
    "StyleTransferConfig",
    "CoreMLConverter",
    "CoreMLOptimizer",
    "InferenceEngine",
    "InferenceConfig",
]
