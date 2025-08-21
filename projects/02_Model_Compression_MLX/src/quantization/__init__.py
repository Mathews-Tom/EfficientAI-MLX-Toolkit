"""
Quantization module for MLX-native model compression.

Provides quantization capabilities including:
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Dynamic quantization
- MLX-optimized quantization methods
"""

from .config import QuantizationConfig
from .quantizer import MLXQuantizer
from .methods import (
    PostTrainingQuantizer,
    DynamicQuantizer,
    QuantizationAwareTrainer,
)
from .calibration import CalibrationDataLoader, CalibrationStrategy
from .utils import (
    calculate_quantization_error,
    estimate_compression_ratio,
    validate_quantization_config,
)

__all__ = [
    "QuantizationConfig",
    "MLXQuantizer",
    "PostTrainingQuantizer",
    "DynamicQuantizer", 
    "QuantizationAwareTrainer",
    "CalibrationDataLoader",
    "CalibrationStrategy",
    "calculate_quantization_error",
    "estimate_compression_ratio", 
    "validate_quantization_config",
]