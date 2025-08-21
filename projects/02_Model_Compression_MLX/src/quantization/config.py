"""
Configuration classes for quantization operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from enum import Enum


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""
    POST_TRAINING = "post_training"
    QUANTIZATION_AWARE = "quantization_aware"
    DYNAMIC = "dynamic"


class CalibrationMethod(str, Enum):
    """Calibration methods for quantization."""
    MINMAX = "minmax"
    ENTROPY = "entropy"
    PERCENTILE = "percentile"
    MSE = "mse"


class QuantizationStrategy(str, Enum):
    """Quantization strategies."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


@dataclass
class QuantizationConfig:
    """Configuration for quantization operations."""
    
    # Basic quantization settings
    target_bits: int = 4
    weight_bits: int = 4
    activation_bits: int = 8
    
    # Quantization strategy
    method: QuantizationMethod = QuantizationMethod.POST_TRAINING
    calibration_method: CalibrationMethod = CalibrationMethod.MINMAX
    calibration_samples: int = 512
    
    # MLX-specific settings
    use_mlx_quantization: bool = True
    mlx_group_size: int = 64
    mlx_precision: str = "float16"
    
    # Advanced options
    symmetric: bool = False
    per_channel: bool = True
    preserve_accuracy_layers: List[str] = field(default_factory=list)
    
    # Calibration dataset
    calibration_dataset_path: Optional[Path] = None
    max_calibration_samples: int = 1000
    
    # Output settings
    quantized_model_path: Optional[Path] = None
    save_quantization_stats: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.target_bits not in [4, 8, 16]:
            raise ValueError(f"target_bits must be 4, 8, or 16, got {self.target_bits}")
        
        if self.weight_bits not in [1, 2, 4, 8, 16]:
            raise ValueError(f"weight_bits must be 1, 2, 4, 8, or 16, got {self.weight_bits}")
            
        if self.activation_bits not in [4, 8, 16, 32]:
            raise ValueError(f"activation_bits must be 4, 8, 16, or 32, got {self.activation_bits}")
        
        if self.calibration_samples <= 0:
            raise ValueError(f"calibration_samples must be positive, got {self.calibration_samples}")
        
        if self.mlx_group_size <= 0:
            raise ValueError(f"mlx_group_size must be positive, got {self.mlx_group_size}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
        """Create configuration from dictionary."""
        # Convert string enums
        if "method" in config_dict:
            config_dict["method"] = QuantizationMethod(config_dict["method"])
        if "calibration_method" in config_dict:
            config_dict["calibration_method"] = CalibrationMethod(config_dict["calibration_method"])
        
        # Convert paths
        if "calibration_dataset_path" in config_dict and config_dict["calibration_dataset_path"]:
            config_dict["calibration_dataset_path"] = Path(config_dict["calibration_dataset_path"])
        if "quantized_model_path" in config_dict and config_dict["quantized_model_path"]:
            config_dict["quantized_model_path"] = Path(config_dict["quantized_model_path"])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def get_compression_ratio(self) -> float:
        """Calculate theoretical compression ratio."""
        # Assuming original model uses 16-bit weights
        original_bits = 16
        return original_bits / self.target_bits
    
    def get_memory_savings(self, original_size_mb: float) -> float:
        """Calculate memory savings in MB."""
        compression_ratio = self.get_compression_ratio()
        return original_size_mb * (1 - 1/compression_ratio)