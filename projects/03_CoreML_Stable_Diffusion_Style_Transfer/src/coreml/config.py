"""Configuration for Core ML conversion and optimization."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CoreMLConfig:
    """Configuration for Core ML conversion and optimization."""

    # Apple Silicon optimization
    optimize_for_apple_silicon: bool = True
    compute_units: str = "all"  # "all", "cpu_only", "cpu_and_gpu"
    precision: str = "float16"  # "float16", "float32"

    # Model conversion settings
    attention_implementation: str = "original"  # "original", "split_einsum"
    convert_unet: bool = True
    convert_vae: bool = True
    convert_text_encoder: bool = True

    # Optimization flags
    use_chunked_inference: bool = True
    chunk_size: int = 2
    use_memory_efficient_attention: bool = True

    # Advanced optimization
    use_palettization: bool = False
    palettization_mode: str = "kmeans"  # "kmeans", "uniform"
    quantization_bits: int = 8  # 8, 4, 2, 1

    # Model-specific optimizations
    reduce_memory_footprint: bool = True
    optimize_for_inference: bool = True
    use_low_precision_weights: bool = True

    # Output settings
    model_format: str = "mlpackage"  # "mlpackage", "mlmodel"
    metadata_description: str = "Style Transfer Model optimized for Apple Silicon"
    metadata_author: str = "EfficientAI-MLX-Toolkit"

    # Validation and testing
    skip_model_validation: bool = False
    generate_test_inputs: bool = True
    test_input_shapes: dict[str, tuple[int, ...]] = None

    def __post_init__(self):
        """Set default test input shapes if not provided."""
        if self.test_input_shapes is None:
            self.test_input_shapes = {
                "image": (1, 3, 512, 512),
                "prompt": (1, 77),
                "timestep": (1,)
            }

    @classmethod
    def from_dict(cls, config_dict: dict[str, any]) -> "CoreMLConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> dict[str, any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def validate(self) -> None:
        """Validate configuration."""
        valid_compute_units = ["all", "cpu_only", "cpu_and_gpu"]
        if self.compute_units not in valid_compute_units:
            raise ValueError(f"compute_units must be one of {valid_compute_units}")

        valid_precisions = ["float16", "float32"]
        if self.precision not in valid_precisions:
            raise ValueError(f"precision must be one of {valid_precisions}")

        valid_attention = ["original", "split_einsum"]
        if self.attention_implementation not in valid_attention:
            raise ValueError(f"attention_implementation must be one of {valid_attention}")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        valid_palettization_modes = ["kmeans", "uniform"]
        if self.palettization_mode not in valid_palettization_modes:
            raise ValueError(f"palettization_mode must be one of {valid_palettization_modes}")

        valid_quantization_bits = [1, 2, 4, 8]
        if self.quantization_bits not in valid_quantization_bits:
            raise ValueError(f"quantization_bits must be one of {valid_quantization_bits}")

        valid_formats = ["mlpackage", "mlmodel"]
        if self.model_format not in valid_formats:
            raise ValueError(f"model_format must be one of {valid_formats}")

    def get_compute_units_enum(self):
        """Get Core ML compute units enum value."""
        import coremltools as ct

        compute_units_map = {
            "all": ct.ComputeUnit.ALL,
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU
        }
        return compute_units_map[self.compute_units]

    def get_precision_enum(self):
        """Get Core ML precision enum value."""
        import coremltools as ct

        precision_map = {
            "float16": ct.precision.FLOAT16,
            "float32": ct.precision.FLOAT32
        }
        return precision_map[self.precision]