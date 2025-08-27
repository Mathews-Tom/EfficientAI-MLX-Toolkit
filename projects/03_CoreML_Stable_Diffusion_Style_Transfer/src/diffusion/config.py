"""Configuration classes for diffusion models."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiffusionConfig:
    """Configuration for Stable Diffusion models."""

    # Model settings
    model_name: str = "runwayml/stable-diffusion-v1-5"
    variant: str = "fp16"
    torch_dtype: str = "float16"

    # Inference settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    scheduler: str = "DPMSolverMultistepScheduler"

    # Safety and performance
    safety_checker: bool = False
    requires_safety_checker: bool = False
    use_attention_slicing: bool = True
    attention_slice_size: str = "auto"
    use_cpu_offload: bool = False

    # Memory optimization
    enable_memory_efficient_attention: bool = True
    enable_xformers_memory_efficient_attention: bool = False

    # Apple Silicon optimizations
    use_mlx: bool = True
    use_mps: bool = True
    device: str = "auto"

    # Caching
    cache_dir: Path | None = None
    offline_mode: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict[str, any]) -> "DiffusionConfig":
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
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        if not 0 <= self.guidance_scale <= 30:
            raise ValueError("guidance_scale must be between 0 and 30")

        if self.variant not in ["fp16", "fp32"]:
            raise ValueError("variant must be 'fp16' or 'fp32'")

        if self.torch_dtype not in ["float16", "float32"]:
            raise ValueError("torch_dtype must be 'float16' or 'float32'")

        valid_schedulers = [
            "DPMSolverMultistepScheduler",
            "DDIMScheduler",
            "PNDMScheduler",
            "LMSDiscreteScheduler",
            "EulerAncestralDiscreteScheduler",
            "EulerDiscreteScheduler",
        ]
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}")
