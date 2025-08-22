"""Configuration for style transfer operations."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class StyleTransferConfig:
    """Configuration for style transfer operations."""

    # Core style transfer settings
    style_strength: float = 0.8
    content_strength: float = 0.6
    output_resolution: tuple[int, int] = (512, 512)
    preserve_aspect_ratio: bool = True
    upscale_factor: float = 1.0

    # Method selection
    method: str = "diffusion"  # "diffusion" or "neural_style"

    # ControlNet settings
    use_controlnet: bool = False
    controlnet_model: str = "lllyasviel/sd-controlnet-canny"
    control_strength: float = 1.0

    # Diffusion-specific settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    scheduler: str = "DPMSolverMultistepScheduler"

    # Neural style transfer settings
    content_weight: float = 1e5
    style_weight: float = 1e10
    total_variation_weight: float = 30
    num_iterations: int = 1000

    # Output settings
    output_format: str = "PNG"
    quality: int = 95
    save_intermediate: bool = False

    # Device and optimization
    device: str = "auto"
    use_mixed_precision: bool = True
    enable_attention_slicing: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict[str, any]) -> "StyleTransferConfig":
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
        if not 0.0 <= self.style_strength <= 1.0:
            raise ValueError("style_strength must be between 0.0 and 1.0")

        if not 0.0 <= self.content_strength <= 1.0:
            raise ValueError("content_strength must be between 0.0 and 1.0")

        if self.method not in ["diffusion", "neural_style"]:
            raise ValueError("method must be 'diffusion' or 'neural_style'")

        if len(self.output_resolution) != 2:
            raise ValueError("output_resolution must be a tuple of (width, height)")

        if any(dim <= 0 for dim in self.output_resolution):
            raise ValueError("output_resolution dimensions must be positive")

        if self.upscale_factor <= 0:
            raise ValueError("upscale_factor must be positive")

        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        if not 0 <= self.guidance_scale <= 30:
            raise ValueError("guidance_scale must be between 0 and 30")

        if self.output_format not in ["PNG", "JPEG", "WEBP"]:
            raise ValueError("output_format must be PNG, JPEG, or WEBP")

        if not 1 <= self.quality <= 100:
            raise ValueError("quality must be between 1 and 100")
