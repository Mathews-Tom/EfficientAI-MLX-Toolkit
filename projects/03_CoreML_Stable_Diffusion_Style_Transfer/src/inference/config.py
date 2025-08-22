"""Configuration for inference operations."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class InferenceConfig:
    """Configuration for inference operations."""
    
    # Model settings
    model_path: Path | str | None = None
    device: str = "auto"
    batch_size: int = 1
    max_batch_size: int = 4
    
    # Performance settings
    use_attention_slicing: bool = True
    attention_slice_size: str = "auto"
    use_cpu_offload: bool = False
    
    # Server settings (for serving command)
    workers: int = 1
    max_requests_per_worker: int = 100
    timeout: int = 30
    
    # Output settings
    output_format: str = "PNG"
    quality: int = 95
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, any]) -> "InferenceConfig":
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
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        
        if self.batch_size > self.max_batch_size:
            raise ValueError("batch_size cannot be greater than max_batch_size")
        
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        
        if not 1 <= self.quality <= 100:
            raise ValueError("quality must be between 1 and 100")
        
        if self.output_format not in ["PNG", "JPEG", "WEBP"]:
            raise ValueError("output_format must be PNG, JPEG, or WEBP")