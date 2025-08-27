"""Configuration for training style transfer models."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training style transfer models."""

    # Basic training settings
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Data settings
    image_size: tuple[int, int] = (512, 512)
    style_dataset_size: int = 1000
    content_dataset_size: int = 5000

    # LoRA fine-tuning settings
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"]
    )

    # Optimization settings
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"  # "fp16", "bf16", "fp32"
    max_grad_norm: float = 1.0

    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant", "polynomial"
    lr_warmup_steps: int = 500
    lr_num_cycles: float = 0.5

    # Loss configuration
    content_loss_weight: float = 1.0
    style_loss_weight: float = 1000.0
    total_variation_weight: float = 10.0
    adversarial_loss_weight: float = 0.1

    # Data augmentation
    use_random_crop: bool = True
    use_horizontal_flip: bool = True
    use_color_jitter: bool = True
    color_jitter_strength: float = 0.1

    # Checkpointing and saving
    save_steps: int = 500
    checkpoint_dir: Path | str = "checkpoints"
    save_total_limit: int = 3
    resume_from_checkpoint: Path | str | None = None

    # Validation and logging
    validation_steps: int = 100
    validation_images: int = 5
    logging_steps: int = 50
    log_dir: Path | str = "logs"

    # Hardware and performance
    device: str = "auto"
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    enable_cpu_offload: bool = False

    # Diffusion-specific settings
    noise_offset: float = 0.1
    snr_gamma: float | None = None
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction"

    # Style transfer specific
    style_mixing_prob: float = 0.5
    content_style_ratio: float = 0.5
    use_style_consistency_loss: bool = True

    # Advanced settings
    use_8bit_adam: bool = False
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "style-transfer-training"
    experiment_name: str | None = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure paths are Path objects
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)

        if isinstance(self.resume_from_checkpoint, str):
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)

    @classmethod
    def from_dict(cls, config_dict: dict[str, any]) -> "TrainingConfig":
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
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if len(self.image_size) != 2:
            raise ValueError("image_size must be a tuple of (width, height)")

        if any(dim <= 0 for dim in self.image_size):
            raise ValueError("image_size dimensions must be positive")

        if not 0 <= self.lora_dropout <= 1:
            raise ValueError("lora_dropout must be between 0 and 1")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")

        valid_mixed_precision = ["fp16", "bf16", "fp32"]
        if self.mixed_precision not in valid_mixed_precision:
            raise ValueError(f"mixed_precision must be one of {valid_mixed_precision}")

        valid_lr_schedulers = ["cosine", "linear", "constant", "polynomial"]
        if self.lr_scheduler not in valid_lr_schedulers:
            raise ValueError(f"lr_scheduler must be one of {valid_lr_schedulers}")

        if self.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be non-negative")

        if not 0 <= self.style_mixing_prob <= 1:
            raise ValueError("style_mixing_prob must be between 0 and 1")

        if not 0 <= self.content_style_ratio <= 1:
            raise ValueError("content_style_ratio must be between 0 and 1")

        if self.save_steps <= 0:
            raise ValueError("save_steps must be positive")

        if self.validation_steps <= 0:
            raise ValueError("validation_steps must be positive")

        if self.logging_steps <= 0:
            raise ValueError("logging_steps must be positive")

        if self.save_total_limit <= 0:
            raise ValueError("save_total_limit must be positive")

        valid_prediction_types = ["epsilon", "v_prediction"]
        if self.prediction_type not in valid_prediction_types:
            raise ValueError(f"prediction_type must be one of {valid_prediction_types}")

    def get_output_dir(self) -> Path:
        """Get the main output directory for this training run."""
        if self.experiment_name:
            return self.checkpoint_dir / self.experiment_name
        else:
            return (
                self.checkpoint_dir / f"run_{self.epochs}epochs_lr{self.learning_rate}"
            )

    def get_device(self) -> str:
        """Get the device to use for training."""
        if self.device != "auto":
            return self.device

        import torch

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
