#!/usr/bin/env python3
"""
Configuration dataclass for CLIP fine-tuning.

Defines all configurable parameters for domain-specific CLIP fine-tuning
with PyTorch MPS backend support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CLIPFinetuningConfig:
    """Configuration for CLIP fine-tuning.

    Attributes:
        model_name: HuggingFace model identifier for CLIP model
        domain: Domain for fine-tuning (e.g., medical, industrial, scientific, general)
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training (None = auto-determined based on memory)
        num_epochs: Number of training epochs
        max_sequence_length: Maximum sequence length for text tokenization
        image_resolution: Target resolution for input images
        use_mps: Whether to use MPS backend for Apple Silicon GPU acceleration
        mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of gradient accumulation steps
        output_dir: Directory for saving model checkpoints and logs
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for gradient clipping
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate model every N steps
        logging_steps: Log metrics every N steps
        seed: Random seed for reproducibility
    """

    # Model configuration
    model_name: str = "openai/clip-vit-base-patch32"
    domain: str = "general"

    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int | None = None  # Auto-determined if None
    num_epochs: int = 10
    max_sequence_length: int = 77
    image_resolution: int = 224

    # Hardware optimization
    use_mps: bool = True
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

    # Output and logging
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # Data pipeline settings
    data_path: Path = field(default_factory=lambda: Path("data"))
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    shuffle: bool = True
    augment_images: bool = True
    augment_text: bool = False

    # Advanced training parameters
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate domain
        valid_domains = {"general", "medical", "industrial", "scientific"}
        if self.domain not in valid_domains:
            raise ValueError(
                f"Invalid domain '{self.domain}'. Must be one of {valid_domains}"
            )

        # Validate learning rate
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")

        # Validate batch size if provided
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")

        # Validate epochs
        if self.num_epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, got {self.num_epochs}")

        # Validate gradient accumulation steps
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"Gradient accumulation steps must be positive, got {self.gradient_accumulation_steps}"
            )

        # Validate data split ratios
        total_split = self.train_split + self.val_split + self.test_split
        if not (0.99 <= total_split <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Data splits must sum to 1.0, got {total_split} "
                f"(train={self.train_split}, val={self.val_split}, test={self.test_split})"
            )

        if self.num_workers < 0:
            raise ValueError(f"Number of workers must be non-negative, got {self.num_workers}")

    def to_dict(self) -> dict[str, object]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "model_name": self.model_name,
            "domain": self.domain,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "max_sequence_length": self.max_sequence_length,
            "image_resolution": self.image_resolution,
            "use_mps": self.use_mps,
            "mixed_precision": self.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "output_dir": str(self.output_dir),
            "data_path": str(self.data_path),
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "augment_images": self.augment_images,
            "augment_text": self.augment_text,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, object]) -> CLIPFinetuningConfig:
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            CLIPFinetuningConfig instance
        """
        return cls(**config_dict)
