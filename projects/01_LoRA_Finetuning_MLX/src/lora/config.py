"""
Configuration classes for LoRA fine-tuning framework.

Provides comprehensive configuration management for LoRA parameters, training settings,
and inference configurations with validation and Apple Silicon optimizations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) parameters."""
    
    # Core LoRA parameters
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    
    # Target modules for LoRA adaptation
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
        "gate_proj", "up_proj", "down_proj"       # MLP projections
    ])
    
    # LoRA-specific settings
    fan_in_fan_out: bool = False
    bias: str = "none"  # "none", "all", "lora_only"
    modules_to_save: list[str] = field(default_factory=list)
    
    # MLX-specific optimizations
    use_mlx_quantization: bool = True
    mlx_precision: str = "float16"  # "float16", "bfloat16", "float32"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("Dropout must be between 0 and 1")
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError("Bias must be 'none', 'all', or 'lora_only'")
        if self.mlx_precision not in ["float16", "bfloat16", "float32"]:
            raise ValueError("MLX precision must be 'float16', 'bfloat16', or 'float32'")
    
    @property
    def scaling_factor(self) -> float:
        """LoRA scaling factor (alpha / rank)."""
        return self.alpha / self.rank
    
    @classmethod
    def from_yaml(cls, path: Path) -> "LoRAConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get("lora", {}))
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {"lora": self.__dict__}
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


@dataclass  
class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Model and data
    model_name: str = "microsoft/DialoGPT-medium"
    dataset_path: str | Path = "data/samples/"
    output_dir: Path = Path("outputs/")
    
    # Training hyperparameters
    batch_size: int = 2
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    
    # Optimization settings
    optimizer: str = "adamw"  # "adamw", "sgd", "adafactor"
    scheduler: str = "linear"  # "linear", "cosine", "polynomial"
    
    # MLX-specific settings
    use_mlx: bool = True
    mlx_memory_limit: int | None = None  # MB, None for auto
    gradient_accumulation_steps: int = 1
    
    # Evaluation and checkpointing
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    max_checkpoints: int = 3
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Data processing
    max_sequence_length: int = 512
    data_preprocessing_num_workers: int = 4
    
    def __post_init__(self):
        """Validate and process configuration."""
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
        
        # Handle null values that might come as strings
        if self.mlx_memory_limit == "null" or self.mlx_memory_limit == "None":
            self.mlx_memory_limit = None
        
        # Ensure numeric types are correct
        self.batch_size = int(self.batch_size)
        self.learning_rate = float(self.learning_rate)
        self.num_epochs = int(self.num_epochs)
        self.warmup_steps = int(self.warmup_steps)
        self.weight_decay = float(self.weight_decay)
        self.gradient_clipping = float(self.gradient_clipping)
        
        # Validation
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.optimizer not in ["adamw", "sgd", "adafactor"]:
            raise ValueError("Optimizer must be 'adamw', 'sgd', or 'adafactor'")
        if self.scheduler not in ["linear", "cosine", "polynomial"]:
            raise ValueError("Scheduler must be 'linear', 'cosine', or 'polynomial'")
    
    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get("training", {}))
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        # Convert Path objects to strings for YAML serialization
        config_dict = self.__dict__.copy()
        config_dict["dataset_path"] = str(config_dict["dataset_path"])
        config_dict["output_dir"] = str(config_dict["output_dir"])
        
        with open(path, 'w') as f:
            yaml.dump({"training": config_dict}, f, default_flow_style=False)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    
    # Model loading
    model_path: Path = Path("outputs/best_model/")
    device: str = "mps"  # "mps", "cpu", "mlx"
    precision: str = "float16"
    
    # Generation parameters
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    
    # Batching and performance
    batch_size: int = 1
    num_beams: int = 1
    use_cache: bool = True
    
    # MLX-specific settings
    mlx_memory_efficient: bool = True
    mlx_compile: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        self.model_path = Path(self.model_path)
        
        if self.device not in ["mps", "cpu", "mlx"]:
            raise ValueError("Device must be 'mps', 'cpu', or 'mlx'")
        if self.precision not in ["float16", "bfloat16", "float32"]:
            raise ValueError("Precision must be 'float16', 'bfloat16', or 'float32'")
        if not 0 < self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0 and 2.0")
        if not 0 < self.top_p <= 1.0:
            raise ValueError("Top-p must be between 0 and 1.0")
        if self.top_k <= 0:
            raise ValueError("Top-k must be positive")
    
    @classmethod
    def from_yaml(cls, path: Path) -> "InferenceConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get("inference", {}))
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = self.__dict__.copy()
        config_dict["model_path"] = str(config_dict["model_path"])
        
        with open(path, 'w') as f:
            yaml.dump({"inference": config_dict}, f, default_flow_style=False)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Search space
    rank_range: tuple[int, int] = (8, 64)
    alpha_range: tuple[float, float] = (8.0, 64.0)
    learning_rate_range: tuple[float, float] = (1e-5, 5e-4)
    dropout_range: tuple[float, float] = (0.0, 0.3)
    
    # Optimization settings
    n_trials: int = 20
    metric: str = "perplexity"
    direction: str = "minimize"  # "minimize" or "maximize"
    
    # Search strategy
    sampler: str = "tpe"  # "tpe", "random", "grid"
    pruner: str = "median"  # "median", "hyperband", "none"
    
    # Early stopping for trials
    min_trials: int = 5
    patience: int = 3
    
    def __post_init__(self):
        """Validate configuration."""
        # Convert lists to tuples and ensure proper types
        if isinstance(self.rank_range, list):
            self.rank_range = tuple(int(x) for x in self.rank_range)
        if isinstance(self.alpha_range, list):
            self.alpha_range = tuple(float(x) for x in self.alpha_range)
        if isinstance(self.learning_rate_range, list):
            self.learning_rate_range = tuple(float(x) for x in self.learning_rate_range)
        if isinstance(self.dropout_range, list):
            self.dropout_range = tuple(float(x) for x in self.dropout_range)
            
        if self.rank_range[0] >= self.rank_range[1]:
            raise ValueError("Invalid rank range")
        if self.alpha_range[0] >= self.alpha_range[1]:
            raise ValueError("Invalid alpha range") 
        if self.learning_rate_range[0] >= self.learning_rate_range[1]:
            raise ValueError("Invalid learning rate range")
        if self.dropout_range[0] >= self.dropout_range[1]:
            raise ValueError("Invalid dropout range")
        if self.direction not in ["minimize", "maximize"]:
            raise ValueError("Direction must be 'minimize' or 'maximize'")
        if self.sampler not in ["tpe", "random", "grid"]:
            raise ValueError("Sampler must be 'tpe', 'random', or 'grid'")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load complete configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    return {
        "lora": LoRAConfig(**config_dict.get("lora", {})),
        "training": TrainingConfig(**config_dict.get("training", {})),
        "inference": InferenceConfig(**config_dict.get("inference", {})),
        "optimization": OptimizationConfig(**config_dict.get("optimization", {})),
    }