"""
Configuration classes for pruning operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
from enum import Enum


class PruningMethod(str, Enum):
    """Supported pruning methods."""
    MAGNITUDE = "magnitude"
    GRADIENT = "gradient"
    FISHER = "fisher"
    RANDOM = "random"
    LOTTERY_TICKET = "lottery_ticket"


class PruningSchedule(str, Enum):
    """Pruning schedules."""
    ONESHOT = "oneshot"
    GRADUAL = "gradual"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"


class PruningCriterion(str, Enum):
    """Pruning criteria."""
    L1 = "l1"
    L2 = "l2"
    FISHER_INFORMATION = "fisher_information"


@dataclass
class PruningConfig:
    """Configuration for pruning operations."""
    
    # Basic pruning settings
    target_sparsity: float = 0.5
    structured: bool = False
    block_size: Tuple[int, int] = (2, 4)
    
    # Pruning strategy
    method: PruningMethod = PruningMethod.MAGNITUDE
    criterion: PruningCriterion = PruningCriterion.L1
    schedule: PruningSchedule = PruningSchedule.GRADUAL
    
    # Gradual pruning settings
    start_epoch: int = 5
    end_epoch: int = 15
    frequency: int = 2
    initial_sparsity: float = 0.0
    
    # Fine-tuning after pruning
    recovery_epochs: int = 10
    recovery_lr: float = 1e-4
    recovery_warmup_steps: int = 100
    
    # Layer-specific settings
    exclude_layers: List[str] = field(default_factory=lambda: ["embedding", "lm_head"])
    layer_wise_sparsity: Dict[str, float] = field(default_factory=dict)
    
    # Advanced options
    global_magnitude_pruning: bool = True
    normalize_by_layer: bool = True
    preserve_output_layer: bool = True
    
    # Recovery training settings
    recovery_optimizer: str = "adamw"
    recovery_weight_decay: float = 0.01
    recovery_gradient_clipping: float = 1.0
    
    # Output settings
    pruned_model_path: Optional[Path] = None
    save_pruning_masks: bool = True
    save_pruning_stats: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.target_sparsity <= 1.0:
            raise ValueError(f"target_sparsity must be between 0 and 1, got {self.target_sparsity}")
        
        if not 0.0 <= self.initial_sparsity <= self.target_sparsity:
            raise ValueError(f"initial_sparsity must be between 0 and target_sparsity")
        
        if self.start_epoch < 0:
            raise ValueError(f"start_epoch must be non-negative, got {self.start_epoch}")
        
        if self.end_epoch <= self.start_epoch:
            raise ValueError(f"end_epoch ({self.end_epoch}) must be greater than start_epoch ({self.start_epoch})")
        
        if self.frequency <= 0:
            raise ValueError(f"frequency must be positive, got {self.frequency}")
        
        if self.recovery_epochs < 0:
            raise ValueError(f"recovery_epochs must be non-negative, got {self.recovery_epochs}")
        
        if self.recovery_lr <= 0:
            raise ValueError(f"recovery_lr must be positive, got {self.recovery_lr}")
        
        # Validate layer-wise sparsity values
        for layer_name, sparsity in self.layer_wise_sparsity.items():
            if not 0.0 <= sparsity <= 1.0:
                raise ValueError(f"layer_wise_sparsity for {layer_name} must be between 0 and 1")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PruningConfig":
        """Create configuration from dictionary."""
        # Convert string enums
        if "method" in config_dict:
            config_dict["method"] = PruningMethod(config_dict["method"])
        if "criterion" in config_dict:
            config_dict["criterion"] = PruningCriterion(config_dict["criterion"])
        if "schedule" in config_dict:
            config_dict["schedule"] = PruningSchedule(config_dict["schedule"])
        
        # Convert paths
        if "pruned_model_path" in config_dict and config_dict["pruned_model_path"]:
            config_dict["pruned_model_path"] = Path(config_dict["pruned_model_path"])
        
        # Convert block_size to tuple if it's a list
        if "block_size" in config_dict and isinstance(config_dict["block_size"], list):
            config_dict["block_size"] = tuple(config_dict["block_size"])
        
        # Ensure numeric fields are proper types
        numeric_fields = ["target_sparsity", "initial_sparsity", "start_epoch", "end_epoch", 
                         "frequency", "recovery_epochs", "recovery_lr", "recovery_warmup_steps",
                         "recovery_weight_decay", "recovery_gradient_clipping"]
        
        for field in numeric_fields:
            if field in config_dict and config_dict[field] is not None:
                if field in ["target_sparsity", "initial_sparsity", "recovery_lr", 
                           "recovery_weight_decay", "recovery_gradient_clipping"]:
                    config_dict[field] = float(config_dict[field])
                else:
                    config_dict[field] = int(config_dict[field])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value
        return result
    
    def get_sparsity_for_layer(self, layer_name: str) -> float:
        """Get sparsity setting for a specific layer."""
        # Check if layer should be excluded
        if any(exclude in layer_name for exclude in self.exclude_layers):
            return 0.0
        
        # Check for layer-specific sparsity
        if layer_name in self.layer_wise_sparsity:
            return self.layer_wise_sparsity[layer_name]
        
        # Return global target sparsity
        return self.target_sparsity
    
    def should_prune_layer(self, layer_name: str) -> bool:
        """Check if a layer should be pruned."""
        # Don't prune excluded layers
        if any(exclude in layer_name for exclude in self.exclude_layers):
            return False
        
        # Don't prune if sparsity is 0
        return self.get_sparsity_for_layer(layer_name) > 0.0
    
    def get_pruning_schedule_steps(self, total_epochs: int) -> List[int]:
        """Get the epochs when pruning should occur."""
        if self.schedule == PruningSchedule.ONESHOT:
            return [self.start_epoch]
        
        steps = []
        for epoch in range(self.start_epoch, min(self.end_epoch + 1, total_epochs), self.frequency):
            steps.append(epoch)
        
        return steps
    
    def calculate_sparsity_for_epoch(self, current_epoch: int, total_epochs: int) -> float:
        """Calculate target sparsity for current epoch based on schedule."""
        if current_epoch < self.start_epoch:
            return self.initial_sparsity
        
        if current_epoch >= self.end_epoch:
            return self.target_sparsity
        
        if self.schedule == PruningSchedule.ONESHOT:
            return self.target_sparsity if current_epoch >= self.start_epoch else self.initial_sparsity
        
        elif self.schedule == PruningSchedule.GRADUAL:
            # Linear interpolation
            progress = (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return self.initial_sparsity + progress * (self.target_sparsity - self.initial_sparsity)
        
        elif self.schedule == PruningSchedule.POLYNOMIAL:
            # Polynomial schedule (power of 3)
            progress = (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            sparsity_increase = (self.target_sparsity - self.initial_sparsity) * (progress ** 3)
            return self.initial_sparsity + sparsity_increase
        
        elif self.schedule == PruningSchedule.EXPONENTIAL:
            # Exponential schedule
            progress = (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            sparsity_increase = (self.target_sparsity - self.initial_sparsity) * (1 - (1 - progress) ** 2)
            return self.initial_sparsity + sparsity_increase
        
        else:
            return self.target_sparsity
    
    def get_compression_ratio(self) -> float:
        """Calculate theoretical compression ratio from sparsity."""
        return 1.0 / (1.0 - self.target_sparsity)
    
    def get_parameter_reduction(self) -> float:
        """Calculate percentage of parameters removed."""
        return self.target_sparsity * 100