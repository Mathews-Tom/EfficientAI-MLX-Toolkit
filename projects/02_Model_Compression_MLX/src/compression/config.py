"""
Configuration for comprehensive model compression.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

from quantization.config import QuantizationConfig
from pruning.config import PruningConfig


@dataclass 
class CompressionConfig:
    """Configuration for comprehensive model compression."""
    
    # General settings
    model_path: Optional[str] = None
    output_dir: Path = Path("outputs/compressed_models/")
    
    # Compression methods to apply
    enabled_methods: List[str] = field(default_factory=lambda: ["quantization", "pruning"])
    
    # Method-specific configurations
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    
    # Compression strategy
    strategy: str = "sequential"  # "sequential", "parallel", "combined"
    
    # Evaluation settings
    run_evaluation: bool = True
    benchmark_methods: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CompressionConfig":
        """Create configuration from dictionary."""
        # Extract method-specific configs
        quant_config = QuantizationConfig.from_dict(config_dict.get("quantization", {}))
        prune_config = PruningConfig.from_dict(config_dict.get("pruning", {}))
        
        # Remove method configs from main dict to avoid duplicate arguments
        config_dict = config_dict.copy()
        config_dict.pop("quantization", None)
        config_dict.pop("pruning", None)
        
        # Remove sections that are not part of CompressionConfig
        extra_sections = ["distillation", "model", "data", "training", "evaluation", 
                         "benchmarking", "hardware", "logging", "advanced"]
        for section in extra_sections:
            config_dict.pop(section, None)
        
        # Convert output_dir to Path
        if "output_dir" in config_dict:
            config_dict["output_dir"] = Path(config_dict["output_dir"])
        
        return cls(
            quantization=quant_config,
            pruning=prune_config,
            **config_dict
        )