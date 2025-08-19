"""
LoRA adapter management and model integration.

Provides comprehensive adapter management for integrating LoRA layers into 
existing models with automated target module detection and replacement.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
import json
import re

from .config import LoRAConfig
from .layers import LoRALinear, LoRAAttention, LoRAEmbedding, LoRAConv1D


class LoRAAdapter:
    """
    Individual LoRA adapter for a specific layer.
    
    Manages the replacement of original layers with LoRA-adapted versions
    while maintaining the original functionality.
    """
    
    def __init__(
        self,
        layer_name: str,
        original_layer: nn.Module,
        config: LoRAConfig,
        layer_type: str = "linear",
    ):
        self.layer_name = layer_name
        self.original_layer = original_layer
        self.config = config
        self.layer_type = layer_type
        self.lora_layer: Optional[nn.Module] = None
        
        self._create_lora_layer()
    
    def _create_lora_layer(self) -> None:
        """Create appropriate LoRA layer based on original layer type."""
        if self.layer_type == "linear" and isinstance(self.original_layer, nn.Linear):
            self.lora_layer = LoRALinear(
                in_features=self.original_layer.weight.shape[1],
                out_features=self.original_layer.weight.shape[0],
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
                bias=self.original_layer.bias is not None,
            )
            
        elif self.layer_type == "attention":
            # For attention layers, we need to extract dimensions
            hidden_size = self.original_layer.weight.shape[0]
            num_heads = getattr(self.original_layer, 'num_heads', 12)  # Default fallback
            
            self.lora_layer = LoRAAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
                target_modules=self.config.target_modules,
            )
            
        elif self.layer_type == "embedding" and isinstance(self.original_layer, nn.Embedding):
            self.lora_layer = LoRAEmbedding(
                num_embeddings=self.original_layer.weight.shape[0],
                embedding_dim=self.original_layer.weight.shape[1],
                rank=self.config.rank,
                alpha=self.config.alpha,
            )
            
        elif self.layer_type == "conv1d":
            # For Conv1D layers (common in GPT-2)
            self.lora_layer = LoRAConv1D(
                in_features=self.original_layer.weight.shape[1],
                out_features=self.original_layer.weight.shape[0],
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
            )
        
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")
        
        # Copy original weights to LoRA layer
        self._copy_original_weights()
    
    def _copy_original_weights(self) -> None:
        """Copy weights from original layer to LoRA layer."""
        if self.lora_layer is None:
            return
            
        # Copy weights based on layer type
        if hasattr(self.lora_layer, 'linear'):
            # For LoRALinear
            self.lora_layer.linear.weight = self.original_layer.weight.copy()
            if hasattr(self.original_layer, 'bias') and self.original_layer.bias is not None:
                self.lora_layer.linear.bias = self.original_layer.bias.copy()
                
        elif hasattr(self.lora_layer, 'embedding'):
            # For LoRAEmbedding
            self.lora_layer.embedding.weight = self.original_layer.weight.copy()
            
        elif hasattr(self.lora_layer, 'conv1d'):
            # For LoRAConv1D
            self.lora_layer.conv1d.weight = self.original_layer.weight.copy()
            if hasattr(self.original_layer, 'bias') and self.original_layer.bias is not None:
                self.lora_layer.conv1d.bias = self.original_layer.bias.copy()
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into the adapted layer."""
        if self.lora_layer and hasattr(self.lora_layer, 'merge_weights'):
            self.lora_layer.merge_weights()
    
    def get_trainable_parameters(self) -> Dict[str, mx.array]:
        """Get trainable LoRA parameters."""
        if self.lora_layer is None:
            return {}
            
        trainable_params = {}
        for name, param in self.lora_layer.named_parameters():
            if 'lora_' in name:  # Only LoRA parameters are trainable
                trainable_params[f"{self.layer_name}.{name}"] = param
                
        return trainable_params


class AdapterManager:
    """
    Manages multiple LoRA adapters across a model.
    
    Handles automatic target module detection, adapter creation, and
    parameter management for efficient training and inference.
    """
    
    def __init__(self, model: nn.Module, config: LoRAConfig):
        self.model = model
        self.config = config
        self.adapters: Dict[str, LoRAAdapter] = {}
        self.target_modules_patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for target modules."""
        patterns = []
        for module_name in self.config.target_modules:
            # Convert glob-like patterns to regex
            pattern = module_name.replace('*', '.*')
            patterns.append(re.compile(pattern))
        return patterns
    
    def _should_adapt_layer(self, layer_name: str, layer: nn.Module) -> tuple[bool, str]:
        """Check if a layer should be adapted and determine its type."""
        # Check if layer name matches target patterns
        matches_pattern = any(
            pattern.search(layer_name) for pattern in self.target_modules_patterns
        )
        
        if not matches_pattern:
            return False, ""
        
        # Determine layer type
        if isinstance(layer, nn.Linear):
            return True, "linear"
        elif isinstance(layer, nn.Embedding):
            return True, "embedding"
        elif hasattr(layer, '__class__') and 'Conv1D' in layer.__class__.__name__:
            return True, "conv1d"
        elif hasattr(layer, 'q_proj') or hasattr(layer, 'attention'):
            return True, "attention"
        
        return False, ""
    
    def create_adapters(self) -> None:
        """Create LoRA adapters for all target modules in the model."""
        for layer_name, layer in self.model.named_modules():
            should_adapt, layer_type = self._should_adapt_layer(layer_name, layer)
            
            if should_adapt:
                try:
                    adapter = LoRAAdapter(
                        layer_name=layer_name,
                        original_layer=layer,
                        config=self.config,
                        layer_type=layer_type,
                    )
                    self.adapters[layer_name] = adapter
                    
                    # Replace original layer with LoRA layer
                    self._replace_layer(layer_name, adapter.lora_layer)
                    
                except Exception as e:
                    print(f"Warning: Failed to create adapter for {layer_name}: {e}")
                    continue
        
        print(f"Created {len(self.adapters)} LoRA adapters")
    
    def _replace_layer(self, layer_path: str, new_layer: nn.Module) -> None:
        """Replace a layer in the model with the LoRA-adapted version."""
        # Split the path into parts
        path_parts = layer_path.split('.')
        
        # Navigate to parent module
        current_module = self.model
        for part in path_parts[:-1]:
            current_module = getattr(current_module, part)
        
        # Replace the final layer
        setattr(current_module, path_parts[-1], new_layer)
    
    def get_trainable_parameters(self) -> Dict[str, mx.array]:
        """Get all trainable LoRA parameters from all adapters."""
        all_params = {}
        for adapter in self.adapters.values():
            all_params.update(adapter.get_trainable_parameters())
        return all_params
    
    def merge_all_weights(self) -> None:
        """Merge all LoRA weights into their respective layers."""
        for adapter in self.adapters.values():
            adapter.merge_weights()
    
    def save_adapters(self, save_path: Path) -> None:
        """Save LoRA adapter weights and configuration."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.to_yaml(save_path / "adapter_config.yaml")
        
        # Save adapter weights
        adapter_weights = {}
        for name, adapter in self.adapters.items():
            adapter_params = adapter.get_trainable_parameters()
            adapter_weights[name] = {
                param_name: param.tolist() for param_name, param in adapter_params.items()
            }
        
        with open(save_path / "adapter_weights.json", 'w') as f:
            json.dump(adapter_weights, f, indent=2)
        
        # Save adapter metadata
        metadata = {
            "adapter_count": len(self.adapters),
            "target_modules": self.config.target_modules,
            "rank": self.config.rank,
            "alpha": self.config.alpha,
        }
        
        with open(save_path / "adapter_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_adapters(self, load_path: Path) -> None:
        """Load LoRA adapter weights from saved files."""
        load_path = Path(load_path)
        
        # Load weights
        with open(load_path / "adapter_weights.json") as f:
            adapter_weights = json.load(f)
        
        # Apply weights to adapters
        for adapter_name, adapter in self.adapters.items():
            if adapter_name in adapter_weights:
                weights_dict = adapter_weights[adapter_name]
                
                for param_name, weight_list in weights_dict.items():
                    # Convert back to MLX array
                    weight_array = mx.array(weight_list)
                    
                    # Set parameter (this is a simplified approach)
                    # In practice, you'd need more sophisticated parameter loading
                    if hasattr(adapter.lora_layer, param_name.split('.')[-1]):
                        setattr(adapter.lora_layer, param_name.split('.')[-1], weight_array)


class ModelAdapter:
    """
    High-level interface for adapting models with LoRA.
    
    Provides simple interface for adding LoRA adaptation to any supported model
    with automatic configuration and management.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[LoRAConfig] = None,
        target_modules: Optional[List[str]] = None,
    ):
        self.model = model
        
        # Use provided config or create default
        if config is None:
            config = LoRAConfig()
        
        # Override target modules if provided
        if target_modules is not None:
            config.target_modules = target_modules
            
        self.config = config
        self.adapter_manager = AdapterManager(model, config)
        self._is_adapted = False
    
    def adapt_model(self) -> None:
        """Adapt the model with LoRA layers."""
        if self._is_adapted:
            print("Model is already adapted")
            return
            
        print("Adapting model with LoRA...")
        self.adapter_manager.create_adapters()
        self._is_adapted = True
        
        # Print adaptation summary
        self.print_adaptation_summary()
    
    def print_adaptation_summary(self) -> None:
        """Print summary of LoRA adaptation."""
        total_params = sum(p.size for p in self.model.parameters())
        trainable_params = sum(p.size for p in self.get_trainable_parameters().values())
        
        print(f"\n=== LoRA Adaptation Summary ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.4f}%")
        print(f"LoRA rank: {self.config.rank}")
        print(f"LoRA alpha: {self.config.alpha}")
        print(f"Scaling factor: {self.config.scaling_factor:.4f}")
        print(f"Adapted modules: {len(self.adapter_manager.adapters)}")
        print("================================\n")
    
    def get_trainable_parameters(self) -> Dict[str, mx.array]:
        """Get all trainable LoRA parameters."""
        if not self._is_adapted:
            return {}
        return self.adapter_manager.get_trainable_parameters()
    
    def save_adapters(self, save_path: Union[str, Path]) -> None:
        """Save LoRA adapters to disk."""
        if not self._is_adapted:
            raise RuntimeError("Model must be adapted before saving")
        
        self.adapter_manager.save_adapters(Path(save_path))
        print(f"LoRA adapters saved to {save_path}")
    
    def load_adapters(self, load_path: Union[str, Path]) -> None:
        """Load LoRA adapters from disk."""
        if not self._is_adapted:
            raise RuntimeError("Model must be adapted before loading")
        
        self.adapter_manager.load_adapters(Path(load_path))
        print(f"LoRA adapters loaded from {load_path}")
    
    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA weights and return the original model."""
        if not self._is_adapted:
            return self.model
        
        print("Merging LoRA weights...")
        self.adapter_manager.merge_all_weights()
        print("LoRA weights merged successfully")
        
        return self.model
    
    @staticmethod
    def from_pretrained(
        model: nn.Module,
        adapter_path: Union[str, Path],
        config: Optional[LoRAConfig] = None,
    ) -> "ModelAdapter":
        """Load a model with pre-trained LoRA adapters."""
        adapter_path = Path(adapter_path)
        
        # Load config if not provided
        if config is None:
            config = LoRAConfig.from_yaml(adapter_path / "adapter_config.yaml")
        
        # Create adapter and load weights
        model_adapter = ModelAdapter(model, config)
        model_adapter.adapt_model()
        model_adapter.load_adapters(adapter_path)
        
        return model_adapter