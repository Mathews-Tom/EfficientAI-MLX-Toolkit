"""
Tests for LoRA layer implementations and adapters.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import tempfile
import json

from src.lora import (
    LoRAConfig,
    LoRALinear,
    LoRAAttention,
    LoRAEmbedding,
    LoRAAdapter,
    AdapterManager,
    ModelAdapter,
)


class TestLoRAConfig:
    """Test LoRA configuration class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = LoRAConfig()
        
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert config.scaling_factor == 2.0
        assert "q_proj" in config.target_modules
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid rank
        with pytest.raises(ValueError, match="LoRA rank must be positive"):
            LoRAConfig(rank=0)
        
        # Test invalid alpha
        with pytest.raises(ValueError, match="LoRA alpha must be positive"):
            LoRAConfig(alpha=-1.0)
        
        # Test invalid dropout
        with pytest.raises(ValueError, match="Dropout must be between 0 and 1"):
            LoRAConfig(dropout=1.5)
    
    def test_config_serialization(self):
        """Test configuration serialization to/from YAML."""
        config = LoRAConfig(rank=32, alpha=64.0, dropout=0.2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            
            # Save config
            config.to_yaml(config_path)
            assert config_path.exists()
            
            # Load config
            loaded_config = LoRAConfig.from_yaml(config_path)
            assert loaded_config.rank == 32
            assert loaded_config.alpha == 64.0
            assert loaded_config.dropout == 0.2


class TestLoRALayers:
    """Test LoRA layer implementations."""
    
    def test_lora_linear_creation(self):
        """Test LoRA linear layer creation."""
        layer = LoRALinear(
            in_features=128,
            out_features=256,
            rank=16,
            alpha=32.0,
            dropout=0.1,
        )
        
        assert layer.in_features == 128
        assert layer.out_features == 256
        assert layer.rank == 16
        assert layer.scaling == 2.0
    
    def test_lora_linear_forward(self):
        """Test LoRA linear layer forward pass."""
        layer = LoRALinear(in_features=64, out_features=32, rank=8)
        
        # Create random input
        x = mx.random.normal((4, 64))
        
        # Forward pass
        output = layer(x)
        
        assert output.shape == (4, 32)
        assert not mx.any(mx.isnan(output))
    
    def test_lora_attention_creation(self):
        """Test LoRA attention layer creation."""
        layer = LoRAAttention(
            hidden_size=512,
            num_heads=8,
            rank=16,
            target_modules=["q_proj", "v_proj"],
        )
        
        assert layer.hidden_size == 512
        assert layer.num_heads == 8
        assert layer.head_dim == 64
        assert len(layer.lora_adapters) == 2
    
    def test_lora_embedding_creation(self):
        """Test LoRA embedding layer creation."""
        layer = LoRAEmbedding(
            num_embeddings=1000,
            embedding_dim=256,
            rank=16,
        )
        
        assert layer.num_embeddings == 1000
        assert layer.embedding_dim == 256
        assert layer.rank == 16
    
    def test_weight_merging(self):
        """Test LoRA weight merging functionality."""
        layer = LoRALinear(in_features=32, out_features=16, rank=4)
        
        # Store original weights
        original_weight = layer.linear.weight.copy()
        
        # Merge weights
        layer.merge_weights()
        
        # Check that rank is reset
        assert layer.rank == 0
        assert layer.lora_A is None
        assert layer.lora_B is None


class TestAdapterManager:
    """Test adapter management functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = LoRAConfig(
            rank=8,
            alpha=16.0,
            target_modules=["linear"],
        )
    
    def test_adapter_creation(self):
        """Test adapter creation for different layer types."""
        # Create simple linear layer
        linear_layer = nn.Linear(64, 32)
        
        adapter = LoRAAdapter(
            layer_name="test_linear",
            original_layer=linear_layer,
            config=self.config,
            layer_type="linear",
        )
        
        assert adapter.layer_name == "test_linear"
        assert adapter.layer_type == "linear"
        assert adapter.lora_layer is not None
    
    def test_trainable_parameters(self):
        """Test trainable parameter extraction."""
        linear_layer = nn.Linear(64, 32)
        
        adapter = LoRAAdapter(
            layer_name="test_linear",
            original_layer=linear_layer,
            config=self.config,
            layer_type="linear",
        )
        
        trainable_params = adapter.get_trainable_parameters()
        
        # Should have LoRA parameters
        assert len(trainable_params) > 0
        assert any("lora_" in name for name in trainable_params.keys())


class TestModelAdapter:
    """Test high-level model adaptation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=[".*linear.*"],  # Match any layer with 'linear' in name
        )
    
    def test_model_adapter_creation(self):
        """Test model adapter creation."""
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 32)
                self.linear2 = nn.Linear(32, 16)
            
            def __call__(self, x):
                x = self.linear1(x)
                x = nn.relu(x)
                return self.linear2(x)
        
        model = SimpleModel()
        adapter = ModelAdapter(model, self.config)
        
        assert adapter.model is model
        assert adapter.config is self.config
        assert not adapter._is_adapted
    
    def test_model_adaptation(self):
        """Test model adaptation process."""
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16)
            
            def __call__(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        adapter = ModelAdapter(model, self.config)
        
        # Adapt model
        adapter.adapt_model()
        
        assert adapter._is_adapted
        assert len(adapter.adapter_manager.adapters) > 0
    
    def test_adapter_serialization(self):
        """Test adapter saving and loading."""
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 8)
        
        model = SimpleModel()
        adapter = ModelAdapter(model, self.config)
        adapter.adapt_model()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "adapters"
            
            # Save adapters
            adapter.save_adapters(save_path)
            
            # Check files exist
            assert (save_path / "adapter_config.yaml").exists()
            assert (save_path / "adapter_weights.json").exists()
            assert (save_path / "adapter_metadata.json").exists()


class TestIntegration:
    """Integration tests for LoRA components."""
    
    def test_end_to_end_adaptation(self):
        """Test complete adaptation workflow."""
        # Create model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear1 = nn.Linear(64, 32)
                self.linear2 = nn.Linear(32, 16)
            
            def __call__(self, x):
                x = self.embedding(x)
                x = self.linear1(x)
                x = nn.relu(x)
                return self.linear2(x)
        
        model = TestModel()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear1", "linear2"],
        )
        
        # Create adapter
        adapter = ModelAdapter(model, config)
        
        # Adapt model
        adapter.adapt_model()
        
        # Test forward pass
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)
        
        assert output.shape == (1, 5, 16)
        assert not mx.any(mx.isnan(output))
        
        # Test trainable parameters
        trainable_params = adapter.get_trainable_parameters()
        assert len(trainable_params) > 0
        
        # Test saving and loading
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_adapters"
            adapter.save_adapters(save_path)
            
            # Create new adapter and load
            new_adapter = ModelAdapter(TestModel(), config)
            new_adapter.adapt_model()
            new_adapter.load_adapters(save_path)


@pytest.mark.benchmark
class TestPerformance:
    """Performance tests for LoRA layers."""
    
    def test_lora_linear_performance(self):
        """Test LoRA linear layer performance."""
        # Create large layer
        layer = LoRALinear(
            in_features=1024,
            out_features=4096,
            rank=64,
        )
        
        # Create large batch
        batch_size = 32
        seq_length = 512
        x = mx.random.normal((batch_size, seq_length, 1024))
        
        # Time forward pass
        import time
        start_time = time.time()
        
        for _ in range(10):
            output = layer(x)
            mx.eval(output)  # Force evaluation
        
        elapsed_time = time.time() - start_time
        print(f"LoRA Linear forward pass time: {elapsed_time:.3f}s for 10 iterations")
        
        assert output.shape == (batch_size, seq_length, 4096)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of LoRA adaptation."""
        # Create model with many parameters
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(2048, 2048) for _ in range(4)]
            
            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        model = LargeModel()
        
        # Count original parameters
        original_params = sum(p.size for p in model.parameters())
        
        # Adapt with LoRA
        config = LoRAConfig(rank=16, target_modules=[".*"])
        adapter = ModelAdapter(model, config)
        adapter.adapt_model()
        
        # Count trainable parameters
        trainable_params = sum(p.size for p in adapter.get_trainable_parameters().values())
        
        # LoRA should significantly reduce trainable parameters
        reduction_ratio = trainable_params / original_params
        print(f"Parameter reduction ratio: {reduction_ratio:.4f}")
        
        assert reduction_ratio < 0.1  # Should be less than 10% of original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])