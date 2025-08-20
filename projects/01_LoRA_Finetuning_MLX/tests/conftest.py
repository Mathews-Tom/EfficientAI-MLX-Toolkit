"""
Pytest configuration and fixtures for LoRA Fine-tuning Framework tests.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import tempfile
import json
from typing import Dict, Any, List
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_lora_config():
    """Create sample LoRA configuration."""
    from lora import LoRAConfig
    
    return LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=["linear", "q_proj", "v_proj"],
    )


@pytest.fixture
def sample_training_config(temp_dir):
    """Create sample training configuration."""
    from lora import TrainingConfig
    
    return TrainingConfig(
        model_name="test-model",
        dataset_path=temp_dir / "data",
        output_dir=temp_dir / "outputs",
        batch_size=1,
        learning_rate=1e-4,
        num_epochs=1,
        warmup_steps=10,
    )


@pytest.fixture
def sample_inference_config(temp_dir):
    """Create sample inference configuration."""
    from lora import InferenceConfig
    
    return InferenceConfig(
        model_path=temp_dir / "model",
        max_length=50,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )


@pytest.fixture
def simple_model():
    """Create simple test model."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 32, bias=True)
            self.linear2 = nn.Linear(32, 16, bias=True) 
            self.embedding = nn.Embedding(100, 64)
            
        def __call__(self, x):
            if len(x.shape) == 2 and x.dtype == mx.int32:
                # Input is token IDs for embedding
                x = self.embedding(x)
                # Take mean over sequence dimension
                x = mx.mean(x, axis=1)
            
            x = self.linear1(x)
            x = nn.relu(x)
            x = self.linear2(x)
            return x
    
    return SimpleModel()


@pytest.fixture
def sample_dataset():
    """Create sample dataset."""
    # Create mock dataset with proper structure
    data = []
    for i in range(10):
        data.append({
            "input_ids": mx.array([1, 2, 3, 4, 5]),
            "attention_mask": mx.array([1, 1, 1, 1, 1]),
            "labels": mx.array([2, 3, 4, 5, 6]),
        })
    return data


@pytest.fixture
def sample_config_file(temp_dir, sample_lora_config, sample_training_config):
    """Create sample configuration file."""
    config_data = {
        "lora": {
            "rank": sample_lora_config.rank,
            "alpha": sample_lora_config.alpha,
            "dropout": sample_lora_config.dropout,
            "target_modules": sample_lora_config.target_modules,
        },
        "training": {
            "model_name": sample_training_config.model_name,
            "dataset_path": str(sample_training_config.dataset_path),
            "output_dir": str(sample_training_config.output_dir),
            "batch_size": sample_training_config.batch_size,
            "learning_rate": sample_training_config.learning_rate,
            "num_epochs": sample_training_config.num_epochs,
        },
    }
    
    config_path = temp_dir / "test_config.yaml"
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return config_path


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            self.pad_token_id = 0
            self.eos_token_id = 2
            
        def encode(self, text: str, **kwargs):
            # Simple mock encoding
            tokens = [hash(char) % self.vocab_size for char in text[:10]]
            if kwargs.get("return_tensors") == "mlx":
                return [mx.array(tokens)]
            return tokens
            
        def decode(self, tokens, **kwargs):
            # Simple mock decoding
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            return f"decoded_{len(tokens)}_tokens"
    
    return MockTokenizer()


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "apple_silicon: marks tests that require Apple Silicon hardware"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip Apple Silicon tests when MLX not available."""
    try:
        import mlx.core as mx
        mlx_available = True
    except ImportError:
        mlx_available = False
    
    if not mlx_available:
        skip_mlx = pytest.mark.skip(reason="MLX not available")
        for item in items:
            if "apple_silicon" in item.keywords:
                item.add_marker(skip_mlx)