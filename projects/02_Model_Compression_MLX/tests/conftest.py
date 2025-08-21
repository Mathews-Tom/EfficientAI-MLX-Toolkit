"""
Test configuration and fixtures for Model Compression tests.
"""

import pytest
from pathlib import Path
import tempfile
import yaml

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture  
def sample_config():
    """Sample configuration for testing."""
    return {
        "quantization": {
            "target_bits": 4,
            "method": "post_training",
            "calibration_samples": 100,
            "use_mlx_quantization": True,
        },
        "pruning": {
            "target_sparsity": 0.5,
            "method": "magnitude",
            "structured": False,
            "recovery_epochs": 5,
        },
        "model": {
            "model_name": "test-model",
            "use_mlx": True,
            "output_dir": "outputs/",
        }
    }


@pytest.fixture
def config_file(temp_dir, sample_config):
    """Create temporary config file."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    class MockModel:
        def __init__(self):
            self.parameters_count = 1000000
            
        def parameters(self):
            return []
            
        def save_weights(self, path):
            pass
    
    return MockModel()


@pytest.fixture
def sample_calibration_data():
    """Sample calibration data for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming various industries.",
        "Apple Silicon provides excellent performance for AI workloads.",
    ]


# Skip markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "apple_silicon: marks tests that require Apple Silicon hardware"
    )
    config.addinivalue_line(
        "markers", "mlx: marks tests that require MLX framework"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several minutes)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
    config.addinivalue_line(
        "markers", "quantization: marks tests related to quantization"
    )
    config.addinivalue_line(
        "markers", "pruning: marks tests related to pruning"
    )
    config.addinivalue_line(
        "markers", "distillation: marks tests related to distillation"
    )