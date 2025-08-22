# Testing Guide

## Overview

The EfficientAI-MLX-Toolkit provides comprehensive testing capabilities through both unified CLI commands and traditional pytest. The testing system is designed to handle Apple Silicon-specific features, optional dependencies, and various hardware configurations.

## Testing Architecture

### Test Organization

```
tests/
├── conftest.py                 # Test configuration and fixtures
├── test_cli.py                # CLI command testing
├── test_inference.py          # Inference engine tests
├── test_training.py           # Training pipeline tests
├── fixtures/                  # Test data and mock objects
│   ├── sample_config.yaml
│   ├── mock_models.py
│   └── test_data.jsonl
└── integration/               # Integration tests
    ├── test_end_to_end.py
    └── test_optimization.py
```

### Test Categories

The testing system uses pytest markers to categorize tests:

- **`apple_silicon`**: Requires Apple Silicon hardware
- **`requires_mlx`**: Needs MLX framework installed
- **`slow`**: Tests that take significant time (>30s)
- **`integration`**: End-to-end workflow tests
- **`benchmark`**: Performance measurement tests
- **`unit`**: Fast unit tests

## Unified CLI Testing

### Basic Testing Commands

```bash
# Run all tests for all projects
uv run efficientai-toolkit test --all

# Run tests for specific project
uv run efficientai-toolkit test lora-finetuning-mlx

# Run with coverage report
uv run efficientai-toolkit test lora-finetuning-mlx --coverage

# Run with verbose output
uv run efficientai-toolkit test lora-finetuning-mlx --verbose

# Run in parallel (faster execution)
uv run efficientai-toolkit test lora-finetuning-mlx --parallel
```

### Advanced Testing Options

```bash
# Filter by test markers
uv run efficientai-toolkit test lora-finetuning-mlx --markers "not slow"
uv run efficientai-toolkit test lora-finetuning-mlx --markers "apple_silicon"
uv run efficientai-toolkit test lora-finetuning-mlx --markers "unit"

# Combine options
uv run efficientai-toolkit test lora-finetuning-mlx \
  --coverage \
  --verbose \
  --parallel \
  --markers "not slow"
```

### Test Status Example

```bash
$ uv run efficientai-toolkit test lora-finetuning-mlx

🧪 Running tests for: lora-finetuning-mlx
📍 Project path: projects/01_LoRA_Finetuning_MLX
🎯 Test command: uv run pytest tests/ -v

=================== test session starts ===================
tests/test_cli.py::test_info_command PASSED        [ 12%]
tests/test_cli.py::test_validate_command PASSED    [ 25%]
tests/test_cli.py::test_train_command PASSED       [ 37%]
tests/test_inference.py::test_engine_init PASSED   [ 50%]
tests/test_inference.py::test_generation PASSED    [ 62%]
tests/test_training.py::test_trainer_init PASSED   [ 75%]
tests/test_training.py::test_training_loop PASSED  [ 87%]
tests/test_training.py::test_model_saving PASSED   [100%]

=============== 56 passed, 0 failed in 24.5s ===============

✅ All tests passed!
```

## Traditional pytest Usage

### Basic pytest Commands

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_training.py

# Run specific test function
uv run pytest tests/test_training.py::test_trainer_init

# Run with coverage
uv run pytest --cov

# Run with detailed coverage report
uv run pytest --cov --cov-report=html
```

### Marker-based Testing

```bash
# Run only fast tests
uv run pytest -m "not slow"

# Run Apple Silicon specific tests
uv run pytest -m apple_silicon

# Run integration tests
uv run pytest -m integration

# Run unit tests only
uv run pytest -m unit

# Combine markers
uv run pytest -m "unit and not slow"
```

### Performance Testing

```bash
# Run benchmark tests
uv run pytest -m benchmark

# Run with performance profiling
uv run pytest --profile

# Memory usage testing
uv run pytest --memray
```

## Test Configuration

### Environment Setup

```bash
# Set test environment variables
export EFFICIENTAI_TEST_MODE=1
export EFFICIENTAI_DEBUG=1
export MLX_MEMORY_LIMIT=4096

# Run tests
uv run efficientai-toolkit test lora-finetuning-mlx
```

### Custom Test Configuration

Create a `pytest.ini` file in your project:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    apple_silicon: Tests requiring Apple Silicon hardware
    requires_mlx: Tests requiring MLX framework
    slow: Tests that take more than 30 seconds
    integration: End-to-end integration tests
    benchmark: Performance benchmark tests
    unit: Fast unit tests
```

## Writing Tests

### Test Structure

```python
# tests/test_example.py
import pytest
import mlx.core as mx
from pathlib import Path
from src.lora.config import LoRAConfig
from src.training.trainer import LoRATrainer

class TestLoRATrainer:
    """Test suite for LoRA trainer functionality."""

    def test_trainer_initialization(self, sample_config):
        """Test trainer can be initialized with valid config."""
        trainer = LoRATrainer(sample_config)
        assert trainer.config == sample_config
        assert trainer.model is None  # Not loaded yet

    @pytest.mark.requires_mlx
    def test_model_loading(self, sample_config):
        """Test model loading with MLX backend."""
        trainer = LoRATrainer(sample_config)
        trainer.load_model()
        assert trainer.model is not None
        assert hasattr(trainer.model, 'parameters')

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_training_pipeline(self, sample_config, temp_dir):
        """Test complete training pipeline."""
        trainer = LoRATrainer(sample_config)
        trainer.train()

        # Check outputs
        assert (temp_dir / "model").exists()
        assert (temp_dir / "adapters").exists()

    @pytest.mark.apple_silicon
    def test_apple_silicon_optimization(self, sample_config):
        """Test Apple Silicon specific optimizations."""
        if not mx.metal.is_available():
            pytest.skip("Apple Silicon not available")

        trainer = LoRATrainer(sample_config)
        trainer.enable_apple_silicon_optimizations()
        assert trainer.use_unified_memory is True
```

### Fixtures and Mocking

```python
# tests/conftest.py
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
from src.lora.config import LoRAConfig, TrainingConfig

@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return LoRAConfig(
        rank=8,
        alpha=16,
        dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_model():
    """Provide a mock MLX model for testing."""
    model = Mock()
    model.parameters.return_value = [Mock(shape=(100, 50))]
    model.named_modules.return_value = [
        ("layer1.q_proj", Mock()),
        ("layer1.v_proj", Mock())
    ]
    return model

@pytest.fixture
def sample_dataset():
    """Provide sample training data."""
    return [
        {"input": "Hello", "target": "Hello, how can I help you?"},
        {"input": "Goodbye", "target": "Goodbye! Have a great day!"}
    ]
```

### Async Testing

```python
# tests/test_async.py
import pytest
import asyncio
from src.inference.serving import InferenceServer

@pytest.mark.asyncio
async def test_async_generation():
    """Test asynchronous text generation."""
    server = InferenceServer(model_path="/path/to/model")
    await server.start()

    result = await server.generate_async(
        prompt="Hello",
        max_length=10
    )

    assert isinstance(result, str)
    assert len(result) > 0

    await server.shutdown()
```

## Hardware-Specific Testing

### Apple Silicon Tests

```python
@pytest.mark.apple_silicon
def test_mlx_memory_optimization():
    """Test MLX memory optimization on Apple Silicon."""
    if not mx.metal.is_available():
        pytest.skip("Apple Silicon required")

    # Test unified memory usage
    model = load_model_with_mlx()
    memory_usage = mx.metal.get_active_memory()

    assert memory_usage < 1024 * 1024 * 1024  # Less than 1GB
```

### Fallback Testing

```python
def test_cpu_fallback():
    """Test CPU fallback when MLX is not available."""
    with mock.patch('mlx.core.metal.is_available', return_value=False):
        trainer = LoRATrainer(config)
        trainer.load_model()

        assert trainer.device == "cpu"
        assert trainer.model.device == "cpu"
```

## Performance Testing

### Benchmark Tests

```python
@pytest.mark.benchmark
def test_training_performance(benchmark):
    """Benchmark training performance."""
    config = LoRAConfig(rank=16, alpha=32)
    trainer = LoRATrainer(config)

    result = benchmark(trainer.train_one_epoch)

    # Assert performance requirements
    assert result.stats.mean < 30.0  # Less than 30 seconds per epoch
```

### Memory Profiling

```python
@pytest.mark.slow
def test_memory_usage():
    """Test memory usage during training."""
    import tracemalloc

    tracemalloc.start()

    trainer = LoRATrainer(config)
    trainer.train()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Assert memory requirements
    assert peak < 16 * 1024 * 1024 * 1024  # Less than 16GB
```

## Test Data Management

### Sample Data Generation

```python
# tests/fixtures/generate_test_data.py
def generate_conversation_data(num_samples=100):
    """Generate sample conversation data for testing."""
    conversations = []
    for i in range(num_samples):
        conversations.append({
            "input": f"User message {i}",
            "target": f"Assistant response {i}",
            "conversation_id": f"conv_{i}"
        })
    return conversations

def save_test_data(data, path):
    """Save test data to JSONL format."""
    import json
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
```

### Mock External Dependencies

```python
# tests/mocks.py
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_huggingface_model():
    """Mock Hugging Face model loading."""
    with patch('transformers.AutoModel.from_pretrained') as mock:
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock.return_value = mock_model
        yield mock

@pytest.fixture
def mock_mlx_operations():
    """Mock MLX operations for testing without hardware."""
    with patch('mlx.core.array') as mock_array:
        mock_array.return_value = Mock()
        yield mock_array
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.12, 3.13]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install uv
      run: pip install uv

    - name: Install dependencies
      run: uv sync

    - name: Run tests
      run: uv run efficientai-toolkit test --all --coverage

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Apple Silicon Testing

```yaml
# .github/workflows/apple-silicon.yml
name: Apple Silicon Tests
on: [push, pull_request]

jobs:
  test-macos:
    runs-on: macos-14  # Apple Silicon runner

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.12

    - name: Install dependencies
      run: |
        pip install uv
        uv sync

    - name: Run Apple Silicon tests
      run: |
        uv run efficientai-toolkit test lora-finetuning-mlx \
          --markers "apple_silicon" \
          --coverage
```

## Best Practices

### Test Organization

1. **Separate concerns**: Unit, integration, and performance tests
2. **Use fixtures**: Share common setup across tests
3. **Mock external dependencies**: Make tests reliable and fast
4. **Test error conditions**: Verify proper error handling
5. **Keep tests fast**: Use markers to separate slow tests

### Hardware-Aware Testing

1. **Graceful fallbacks**: Test behavior without specialized hardware
2. **Feature detection**: Skip tests when hardware features unavailable
3. **Environment validation**: Verify test environment before running
4. **Resource cleanup**: Ensure tests don't leak memory or resources

### Debugging Tests

```bash
# Run single test with debugging
uv run pytest tests/test_training.py::test_trainer_init -vvv -s

# Drop into debugger on failure
uv run pytest --pdb

# Enable debug logging
EFFICIENTAI_DEBUG=1 uv run pytest tests/ -s

# Profile test performance
uv run pytest --profile tests/test_training.py
```

---

This testing framework ensures comprehensive coverage while being efficient and hardware-aware. Use the unified CLI for everyday testing and pytest directly for development and debugging.