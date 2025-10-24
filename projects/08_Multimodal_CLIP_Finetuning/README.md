# Multimodal CLIP Fine-tuning Framework

Domain-specific CLIP fine-tuning with PyTorch MPS optimization for specialized image-text understanding on Apple Silicon.

## Overview

This project implements a CLIP fine-tuning framework optimized for Apple Silicon (M1/M2/M3) with MPS backend support. It enables domain-specific adaptation of CLIP models for medical, industrial, scientific, and other specialized domains.

## Features

- **MPS Optimization**: Automatic detection and utilization of Apple Silicon GPU acceleration
- **Domain-Specific Fine-tuning**: Pre-configured settings for medical, industrial, and scientific domains
- **Memory-Efficient Training**: Dynamic batch sizing and mixed precision support
- **Flexible Configuration**: YAML-based configuration with sensible defaults

## Current Implementation Status

**MULT-002: CLIP Model Integration** (Completed)

This ticket implements the foundation for CLIP fine-tuning:

- ✅ CLIPFinetuningConfig dataclass with comprehensive validation
- ✅ DeviceManager for MPS backend detection and configuration
- ✅ CLIPFinetuningController for model loading and setup
- ✅ Comprehensive test suite with hardware-specific markers
- ✅ Default configuration file

**Next Steps** (Future Tickets):

- MULT-003: Multimodal Data Pipeline
- MULT-004: Contrastive Learning Implementation
- MULT-005: Memory Management System
- MULT-006: Training Loop Implementation

## Requirements

- Python 3.11+
- PyTorch 2.0+ with MPS support
- transformers 4.30+
- PIL/Pillow
- Apple Silicon hardware (M1/M2/M3) for MPS acceleration

## Installation

From the toolkit root:

```bash
# Install dependencies
uv sync

# Or from this project directory
cd projects/08_Multimodal_CLIP_Finetuning
uv sync
```

## Dependencies

The project uses the following key dependencies (defined in toolkit's `pyproject.toml`):

- `torch>=2.0.0` - PyTorch with MPS backend
- `transformers>=4.30.0` - HuggingFace transformers for CLIP models
- `pillow` - Image processing

## Usage

### Basic Usage

```python
from pathlib import Path
from config import CLIPFinetuningConfig
from model import CLIPFinetuningController

# Create configuration
config = CLIPFinetuningConfig(
    model_name="openai/clip-vit-base-patch32",
    domain="medical",
    learning_rate=5e-5,
    batch_size=16,
    num_epochs=10,
    use_mps=True,  # Enable MPS acceleration
    mixed_precision=True,  # Enable mixed precision training
)

# Initialize controller
controller = CLIPFinetuningController(config)

# Setup model (loads from HuggingFace, moves to MPS device)
controller.setup()

# Get model state
state = controller.get_model_state()
print(f"Model running on: {state['device_type']}")
print(f"Apple Silicon detected: {state['is_apple_silicon']}")
```

### Text and Image Encoding

```python
from PIL import Image

# Encode text
text_features = controller.encode_text(["a photo of a cat", "a photo of a dog"])

# Encode images
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
image_features = controller.encode_image(images)

# Compute similarity
similarity = controller.compute_similarity(
    text=["a medical scan", "a healthy tissue"],
    images=[Image.open("scan1.jpg"), Image.open("scan2.jpg")]
)
```

### Configuration from YAML

```python
import yaml
from config import CLIPFinetuningConfig

# Load from YAML
with open("configs/default.yaml") as f:
    config_dict = yaml.safe_load(f)

config = CLIPFinetuningConfig.from_dict(config_dict)
```

## Configuration

The default configuration is in `configs/default.yaml`. Key parameters:

### Model Configuration

- `model_name`: HuggingFace model identifier (default: "openai/clip-vit-base-patch32")
- `domain`: Target domain (general, medical, industrial, scientific)

### Training Hyperparameters

- `learning_rate`: Learning rate for optimizer (default: 5e-5)
- `batch_size`: Batch size (null = auto-determined based on memory)
- `num_epochs`: Number of training epochs (default: 10)
- `max_sequence_length`: Maximum text sequence length (default: 77)
- `image_resolution`: Target image resolution (default: 224)

### Hardware Optimization

- `use_mps`: Enable MPS backend for Apple Silicon (default: true)
- `mixed_precision`: Use mixed precision training (default: true)
- `gradient_accumulation_steps`: Gradient accumulation steps (default: 1)

### Advanced Parameters

- `warmup_steps`: Learning rate warmup steps (default: 500)
- `weight_decay`: Weight decay for optimizer (default: 0.01)
- `max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)
- `save_steps`: Save checkpoint frequency (default: 1000)
- `eval_steps`: Evaluation frequency (default: 500)
- `logging_steps`: Logging frequency (default: 100)
- `seed`: Random seed (default: 42)

## Testing

The project includes comprehensive tests with hardware-specific markers.

### Run All Tests

```bash
# From toolkit root
uv run pytest projects/08_Multimodal_CLIP_Finetuning/tests/ -v

# From project directory
cd projects/08_Multimodal_CLIP_Finetuning
uv run pytest tests/ -v
```

### Run Specific Tests

```bash
# Test configuration
uv run pytest tests/test_config.py -v

# Test device manager
uv run pytest tests/test_device.py -v

# Test model controller
uv run pytest tests/test_model.py -v
```

### Apple Silicon Tests

Tests marked with `@pytest.mark.apple_silicon` require Apple Silicon hardware:

```bash
# Run only Apple Silicon tests
uv run pytest tests/ -v -m apple_silicon

# Skip Apple Silicon tests
uv run pytest tests/ -v -m "not apple_silicon"
```

## Project Structure

```
08_Multimodal_CLIP_Finetuning/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # CLIPFinetuningConfig dataclass
│   ├── device_manager.py     # MPS device detection and management
│   └── model.py              # CLIPFinetuningController class
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures
│   ├── test_config.py        # Configuration tests
│   ├── test_device.py        # Device manager tests
│   └── test_model.py         # Model controller tests
├── configs/
│   └── default.yaml          # Default configuration
├── outputs/                  # Training outputs (created automatically)
└── README.md                 # This file
```

## Architecture

### Core Components

1. **CLIPFinetuningConfig**: Configuration dataclass with validation
   - Defines all training and model parameters
   - Validates domains, learning rates, batch sizes
   - Supports serialization to/from dictionaries

2. **DeviceManager**: MPS device detection and management
   - Detects Apple Silicon hardware
   - Checks MPS backend availability
   - Selects appropriate device (MPS/CPU)
   - Logs device information

3. **CLIPFinetuningController**: Main controller for CLIP fine-tuning
   - Loads CLIP model and processor from HuggingFace
   - Manages device placement (MPS/CPU)
   - Enables mixed precision training
   - Provides text/image encoding methods
   - Computes image-text similarity

### Hardware Support

- **Apple Silicon (M1/M2/M3)**: Uses MPS backend for GPU acceleration
- **Intel Macs**: Falls back to CPU
- **Linux/Windows**: CPU only (no MPS support)

The framework automatically detects hardware capabilities and selects the optimal device.

## Domain-Specific Configuration

### Medical Domain

```yaml
domain: "medical"
learning_rate: 3.0e-5
num_epochs: 15
```

### Industrial Domain

```yaml
domain: "industrial"
learning_rate: 5.0e-5
num_epochs: 12
```

### Scientific Domain

```yaml
domain: "scientific"
learning_rate: 4.0e-5
num_epochs: 10
```

## Performance Notes

- **MPS Backend**: Provides 2-3x speedup over CPU on Apple Silicon
- **Mixed Precision**: Reduces memory usage and speeds up training
- **Dynamic Batch Sizing**: Automatically adjusts batch size based on available memory
  - MPS default: 16
  - CPU default: 8

## Limitations (Current Implementation)

- Training loop not yet implemented (future ticket MULT-006)
- Custom loss functions not yet implemented (future ticket MULT-004)
- Data pipeline not yet implemented (future ticket MULT-003)
- Model serving API not yet implemented (future ticket)

## Future Work

See the specification document (`docs/specs/multimodal-clip-finetuning/spec.md`) for planned features:

- Custom contrastive loss functions
- Multi-resolution training
- Memory management optimizations
- FastAPI serving endpoints
- MLOps integration

## Contributing

This project is part of the EfficientAI-MLX-Toolkit. Follow the toolkit's contribution guidelines.

## License

Part of EfficientAI-MLX-Toolkit. See main repository for license information.

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

**Implementation Status**: MULT-002 Completed (CLIP Model Integration)
**Next Ticket**: MULT-003 (Multimodal Data Pipeline)
