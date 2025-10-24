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

**MULT-003: Multimodal Data Pipeline** (Completed)

This ticket implements the complete data loading and preprocessing pipeline:

- ✅ Dataset loading from multiple formats (CSV, JSON, directories, HuggingFace)
- ✅ Image transforms with augmentation support
- ✅ Text tokenization and preprocessing
- ✅ PyTorch Dataset and DataLoader with collation
- ✅ Multi-caption dataset support
- ✅ Prefetch DataLoader for performance
- ✅ 62 comprehensive tests (100% pass rate)

**MULT-004: Contrastive Learning Implementation** (Completed)

This ticket implements custom contrastive loss functions for CLIP fine-tuning:

- ✅ Standard CLIP contrastive loss with temperature scaling
- ✅ Hard negative mining loss (semi-hard, hard, weighted strategies)
- ✅ Domain-specific loss adaptations (medical, industrial, scientific)
- ✅ Multi-scale contrastive learning
- ✅ Temperature scheduling (constant, warmup, cosine, exponential, adaptive)
- ✅ 88 comprehensive tests (100% pass rate)

**Next Steps** (Future Tickets):

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

### Data Pipeline Usage

#### Loading Data from CSV

```python
from pathlib import Path
from data.loaders import DatasetLoader
from data.dataset import CLIPDataset
from data.dataloader import create_dataloader

# Load image-text pairs from CSV
csv_path = Path("data/dataset.csv")
pairs = DatasetLoader.from_csv(csv_path)

# Create dataset
dataset = CLIPDataset(pairs, controller.processor, config)

# Create DataLoader
dataloader = create_dataloader(dataset, config, batch_size=16)

# Iterate over batches
for batch in dataloader:
    pixel_values = batch["pixel_values"]  # [batch_size, 3, 224, 224]
    input_ids = batch["input_ids"]        # [batch_size, 77]
    attention_mask = batch["attention_mask"]  # [batch_size, 77]
```

#### Loading Data from JSON

```python
# Load from JSON format
json_path = Path("data/dataset.json")
pairs = DatasetLoader.from_json(json_path)

dataset = CLIPDataset(pairs, controller.processor, config)
```

#### Loading Data from Directory

```python
# Load from directory with caption files
data_dir = Path("data/images")
pairs = DatasetLoader.from_directory(data_dir)

dataset = CLIPDataset(pairs, controller.processor, config)
```

#### Custom Image Transforms

```python
from data.transforms import ImageTransform

# Create image transform with augmentation
image_transform = ImageTransform(config, augment=True)

# Use in dataset
dataset = CLIPDataset(
    pairs,
    controller.processor,
    config,
    image_transform=image_transform,
)
```

#### Multi-Resolution Training

```python
from data.transforms import MultiResolutionImageTransform

# Create multi-resolution transform
transform = MultiResolutionImageTransform(
    config,
    resolutions=[224, 288, 384],
    augment=True,
)

# Apply at different resolutions
image_224 = transform(image, resolution=224)
image_288 = transform(image, resolution=288)
```

#### Creating Train/Val Splits

```python
from data.dataloader import create_train_val_dataloaders

# Split data
train_size = int(0.8 * len(pairs))
train_pairs = pairs[:train_size]
val_pairs = pairs[train_size:]

# Create datasets
train_dataset = CLIPDataset(train_pairs, controller.processor, config)
val_dataset = CLIPDataset(val_pairs, controller.processor, config)

# Create DataLoaders
train_loader, val_loader = create_train_val_dataloaders(
    train_dataset,
    val_dataset,
    config,
    batch_size=16,
)
```

### Loss Functions Usage

#### Standard Contrastive Loss

```python
from losses import CLIPContrastiveLoss

# Create loss function
loss_fn = CLIPContrastiveLoss(temperature=0.07, learnable_temp=False)

# Get embeddings from CLIP model
image_embeds = model.encode_image(images)  # [batch_size, embed_dim]
text_embeds = model.encode_text(texts)     # [batch_size, embed_dim]

# Compute loss
output = loss_fn(image_embeds, text_embeds)
loss = output["loss"]                       # Total loss
i2t_loss = output["image_to_text_loss"]    # Image-to-text loss
t2i_loss = output["text_to_image_loss"]    # Text-to-image loss
```

#### Hard Negative Mining

```python
from losses import HardNegativeMiningLoss

# Create loss with hard negative mining
loss_fn = HardNegativeMiningLoss(
    temperature=0.07,
    hard_negative_ratio=0.5,
    mining_strategy="semi-hard",  # semi-hard, hard, or weighted
    hard_negative_weight=2.0,
)

output = loss_fn(image_embeds, text_embeds)
loss = output["loss"]
hard_neg_count = output["hard_negative_count"]
hard_neg_ratio = output["hard_negative_ratio_actual"]
```

#### Domain-Specific Loss

```python
from losses import DomainSpecificLoss

# Medical domain
loss_fn = DomainSpecificLoss(
    domain="medical",  # medical, industrial, scientific, general
    temperature=0.07,
    domain_weight=1.5,
)

output = loss_fn(image_embeds, text_embeds)
loss = output["loss"]
```

#### Multi-Scale Loss

```python
from losses import MultiScaleLoss

# Multi-scale contrastive learning
loss_fn = MultiScaleLoss(
    scales=[1.0, 0.75, 0.5],
    base_temperature=0.07,
    scale_weights=[1.0, 0.75, 0.5],
    normalize_weights=True,
)

output = loss_fn(image_embeds, text_embeds)
loss = output["loss"]
scale_losses = output["scale_losses"]      # Loss at each scale
temperatures = output["temperatures"]      # Temperature at each scale
```

#### Temperature Scheduling

```python
from losses import TemperatureScheduler

# Create scheduler
scheduler = TemperatureScheduler(
    initial_temp=0.07,
    min_temp=0.01,
    max_temp=0.1,
    warmup_steps=1000,
    total_steps=10000,
    schedule_type="cosine",  # constant, warmup, cosine, exponential, adaptive
)

# During training loop
for step in range(num_steps):
    # Update temperature
    if config.temperature_scheduling:
        current_temp = scheduler.step(step, loss=loss.item())
        loss_fn.temperature = current_temp

    # ... rest of training ...
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

### Data Pipeline Settings

- `data_path`: Path to dataset directory (default: "data")
- `train_split`: Training data split ratio (default: 0.8)
- `val_split`: Validation data split ratio (default: 0.1)
- `test_split`: Test data split ratio (default: 0.1)
- `num_workers`: Number of data loading workers (default: 4)
- `shuffle`: Whether to shuffle training data (default: true)
- `augment_images`: Apply image augmentation (default: true)
- `augment_text`: Apply text augmentation (default: false)

### Loss Function Settings

- `loss_type`: Type of loss function (contrastive, hard_negative, domain_specific, multi_scale)
- `temperature`: Temperature for scaling logits (default: 0.07)
- `learnable_temp`: Whether temperature is learnable (default: false)
- `hard_negative_ratio`: Ratio of negatives to treat as hard (default: 0.5)
- `hard_negative_mining_strategy`: Strategy for mining (semi-hard, hard, weighted)
- `hard_negative_weight`: Weight multiplier for hard negatives (default: 2.0)
- `domain_weight`: Weight multiplier for domain adjustments (default: 1.0)
- `temperature_scheduling`: Enable temperature scheduling (default: false)
- `temperature_schedule_type`: Scheduling strategy (constant, warmup, cosine, exponential, adaptive)
- `multi_scale_scales`: List of temperature scales (default: [1.0, 0.75, 0.5])
- `multi_scale_weights`: Weights for each scale (default: [1.0, 0.75, 0.5])

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

# Test data pipeline
uv run pytest tests/data/ -v
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
│   ├── model.py              # CLIPFinetuningController class
│   ├── data/                 # Data pipeline module
│   │   ├── __init__.py       # Data module exports
│   │   ├── dataset.py        # CLIPDataset and MultiCaptionCLIPDataset
│   │   ├── transforms.py     # Image and text transforms
│   │   ├── dataloader.py     # DataLoader utilities and collation
│   │   └── loaders.py        # Dataset loaders (CSV, JSON, directory, HF)
│   └── losses/               # Loss functions module
│       ├── __init__.py       # Loss module exports
│       ├── contrastive.py    # Standard CLIP contrastive loss
│       ├── hard_negative.py  # Hard negative mining loss
│       ├── domain_specific.py # Domain-adapted loss
│       ├── multi_scale.py    # Multi-scale contrastive loss
│       └── temperature_scheduler.py # Temperature scheduling
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures
│   ├── test_config.py        # Configuration tests
│   ├── test_device.py        # Device manager tests
│   ├── test_model.py         # Model controller tests
│   ├── data/                 # Data pipeline tests
│   │   ├── __init__.py
│   │   ├── test_dataset.py   # Dataset tests (28 tests)
│   │   ├── test_transforms.py # Transform tests (15 tests)
│   │   ├── test_dataloader.py # DataLoader tests (17 tests)
│   │   └── test_loaders.py   # Loader tests (14 tests)
│   ├── losses/               # Loss function tests
│   │   ├── __init__.py
│   │   ├── test_contrastive.py # Contrastive loss tests (16 tests)
│   │   ├── test_hard_negative.py # Hard negative tests (14 tests)
│   │   ├── test_domain_specific.py # Domain-specific tests (16 tests)
│   │   ├── test_multi_scale.py # Multi-scale tests (19 tests)
│   │   └── test_temperature_scheduler.py # Scheduler tests (23 tests)
│   └── fixtures/             # Test fixtures
│       ├── test_data.csv     # Sample CSV data
│       ├── test_captions.json # Sample JSON data
│       └── test_image_*.jpg  # Sample test images
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
   - Includes data pipeline configuration

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

4. **Data Pipeline**: Complete data loading and preprocessing
   - **DatasetLoader**: Load from CSV, JSON, directories, HuggingFace
   - **CLIPDataset**: PyTorch Dataset with validation
   - **ImageTransform**: Image preprocessing and augmentation
   - **TextTransform**: Text tokenization using CLIP processor
   - **DataLoader utilities**: Collation, batching, prefetching
   - **MultiCaptionCLIPDataset**: Support for multiple captions per image

5. **Loss Functions**: Custom contrastive learning losses
   - **CLIPContrastiveLoss**: Standard CLIP contrastive loss with temperature
   - **HardNegativeMiningLoss**: Focus learning on challenging negatives
   - **DomainSpecificLoss**: Domain-adapted loss for specialized tasks
   - **MultiScaleLoss**: Multi-scale contrastive learning
   - **TemperatureScheduler**: Adaptive temperature scheduling

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

## Data Format Support

### CSV Format

```csv
image_path,caption
image1.jpg,A cat sitting on a mat
image2.jpg,A dog playing in the park
```

### JSON Format

```json
[
  {"image": "image1.jpg", "caption": "A cat sitting on a mat"},
  {"image": "image2.jpg", "caption": "A dog playing in the park"}
]
```

Or with `data` wrapper:

```json
{
  "data": [
    {"image": "image1.jpg", "caption": "A cat sitting on a mat"}
  ]
}
```

### Directory Format

**Option 1: Global caption file (captions.txt)**
```
directory/
  image1.jpg
  image2.jpg
  captions.txt  # Format: "image1.jpg\tCaption text"
```

**Option 2: Individual caption files**
```
directory/
  image1.jpg
  image1.txt  # Contains caption for image1.jpg
  image2.jpg
  image2.txt  # Contains caption for image2.jpg
```

## Limitations (Current Implementation)

- Training loop not yet implemented (future ticket MULT-006)
- Memory management optimizations not yet implemented (future ticket MULT-005)
- Model serving API not yet implemented (future ticket)

## Future Work

See the specification document (`docs/specs/multimodal-clip-finetuning/spec.md`) for planned features:

- ✅ Custom contrastive loss functions (MULT-004 completed)
- Multi-resolution training
- Memory management optimizations (MULT-005)
- Training loop implementation (MULT-006)
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

**Implementation Status**: MULT-004 Completed (Contrastive Learning Implementation)
**Next Ticket**: MULT-005 (Memory Management System)
