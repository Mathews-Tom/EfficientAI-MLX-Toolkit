# Configuration Management

## Overview

The EfficientAI-MLX-Toolkit uses a comprehensive configuration system that supports multiple formats, profiles, and environment variable overrides. All projects utilize standardized configuration patterns for consistency and maintainability.

## Configuration Formats

### Supported Formats

- **YAML** (recommended): Human-readable, supports comments
- **TOML**: Configuration-focused format
- **JSON**: Universal compatibility

### Configuration Structure

```yaml
# Example: projects/01_LoRA_Finetuning_MLX/configs/default.yaml

# LoRA Configuration
lora:
  rank: 16                    # LoRA rank (8-64)
  alpha: 32                   # Scaling factor (typically 2x rank)
  dropout: 0.1                # Dropout rate (0.0-0.3)
  target_modules:             # Modules to adapt
    - "q_proj"
    - "v_proj" 
    - "o_proj"

# Training Configuration
training:
  model_name: "microsoft/DialoGPT-medium"
  batch_size: 2
  learning_rate: 2e-4
  num_epochs: 3
  optimizer: "adamw"          # adamw, sgd, adafactor
  scheduler: "linear"         # linear, cosine, polynomial
  warmup_steps: 100
  weight_decay: 0.01
  gradient_clip_norm: 1.0
  save_steps: 500
  eval_steps: 100

# Dataset Configuration  
data:
  dataset_path: "data/samples/"
  validation_split: 0.1
  max_length: 512
  preprocessing:
    truncation: true
    padding: "max_length"

# Inference Configuration
inference:
  max_length: 100
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  do_sample: true

# Optimization Configuration
optimization:
  rank_range: [8, 64]
  alpha_range: [8.0, 64.0]
  learning_rate_range: [1e-5, 5e-4]
  dropout_range: [0.0, 0.3]
  batch_size_options: [1, 2, 4]
  warmup_steps_range: [50, 200]
  weight_decay_range: [0.0, 0.1]
  optimizer_options: ["adamw", "sgd", "adafactor"]
  scheduler_options: ["linear", "cosine"]

# Output Configuration
output:
  output_dir: "outputs/"
  model_name: "lora-finetuned-model"
  save_model: true
  save_adapters: true
  create_merged_model: false

# MLX Specific Settings
mlx:
  precision: "float16"        # float16, float32, bfloat16
  memory_limit: null          # MB, null for auto
  compile_model: true         # Enable MLX compilation
  use_unified_memory: true    # Apple Silicon optimization
```

## Configuration Profiles

### Profile System

Use profiles for different environments and use cases:

```yaml
# config.yaml
default:
  lora:
    rank: 16
    alpha: 32

# Development profile  
development:
  lora:
    rank: 8        # Faster training
    alpha: 16
  training:
    batch_size: 1  # Lower memory usage
    num_epochs: 1  # Quick testing

# Production profile
production:
  lora:
    rank: 32       # Higher quality
    alpha: 64
  training:
    batch_size: 4
    num_epochs: 10
  mlx:
    compile_model: true
    precision: "float16"

# Optimization profile
optimization:
  optimization:
    rank_range: [8, 16]        # Narrow search space
    alpha_range: [16.0, 32.0]
    learning_rate_range: [1e-4, 3e-4]
```

### Using Profiles

```bash
# Use specific profile
uv run efficientai-toolkit projects lora-finetuning-mlx train --config configs/default.yaml --profile production

# Environment variable
export EFFICIENTAI_PROFILE=development
uv run efficientai-toolkit projects lora-finetuning-mlx train
```

## Environment Variables

### Configuration Overrides

Environment variables can override configuration values using dot notation:

```bash
# Override LoRA rank
export LORA_RANK=32

# Override training batch size  
export TRAINING_BATCH_SIZE=4

# Override learning rate
export TRAINING_LEARNING_RATE=3e-4

# Override model name
export TRAINING_MODEL_NAME="microsoft/DialoGPT-large"
```

### System Environment Variables

```bash
# Debug logging
export EFFICIENTAI_DEBUG=1

# Default configuration directory
export EFFICIENTAI_CONFIG_PATH=/path/to/configs/

# MLX memory limit (MB)
export MLX_MEMORY_LIMIT=8192

# Apple Silicon optimizations
export MLX_ENABLE_UNIFIED_MEMORY=1
export MLX_PRECISION=float16
```

## Configuration Validation

### Automatic Validation

All configurations are automatically validated on load:

```python
from src.lora.config import LoRAConfig, TrainingConfig

# Validation occurs during instantiation
lora_config = LoRAConfig(rank=16, alpha=32, dropout=0.1)
training_config = TrainingConfig(batch_size=2, learning_rate=2e-4)
```

### Manual Validation

```bash
# Validate configuration file
uv run efficientai-toolkit projects lora-finetuning-mlx validate --config configs/custom.yaml

# Validate with specific profile
uv run efficientai-toolkit projects lora-finetuning-mlx validate --config configs/default.yaml --profile production
```

### Validation Rules

**LoRA Configuration:**
- `rank`: Must be positive integer ≤ 128
- `alpha`: Must be positive float, typically 2x rank
- `dropout`: Must be between 0.0 and 1.0
- `target_modules`: Must be non-empty list of valid module names

**Training Configuration:**
- `batch_size`: Must be positive integer
- `learning_rate`: Must be positive float between 1e-6 and 1e-1
- `num_epochs`: Must be positive integer
- `optimizer`: Must be one of ["adamw", "sgd", "adafactor"]
- `scheduler`: Must be one of ["linear", "cosine", "polynomial", "constant"]

**Optimization Ranges:**
- All ranges must be 2-element lists [min, max] where min < max
- Values must be within valid parameter bounds

## Configuration Utilities

### Python API

```python
from utils import ConfigManager
from pathlib import Path

# Load configuration with profile
config = ConfigManager(
    config_path=Path("configs/default.yaml"),
    profile="development"
)

# Type-safe access
batch_size = config.get_with_type("training.batch_size", int, default=2)
learning_rate = config.get_with_type("training.learning_rate", float)
model_name = config.get_with_type("training.model_name", str)

# Environment variable override
debug = config.get_with_type("debug", bool, env_var="EFFICIENTAI_DEBUG")

# Update configuration
config.update("training.num_epochs", 5)
config.save(Path("configs/modified.yaml"))
```

### CLI Configuration Override

```bash
# Override single values
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --config configs/default.yaml \
  --override training.learning_rate=3e-4 \
  --override lora.rank=32

# Multiple overrides
uv run efficientai-toolkit projects lora-finetuning-mlx optimize \
  --model microsoft/DialoGPT-medium \
  --data /path/to/data.jsonl \
  --override optimization.rank_range=[8,32] \
  --override optimization.alpha_range=[16,64]
```

## Best Practices

### Configuration Management

1. **Use Version Control**: Track configuration files in git
2. **Environment Separation**: Separate configs for dev/staging/prod
3. **Documentation**: Comment configuration options thoroughly
4. **Validation**: Always validate configurations before training
5. **Secrets Management**: Never commit sensitive data to configs

### Performance Optimization

1. **Apple Silicon Settings**:
   ```yaml
   mlx:
     precision: "float16"      # Faster inference
     use_unified_memory: true  # Better memory utilization
     compile_model: true       # JIT compilation
   ```

2. **Memory Optimization**:
   ```yaml
   training:
     batch_size: 1            # Start small
     gradient_accumulation: 4  # Effective larger batches
   mlx:
     memory_limit: 8192       # Limit MLX memory usage
   ```

3. **Training Efficiency**:
   ```yaml
   lora:
     rank: 16                 # Balance quality vs speed
     target_modules:          # Fewer modules = faster
       - "q_proj"
       - "v_proj"
   training:
     scheduler: "cosine"      # Better convergence
     warmup_steps: 100        # Stable training start
   ```

### Troubleshooting

**Common Configuration Issues:**

1. **YAML Parsing Error**:
   ```
   Error: Invalid YAML syntax
   ```
   - Check indentation (use spaces, not tabs)
   - Verify quotation marks for strings
   - Ensure proper list formatting

2. **Validation Error**:
   ```
   Error: rank must be between 1 and 128
   ```
   - Check parameter bounds in documentation
   - Verify data types match expected format
   - Use validation command to identify issues

3. **Missing Configuration**:
   ```
   Error: Configuration file not found
   ```
   - Use absolute paths or ensure working directory is correct
   - Check file permissions
   - Verify configuration file exists

4. **Environment Override Not Working**:
   ```
   Warning: Environment variable ignored
   ```
   - Check variable naming (use uppercase with underscores)
   - Ensure proper dot notation for nested values
   - Verify environment variable is exported

### Configuration Templates

Generate configuration templates for new projects:

```bash
# Generate default configuration
uv run efficientai-toolkit projects lora-finetuning-mlx generate-config \
  --output configs/my_experiment.yaml \
  --profile development

# Generate optimization configuration
uv run efficientai-toolkit projects lora-finetuning-mlx generate-config \
  --output configs/optimization.yaml \
  --profile optimization \
  --model microsoft/DialoGPT-medium
```

---

This configuration system provides flexible, validated, and environment-aware configuration management for all toolkit projects.