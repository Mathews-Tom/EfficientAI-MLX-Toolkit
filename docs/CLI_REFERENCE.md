# EfficientAI-MLX-Toolkit CLI Reference

## Overview

The EfficientAI-MLX-Toolkit provides a **unified command-line interface** that seamlessly integrates all projects and utilities. The CLI automatically discovers projects and exposes their commands through a single entry point.

## Command Structure

```bash
uv run efficientai-toolkit <command> [options]
uv run efficientai-toolkit projects <project-name> <command> [options]
```

## Global Commands

### System Information

#### `info`
Display system information and hardware capabilities.

```bash
uv run efficientai-toolkit info
```

**Output:**
- System platform and version
- Apple Silicon detection (M1/M2/M3)
- MLX framework availability
- Memory information
- Python environment details

#### `setup`
Configure the development environment for Apple Silicon optimization.

```bash
uv run efficientai-toolkit setup
```

**Features:**
- MLX framework installation verification
- Apple Silicon optimization setup
- Environment variable configuration
- Development dependencies check

### Project Management

#### `projects-list`
List all available projects with their status and CLI availability.

```bash
uv run efficientai-toolkit projects-list
```

**Output:**
- Project name mapping
- File system path
- CLI availability status
- Brief description

### Testing

#### `test`
Run tests for specific projects or all projects.

```bash
# Test all projects
uv run efficientai-toolkit test --all

# Test specific project
uv run efficientai-toolkit test <project-name>

# Test with coverage
uv run efficientai-toolkit test <project-name> --coverage

# Test with verbose output
uv run efficientai-toolkit test <project-name> --verbose

# Test with specific markers
uv run efficientai-toolkit test <project-name> --markers "not slow"

# Test in parallel
uv run efficientai-toolkit test <project-name> --parallel
```

**Options:**
- `--all`: Run tests for all discovered projects
- `--coverage, -c`: Generate coverage report
- `--verbose, -v`: Enable verbose output
- `--markers, -m`: Filter tests by pytest markers
- `--parallel, -p`: Run tests in parallel

## Project-Specific Commands

### LoRA Fine-tuning MLX (`lora-finetuning-mlx`)

#### Information Commands

##### `info`
Display LoRA project configuration and status.

```bash
uv run efficientai-toolkit projects lora-finetuning-mlx info [--config CONFIG]
```

**Options:**
- `--config`: Custom configuration file path (default: `configs/default.yaml`)

##### `validate`
Validate configuration files and dependencies.

```bash
uv run efficientai-toolkit projects lora-finetuning-mlx validate [--config CONFIG]
```

**Validates:**
- LoRA configuration parameters
- Training hyperparameters
- Inference settings
- Optimization ranges
- File paths and dependencies

#### Training Commands

##### `train`
Train a LoRA model with specified configuration.

```bash
uv run efficientai-toolkit projects lora-finetuning-mlx train [OPTIONS]
```

**Options:**
- `--config`: Configuration file path (default: `configs/default.yaml`)
- `--model`: Override model name
- `--data`: Override dataset path
- `--output`: Override output directory
- `--epochs`: Override number of epochs
- `--batch-size`: Override batch size
- `--learning-rate`: Override learning rate
- `--rank`: Override LoRA rank
- `--alpha`: Override LoRA alpha

**Examples:**
```bash
# Basic training
uv run efficientai-toolkit projects lora-finetuning-mlx train

# Custom parameters
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --epochs 5 --batch-size 4 --learning-rate 3e-4

# Override model and data
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --model microsoft/DialoGPT-medium \
  --data /path/to/data
```

#### Optimization Commands

##### `optimize`
Run automated hyperparameter optimization.

```bash
uv run efficientai-toolkit projects lora-finetuning-mlx optimize [OPTIONS]
```

**Required Options:**
- `--model`: Model name or path
- `--data`: Dataset path

**Optional Options:**
- `--output`: Output directory (default: `optimization_results/`)
- `--trials`: Number of optimization trials (default: 20)

**Examples:**
```bash
# Basic optimization
uv run efficientai-toolkit projects lora-finetuning-mlx optimize \
  --model microsoft/DialoGPT-medium \
  --data projects/01_LoRA_Finetuning_MLX/data/samples/sample_conversations.jsonl

# Quick optimization
uv run efficientai-toolkit projects lora-finetuning-mlx optimize \
  --model microsoft/DialoGPT-medium \
  --data /path/to/data.jsonl \
  --trials 5
```

#### Inference Commands

##### `generate`
Generate text using a trained LoRA model.

```bash
uv run efficientai-toolkit projects lora-finetuning-mlx generate [OPTIONS]
```

**Required Options:**
- `--model-path`: Path to trained model
- `--prompt`: Input text prompt

**Optional Options:**
- `--adapter-path`: Path to LoRA adapters
- `--max-length`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Nucleus sampling parameter (default: 0.9)
- `--top-k`: Top-k sampling parameter (default: 50)
- `--repetition-penalty`: Repetition penalty (default: 1.1)

**Examples:**
```bash
# Basic generation
uv run efficientai-toolkit projects lora-finetuning-mlx generate \
  --model-path /path/to/model \
  --prompt "Hello, how are you?"

# Custom parameters
uv run efficientai-toolkit projects lora-finetuning-mlx generate \
  --model-path /path/to/model \
  --prompt "The future of AI is" \
  --max-length 50 \
  --temperature 0.8
```

##### `serve`
Start a production inference server.

```bash
uv run efficientai-toolkit projects lora-finetuning-mlx serve [OPTIONS]
```

**Required Options:**
- `--model-path`: Path to trained model

**Optional Options:**
- `--adapter-path`: Path to LoRA adapters
- `--host`: Server host (default: `0.0.0.0`)
- `--port`: Server port (default: 8000)

**Examples:**
```bash
# Basic server
uv run efficientai-toolkit projects lora-finetuning-mlx serve \
  --model-path /path/to/model

# Custom host and port
uv run efficientai-toolkit projects lora-finetuning-mlx serve \
  --model-path /path/to/model \
  --host 127.0.0.1 \
  --port 8001

# With LoRA adapters
uv run efficientai-toolkit projects lora-finetuning-mlx serve \
  --model-path /path/to/base/model \
  --adapter-path /path/to/adapters
```

## Configuration Files

### LoRA Project Configuration

The LoRA project uses a comprehensive YAML configuration system located at:
- **Default**: `projects/01_LoRA_Finetuning_MLX/configs/default.yaml`
- **Custom**: Specify with `--config` option

#### Key Configuration Sections:

1. **LoRA Settings** (`lora`)
   - `rank`: LoRA rank (8-64)
   - `alpha`: Scaling factor (typically 2x rank)
   - `dropout`: Dropout rate (0.0-0.3)
   - `target_modules`: List of modules to adapt

2. **Training Settings** (`training`)
   - `model_name`: Base model identifier
   - `batch_size`, `learning_rate`, `num_epochs`
   - `optimizer`: "adamw", "sgd", "adafactor"
   - `scheduler`: "linear", "cosine", "polynomial"

3. **Inference Settings** (`inference`)
   - `max_length`, `temperature`, `top_p`, `top_k`
   - `repetition_penalty`
   - MLX-specific optimizations

4. **Optimization Settings** (`optimization`)
   - Search space ranges for hyperparameters
   - Number of trials and search strategy
   - Pruning and early stopping configuration

## Environment Variables

The CLI respects the following environment variables:

- `EFFICIENTAI_DEBUG`: Enable debug logging
- `EFFICIENTAI_CONFIG_PATH`: Default configuration directory
- `MLX_MEMORY_LIMIT`: MLX memory limit in MB
- `PYTHONPATH`: Include project paths for module discovery

## Error Handling

### Common Issues

1. **Configuration File Not Found**
   ```
   ❌ Configuration file not found: configs/default.yaml
   ```
   **Solution**: Ensure you're in the correct directory or specify absolute path

2. **MLX Framework Not Available**
   ```
   ⚠️  MLX framework not detected. Falling back to CPU mode.
   ```
   **Solution**: Install MLX on Apple Silicon or use CPU-compatible commands

3. **Project Not Found**
   ```
   ❌ Project 'project-name' not found
   ```
   **Solution**: Check available projects with `uv run efficientai-toolkit projects-list`

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export EFFICIENTAI_DEBUG=1
uv run efficientai-toolkit projects lora-finetuning-mlx info
```

## Performance Tips

### Apple Silicon Optimization

1. **Enable MLX Acceleration**
   - Ensure MLX is properly installed
   - Use appropriate precision (float16 recommended)
   - Enable memory-efficient mode

2. **Memory Management**
   - Start with smaller batch sizes (1-2)
   - Use gradient accumulation for effective larger batches
   - Monitor memory usage during training

3. **Training Efficiency**
   - Use lower LoRA ranks (8-16) for faster training
   - Enable MLX compilation for inference
   - Leverage unified memory architecture

### Testing Performance

- Use `--markers "not slow"` to skip slow tests during development
- Enable parallel testing with `--parallel` for faster execution
- Use coverage sparingly as it adds overhead

## Integration Examples

### CI/CD Pipeline

```bash
# Validate configuration
uv run efficientai-toolkit projects lora-finetuning-mlx validate

# Run tests
uv run efficientai-toolkit test lora-finetuning-mlx --coverage

# Quick training test
uv run efficientai-toolkit projects lora-finetuning-mlx train --epochs 1

# Generation test
uv run efficientai-toolkit projects lora-finetuning-mlx generate \
  --model-path /tmp/test-model \
  --prompt "Test generation" \
  --max-length 10
```

### Development Workflow

```bash
# 1. Check project status
uv run efficientai-toolkit projects lora-finetuning-mlx info

# 2. Validate configuration
uv run efficientai-toolkit projects lora-finetuning-mlx validate

# 3. Run tests
uv run efficientai-toolkit test lora-finetuning-mlx

# 4. Training experiment
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --epochs 3 --batch-size 2

# 5. Optimization run
uv run efficientai-toolkit projects lora-finetuning-mlx optimize \
  --model microsoft/DialoGPT-medium \
  --data projects/01_LoRA_Finetuning_MLX/data/samples/sample_conversations.jsonl \
  --trials 10
```

## Advanced Usage

### Custom Configuration

Create custom configuration files for different experiments:

```bash
# Copy default configuration
cp projects/01_LoRA_Finetuning_MLX/configs/default.yaml my_experiment.yaml

# Edit configuration
# ... modify my_experiment.yaml ...

# Use custom configuration
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --config my_experiment.yaml
```

### Batch Processing

Process multiple models or datasets:

```bash
# Script for batch processing
for model in microsoft/DialoGPT-medium microsoft/DialoGPT-small; do
    uv run efficientai-toolkit projects lora-finetuning-mlx optimize \
      --model $model \
      --data /path/to/data.jsonl \
      --trials 5 \
      --output "results_$(basename $model)"
done
```

---

This CLI reference provides comprehensive coverage of all available commands and options. For additional help, use the `--help` flag with any command:

```bash
uv run efficientai-toolkit --help
uv run efficientai-toolkit projects lora-finetuning-mlx --help
uv run efficientai-toolkit projects lora-finetuning-mlx train --help
```