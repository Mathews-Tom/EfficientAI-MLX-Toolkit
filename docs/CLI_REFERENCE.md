# EfficientAI-MLX-Toolkit CLI Reference

## Overview

The EfficientAI-MLX-Toolkit provides a **unified command-line interface** that seamlessly integrates all projects and utilities. The CLI automatically discovers projects and exposes their commands through a single entry point using a `namespace:command` syntax.

This document outlines the two primary ways to interact with the toolkit: the unified CLI and direct project execution for development.

## 1. Unified CLI Execution

The recommended method for interacting with the toolkit. It provides a consistent interface for all projects.

### Command Structure

```bash
# General command structure
uv run efficientai-toolkit <namespace>:<command> [options]

# Global commands
uv run efficientai-toolkit <command> [options]
```

- **`<namespace>`**: The unique identifier for a project (e.g., `lora-finetuning-mlx`).
- **`<command>`**: The specific action to perform within that project.

## 2. Direct Project Execution (Hybrid Approach)

For development and debugging, you can also run a project's CLI directly. This bypasses the unified entry point.

### Command Structure

```bash
# Navigate to the project directory
cd projects/<project-directory>

# Execute its CLI script
uv run python src/cli.py <command> [options]
```

**Example:**

```bash
cd projects/01_LoRA_Finetuning_MLX
uv run python src/cli.py train --epochs 3
```

---

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

#### `projects`

List all available projects with their status and CLI availability.

```bash
uv run efficientai-toolkit projects
```

**Output:**

- Project name (namespace)
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
uv run efficientai-toolkit test <namespace>

# Test with coverage
uv run efficientai-toolkit test <namespace> --coverage

# Test with verbose output
uv run efficientai-toolkit test <namespace> --verbose

# Test with specific markers
uv run efficientai-toolkit test <namespace> --markers "not slow"

# Test in parallel
uv run efficientai-toolkit test <namespace> --parallel
```

**Options:**

- `--all`: Run tests for all discovered projects
- `--coverage, -c`: Generate coverage report
- `--verbose, -v`: Enable verbose output
- `--markers, -m`: Filter tests by pytest markers
- `--parallel, -p`: Run tests in parallel

## Project-Specific Commands

### LoRA Fine-tuning MLX (`lora-finetuning-mlx`)

**Status**: ✅ Completed and Production-Ready

#### Information Commands

##### `info`

Display LoRA project configuration and status.

```bash
uv run efficientai-toolkit lora-finetuning-mlx:info [--config CONFIG]
```

**Options:**

- `--config`: Custom configuration file path (default: `configs/default.yaml`)

##### `validate`

Validate configuration files and dependencies.

```bash
uv run efficientai-toolkit lora-finetuning-mlx:validate [--config CONFIG]
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
uv run efficientai-toolkit lora-finetuning-mlx:train [OPTIONS]
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
uv run efficientai-toolkit lora-finetuning-mlx:train

# Custom parameters
uv run efficientai-toolkit lora-finetuning-mlx:train \
  --epochs 5 --batch-size 4 --learning-rate 3e-4

# Override model and data
uv run efficientai-toolkit lora-finetuning-mlx:train \
  --model microsoft/DialoGPT-medium \
  --data /path/to/data
```

#### Optimization Commands

##### `optimize`

Run automated hyperparameter optimization.

```bash
uv run efficientai-toolkit lora-finetuning-mlx:optimize [OPTIONS]
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
uv run efficientai-toolkit lora-finetuning-mlx:optimize \
  --model microsoft/DialoGPT-medium \
  --data projects/01_LoRA_Finetuning_MLX/data/samples/sample_conversations.jsonl

# Quick optimization
uv run efficientai-toolkit lora-finetuning-mlx:optimize \
  --model microsoft/DialoGPT-medium \
  --data /path/to/data.jsonl \
  --trials 5
```

#### Inference Commands

##### `generate`

Generate text using a trained LoRA model.

```bash
uv run efficientai-toolkit lora-finetuning-mlx:generate [OPTIONS]
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
uv run efficientai-toolkit lora-finetuning-mlx:generate \
  --model-path /path/to/model \
  --prompt "Hello, how are you?"

# Custom parameters
uv run efficientai-toolkit lora-finetuning-mlx:generate \
  --model-path /path/to/model \
  --prompt "The future of AI is" \
  --max-length 50 \
  --temperature 0.8
```

##### `serve`

Start a production inference server.

```bash
uv run efficientai-toolkit lora-finetuning-mlx:serve [OPTIONS]
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
uv run efficientai-toolkit lora-finetuning-mlx:serve \
  --model-path /path/to/model

# Custom host and port
uv run efficientai-toolkit lora-finetuning-mlx:serve \
  --model-path /path/to/model \
  --host 127.0.0.1 \
  --port 8001

# With LoRA adapters
uv run efficientai-toolkit lora-finetuning-mlx:serve \
  --model-path /path/to/base/model \
  --adapter-path /path/to/adapters
```

### Model Compression MLX (`model-compression-mlx`)

**Status**: ✅ Completed and Production-Ready

The Model Compression framework provides comprehensive model optimization techniques including quantization, pruning, knowledge distillation, and performance benchmarking. All features are implemented with real MLX operations for Apple Silicon optimization.

#### Information Commands

##### `info`

Display Model Compression framework configuration and status.

```bash
uv run efficientai-toolkit model-compression-mlx:info [--config CONFIG]
```

**Options:**

- `--config`: Custom configuration file path (default: `configs/default.yaml`)

**Output:**
- Quantization configuration (target bits, method, MLX status)
- Pruning configuration (sparsity, method, structured/unstructured)
- Knowledge distillation settings (temperature, alpha, beta)
- Model configuration and output directories

##### `validate`

Validate configuration files and dependencies.

```bash
uv run efficientai-toolkit model-compression-mlx:validate [--config CONFIG]
```

**Validates:**
- Quantization configuration parameters
- Pruning configuration settings
- Compression configuration consistency
- MLX framework availability
- File paths and dependencies

#### Compression Commands

##### `quantize`

Apply quantization to reduce model precision and size.

```bash
uv run efficientai-toolkit model-compression-mlx:quantize [OPTIONS]
```

**Required Options:**
- `--model-path`: Model name or path to quantize

**Optional Options:**
- `--config`: Configuration file path (default: `configs/default.yaml`)
- `--output`: Override output directory
- `--bits`: Override target bits (4, 8, 16)
- `--method`: Override quantization method (`post_training`, `dynamic`, `qat`)
- `--calibration-data`: Path to calibration data for PTQ

**Examples:**

```bash
# Basic quantization
uv run efficientai-toolkit model-compression-mlx:quantize \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit

# Custom quantization settings
uv run efficientai-toolkit model-compression-mlx:quantize \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \
  --bits 8 --method post_training \
  --output outputs/quantized_8bit

# With calibration data
uv run efficientai-toolkit model-compression-mlx:quantize \
  --model-path my-model \
  --calibration-data data/calibration.jsonl \
  --bits 4
```

##### `prune`

Apply pruning to remove less important weights from the model.

```bash
uv run efficientai-toolkit model-compression-mlx:prune [OPTIONS]
```

**Required Options:**
- `--model-path`: Model name or path to prune

**Optional Options:**
- `--config`: Configuration file path (default: `configs/default.yaml`)
- `--output`: Override output directory
- `--sparsity`: Override target sparsity (0.0-0.9)
- `--method`: Override pruning method (`magnitude`, `gradient`, `structured`)
- `--structured/--no-structured`: Enable/disable structured pruning

**Examples:**

```bash
# Basic pruning
uv run efficientai-toolkit model-compression-mlx:prune \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit

# Custom sparsity
uv run efficientai-toolkit model-compression-mlx:prune \
  --model-path my-model \
  --sparsity 0.7 --method magnitude

# Structured pruning
uv run efficientai-toolkit model-compression-mlx:prune \
  --model-path my-model \
  --structured \
  --sparsity 0.5
```

##### `compress`

Apply comprehensive compression with multiple techniques.

```bash
uv run efficientai-toolkit model-compression-mlx:compress [OPTIONS]
```

**Required Options:**
- `--model-path`: Model name or path to compress

**Optional Options:**
- `--config`: Configuration file path (default: `configs/default.yaml`)
- `--output`: Override output directory
- `--methods`: Compression methods to apply (comma-separated: `quantization`, `pruning`, `distillation`)

**Examples:**

```bash
# Comprehensive compression
uv run efficientai-toolkit model-compression-mlx:compress \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit

# Specific methods only
uv run efficientai-toolkit model-compression-mlx:compress \
  --model-path my-model \
  --methods quantization,pruning

# Custom output
uv run efficientai-toolkit model-compression-mlx:compress \
  --model-path my-model \
  --output results/compressed_model
```

##### `benchmark`

Benchmark different compression methods and measure performance.

```bash
uv run efficientai-toolkit model-compression-mlx:benchmark [OPTIONS]
```

**Required Options:**
- `--model-path`: Model name or path to benchmark

**Optional Options:**
- `--config`: Configuration file path (default: `configs/default.yaml`)
- `--methods`: Compression methods to benchmark (default: `quantization,pruning`)
- `--output`: Output directory for detailed results

**Output Metrics:**
- Model size comparison (before/after compression)
- Inference speed measurements
- Memory usage analysis
- Accuracy/perplexity evaluation
- Compression ratios achieved

**Examples:**

```bash
# Basic benchmarking
uv run efficientai-toolkit model-compression-mlx:benchmark \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit

# Specific methods
uv run efficientai-toolkit model-compression-mlx:benchmark \
  --model-path my-model \
  --methods quantization,pruning,distillation

# Save detailed results
uv run efficientai-toolkit model-compression-mlx:benchmark \
  --model-path my-model \
  --output benchmark_results/
```

#### Advanced Features

##### Knowledge Distillation

The framework includes a complete knowledge distillation system for creating smaller, efficient models:

- **Teacher-Student Training**: Transfer knowledge from larger to smaller models
- **Temperature Scaling**: Configurable softmax temperature for distillation
- **Loss Combination**: Balanced distillation and student losses
- **Performance Monitoring**: Real-time training metrics and evaluation

##### Real MLX Operations

All compression techniques use genuine MLX operations:

- **Quantization**: Real tensor quantization with per-channel/per-tensor support
- **Pruning**: Actual weight analysis and mask application
- **Evaluation**: True performance measurement with MLX streams
- **I/O**: Native MLX model loading and saving

## Configuration Files

### LoRA Project Configuration

The LoRA project uses a comprehensive YAML configuration system located at:

- **Default**: `projects/01_LoRA_Finetuning_MLX/configs/default.yaml`
- **Custom**: Specify with `--config` option

#### Key Configuration Sections

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

### Model Compression Project Configuration

The Model Compression project uses a comprehensive YAML configuration system located at:

- **Default**: `projects/02_Model_Compression_MLX/configs/default.yaml`
- **Custom**: Specify with `--config` option

#### Key Configuration Sections

1. **Quantization Settings** (`quantization`)
   - `target_bits`: Target bit precision (4, 8, 16)
   - `method`: Quantization method (`post_training`, `dynamic`, `qat`)
   - `calibration_samples`: Number of calibration samples (512)
   - `calibration_method`: Calibration approach (`minmax`, `entropy`)
   - `per_channel`: Enable per-channel quantization
   - `use_mlx_quantization`: MLX-native quantization

2. **Pruning Settings** (`pruning`)
   - `target_sparsity`: Target sparsity level (0.0-0.9)
   - `method`: Pruning method (`magnitude`, `gradient`, `structured`)
   - `structured`: Enable structured pruning
   - `recovery_epochs`: Fine-tuning epochs after pruning
   - `gradual_pruning`: Enable gradual pruning schedule

3. **Knowledge Distillation Settings** (`distillation`)
   - `temperature`: Softmax temperature for distillation (4.0)
   - `alpha`: Weight for distillation loss (0.7)
   - `beta`: Weight for student loss (0.3)
   - `use_mlx_distillation`: MLX-native distillation

4. **Model Settings** (`model`)
   - `model_name`: Default model identifier
   - `output_dir`: Output directory for compressed models
   - `use_mlx`: Enable MLX acceleration
   - `precision`: Model precision (`float16`, `float32`)

5. **Benchmarking Settings** (`benchmarking`)
   - `num_samples`: Number of samples for evaluation (100)
   - `batch_size`: Batch size for benchmarking (1)
   - `metrics`: Metrics to compute (`accuracy`, `perplexity`, `speed`, `memory`)
   - `save_results`: Save detailed benchmark results

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

   **Solution**: Check available projects with `uv run efficientai-toolkit projects`

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export EFFICIENTAI_DEBUG=1
uv run efficientai-toolkit lora-finetuning-mlx:info
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
uv run efficientai-toolkit lora-finetuning-mlx:validate

# Run tests
uv run efficientai-toolkit test lora-finetuning-mlx --coverage

# Quick training test
uv run efficientai-toolkit lora-finetuning-mlx:train --epochs 1

# Generation test
uv run efficientai-toolkit lora-finetuning-mlx:generate \
  --model-path /tmp/test-model \
  --prompt "Test generation" \
  --max-length 10
```

### Development Workflow - LoRA

```bash
# 1. Check project status
uv run efficientai-toolkit lora-finetuning-mlx:info

# 2. Validate configuration
uv run efficientai-toolkit lora-finetuning-mlx:validate

# 3. Run tests
uv run efficientai-toolkit test lora-finetuning-mlx

# 4. Training experiment
uv run efficientai-toolkit lora-finetuning-mlx:train \
  --epochs 3 --batch-size 2

# 5. Optimization run
uv run efficientai-toolkit lora-finetuning-mlx:optimize \
  --model microsoft/DialoGPT-medium \
  --data projects/01_LoRA_Finetuning_MLX/data/samples/sample_conversations.jsonl \
  --trials 10
```

### Development Workflow - Model Compression

```bash
# 1. Check project status
uv run efficientai-toolkit model-compression-mlx:info

# 2. Validate configuration
uv run efficientai-toolkit model-compression-mlx:validate

# 3. Run tests
uv run efficientai-toolkit test model-compression-mlx

# 4. Quick quantization experiment
uv run efficientai-toolkit model-compression-mlx:quantize \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \
  --bits 8

# 5. Pruning experiment
uv run efficientai-toolkit model-compression-mlx:prune \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \
  --sparsity 0.5

# 6. Comprehensive compression
uv run efficientai-toolkit model-compression-mlx:compress \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit

# 7. Performance benchmarking
uv run efficientai-toolkit model-compression-mlx:benchmark \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \
  --output benchmark_results/
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
uv run efficientai-toolkit lora-finetuning-mlx:train \
  --config my_experiment.yaml
```

### Batch Processing

Process multiple models or datasets:

```bash
# Script for batch processing
for model in microsoft/DialoGPT-medium microsoft/DialoGPT-small; do
    uv run efficientai-toolkit lora-finetuning-mlx:optimize \
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
uv run efficientai-toolkit lora-finetuning-mlx:train --help
```
