# Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will get you up and running with the EfficientAI-MLX-Toolkit in minutes. We'll walk through installation, basic usage, and your first LoRA fine-tuning experiment.

## Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3) for optimal performance
- **Python 3.12+**
- **16GB+ RAM** recommended for model training

## Installation

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/Mathews-Tom/EfficientAI-MLX-Toolkit.git
cd EfficientAI-MLX-Toolkit

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### 2. Verify Installation

```bash
# Check system information
uv run efficientai-toolkit info

# Expected output:
# ✅ Apple Silicon detected: M1 Pro
# ✅ MLX framework available
# ✅ Python 3.12.0
# ✅ Memory: 16.0 GB
```

### 3. Setup Environment

```bash
# Configure for Apple Silicon optimization
uv run efficientai-toolkit setup

# Expected output:
# ✅ MLX framework configured
# ✅ Apple Silicon optimizations enabled
# ✅ Environment ready for training
```

## Your First LoRA Fine-tuning

### 1. Check Project Status

```bash
# List available projects
uv run efficientai-toolkit projects-list

# Get LoRA project information
uv run efficientai-toolkit projects lora-finetuning-mlx info
```

### 2. Validate Configuration

```bash
# Validate default configuration
uv run efficientai-toolkit projects lora-finetuning-mlx validate

# Expected output:
# ✅ Configuration validation passed
# ✅ LoRA parameters valid
# ✅ Training settings valid
# ✅ Dependencies available
```

### 3. Quick Training Run

```bash
# Train with default settings (fast test)
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --epochs 1 \
  --batch-size 1 \
  --learning-rate 2e-4

# Expected output:
# 🚀 Starting LoRA fine-tuning...
# 📊 Model: microsoft/DialoGPT-medium
# 🔧 LoRA rank: 16, alpha: 32
# ⏱️  Epoch 1/1: 100%|████████| 10/10 [00:45<00:00, 4.5s/it]
# ✅ Training completed successfully
# 💾 Model saved to: outputs/lora-finetuned-model
```

### 4. Test Text Generation

```bash
# Generate text with trained model
uv run efficientai-toolkit projects lora-finetuning-mlx generate \
  --model-path outputs/lora-finetuned-model \
  --prompt "Hello, how are you today?" \
  --max-length 50

# Expected output:
# 🤖 Generated text:
# Hello, how are you today? I'm doing well, thank you for asking. 
# How about you? Are you having a good day so far?
```

## Comprehensive Example Workflows

### Workflow 1: Basic Training Pipeline

```bash
# Step 1: Validate everything is ready
uv run efficientai-toolkit projects lora-finetuning-mlx validate

# Step 2: Quick training test
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --epochs 2 --batch-size 2 --learning-rate 3e-4

# Step 3: Test generation
uv run efficientai-toolkit projects lora-finetuning-mlx generate \
  --model-path outputs/lora-finetuned-model \
  --prompt "The future of AI is" \
  --max-length 30

# Step 4: Run tests to verify everything works
uv run efficientai-toolkit test lora-finetuning-mlx
```

### Workflow 2: Hyperparameter Optimization

```bash
# Step 1: Quick optimization run (5 trials)
uv run efficientai-toolkit projects lora-finetuning-mlx optimize \
  --model microsoft/DialoGPT-medium \
  --data projects/01_LoRA_Finetuning_MLX/data/samples/sample_conversations.jsonl \
  --trials 5

# Step 2: Train with best parameters
# (Use the best hyperparameters from optimization output)
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --rank 32 --alpha 64 --learning-rate 1.5e-4 \
  --epochs 5 --batch-size 2

# Step 3: Start inference server
uv run efficientai-toolkit projects lora-finetuning-mlx serve \
  --model-path outputs/lora-finetuned-model \
  --host 127.0.0.1 --port 8000
```

### Workflow 3: Production Deployment

```bash
# Step 1: Train production model
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --config configs/production.yaml \
  --epochs 10 --batch-size 4

# Step 2: Run comprehensive tests
uv run efficientai-toolkit test lora-finetuning-mlx --coverage

# Step 3: Start production server
uv run efficientai-toolkit projects lora-finetuning-mlx serve \
  --model-path outputs/production-model \
  --host 0.0.0.0 --port 8000

# Step 4: Test API endpoint
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, I need help with",
    "max_length": 50,
    "temperature": 0.7
  }'
```

## Common Usage Patterns

### Custom Model Training

```bash
# Train with different model
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --model microsoft/DialoGPT-large \
  --data /path/to/your/data.jsonl \
  --epochs 5 --batch-size 2 \
  --output custom_model_output/

# Train with custom LoRA settings
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --rank 64 --alpha 128 --dropout 0.2 \
  --learning-rate 1e-4 --epochs 3
```

### Batch Processing

```bash
# Process multiple models
for model in microsoft/DialoGPT-medium microsoft/DialoGPT-large; do
    echo "Training with model: $model"
    uv run efficientai-toolkit projects lora-finetuning-mlx optimize \
      --model $model \
      --data projects/01_LoRA_Finetuning_MLX/data/samples/sample_conversations.jsonl \
      --trials 3 \
      --output "results_$(basename $model)"
done
```

### Development Testing

```bash
# Quick development cycle
uv run efficientai-toolkit projects lora-finetuning-mlx train --epochs 1 --batch-size 1
uv run efficientai-toolkit test lora-finetuning-mlx --markers "not slow"
uv run efficientai-toolkit projects lora-finetuning-mlx generate \
  --model-path outputs/lora-finetuned-model \
  --prompt "Quick test" --max-length 20
```

## Performance Tips

### Apple Silicon Optimization

```bash
# Enable all Apple Silicon optimizations
export MLX_ENABLE_UNIFIED_MEMORY=1
export MLX_PRECISION=float16

# Train with optimized settings
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --batch-size 4 --learning-rate 3e-4 \
  --config configs/apple_silicon.yaml
```

### Memory Management

```bash
# Start with small batch size and increase gradually
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --batch-size 1 --epochs 1  # Test memory usage

# If successful, increase batch size
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --batch-size 2 --epochs 1

# Monitor memory with system tools
# Activity Monitor or: htop, or: watch -n 1 'ps aux | grep python'
```

### Training Efficiency

```bash
# Fast training for experimentation
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --rank 8 --alpha 16 \        # Lower rank = faster
  --batch-size 1 \             # Lower memory usage
  --epochs 2                   # Quick testing

# High quality training
uv run efficientai-toolkit projects lora-finetuning-mlx train \
  --rank 32 --alpha 64 \       # Higher rank = better quality
  --batch-size 4 \             # Higher throughput
  --epochs 10                  # More training
```

## Troubleshooting Quick Fixes

### Common Issues

1. **MLX Not Available**:
   ```bash
   # Check MLX installation
   python -c "import mlx.core as mx; print('MLX available:', mx.metal.is_available())"
   
   # Reinstall if needed
   uv add mlx
   ```

2. **Memory Issues**:
   ```bash
   # Reduce batch size
   uv run efficientai-toolkit projects lora-finetuning-mlx train --batch-size 1
   
   # Set memory limit
   export MLX_MEMORY_LIMIT=8192
   ```

3. **Configuration Errors**:
   ```bash
   # Validate configuration
   uv run efficientai-toolkit projects lora-finetuning-mlx validate
   
   # Use default configuration
   uv run efficientai-toolkit projects lora-finetuning-mlx train --config configs/default.yaml
   ```

### Debug Mode

```bash
# Enable debug logging
export EFFICIENTAI_DEBUG=1
uv run efficientai-toolkit projects lora-finetuning-mlx info

# Check all system information
uv run efficientai-toolkit info
```

## Next Steps

### Learn More

- **[CLI Reference](CLI_REFERENCE.md)**: Complete command documentation
- **[Configuration Guide](CONFIGURATION.md)**: Advanced configuration options
- **[Testing Guide](TESTING.md)**: Comprehensive testing strategies
- **[Troubleshooting](TROUBLESHOOTING.md)**: Solutions to common problems

### Advanced Features

- **Hyperparameter Optimization**: Automated parameter tuning
- **Model Comparison**: Compare different models and methods
- **Production Deployment**: FastAPI serving and monitoring
- **Custom Datasets**: Work with your own data

### Community

- **Issues**: Report bugs and request features
- **Discussions**: Join the community discussions
- **Contributions**: Help improve the toolkit

---

**You're ready to go! 🚀 Start with the basic training workflow and explore advanced features as needed.**