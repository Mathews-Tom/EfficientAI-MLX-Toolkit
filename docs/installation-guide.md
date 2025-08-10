# Installation Guide

## Prerequisites

### System Requirements

- **Hardware**: Apple Silicon Mac (M-series chips)
- **Operating System**: macOS 12.0 (Monterey) or later
- **Python**: 3.11 or later
- **Memory**: Minimum 16GB RAM recommended (32GB for advanced projects)
- **Storage**: At least 50GB free space for models and datasets

### Software Dependencies

- **UV Package Manager**: Modern Python package manager
- **Git**: Version control system
- **Xcode Command Line Tools**: Required for some native dependencies

## Installation Steps

### 1. Install System Dependencies

#### Install Xcode Command Line Tools

```bash
xcode-select --install
```

#### Install UV Package Manager

```bash
# Using Homebrew (recommended)
brew install uv

# Or using pip (if you have Python already)
pip install uv

# Or using the installer script
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Verify UV Installation

```bash
uv --version
# Should output: uv 0.x.x
```

### 2. Clone and Setup Main Repository

#### Clone Repository

```bash
gh repo clone Mathews-Tom/EfficientAI-MLX-Toolkit
cd EfficientAI-MLX-Toolkit
```

#### Create Virtual Environment

```bash
# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate

# Verify activation (should show .venv path)
which python
```

#### Install Main Dependencies

```bash
# Install all dependencies including development and Apple Silicon extras
uv sync --extra dev --extra apple-silicon

# Or install only production dependencies
uv sync
```

### 3. Verify Apple Silicon Optimization

#### Test MLX Installation

```bash
uv run python3 -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}'); print(f'Metal available: {mx.metal.is_available()}')"
```

Expected output:

```
MLX version: 0.x.x
Metal available: True
```

#### Test Core ML Tools

```bash
python -c "import coremltools; print(f'Core ML Tools version: {coremltools.__version__}')"
```

#### Test PyTorch MPS

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 4. Install Individual Projects

#### Option A: Install All Projects

```bash
# Install all projects in development mode
for project in projects/*/; do
    echo "Installing $(basename "$project")..."
    cd "$project"
    uv sync
    cd ../..
done
```

#### Option B: Install Specific Project

```bash
# Example: Install LoRA Fine-tuning project
cd projects/01_LoRA_Finetuning_MLX
uv sync
cd ../..
```

### 5. Verify Installation

#### Run Test Suite

```bash
# Run all tests
uv run pytest tests/ -v

# Run Apple Silicon specific tests
uv run pytest tests/test_apple_silicon.py -v

# Run integration tests
uv run pytest tests/integration/ -v
```

#### Run Benchmark Suite

```bash
# Run quick benchmarks to verify performance
uv run python -m utils.benchmark_runner --quick

# Run hardware detection
uv run python -c "from utils.config_manager import detect_hardware; print(detect_hardware())"
```

## Project-Specific Setup

### LoRA Fine-tuning Framework

```bash
cd projects/01_LoRA_Finetuning_MLX
uv sync

# Download sample model (optional)
uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')"
```

### Core ML Stable Diffusion

```bash
cd projects/02_CoreML_StableDiffusion
uv sync

# Download Core ML Stable Diffusion models
uv run python -m python_coreml_stable_diffusion.torch2coreml --convert-unet --convert-text-encoder --convert-vae-decoder --bundle-resources-for-swift-cli
```

### Model Compression Pipeline

```bash
cd projects/04_CPU_Model_Compression
uv sync

# Install ONNX Runtime for CPU optimization
uv add onnxruntime
```

## Troubleshooting

### Common Issues

#### UV Not Found

```bash
# Add UV to PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### MLX Installation Issues

```bash
# Ensure you're on Apple Silicon
uname -m
# Should output: arm64

# Update to latest MLX
uv add mlx --upgrade
```

#### Memory Issues

```bash
# Check available memory
system_profiler SPHardwareDataType | grep Memory

# For 16GB systems, use smaller batch sizes
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

#### Core ML Tools Issues

```bash
# Install with specific version
uv add "coremltools>=7.0"

# Verify installation
python -c "import coremltools as ct; print(ct.__version__)"
```

### Performance Issues

#### Slow Training

1. Verify Apple Silicon optimization is enabled
2. Check memory usage and reduce batch size if needed
3. Enable mixed precision training
4. Use gradient checkpointing for memory efficiency

#### Model Loading Issues

1. Ensure sufficient disk space for model downloads
2. Check internet connection for model downloads
3. Use local model paths when possible
4. Clear model cache if corrupted: `rm -rf ~/.cache/huggingface/`

### Environment Issues

#### Virtual Environment Problems

```bash
# Remove and recreate virtual environment
rm -rf .venv
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra dev --extra apple-silicon
```

#### Dependency Conflicts

```bash
# Update all dependencies
uv sync --upgrade

# Check for conflicts
uv tree

# Reset to clean state
rm uv.lock
uv sync
```

## Development Setup

### IDE Configuration

#### VS Code

Install recommended extensions:

- Python
- Pylance
- Black Formatter
- isort

#### PyCharm

Configure interpreter to use `.venv/bin/python`

### Pre-commit Hooks

```bash
# Install pre-commit
uv add --dev pre-commit

# Setup hooks
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

### Environment Variables

```bash
# Add to ~/.zshrc or ~/.bashrc
export UV_CACHE_DIR="$HOME/.cache/uv"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MLX_METAL_DEBUG=1  # For debugging MLX issues
```

## Next Steps

After successful installation:

1. **Read Project Overview**: Review `docs/project-overview.md`
2. **Choose Starting Project**: Begin with Project #1 for beginners
3. **Review Steering Rules**: Check `.kiro/steering/` for development guidelines
4. **Run Benchmarks**: Establish performance baselines
5. **Explore Examples**: Check individual project notebooks and examples

For support, check the troubleshooting section or create an issue in the repository.
