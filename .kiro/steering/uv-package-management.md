---
inclusion: always
---

# UV Package Management Guidelines

## Core UV Usage

Replace all `pip` and `conda` commands with `uv` equivalents throughout the project:

### Environment Setup

```bash
# Instead of: conda create -n m1-ai python=3.11
uv venv m1-ai --python 3.11

# Instead of: conda activate m1-ai
source m1-ai/bin/activate  # or m1-ai\Scripts\activate on Windows

# Instead of: pip install package
uv add package

# Instead of: pip install -r requirements.txt
uv sync
```

### Project Configuration

Use `pyproject.toml` instead of `requirements.txt` for dependency management:

```toml
[project]
name = "efficientai-mlx-toolkit"
version = "0.1.0"
description = "Apple Silicon optimized AI toolkit"
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.0.1",
    "transformers>=4.30.0",
    "datasets>=2.12.0",
    "fastapi>=0.100.0",
    "gradio>=3.35.0",
    "pathlib-mate>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.4.0",
]
apple-silicon = [
    "coremltools>=7.0.0",
    "torch>=2.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "jupyter>=1.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

### Individual Project Setup

Each project should have its own `pyproject.toml`:

```toml
[project]
name = "lora-finetuning-mlx"
version = "0.1.0"
description = "MLX-native LoRA fine-tuning framework"
requires-python = ">=3.11"
dependencies = [
    "mlx-lm>=0.1.0",
    "transformers>=4.30.0",
    "datasets>=2.12.0",
    "efficientai-shared-utils",  # Reference to shared utilities
]

[project.scripts]
lora-train = "lora_finetuning_mlx.cli:train"
lora-infer = "lora_finetuning_mlx.cli:infer"
```

## Installation Commands

### Development Setup

```bash
# Clone and setup main project
gh repo clone Mathews-Tom/EfficientAI-MLX-Toolkit
cd EfficientAI-MLX-Toolkit

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install main project with dev dependencies
uv sync --extra dev --extra apple-silicon

# Install individual project
cd projects/01_LoRA_Finetuning_MLX
uv sync
```

### Production Setup

```bash
# Install only production dependencies
uv sync --no-dev

# Install specific project
uv add --editable projects/01_LoRA_Finetuning_MLX
```

## Dependency Management Best Practices

### Version Pinning

- Pin major versions for stability
- Use compatible release specifiers (~=) for minor updates
- Lock exact versions in production environments

### Shared Dependencies

- Define shared dependencies in the main `pyproject.toml`
- Reference shared utilities as local dependencies in individual projects
- Use workspace configuration for multi-project setups

### Apple Silicon Specific Dependencies

```toml
[project.optional-dependencies]
apple-silicon = [
    "mlx>=0.0.1",
    "mlx-lm>=0.1.0",
    "coremltools>=7.0.0",
    "torch>=2.0.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
]
```

## Migration from pip/conda

### Replace Common Commands

```bash
# Old: pip install package
# New: uv add package

# Old: pip install -e .
# New: uv add --editable .

# Old: pip freeze > requirements.txt
# New: uv export --format requirements-txt > requirements.txt

# Old: pip install -r requirements.txt
# New: uv sync

# Old: conda env export > environment.yml
# New: uv export > uv.lock
```

### Environment Variables

```bash
# Set UV cache directory
export UV_CACHE_DIR="$HOME/.cache/uv"

# Use specific Python version
export UV_PYTHON="3.11"

# Enable verbose output
export UV_VERBOSE=1
```
