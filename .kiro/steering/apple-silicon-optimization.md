---
inclusion: always
---

# Apple Silicon Optimization Guidelines

## Core Principles

When developing for the EfficientAI-MLX-Toolkit, always prioritize Apple Silicon optimization:

### MLX Framework Usage
- Use MLX as the primary framework for Apple Silicon optimization
- Leverage MLX's native operations for 3-5x performance improvements over PyTorch
- Implement MLX-native operations wherever possible instead of fallback implementations

### Memory Management
- Optimize for Apple Silicon's unified memory architecture
- Use gradient checkpointing and mixed precision training for memory efficiency
- Implement dynamic batch sizing based on available memory
- Start with batch_size=1 and scale up based on memory constraints

### Hardware Detection
- Always detect Apple Silicon hardware (M1/M2) and configure optimizations automatically
- Provide fallback implementations for non-Apple Silicon hardware
- Use MPS backend for PyTorch when MLX is not available

### Performance Optimization
- Benchmark all implementations on actual Apple Silicon hardware
- Measure and optimize for CPU, MPS GPU, and ANE performance separately
- Provide performance comparisons against baseline implementations

## Code Standards

### Package Management
- Use `uv` as the primary package manager instead of pip or conda
- Configure all projects with `pyproject.toml` using uv specifications
- Maintain isolated environments for each project using uv

### File Operations
- Use `pathlib` for all file and directory operations
- Never use string-based path operations
- Ensure cross-platform compatibility through pathlib usage

### Error Handling
- Implement graceful fallbacks when Apple Silicon optimizations fail
- Provide clear error messages for hardware compatibility issues
- Log optimization status and performance metrics

## Example Implementation Patterns

```python
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

# Correct: Use pathlib for file operations
config_path = Path("config.yaml")
model_path = Path("models") / "checkpoint.mlx"

# Correct: MLX-native implementation
def mlx_optimized_function(x):
    return mx.softmax(x, axis=-1)

# Correct: Hardware detection and optimization
def setup_apple_silicon_optimization():
    if mx.metal.is_available():
        mx.metal.set_memory_limit(16 * 1024**3)  # 16GB limit
        return "mlx"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```