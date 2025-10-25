"""P0 Projects MLOps Integration

Integration examples and utilities for connecting P0 projects
with the MLOps infrastructure:

1. LoRA Fine-tuning MLX
2. Model Compression MLX
3. CoreML Stable Diffusion Style Transfer

Each integration provides:
- Experiment tracking with MLFlow
- Data versioning with DVC
- Model deployment with BentoML
- Performance monitoring with Evidently
- Apple Silicon optimized metrics
"""

from __future__ import annotations

__all__ = [
    "lora_finetuning",
    "model_compression",
    "coreml_diffusion",
]
