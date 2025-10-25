"""CoreML Stable Diffusion Style Transfer MLOps Integration

Integration utilities for connecting CoreML Diffusion project with MLOps infrastructure.
Provides wrapper functions and examples for experiment tracking, model versioning,
deployment, and performance monitoring.
"""

from __future__ import annotations

__all__ = [
    "DiffusionMLOpsTracker",
    "create_diffusion_mlops_client",
]
