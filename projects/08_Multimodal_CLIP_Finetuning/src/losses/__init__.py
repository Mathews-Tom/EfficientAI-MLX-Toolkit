"""
Contrastive loss functions for CLIP fine-tuning.

This module provides various contrastive learning loss functions including:
- Standard CLIP contrastive loss with temperature scaling
- Hard negative mining loss
- Domain-specific loss adaptations
- Multi-scale contrastive learning
- Temperature scheduling utilities
"""

from __future__ import annotations

from .contrastive import CLIPContrastiveLoss
from .domain_specific import DomainSpecificLoss
from .hard_negative import HardNegativeMiningLoss
from .multi_scale import MultiScaleLoss
from .temperature_scheduler import TemperatureScheduler

__all__ = [
    "CLIPContrastiveLoss",
    "HardNegativeMiningLoss",
    "DomainSpecificLoss",
    "MultiScaleLoss",
    "TemperatureScheduler",
]
