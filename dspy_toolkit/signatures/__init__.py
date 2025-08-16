"""
Project-specific DSPy signatures for EfficientAI-MLX-Toolkit.
"""

from .base_signatures import (
    BaseDeploymentSignature,
    BaseEvaluationSignature,
    BaseOptimizationSignature,
    BaseTrainingSignature,
)
from .clip_signatures import CLIPDomainAdaptationSignature, ContrastiveLossSignature
from .diffusion_signatures import (
    DiffusionOptimizationSignature,
    SamplingScheduleSignature,
)
from .federated_signatures import ClientUpdateSignature, FederatedLearningSignature
from .lora_signatures import LoRAOptimizationSignature, LoRATrainingSignature

__all__ = [
    # LoRA signatures
    "LoRAOptimizationSignature",
    "LoRATrainingSignature",
    # Diffusion signatures
    "DiffusionOptimizationSignature",
    "SamplingScheduleSignature",
    # CLIP signatures
    "CLIPDomainAdaptationSignature",
    "ContrastiveLossSignature",
    # Federated learning signatures
    "FederatedLearningSignature",
    "ClientUpdateSignature",
    # Base signatures
    "BaseOptimizationSignature",
    "BaseTrainingSignature",
    "BaseEvaluationSignature",
    "BaseDeploymentSignature",
]
