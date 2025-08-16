"""
DSPy Integration Framework for EfficientAI-MLX-Toolkit

This package provides intelligent prompt optimization and workflow automation
for all EfficientAI-MLX-Toolkit projects using DSPy's signature-based programming model.
"""

from .framework import DSPyFramework
from .manager import ModuleManager
from .optimizer import OptimizerEngine
from .providers import MLXLLMProvider
from .registry import SignatureRegistry
from .signatures import (
    ClientUpdateSignature,
    CLIPDomainAdaptationSignature,
    ContrastiveLossSignature,
    DiffusionOptimizationSignature,
    FederatedLearningSignature,
    LoRAOptimizationSignature,
    LoRATrainingSignature,
    SamplingScheduleSignature,
)
from .types import DSPyConfig

__version__ = "0.1.0"
__all__ = [
    "DSPyFramework",
    "DSPyConfig",
    "MLXLLMProvider",
    "OptimizerEngine",
    "SignatureRegistry",
    "ModuleManager",
    # Signatures
    "LoRAOptimizationSignature",
    "LoRATrainingSignature",
    "DiffusionOptimizationSignature",
    "SamplingScheduleSignature",
    "CLIPDomainAdaptationSignature",
    "ContrastiveLossSignature",
    "FederatedLearningSignature",
    "ClientUpdateSignature",
]
