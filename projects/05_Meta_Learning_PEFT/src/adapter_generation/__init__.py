"""Adapter generation and PEFT integration for meta-learning.

This module provides integration between meta-learning algorithms (MAML, Reptile)
and parameter-efficient fine-tuning (PEFT) methods like LoRA. Instead of meta-learning
full model parameters, we meta-learn adapter parameters for efficient few-shot adaptation.
"""

from .peft_integration import LoRAMetaLearner, PEFTConfig
from .adapter_factory import AdapterFactory, PEFTMethod

__all__ = [
    "LoRAMetaLearner",
    "PEFTConfig",
    "AdapterFactory",
    "PEFTMethod",
]
