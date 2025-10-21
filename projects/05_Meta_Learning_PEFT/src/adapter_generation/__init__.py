"""Adapter generation and PEFT integration for meta-learning.

This module provides integration between meta-learning algorithms (MAML, Reptile)
and parameter-efficient fine-tuning (PEFT) methods like LoRA. Instead of meta-learning
full model parameters, we meta-learn adapter parameters for efficient few-shot adaptation.
"""

from .peft_integration import LoRAMetaLearner, PEFTConfig, LoRALayer, SimpleLoRAModel
from .adapter_factory import AdapterFactory, PEFTMethod
from .adalora import AdaLoRALayer, AdaLoRAModel, AdaLoRAMetaLearner
from .prefix_prompt_tuning import (
    PrefixTuningLayer,
    PromptTuningLayer,
    PromptTuningModel,
    PromptTuningMetaLearner,
)
from .task_conditional import (
    LoRAHyperNetwork,
    TaskConditionalLoRALayer,
    TaskConditionalAdapterModel,
    AdapterHyperparameterOptimizer,
    auto_select_peft_method,
)

__all__ = [
    "LoRAMetaLearner",
    "PEFTConfig",
    "AdapterFactory",
    "PEFTMethod",
    "LoRALayer",
    "SimpleLoRAModel",
    "AdaLoRALayer",
    "AdaLoRAModel",
    "AdaLoRAMetaLearner",
    "PrefixTuningLayer",
    "PromptTuningLayer",
    "PromptTuningModel",
    "PromptTuningMetaLearner",
    "LoRAHyperNetwork",
    "TaskConditionalLoRALayer",
    "TaskConditionalAdapterModel",
    "AdapterHyperparameterOptimizer",
    "auto_select_peft_method",
]
