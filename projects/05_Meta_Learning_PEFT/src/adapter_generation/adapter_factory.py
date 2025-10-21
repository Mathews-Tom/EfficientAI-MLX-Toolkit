"""Adapter factory for creating different PEFT method integrations.

This module provides a factory pattern for creating meta-learners with different
PEFT methods (LoRA, AdaLoRA, prompt tuning, etc.).
"""

from enum import Enum
from typing import Any

import mlx.nn as nn

from adapter_generation.peft_integration import (
    LoRAMetaLearner,
    PEFTConfig,
    SimpleLoRAModel,
)
from adapter_generation.adalora import AdaLoRAModel, AdaLoRAMetaLearner
from adapter_generation.prefix_prompt_tuning import PromptTuningModel, PromptTuningMetaLearner
from meta_learning.maml import MAMLLearner
from utils.logging import get_logger

logger = get_logger(__name__)


class PEFTMethod(Enum):
    """Supported PEFT methods for meta-learning."""

    LORA = "lora"
    ADALORA = "adalora"
    PROMPT_TUNING = "prompt_tuning"
    PREFIX_TUNING = "prefix_tuning"
    P_TUNING = "p_tuning"


class AdapterFactory:
    """Factory for creating meta-learners with different PEFT methods.

    This factory simplifies the creation of meta-learning + PEFT combinations,
    automatically configuring the appropriate adapter layers and meta-learner.
    """

    @staticmethod
    def create_lora_meta_learner(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        meta_batch_size: int = 4,
        first_order: bool = False,
    ) -> LoRAMetaLearner:
        """Create a MAML meta-learner with LoRA adapters.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (number of classes).
            lora_rank: Rank for LoRA low-rank decomposition.
            lora_alpha: Scaling factor for LoRA.
            inner_lr: Inner loop learning rate.
            outer_lr: Outer loop (meta) learning rate.
            num_inner_steps: Number of gradient steps per task.
            meta_batch_size: Number of tasks per meta-update.
            first_order: Whether to use first-order approximation (FOMAML).

        Returns:
            Configured LoRA meta-learner.
        """
        # Create LoRA-adapted model
        model = SimpleLoRAModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # Create PEFT config
        peft_config = PEFTConfig(
            method="lora",
            rank=lora_rank,
            alpha=lora_alpha,
            freeze_base_model=True,
        )

        # Create meta-learner
        learner = LoRAMetaLearner(
            model=model,
            peft_config=peft_config,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            num_inner_steps=num_inner_steps,
            meta_batch_size=meta_batch_size,
            first_order=first_order,
        )

        param_info = learner.count_trainable_parameters()
        logger.info(
            f"Created LoRA meta-learner: "
            f"trainable={param_info['trainable']:,}, "
            f"total={param_info['total']:,}, "
            f"reduction={param_info['reduction_ratio']:.1f}x"
        )

        return learner

    @staticmethod
    def create_adalora_meta_learner(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        **kwargs: Any,
    ) -> AdaLoRAMetaLearner:
        """Create AdaLoRA meta-learner with adaptive rank allocation.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            lora_rank: Maximum rank for AdaLoRA
            lora_alpha: Scaling factor
            inner_lr: Inner loop learning rate
            outer_lr: Outer loop learning rate
            num_inner_steps: Number of gradient steps per task
            **kwargs: Additional arguments

        Returns:
            Configured AdaLoRA meta-learner
        """
        model = AdaLoRAModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        learner = AdaLoRAMetaLearner(
            model=model,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            num_inner_steps=num_inner_steps,
        )

        logger.info(f"Created AdaLoRA meta-learner with max rank={lora_rank}")

        return learner

    @staticmethod
    def create_prompt_tuning_meta_learner(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_prompts: int = 10,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        **kwargs: Any,
    ) -> PromptTuningMetaLearner:
        """Create Prompt Tuning meta-learner.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            num_prompts: Number of prompt tokens
            inner_lr: Inner loop learning rate
            outer_lr: Outer loop learning rate
            num_inner_steps: Number of gradient steps per task
            **kwargs: Additional arguments

        Returns:
            Configured Prompt Tuning meta-learner
        """
        model = PromptTuningModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_prompts=num_prompts,
        )

        learner = PromptTuningMetaLearner(
            model=model,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            num_inner_steps=num_inner_steps,
        )

        logger.info(f"Created Prompt Tuning meta-learner with {num_prompts} prompts")

        return learner

    @staticmethod
    def create_meta_learner(
        method: PEFTMethod | str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        **kwargs: Any,
    ) -> MAMLLearner:
        """Create a meta-learner with the specified PEFT method.

        Args:
            method: PEFT method to use.
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension.
            **kwargs: Additional arguments for the meta-learner.

        Returns:
            Configured meta-learner.

        Raises:
            ValueError: If unsupported PEFT method.
        """
        # Convert string to enum if needed
        if isinstance(method, str):
            try:
                method = PEFTMethod(method.lower())
            except ValueError:
                raise ValueError(
                    f"Unsupported PEFT method: {method}. "
                    f"Supported: {[m.value for m in PEFTMethod]}"
                )

        if method == PEFTMethod.LORA:
            return AdapterFactory.create_lora_meta_learner(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                **kwargs,
            )
        elif method == PEFTMethod.ADALORA:
            return AdapterFactory.create_adalora_meta_learner(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                **kwargs,
            )
        elif method == PEFTMethod.PROMPT_TUNING:
            return AdapterFactory.create_prompt_tuning_meta_learner(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                **kwargs,
            )
        elif method == PEFTMethod.PREFIX_TUNING:
            # Prefix tuning shares implementation with prompt tuning
            return AdapterFactory.create_prompt_tuning_meta_learner(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                **kwargs,
            )
        elif method == PEFTMethod.P_TUNING:
            # P-tuning is a variant of prompt tuning
            return AdapterFactory.create_prompt_tuning_meta_learner(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown PEFT method: {method}")

    @staticmethod
    def list_supported_methods() -> list[str]:
        """List all supported PEFT methods.

        Returns:
            List of method names.
        """
        return [method.value for method in PEFTMethod]
