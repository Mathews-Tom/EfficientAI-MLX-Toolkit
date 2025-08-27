"""
Utility functions for pruning operations.
"""

import logging
from typing import Any

import numpy as np

# Import MLX for actual model operations
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger

    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


logger = get_logger(__name__)

# Constants
EPSILON = 1e-8


def calculate_sparsity(model: Any) -> float:
    """Calculate the sparsity (proportion of zero weights) of a model."""
    if not MLX_AVAILABLE:
        logger.warning("MLX not available, cannot calculate sparsity accurately")
        return 0.0

    try:
        total_params = 0
        zero_params = 0

        if hasattr(model, 'named_modules'):
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    weight = module.weight
                    if isinstance(weight, mx.array):
                        # Count total parameters
                        total_params += weight.size
                        # Count zero parameters (with small epsilon for numerical stability)
                        zero_params += mx.sum(mx.abs(weight) < EPSILON).item()

        elif hasattr(model, 'parameters'):
            for param in model.parameters():
                if isinstance(param, mx.array):
                    total_params += param.size
                    zero_params += mx.sum(mx.abs(param) < EPSILON).item()

        if total_params > 0:
            sparsity = zero_params / total_params
            logger.debug(f"Calculated sparsity: {sparsity:.4f} ({zero_params}/{total_params})")
            return sparsity
        else:
            logger.warning("No parameters found in model for sparsity calculation")
            return 0.0

    except Exception as e:
        logger.error("Failed to calculate sparsity: %s", e)
        return 0.0


def create_pruning_mask(weights: Any, sparsity: float) -> Any:
    """Create pruning mask based on weight magnitudes and target sparsity."""
    if not MLX_AVAILABLE:
        logger.warning("MLX not available, returning identity mask")
        return weights

    try:
        if not isinstance(weights, mx.array):
            logger.warning("Weights are not MLX arrays, cannot create mask")
            return weights

        # Calculate magnitude of weights
        magnitude = mx.abs(weights)

        # Flatten weights for threshold calculation
        flat_magnitude = magnitude.flatten()

        # Calculate threshold for target sparsity
        num_elements = flat_magnitude.size
        num_to_prune = int(sparsity * num_elements)

        if num_to_prune <= 0:
            # No pruning needed
            return mx.ones_like(weights, dtype=mx.bool_)
        elif num_to_prune >= num_elements:
            # Prune everything
            return mx.zeros_like(weights, dtype=mx.bool_)

        # Find threshold value (elements below this will be pruned)
        sorted_magnitudes = mx.sort(flat_magnitude)
        threshold = sorted_magnitudes[num_to_prune]

        # Create mask (True = keep, False = prune)
        mask = magnitude > threshold

        logger.debug(f"Created pruning mask with {sparsity:.4f} target sparsity")
        return mask

    except Exception as e:
        logger.error("Failed to create pruning mask: %s", e)
        return mx.ones_like(weights, dtype=mx.bool_) if isinstance(weights, mx.array) else weights


def apply_pruning_mask(weights: Any, mask: Any) -> Any:
    """Apply pruning mask to weights, setting pruned weights to zero."""
    if not MLX_AVAILABLE:
        logger.warning("MLX not available, returning original weights")
        return weights

    try:
        if not isinstance(weights, mx.array) or not isinstance(mask, mx.array):
            logger.warning("Weights or mask are not MLX arrays")
            return weights

        if weights.shape != mask.shape:
            logger.error("Weights and mask shape mismatch: %s vs %s", weights.shape, mask.shape)
            return weights

        # Apply mask (element-wise multiplication)
        pruned_weights = weights * mask.astype(weights.dtype)

        # Calculate resulting sparsity
        total_elements = weights.size
        zero_elements = mx.sum(mx.abs(pruned_weights) < EPSILON).item()
        resulting_sparsity = zero_elements / total_elements if total_elements > 0 else 0

        logger.debug(f"Applied pruning mask, resulting sparsity: {resulting_sparsity:.4f}")
        return pruned_weights

    except Exception as e:
        logger.error("Failed to apply pruning mask: %s", e)
        return weights


def analyze_pruning_impact(original_model: Any, pruned_model: Any) -> dict[str, float]:
    """Analyze the impact of pruning by comparing original and pruned models."""
    try:
        results = {}

        # Calculate sparsity of both models
        original_sparsity = calculate_sparsity(original_model)
        pruned_sparsity = calculate_sparsity(pruned_model)

        results["original_sparsity"] = original_sparsity
        results["pruned_sparsity"] = pruned_sparsity
        results["sparsity_increase"] = pruned_sparsity - original_sparsity

        # Calculate model sizes (parameter counts)
        original_params = _count_model_parameters(original_model)
        pruned_params = _count_model_parameters(pruned_model)

        if original_params > 0:
            effective_params_original = original_params * (1 - original_sparsity)
            effective_params_pruned = pruned_params * (1 - pruned_sparsity)

            results["original_parameters"] = original_params
            results["pruned_parameters"] = pruned_params
            results["effective_original_parameters"] = effective_params_original
            results["effective_pruned_parameters"] = effective_params_pruned

            if effective_params_original > 0:
                results["effective_size_reduction"] = (
                    (effective_params_original - effective_params_pruned) / effective_params_original
                )
            else:
                results["effective_size_reduction"] = 0.0

            # Theoretical memory reduction (assuming sparse storage benefits)
            results["theoretical_memory_reduction"] = pruned_sparsity
        else:
            results["effective_size_reduction"] = 0.0
            results["theoretical_memory_reduction"] = 0.0

        # Note: Accuracy impact would require actual evaluation with test data
        # This would need to be measured separately with model.evaluate() or similar
        results["accuracy_drop_note"] = "Requires separate evaluation with test data"

        logger.info("Pruning impact analysis completed")
        return results

    except Exception as e:
        logger.error("Failed to analyze pruning impact: %s", e)
        return {
            "error": str(e),
            "sparsity_achieved": 0.0,
            "effective_size_reduction": 0.0,
        }


def _count_model_parameters(model: Any) -> int:
    """Count total parameters in a model."""
    try:
        total_params = 0

        if hasattr(model, 'named_modules'):
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    weight = module.weight
                    if isinstance(weight, mx.array):
                        total_params += weight.size
                    if hasattr(module, 'bias') and module.bias is not None:
                        total_params += module.bias.size

        elif hasattr(model, 'parameters'):
            for param in model.parameters():
                if isinstance(param, mx.array):
                    total_params += param.size

        return total_params

    except Exception as e:
        logger.error("Failed to count model parameters: %s", e)
        return 0
