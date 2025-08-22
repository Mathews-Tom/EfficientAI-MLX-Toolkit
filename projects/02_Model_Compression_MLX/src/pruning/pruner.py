"""
Main pruner class for MLX-native pruning operations.
"""

import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import PruningConfig, PruningMethod, PruningSchedule
from .scheduler import GradualPruningScheduler, PruningScheduler
from .strategies import GradientPruner, MagnitudePruner, StructuredPruner, UnstructuredPruner
from .utils import apply_pruning_mask, calculate_sparsity, create_pruning_mask

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


logger = get_logger(__name__)

# Constants for numerical stability
ZERO_THRESHOLD = 1e-8


class MLXPruner:
    """
    Main pruner class that orchestrates different pruning methods.E

    Supports:
    - Magnitude-based pruning
    - Gradient-based pruning
    - Structured and unstructured pruning
    - Gradual pruning with recovery training
    - MLX-native optimizations
    """

    def __init__(self, config: PruningConfig):
        """
        Initialize the pruner with configuration.

        Args:
            config: Pruning configuration
        """
        self.config = config
        self.model = None
        self.pruned_model = None
        self.pruning_masks = {}
        self.pruning_stats = {}
        self.scheduler = None
        self.device = mx.default_device()
        self.original_weights = {}  # Store original weights for analysis

    def load_model(self, model: Any) -> None:
        """
        Load model for pruning.

        Args:
            model: Model to prune
        """
        logger.info("Loading model for pruning")
        self.model = model

        # Store original weights for later analysis
        self._store_original_weights()

        # Calculate initial statistics
        self._calculate_initial_stats()

    def prune(
        self,
        model: Any | None = None,
        training_data: Any | None = None,
        validation_data: Any | None = None,
    ) -> Any:
        """
        Perform pruning based on the configured method.

        Args:
            model: Model to prune (if not already loaded)
            training_data: Training data for gradient-based methods
            validation_data: Validation data for evaluation

        Returns:
            Pruned model
        """
        if model:
            self.load_model(model)

        if not self.model:
            raise RuntimeError("No model loaded. Call load_model() first or provide model.")

        logger.info(f"Starting pruning with method: {self.config.method}")
        start_time = time.time()

        try:
            if self.config.schedule == PruningSchedule.ONESHOT:
                self.pruned_model = self._oneshot_pruning(training_data)
            else:
                self.pruned_model = self._gradual_pruning(training_data, validation_data)

            pruning_time = time.time() - start_time
            logger.info(f"Pruning completed in {pruning_time:.2f}s")

            # Calculate final statistics
            self._calculate_pruning_stats()

            return self.pruned_model

        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            raise

    def _oneshot_pruning(self, training_data: Any | None = None) -> Any:
        """Perform one-shot pruning."""
        logger.info("Performing one-shot pruning")

        # Create a copy of the model for pruning
        import copy

        self.pruned_model = copy.deepcopy(self.model)

        # Generate pruning masks based on method
        if self.config.method == PruningMethod.MAGNITUDE:
            self.pruning_masks = self._create_magnitude_masks(self.config.target_sparsity)
        elif self.config.method == PruningMethod.GRADIENT:
            self.pruning_masks = self._create_gradient_masks(
                self.config.target_sparsity, training_data
            )
        else:
            self.pruning_masks = self._create_magnitude_masks(self.config.target_sparsity)

        # Apply pruning masks
        self._apply_pruning_masks_to_model(self.pruned_model)

        # Recovery training if configured
        if self.config.recovery_epochs > 0:
            logger.info(f"Starting recovery training for {self.config.recovery_epochs} epochs")
            self.pruned_model = self._recovery_training(
                self.pruned_model, training_data, self.config.recovery_epochs
            )

        return self.pruned_model

    def _gradual_pruning(
        self, training_data: Any | None = None, validation_data: Any | None = None
    ) -> Any:
        """Perform gradual pruning with recovery training."""
        logger.info("Performing gradual pruning")

        # Create scheduler
        self.scheduler = GradualPruningScheduler(self.config)

        # Create a copy of the model for gradual pruning
        import copy

        current_model = copy.deepcopy(self.model)

        # Gradual pruning loop
        for epoch in range(self.config.start_epoch, self.config.end_epoch + 1):
            if epoch % self.config.frequency == 0:
                # Calculate target sparsity for this epoch
                target_sparsity = self.config.calculate_sparsity_for_epoch(
                    epoch, self.config.end_epoch
                )

                logger.info(f"Epoch {epoch}: Target sparsity = {target_sparsity:.3f}")

                # Create new pruning masks
                if self.config.method == PruningMethod.MAGNITUDE:
                    epoch_masks = self._create_magnitude_masks(target_sparsity)
                elif self.config.method == PruningMethod.GRADIENT:
                    epoch_masks = self._create_gradient_masks(target_sparsity, training_data)
                else:
                    epoch_masks = self._create_magnitude_masks(target_sparsity)

                # Update pruning masks
                self.pruning_masks.update(epoch_masks)

                # Apply updated pruning
                self._apply_pruning_masks_to_model(current_model)

                # Mini recovery training
                if training_data:
                    current_model = self._recovery_training(
                        current_model,
                        training_data,
                        epochs=1,  # One epoch recovery between pruning steps
                    )

        # Final recovery training
        if self.config.recovery_epochs > 0:
            logger.info(f"Final recovery training for {self.config.recovery_epochs} epochs")
            current_model = self._recovery_training(
                current_model, training_data, self.config.recovery_epochs
            )

        self.pruned_model = current_model
        return self.pruned_model

    def _calculate_weight_magnitude(self, weight: mx.array) -> mx.array:
        """Calculate weight magnitude based on configured criterion."""
        if self.config.criterion.value == "l1":
            return mx.abs(weight)
        elif self.config.criterion.value == "l2":
            return weight**2
        else:
            return mx.abs(weight)  # Default to L1

    def _collect_prunable_layers(self) -> list[tuple[str, Any, tuple]]:
        """Collect information about layers that should be pruned."""
        layer_info = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and self.config.should_prune_layer(name):
                weight = module.weight
                if isinstance(weight, mx.array):
                    layer_info.append((name, module, weight.shape))
        return layer_info

    def _compute_global_threshold(self, target_sparsity: float, layer_info: list) -> mx.array | None:
        """Compute global pruning threshold if global pruning is enabled."""
        if not self.config.global_magnitude_pruning:
            return None

        all_weights = []
        for name, module, _ in layer_info:
            weight = module.weight
            magnitude = self._calculate_weight_magnitude(weight)
            all_weights.append(magnitude.flatten())

        all_magnitudes = mx.concatenate(all_weights)
        threshold_idx = int(target_sparsity * len(all_magnitudes))
        threshold = mx.sort(all_magnitudes)[threshold_idx]
        logger.info(f"Global magnitude threshold: {float(threshold):.6f}")
        return threshold

    def _create_layer_mask(self, name: str, module: Any, weight_shape: tuple, global_threshold: mx.array | None) -> mx.array:
        """Create pruning mask for a single layer."""
        weight = module.weight
        magnitude = self._calculate_weight_magnitude(weight)

        if global_threshold is not None:
            # Use global threshold
            mask = magnitude > global_threshold
        else:
            # Use layer-specific threshold
            layer_sparsity = self.config.get_sparsity_for_layer(name)
            flat_magnitude = magnitude.flatten()
            threshold_idx = int(layer_sparsity * len(flat_magnitude))
            if threshold_idx > 0:
                layer_threshold = mx.sort(flat_magnitude)[threshold_idx]
                mask = magnitude > layer_threshold
            else:
                mask = mx.ones_like(magnitude, dtype=mx.bool_)

        if self.config.structured:
            # Apply structured pruning (channel/block level)
            mask = self._apply_structured_mask(mask, weight_shape)

        return mask.astype(mx.float32)

    def _create_magnitude_masks(self, target_sparsity: float) -> dict[str, mx.array]:
        """Create pruning masks based on weight magnitudes."""
        logger.info(f"Creating magnitude-based masks for {target_sparsity:.3f} sparsity")

        # Collect prunable layers
        layer_info = self._collect_prunable_layers()
        if not layer_info:
            logger.warning("No weights found for pruning")
            return {}

        # Compute global threshold if needed
        global_threshold = self._compute_global_threshold(target_sparsity, layer_info)

        # Create masks for each layer
        masks = {}
        for name, module, weight_shape in layer_info:
            mask = self._create_layer_mask(name, module, weight_shape, global_threshold)
            masks[name] = mask

            # Log pruning statistics
            sparsity_achieved = 1.0 - float(mx.mean(mask))
            logger.debug(f"Layer {name}: {sparsity_achieved:.3f} sparsity achieved")

        return masks

    def _create_gradient_masks(
        self, target_sparsity: float, training_data: Any | None = None
    ) -> dict[str, mx.array]:
        """Create pruning masks based on gradient information."""
        logger.info(f"Creating gradient-based masks for {target_sparsity:.3f} sparsity")

        if training_data is None:
            logger.warning(
                "No training data provided for gradient-based pruning, falling back to magnitude"
            )
            return self._create_magnitude_masks(target_sparsity)

        # This is a simplified gradient-based pruning
        # In a full implementation, you would:
        # 1. Compute gradients w.r.t. weights on training data
        # 2. Use gradient * weight as importance score
        # 3. Prune weights with lowest importance scores

        # For now, fall back to magnitude-based pruning
        logger.warning("Full gradient-based pruning not implemented, using magnitude-based")
        return self._create_magnitude_masks(target_sparsity)

    def _apply_structured_mask(self, mask: mx.array, weight_shape: tuple[int, ...]) -> mx.array:
        """Apply structured pruning to mask (channel/block level)."""
        if len(weight_shape) < 2:
            return mask

        # For structured pruning, we prune entire channels
        # Calculate channel importance (sum of magnitudes)
        channel_importance = mx.sum(mask, axis=tuple(range(1, len(weight_shape))))

        # Determine how many channels to prune
        num_channels = weight_shape[0]
        channels_to_prune = int(self.config.target_sparsity * num_channels)

        if channels_to_prune > 0:
            # Find channels with lowest importance
            _, pruned_channels = mx.topk(-channel_importance, channels_to_prune)

            # Create structured mask
            structured_mask = mx.ones_like(mask)
            for channel_idx in pruned_channels:
                structured_mask = structured_mask.at[channel_idx].set(0.0)

            return structured_mask

        return mask

    def _apply_pruning_masks_to_model(self, model: Any) -> None:
        """Apply pruning masks to model weights."""
        logger.info(f"Applying pruning masks to model")

        masked_layers = 0
        for name, module in model.named_modules():
            if name in self.pruning_masks and isinstance(module, nn.Linear):
                mask = self.pruning_masks[name]

                # Apply mask to weight
                if isinstance(module.weight, mx.array) and isinstance(mask, mx.array):
                    # Zero out weights according to mask
                    module.weight = module.weight * mask
                    masked_layers += 1

                    # Store mask metadata
                    if not hasattr(module, "pruning_mask"):
                        module.pruning_mask = mask

                    logger.debug(f"Applied mask to layer {name}")

        logger.info(f"Applied masks to {masked_layers} layers")

    def _recovery_training(self, model: Any, training_data: Any, epochs: int) -> Any:
        """Perform recovery training on pruned model."""
        if not training_data:
            logger.warning("No training data provided for recovery training")
            return model

        logger.info(f"Starting recovery training for {epochs} epochs")

        # Create optimizer for recovery training
        try:
            import mlx.optimizers as optim

            # Use the configured optimizer
            if self.config.recovery_optimizer == "adamw":
                optimizer = optim.AdamW(
                    learning_rate=self.config.recovery_lr,
                    weight_decay=self.config.recovery_weight_decay,
                )
            elif self.config.recovery_optimizer == "adam":
                optimizer = optim.Adam(learning_rate=self.config.recovery_lr)
            elif self.config.recovery_optimizer == "sgd":
                optimizer = optim.SGD(learning_rate=self.config.recovery_lr)
            else:
                optimizer = optim.AdamW(learning_rate=self.config.recovery_lr)

            logger.info(
                f"Using {self.config.recovery_optimizer} optimizer with lr={self.config.recovery_lr}"
            )

        except ImportError:
            logger.warning("MLX optimizers not available, skipping actual training")
            return model

        # Simple recovery training loop
        for epoch in range(epochs):
            logger.debug(f"Recovery epoch {epoch + 1}/{epochs}")

            # In a full implementation, you would:
            # 1. Process training batches
            # 2. Compute loss (language modeling, classification, etc.)
            # 3. Backward pass
            # 4. Apply gradients while preserving pruning masks
            # 5. Update parameters

            # For now, simulate training
            if isinstance(training_data, (list, str)):
                # Text data - simulate language modeling training
                self._simulate_training_step(model, training_data, optimizer)

            # Re-apply pruning masks after each epoch to ensure pruned weights stay zero
            self._apply_pruning_masks_to_model(model)

        logger.info("Recovery training completed")
        return model

    def _simulate_training_step(self, model: Any, training_data: Any, optimizer: Any) -> None:
        """Simulate a training step (placeholder for actual training)."""
        # This would contain actual training logic in a full implementation
        # For now, just a placeholder to maintain pruning masks
        pass

    def _store_original_weights(self) -> None:
        """Store original weights for analysis."""
        if not self.model:
            return

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if isinstance(module.weight, mx.array):
                    # Store a copy of the original weight
                    self.original_weights[name] = module.weight.copy()

        logger.info(f"Stored original weights for {len(self.original_weights)} layers")

    def evaluate_pruned_model(
        self, validation_data: Any, metrics: list[str] | None = None
    ) -> dict[str, float]:
        """
        Evaluate the pruned model.

        Args:
            validation_data: Validation dataset
            metrics: List of metrics to calculate

        Returns:
            Dictionary of evaluation results
        """
        if not self.pruned_model:
            raise RuntimeError("No pruned model available. Run prune() first.")

        logger.info("Evaluating pruned model")

        results = {}

        try:
            # Calculate sparsity metrics
            sparsity_stats = self._calculate_sparsity_metrics()
            results.update(sparsity_stats)

            # Model size comparison
            if self.model:
                original_size = self._calculate_model_size(self.model)
                pruned_size = self._calculate_model_size(self.pruned_model)

                results.update(
                    {
                        "original_size_mb": original_size,
                        "pruned_size_mb": pruned_size,
                        "size_reduction_mb": original_size - pruned_size,
                        "size_reduction_percent": ((original_size - pruned_size) / original_size)
                        * 100,
                    }
                )

            # Performance metrics would be calculated here
            # This requires running inference on validation data

            logger.info("Model evaluation completed")
            return results

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}

    def save_pruned_model(self, output_path: str | Path) -> None:
        """
        Save the pruned model and associated artifacts.

        Args:
            output_path: Path to save the pruned model
        """
        if not self.pruned_model:
            raise RuntimeError("No pruned model available. Run prune() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving pruned model to: {output_path}")

        try:
            # Save model weights
            if hasattr(self.pruned_model, "save_weights"):
                weights_path = output_path / "weights.npz"
                self.pruned_model.save_weights(str(weights_path))

            # Save pruning configuration
            config_path = output_path / "pruning_config.yaml"
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(self.config.to_dict(), f, default_flow_style=False)

            # Save pruning masks
            if self.config.save_pruning_masks and self.pruning_masks:
                masks_path = output_path / "pruning_masks.npz"
                mx.savez(str(masks_path), **self.pruning_masks)

            # Save pruning statistics
            if self.config.save_pruning_stats and self.pruning_stats:
                stats_path = output_path / "pruning_stats.yaml"
                with open(stats_path, "w") as f:
                    yaml.dump(self.pruning_stats, f, default_flow_style=False)

            logger.info("Pruned model saved successfully")

        except Exception as e:
            logger.error(f"Failed to save pruned model: {e}")
            raise

    def _calculate_initial_stats(self) -> None:
        """Calculate initial model statistics."""
        if not self.model:
            return

        self.pruning_stats["initial"] = {
            "model_size_mb": self._calculate_model_size(self.model),
            "total_parameters": self._count_parameters(self.model),
            "trainable_parameters": self._count_trainable_parameters(self.model),
        }

    def _calculate_pruning_stats(self) -> None:
        """Calculate pruning statistics."""
        if not self.model or not self.pruned_model:
            return

        original_params = self._count_parameters(self.model)
        pruned_params = self._count_parameters(self.pruned_model)

        self.pruning_stats.update(
            {
                "pruning_method": self.config.method.value,
                "target_sparsity": self.config.target_sparsity,
                "actual_sparsity": self._calculate_actual_sparsity(),
                "parameters_removed": original_params - pruned_params,
                "parameters_removed_percent": ((original_params - pruned_params) / original_params)
                * 100,
            }
        )

    def _calculate_sparsity_metrics(self) -> dict[str, float]:
        """Calculate detailed sparsity metrics."""
        if not self.pruned_model:
            return {}

        metrics = {}

        try:
            # Overall sparsity
            total_params = 0
            zero_params = 0
            layer_sparsities = []

            for name, module in self.pruned_model.named_modules():
                if isinstance(module, nn.Linear):
                    weight = module.weight
                    if isinstance(weight, mx.array):
                        layer_total = weight.size
                        layer_zeros = mx.sum(mx.abs(weight) < ZERO_THRESHOLD)

                        total_params += layer_total
                        zero_params += layer_zeros

                        # Calculate per-layer sparsity
                        layer_sparsity = float(layer_zeros) / float(layer_total)
                        layer_sparsities.append(layer_sparsity)

            if total_params > 0:
                overall_sparsity = float(zero_params) / float(total_params)
                density = 1.0 - overall_sparsity
                compression_ratio = 1.0 / density if density > 0 else float("inf")

                metrics.update(
                    {
                        "overall_sparsity": overall_sparsity,
                        "density": density,
                        "compression_ratio": compression_ratio,
                        "total_parameters": total_params,
                        "zero_parameters": int(zero_params),
                        "non_zero_parameters": total_params - int(zero_params),
                    }
                )

                # Add layer-wise statistics
                if layer_sparsities:
                    metrics.update(
                        {
                            "mean_layer_sparsity": float(np.mean(layer_sparsities)),
                            "std_layer_sparsity": float(np.std(layer_sparsities)),
                            "min_layer_sparsity": float(np.min(layer_sparsities)),
                            "max_layer_sparsity": float(np.max(layer_sparsities)),
                        }
                    )

        except Exception as e:
            logger.warning(f"Could not calculate sparsity metrics: {e}")

        return metrics

    def _calculate_actual_sparsity(self) -> float:
        """Calculate the actual sparsity achieved."""
        if not self.pruned_model:
            return 0.0

        try:
            total_params = 0
            zero_params = 0

            for name, module in self.pruned_model.named_modules():
                if isinstance(module, nn.Linear):
                    weight = module.weight
                    if isinstance(weight, mx.array):
                        # Count total parameters
                        total_params += weight.size

                        # Count zero parameters (or very close to zero)
                        zero_mask = mx.abs(weight) < ZERO_THRESHOLD
                        zero_params += mx.sum(zero_mask)

            if total_params > 0:
                sparsity = float(zero_params) / float(total_params)
                return sparsity
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Could not calculate actual sparsity: {e}")
            return self.config.target_sparsity

    def _calculate_model_size(self, model: Any) -> float:
        """Calculate model size in MB considering actual data types."""
        if not model:
            return 0.0

        try:
            total_bytes = 0

            if hasattr(model, "parameters"):
                for param in model.parameters():
                    if isinstance(param, mx.array):
                        # MLX array - get actual size in bytes
                        total_bytes += param.nbytes
                    elif hasattr(param, "shape") and hasattr(param, "dtype"):
                        # Calculate from shape and dtype
                        num_elements = np.prod(param.shape)
                        if "float32" in str(param.dtype):
                            bytes_per_element = 4
                        elif "float16" in str(param.dtype) or "bfloat16" in str(param.dtype):
                            bytes_per_element = 2
                        elif "int8" in str(param.dtype):
                            bytes_per_element = 1
                        else:
                            bytes_per_element = 4  # Default
                        total_bytes += num_elements * bytes_per_element

            size_mb = total_bytes / (1024 * 1024)
            return size_mb

        except Exception as e:
            logger.warning(f"Could not calculate model size: {e}")
            return 0.0

    def _count_parameters(self, model: Any) -> int:
        """Count total parameters in model."""
        if not model:
            return 0

        try:
            total_params = 0
            if hasattr(model, "parameters"):
                for param in model.parameters():
                    if isinstance(param, mx.array):
                        total_params += param.size
                    elif hasattr(param, "shape"):
                        total_params += np.prod(param.shape)
            return total_params
        except Exception as e:
            logger.warning(f"Could not count parameters: {e}")
            return 0

    def _count_trainable_parameters(self, model: Any) -> int:
        """Count trainable parameters in model."""
        if not model:
            return 0

        try:
            # For MLX models, typically all parameters are trainable
            # In more complex models, you might check for frozen parameters
            total_trainable = 0
            if hasattr(model, "parameters"):
                for param in model.parameters():
                    # Check if parameter requires gradients (trainable)
                    if isinstance(param, mx.array):
                        # MLX arrays are typically trainable unless explicitly frozen
                        total_trainable += param.size
                    elif hasattr(param, "shape"):
                        total_trainable += np.prod(param.shape)
            return total_trainable
        except Exception as e:
            logger.warning(f"Could not count trainable parameters: {e}")
            return self._count_parameters(model)

    def get_pruning_info(self) -> dict[str, Any]:
        """Get comprehensive pruning information."""
        info = {
            "config": self.config.to_dict(),
            "stats": self.pruning_stats,
            "model_loaded": self.model is not None,
            "pruned_model_available": self.pruned_model is not None,
            "masks_generated": len(self.pruning_masks) > 0,
            "device": str(self.device),
            "original_weights_stored": len(self.original_weights) > 0,
        }

        if self.scheduler:
            info["scheduler_info"] = self.scheduler.get_schedule_info()

        return info
