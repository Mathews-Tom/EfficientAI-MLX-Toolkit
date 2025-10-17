"""Federated server coordinator for managing distributed training."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from federated.config import FederatedConfig
from federated.server.round_manager import RoundManager
from federated.types import (
    ClientInfo,
    ModelUpdate,
    RoundResults,
    RoundStatus,
    TrainingMetrics,
)

logger = logging.getLogger(__name__)


class FederatedServer:
    """Main federated learning server coordinator.

    Manages federated training rounds, client selection, model aggregation,
    and checkpointing.

    Args:
        config: Federated learning configuration
        model_factory: Callable that creates a new model instance
        loss_fn: Loss function for evaluation
    """

    def __init__(
        self,
        config: FederatedConfig,
        model_factory: Callable[[], nn.Module],
        loss_fn: Callable[[mx.array, mx.array], mx.array] | None = None,
    ):
        config.validate()
        self.config = config
        self.model_factory = model_factory

        # Initialize global model
        self.global_model = model_factory()
        self.loss_fn = loss_fn

        # Initialize components
        self.round_manager = RoundManager(config)

        # Training state
        self.current_round = 0
        self.training_history: list[TrainingMetrics] = []

        # Checkpointing
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized FederatedServer with {config.num_clients} clients, "
            f"{config.clients_per_round} per round"
        )

    def get_global_parameters(self) -> dict[str, mx.array]:
        """Get current global model parameters.

        Returns:
            Dictionary mapping parameter names to arrays
        """
        params = {}

        def flatten_params(prefix: str, module_params: dict) -> None:
            """Recursively flatten nested parameter dictionaries."""
            for name, value in module_params.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(value, dict):
                    flatten_params(full_name, value)
                else:
                    params[full_name] = mx.array(value)

        flatten_params("", self.global_model.parameters())
        return params

    def set_global_parameters(self, parameters: dict[str, mx.array]) -> None:
        """Update global model with new parameters.

        Args:
            parameters: Dictionary mapping parameter names to arrays
        """
        # Build nested dictionary structure
        nested_params = {}
        for full_name, value in parameters.items():
            parts = full_name.split(".")
            current = nested_params
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        # Update model parameters
        self.global_model.update(nested_params)

    def train_federated(
        self,
        num_rounds: int | None = None,
        validation_data: tuple[mx.array, mx.array] | None = None,
    ) -> dict[str, Any]:
        """Execute federated training rounds.

        Args:
            num_rounds: Number of rounds to train (uses config if None)
            validation_data: Optional validation data (X, y) for evaluation

        Returns:
            Dictionary containing training history and final metrics
        """
        if num_rounds is None:
            num_rounds = self.config.num_rounds

        logger.info(f"Starting federated training for {num_rounds} rounds")

        for round_id in range(num_rounds):
            self.current_round = round_id
            round_start = time.time()

            # Execute training round
            round_results = self.round_manager.execute_round(
                round_id=round_id,
                global_parameters=self.get_global_parameters(),
            )

            # Handle failed round
            if round_results.status == RoundStatus.FAILED:
                logger.error(f"Round {round_id} failed, skipping aggregation")
                continue

            # Aggregate updates (will be implemented in FEDE-004)
            aggregated_params = self._aggregate_updates(round_results.client_updates)
            self.set_global_parameters(aggregated_params)

            # Evaluate on validation data if provided
            val_loss = None
            val_accuracy = None
            if validation_data is not None:
                val_loss, val_accuracy = self._evaluate(validation_data)

            # Record metrics
            round_time = time.time() - round_start
            metrics = TrainingMetrics(
                round_id=round_id,
                loss=round_results.get_average_loss(),
                accuracy=val_accuracy,
                num_clients=round_results.num_clients,
                training_time=round_time,
            )
            self.training_history.append(metrics)

            # Logging
            if round_id % self.config.log_frequency == 0:
                logger.info(
                    f"Round {round_id}: loss={metrics.loss:.4f}, "
                    f"clients={metrics.num_clients}, time={round_time:.2f}s"
                )
                if val_accuracy is not None:
                    logger.info(f"  Validation accuracy: {val_accuracy:.4f}")

            # Checkpointing
            if round_id % self.config.save_frequency == 0:
                self.save_checkpoint(round_id)

        logger.info("Federated training completed")

        return {
            "history": [m.to_dict() for m in self.training_history],
            "final_loss": self.training_history[-1].loss if self.training_history else None,
            "final_accuracy": self.training_history[-1].accuracy if self.training_history else None,
            "total_rounds": len(self.training_history),
        }

    def _aggregate_updates(
        self, client_updates: list[ModelUpdate]
    ) -> dict[str, mx.array]:
        """Aggregate client model updates (placeholder for FEDE-004).

        Args:
            client_updates: List of client model updates

        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            logger.warning("No client updates to aggregate, returning current parameters")
            return self.get_global_parameters()

        # Simple federated averaging (will be enhanced in FEDE-004)
        total_samples = sum(update.num_samples for update in client_updates)
        aggregated = {}

        # Get parameter names from first update
        param_names = list(client_updates[0].parameters.keys())

        for name in param_names:
            weighted_sum = mx.zeros_like(client_updates[0].parameters[name])

            for update in client_updates:
                weight = update.num_samples / total_samples
                weighted_sum = weighted_sum + update.parameters[name] * weight

            aggregated[name] = weighted_sum

        return aggregated

    def _evaluate(
        self, validation_data: tuple[mx.array, mx.array]
    ) -> tuple[float, float]:
        """Evaluate global model on validation data.

        Args:
            validation_data: Tuple of (X, y) validation data

        Returns:
            Tuple of (loss, accuracy)
        """
        X, y = validation_data

        # Forward pass
        logits = self.global_model(X)

        # Calculate loss
        if self.loss_fn is not None:
            loss = self.loss_fn(logits, y)
        else:
            # Default cross-entropy loss
            loss = mx.mean(nn.losses.cross_entropy(logits, y))

        # Calculate accuracy
        predictions = mx.argmax(logits, axis=1)
        accuracy = mx.mean(predictions == y)

        return float(loss), float(accuracy)

    def save_checkpoint(self, round_id: int) -> Path:
        """Save model checkpoint.

        Args:
            round_id: Current training round

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = (
            self.config.checkpoint_dir / f"server_round_{round_id}.safetensors"
        )

        # Save model weights
        self.global_model.save_weights(str(checkpoint_path))

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.global_model.load_weights(str(checkpoint_path))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def get_client_info(self, client_id: str) -> ClientInfo | None:
        """Get information about a specific client.

        Args:
            client_id: Client identifier

        Returns:
            ClientInfo if client exists, None otherwise
        """
        return self.round_manager.client_manager.get_client_info(client_id)

    def get_all_clients(self) -> list[ClientInfo]:
        """Get information about all registered clients.

        Returns:
            List of all client information
        """
        return self.round_manager.client_manager.get_all_clients()
