"""Federated learning client implementation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from federated.client.local_trainer import LocalTrainer
from federated.config import ClientConfig
from federated.types import ModelUpdate

logger = logging.getLogger(__name__)


class FederatedClient:
    """Federated learning client for local training.

    Handles local training on client data and communication with server.

    Args:
        config: Client configuration
        model_factory: Callable that creates a new model instance
        train_data: Tuple of (X, y) training data
        val_data: Optional tuple of (X, y) validation data
    """

    def __init__(
        self,
        config: ClientConfig,
        model_factory: Callable[[], nn.Module],
        train_data: tuple[mx.array, mx.array],
        val_data: tuple[mx.array, mx.array] | None = None,
    ):
        config.validate()
        self.config = config
        self.model_factory = model_factory
        self.train_data = train_data
        self.val_data = val_data

        # Initialize local model
        self.local_model = model_factory()

        # Initialize local trainer
        self.local_trainer = LocalTrainer(
            model=self.local_model,
            train_data=train_data,
            val_data=val_data,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
        )

        # Client state
        self.current_round = 0
        self.training_history: list[dict[str, Any]] = []

        logger.info(
            f"Initialized FederatedClient {config.client_id} "
            f"with {len(train_data[0])} samples"
        )

    def get_num_samples(self) -> int:
        """Get number of training samples.

        Returns:
            Number of samples
        """
        return len(self.train_data[0])

    def update_model(self, global_parameters: dict[str, mx.array]) -> None:
        """Update local model with global parameters.

        Args:
            global_parameters: Global model parameters from server
        """
        # Build nested dictionary structure
        nested_params = {}
        for full_name, value in global_parameters.items():
            parts = full_name.split(".")
            current = nested_params
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        # Update model
        self.local_model.update(nested_params)
        logger.debug(f"Updated local model with {len(global_parameters)} parameters")

    def train_local(
        self, round_id: int, epochs: int | None = None
    ) -> ModelUpdate:
        """Train local model for one federated round.

        Args:
            round_id: Current round identifier
            epochs: Number of local epochs (uses config if None)

        Returns:
            ModelUpdate containing trained parameters and metrics
        """
        if epochs is None:
            epochs = self.config.local_epochs

        self.current_round = round_id
        logger.info(
            f"Client {self.config.client_id} starting local training "
            f"for round {round_id} ({epochs} epochs)"
        )

        training_start = time.time()

        # Train locally
        train_metrics = self.local_trainer.train(epochs=epochs)

        training_time = time.time() - training_start

        # Get updated parameters
        params = self._get_model_parameters()

        # Create model update
        update = ModelUpdate(
            client_id=self.config.client_id,
            round_id=round_id,
            parameters=params,
            num_samples=self.get_num_samples(),
            loss=train_metrics["final_loss"],
            metrics={
                "accuracy": train_metrics.get("final_accuracy"),
                "training_time": training_time,
                "epochs": epochs,
            },
        )

        # Record history
        self.training_history.append({
            "round_id": round_id,
            "loss": update.loss,
            "accuracy": update.metrics.get("accuracy"),
            "time": training_time,
        })

        logger.info(
            f"Client {self.config.client_id} completed round {round_id}: "
            f"loss={update.loss:.4f}, time={training_time:.2f}s"
        )

        return update

    def _get_model_parameters(self) -> dict[str, mx.array]:
        """Get current model parameters.

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

        flatten_params("", self.local_model.parameters())
        return params

    def evaluate(self) -> dict[str, float]:
        """Evaluate local model on validation data.

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.val_data is None:
            logger.warning("No validation data available")
            return {}

        return self.local_trainer.evaluate(self.val_data)

    def get_client_info(self) -> dict[str, Any]:
        """Get client information.

        Returns:
            Dictionary containing client metadata
        """
        return {
            "client_id": self.config.client_id,
            "num_samples": self.get_num_samples(),
            "current_round": self.current_round,
            "num_rounds_completed": len(self.training_history),
        }
