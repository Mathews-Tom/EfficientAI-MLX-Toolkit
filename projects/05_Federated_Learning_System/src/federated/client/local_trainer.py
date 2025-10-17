"""Local trainer for federated learning clients."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger(__name__)


class LocalTrainer:
    """Local training engine for federated learning clients.

    Handles local SGD training on client data with MLX optimization.

    Args:
        model: Model to train
        train_data: Tuple of (X, y) training data
        val_data: Optional tuple of (X, y) validation data
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: tuple[mx.array, mx.array],
        val_data: tuple[mx.array, mx.array] | None = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Initialize optimizer
        self.optimizer = optim.SGD(learning_rate=learning_rate)

    def loss_fn(self, model: nn.Module, X: mx.array, y: mx.array) -> mx.array:
        """Compute loss for batch.

        Args:
            model: Model to evaluate
            X: Input data
            y: Target labels

        Returns:
            Loss value
        """
        logits = model(X)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    def train(self, epochs: int = 1) -> dict[str, Any]:
        """Train model for specified epochs.

        Args:
            epochs: Number of epochs to train

        Returns:
            Dictionary containing training metrics
        """
        X_train, y_train = self.train_data
        num_samples = len(X_train)

        history = {
            "loss": [],
            "accuracy": [],
        }

        for epoch in range(epochs):
            # Shuffle data
            indices = mx.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for i in range(0, num_samples, self.batch_size):
                batch_X = X_shuffled[i:i + self.batch_size]
                batch_y = y_shuffled[i:i + self.batch_size]

                # Forward and backward pass
                loss, grads = self._compute_loss_and_grads(batch_X, batch_y)

                # Update parameters
                self.optimizer.update(self.model, grads)

                epoch_loss += float(loss)
                num_batches += 1

            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            history["loss"].append(avg_loss)

            # Compute accuracy on training data
            if epoch == epochs - 1:  # Only on last epoch to save compute
                train_accuracy = self._compute_accuracy(X_train, y_train)
                history["accuracy"].append(train_accuracy)

        return {
            "history": history,
            "final_loss": history["loss"][-1],
            "final_accuracy": history["accuracy"][-1] if history["accuracy"] else None,
            "num_epochs": epochs,
        }

    def _compute_loss_and_grads(
        self, X: mx.array, y: mx.array
    ) -> tuple[mx.array, dict]:
        """Compute loss and gradients.

        Args:
            X: Input data
            y: Target labels

        Returns:
            Tuple of (loss, gradients)
        """
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, X, y)
        return loss, grads

    def _compute_accuracy(self, X: mx.array, y: mx.array) -> float:
        """Compute accuracy on dataset.

        Args:
            X: Input data
            y: Target labels

        Returns:
            Accuracy as float
        """
        logits = self.model(X)
        predictions = mx.argmax(logits, axis=1)
        accuracy = mx.mean(predictions == y)
        return float(accuracy)

    def evaluate(self, data: tuple[mx.array, mx.array]) -> dict[str, float]:
        """Evaluate model on dataset.

        Args:
            data: Tuple of (X, y) data

        Returns:
            Dictionary containing evaluation metrics
        """
        X, y = data
        loss = float(self.loss_fn(self.model, X, y))
        accuracy = self._compute_accuracy(X, y)

        return {
            "loss": loss,
            "accuracy": accuracy,
        }
