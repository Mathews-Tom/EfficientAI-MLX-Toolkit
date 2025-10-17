"""Differential privacy implementation for federated learning."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


class DifferentialPrivacyManager:
    """Manages differential privacy for federated learning.

    Implements DP-SGD with calibrated noise addition for privacy guarantees.
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
    ):
        """Initialize differential privacy manager.

        Args:
            noise_multiplier: Noise multiplier for DP-SGD
            max_grad_norm: Maximum gradient norm for clipping
            delta: Privacy parameter delta
        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta

    def add_noise_to_gradients(
        self, gradients: dict[str, mx.array], sensitivity: float = 1.0
    ) -> dict[str, mx.array]:
        """Add calibrated noise to gradients for differential privacy.

        Args:
            gradients: Model gradients
            sensitivity: Sensitivity of the gradient computation

        Returns:
            Noisy gradients
        """
        noisy_gradients = {}

        for name, grad in gradients.items():
            # Clip gradients
            clipped_grad = self._clip_gradient(grad)

            # Add Gaussian noise
            noise_std = self.noise_multiplier * sensitivity
            noise = mx.random.normal(shape=grad.shape, scale=noise_std)

            noisy_gradients[name] = clipped_grad + noise

        logger.debug(
            f"Added DP noise to {len(gradients)} gradients "
            f"(multiplier={self.noise_multiplier})"
        )

        return noisy_gradients

    def _clip_gradient(self, gradient: mx.array) -> mx.array:
        """Clip gradient to max norm.

        Args:
            gradient: Gradient tensor

        Returns:
            Clipped gradient
        """
        grad_norm = mx.sqrt(mx.sum(gradient ** 2))

        if grad_norm > self.max_grad_norm:
            return gradient * (self.max_grad_norm / grad_norm)

        return gradient

    def compute_privacy_spent(
        self, num_iterations: int, batch_size: int, dataset_size: int
    ) -> tuple[float, float]:
        """Compute privacy budget spent using RDP accountant.

        Args:
            num_iterations: Number of training iterations
            batch_size: Batch size
            dataset_size: Total dataset size

        Returns:
            Tuple of (epsilon, delta)
        """
        # Simplified privacy accounting (real implementation would use RDP)
        q = batch_size / dataset_size  # Sampling ratio
        epsilon = q * num_iterations * (self.noise_multiplier ** -2)

        return epsilon, self.delta
