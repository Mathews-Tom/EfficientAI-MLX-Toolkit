"""Secure aggregation protocol for privacy-preserving federated learning."""

from __future__ import annotations

import logging

import mlx.core as mx

from federated.types import ModelUpdate

logger = logging.getLogger(__name__)


class SecureAggregation:
    """Implements secure multi-party aggregation protocol.

    Simplified implementation of secure aggregation for privacy preservation.
    """

    def __init__(self):
        """Initialize secure aggregation protocol."""
        pass

    def encrypt_update(
        self, update: ModelUpdate, mask: dict[str, mx.array] | None = None
    ) -> dict[str, mx.array]:
        """Encrypt model update with additive masking.

        Args:
            update: Model update to encrypt
            mask: Optional pre-generated mask

        Returns:
            Masked parameters
        """
        if mask is None:
            mask = self._generate_mask(update.parameters)

        encrypted = {}
        for name, param in update.parameters.items():
            encrypted[name] = param + mask[name]

        logger.debug(f"Encrypted update from {update.client_id}")
        return encrypted

    def aggregate_encrypted(
        self, encrypted_updates: list[dict[str, mx.array]]
    ) -> dict[str, mx.array]:
        """Aggregate encrypted updates.

        Args:
            encrypted_updates: List of encrypted parameter dictionaries

        Returns:
            Aggregated encrypted parameters
        """
        if not encrypted_updates:
            raise ValueError("No encrypted updates provided")

        param_names = list(encrypted_updates[0].keys())
        aggregated = {}

        for name in param_names:
            total = mx.zeros_like(encrypted_updates[0][name])
            for update in encrypted_updates:
                total = total + update[name]

            aggregated[name] = total / len(encrypted_updates)

        logger.info(f"Aggregated {len(encrypted_updates)} encrypted updates")
        return aggregated

    def _generate_mask(
        self, parameters: dict[str, mx.array]
    ) -> dict[str, mx.array]:
        """Generate random mask for encryption.

        Args:
            parameters: Parameters to mask

        Returns:
            Random mask dictionary
        """
        mask = {}
        for name, param in parameters.items():
            mask[name] = mx.random.normal(shape=param.shape) * 0.1

        return mask
