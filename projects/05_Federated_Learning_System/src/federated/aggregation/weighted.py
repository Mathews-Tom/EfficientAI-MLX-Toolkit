"""Weighted aggregation strategy."""

from __future__ import annotations

import logging

import mlx.core as mx

from federated.types import ModelUpdate

logger = logging.getLogger(__name__)


class WeightedAggregator:
    """Custom weighted aggregation strategy."""

    def __init__(self, weight_fn: callable | None = None):
        """Initialize weighted aggregator.

        Args:
            weight_fn: Optional function to compute custom weights
        """
        self.weight_fn = weight_fn or self._default_weight_fn

    def _default_weight_fn(self, update: ModelUpdate) -> float:
        """Default weight function based on number of samples.

        Args:
            update: Model update

        Returns:
            Weight value
        """
        return float(update.num_samples)

    def aggregate(
        self, client_updates: list[ModelUpdate]
    ) -> dict[str, mx.array]:
        """Aggregate using custom weights.

        Args:
            client_updates: List of client updates

        Returns:
            Aggregated parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        # Compute weights
        weights = [self.weight_fn(u) for u in client_updates]
        total_weight = sum(weights)

        if total_weight == 0:
            raise ValueError("Total weight is zero")

        # Get parameter names
        param_names = list(client_updates[0].parameters.keys())

        # Weighted aggregation
        aggregated = {}
        for name in param_names:
            weighted_sum = mx.zeros_like(client_updates[0].parameters[name])

            for update, weight in zip(client_updates, weights):
                normalized_weight = weight / total_weight
                weighted_sum = weighted_sum + update.parameters[name] * normalized_weight

            aggregated[name] = weighted_sum

        logger.info(f"Aggregated {len(client_updates)} updates with custom weights")
        return aggregated
