"""Federated Averaging (FedAvg) aggregation strategy."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from federated.types import ModelUpdate

logger = logging.getLogger(__name__)


class FederatedAvgAggregator:
    """Federated Averaging aggregation strategy.

    Implements the FedAvg algorithm from McMahan et al. 2017:
    Communication-Efficient Learning of Deep Networks from Decentralized Data
    """

    def __init__(self):
        """Initialize FedAvg aggregator."""
        pass

    def aggregate(
        self, client_updates: list[ModelUpdate]
    ) -> dict[str, mx.array]:
        """Aggregate client updates using weighted averaging.

        Args:
            client_updates: List of client model updates

        Returns:
            Aggregated global model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")

        # Calculate total samples across all clients
        total_samples = sum(update.num_samples for update in client_updates)

        if total_samples == 0:
            raise ValueError("Total samples is zero, cannot aggregate")

        # Get parameter names from first update
        param_names = list(client_updates[0].parameters.keys())

        # Weighted average of parameters
        aggregated = {}
        for name in param_names:
            weighted_sum = mx.zeros_like(client_updates[0].parameters[name])

            for update in client_updates:
                weight = update.num_samples / total_samples
                weighted_sum = weighted_sum + update.parameters[name] * weight

            aggregated[name] = weighted_sum

        logger.info(
            f"Aggregated {len(client_updates)} client updates "
            f"({total_samples} total samples)"
        )

        return aggregated

    def compute_aggregated_metrics(
        self, client_updates: list[ModelUpdate]
    ) -> dict[str, Any]:
        """Compute aggregated metrics from client updates.

        Args:
            client_updates: List of client model updates

        Returns:
            Dictionary of aggregated metrics
        """
        if not client_updates:
            return {}

        total_samples = sum(u.num_samples for u in client_updates)

        # Weighted average loss
        avg_loss = sum(
            u.loss * u.num_samples for u in client_updates
        ) / total_samples

        # Collect accuracy if available
        accuracies = [
            u.metrics.get("accuracy")
            for u in client_updates
            if u.metrics.get("accuracy") is not None
        ]

        avg_accuracy = None
        if accuracies:
            # Weighted average accuracy
            total_acc_samples = sum(
                u.num_samples
                for u in client_updates
                if u.metrics.get("accuracy") is not None
            )
            avg_accuracy = sum(
                u.metrics["accuracy"] * u.num_samples
                for u in client_updates
                if u.metrics.get("accuracy") is not None
            ) / total_acc_samples

        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "num_clients": len(client_updates),
            "total_samples": total_samples,
        }
