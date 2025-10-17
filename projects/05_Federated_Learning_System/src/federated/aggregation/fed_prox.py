"""FedProx aggregation with proximal term."""

from __future__ import annotations

import logging

import mlx.core as mx

from federated.aggregation.fed_avg import FederatedAvgAggregator
from federated.types import ModelUpdate

logger = logging.getLogger(__name__)


class FederatedProxAggregator(FederatedAvgAggregator):
    """FedProx aggregation strategy.

    Extends FedAvg with proximal term for heterogeneous clients.
    From Li et al. 2020: Federated Optimization in Heterogeneous Networks
    """

    def __init__(self, mu: float = 0.01):
        """Initialize FedProx aggregator.

        Args:
            mu: Proximal term coefficient
        """
        super().__init__()
        self.mu = mu

    def aggregate(
        self, client_updates: list[ModelUpdate]
    ) -> dict[str, mx.array]:
        """Aggregate with FedProx strategy.

        Args:
            client_updates: List of client model updates

        Returns:
            Aggregated parameters
        """
        # FedProx uses same aggregation as FedAvg
        # The proximal term is applied during client training
        return super().aggregate(client_updates)
