"""Federated aggregation strategies."""

from federated.aggregation.fed_avg import FederatedAvgAggregator
from federated.aggregation.fed_prox import FederatedProxAggregator
from federated.aggregation.weighted import WeightedAggregator

__all__ = [
    "FederatedAvgAggregator",
    "FederatedProxAggregator",
    "WeightedAggregator",
]
