"""Federated aggregation strategies."""

from federated.aggregation.byzantine import ByzantineTolerantAggregator
from federated.aggregation.fed_avg import FederatedAvgAggregator
from federated.aggregation.fed_prox import FederatedProxAggregator
from federated.aggregation.weighted import WeightedAggregator

__all__ = [
    "ByzantineTolerantAggregator",
    "FederatedAvgAggregator",
    "FederatedProxAggregator",
    "WeightedAggregator",
]
