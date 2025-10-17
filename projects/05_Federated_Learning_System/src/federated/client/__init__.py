"""Federated learning client components."""

from federated.client.fl_client import FederatedClient
from federated.client.local_trainer import LocalTrainer

__all__ = [
    "FederatedClient",
    "LocalTrainer",
]
