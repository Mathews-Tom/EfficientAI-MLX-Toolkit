"""Communication protocols for federated learning."""

from federated.communication.compression import GradientCompressor
from federated.communication.protocol import CommunicationProtocol

__all__ = [
    "GradientCompressor",
    "CommunicationProtocol",
]
