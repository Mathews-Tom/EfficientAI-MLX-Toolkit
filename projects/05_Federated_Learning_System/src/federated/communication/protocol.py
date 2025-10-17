"""Communication protocol for federated learning."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from federated.types import ModelUpdate

logger = logging.getLogger(__name__)


class CommunicationProtocol:
    """Handles client-server communication for federated learning."""

    def __init__(self, compression_enabled: bool = True):
        """Initialize communication protocol.

        Args:
            compression_enabled: Enable gradient compression
        """
        self.compression_enabled = compression_enabled

    def serialize_update(self, update: ModelUpdate) -> dict[str, Any]:
        """Serialize model update for transmission.

        Args:
            update: Model update to serialize

        Returns:
            Serialized update dictionary
        """
        serialized = {
            "client_id": update.client_id,
            "round_id": update.round_id,
            "num_samples": update.num_samples,
            "loss": update.loss,
            "metrics": update.metrics,
            "parameters": self._serialize_parameters(update.parameters),
        }

        return serialized

    def deserialize_update(self, data: dict[str, Any]) -> ModelUpdate:
        """Deserialize model update from transmission.

        Args:
            data: Serialized update data

        Returns:
            Model update
        """
        parameters = self._deserialize_parameters(data["parameters"])

        return ModelUpdate(
            client_id=data["client_id"],
            round_id=data["round_id"],
            parameters=parameters,
            num_samples=data["num_samples"],
            loss=data["loss"],
            metrics=data["metrics"],
        )

    def _serialize_parameters(
        self, parameters: dict[str, mx.array]
    ) -> dict[str, Any]:
        """Serialize parameters to transmittable format.

        Args:
            parameters: Model parameters

        Returns:
            Serialized parameters
        """
        serialized = {}
        for name, param in parameters.items():
            serialized[name] = {
                "data": param.tolist(),  # Convert to list for serialization
                "shape": param.shape,
                "dtype": str(param.dtype),
            }

        return serialized

    def _deserialize_parameters(
        self, data: dict[str, Any]
    ) -> dict[str, mx.array]:
        """Deserialize parameters from transmission.

        Args:
            data: Serialized parameter data

        Returns:
            Model parameters
        """
        parameters = {}
        for name, param_data in data.items():
            parameters[name] = mx.array(
                param_data["data"]
            ).reshape(param_data["shape"])

        return parameters

    def estimate_bandwidth(self, update: ModelUpdate) -> int:
        """Estimate bandwidth required for update transmission.

        Args:
            update: Model update

        Returns:
            Estimated bytes
        """
        # Estimate based on parameter sizes
        total_params = sum(
            param.size for param in update.parameters.values()
        )

        # Assume 4 bytes per float32
        bytes_required = total_params * 4

        # Add overhead for metadata
        overhead = 1024  # 1KB overhead

        return int(bytes_required + overhead)
