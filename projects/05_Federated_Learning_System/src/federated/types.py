"""Type definitions for federated learning system."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import mlx.core as mx


class ClientStatus(Enum):
    """Client connection status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    DISCONNECTED = "disconnected"


class RoundStatus(Enum):
    """Training round status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ClientInfo:
    """Information about a federated learning client."""

    client_id: str
    status: ClientStatus
    num_samples: int
    last_seen: float  # timestamp
    performance_score: float = 0.0
    num_failures: int = 0

    def is_available(self) -> bool:
        """Check if client is available for training."""
        return self.status == ClientStatus.ACTIVE


@dataclass
class ModelUpdate:
    """Model update from a client."""

    client_id: str
    round_id: int
    parameters: dict[str, mx.array]
    num_samples: int
    loss: float
    metrics: dict[str, Any]

    def get_weight(self) -> float:
        """Get aggregation weight based on number of samples."""
        return float(self.num_samples)


@dataclass
class RoundResults:
    """Results from a federated learning round."""

    round_id: int
    status: RoundStatus
    num_clients: int
    aggregated_loss: float
    aggregated_metrics: dict[str, Any]
    client_updates: list[ModelUpdate]

    def get_average_loss(self) -> float:
        """Get weighted average loss."""
        if not self.client_updates:
            return float("inf")

        total_samples = sum(u.num_samples for u in self.client_updates)
        if total_samples == 0:
            return float("inf")

        weighted_loss = sum(
            u.loss * u.num_samples for u in self.client_updates
        )
        return weighted_loss / total_samples


@dataclass
class TrainingMetrics:
    """Training metrics for monitoring."""

    round_id: int
    loss: float
    accuracy: float | None = None
    num_clients: int = 0
    communication_cost: float = 0.0  # bytes
    training_time: float = 0.0  # seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_id": self.round_id,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "num_clients": self.num_clients,
            "communication_cost": self.communication_cost,
            "training_time": self.training_time,
        }
