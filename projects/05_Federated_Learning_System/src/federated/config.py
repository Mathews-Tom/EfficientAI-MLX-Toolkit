"""Configuration classes for federated learning system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""

    # Server configuration
    num_clients: int = 10
    clients_per_round: int = 5
    num_rounds: int = 100
    server_address: str = "localhost:8080"

    # Training configuration
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32

    # Privacy configuration
    privacy_budget: float = 1.0
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0

    # Communication configuration
    compression_ratio: float = 0.1
    use_gradient_compression: bool = True
    use_sparse_gradients: bool = False

    # Client selection
    selection_strategy: str = "random"  # random, performance, adaptive
    min_clients: int = 5

    # Fault tolerance
    client_timeout: int = 300  # seconds
    max_retries: int = 3
    byzantine_tolerance: bool = True

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_frequency: int = 10  # rounds

    # Monitoring
    log_frequency: int = 1  # rounds
    enable_tensorboard: bool = True

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_clients < self.clients_per_round:
            raise ValueError(
                f"num_clients ({self.num_clients}) must be >= "
                f"clients_per_round ({self.clients_per_round})"
            )

        if self.clients_per_round < self.min_clients:
            raise ValueError(
                f"clients_per_round ({self.clients_per_round}) must be >= "
                f"min_clients ({self.min_clients})"
            )

        if self.privacy_budget <= 0:
            raise ValueError(f"privacy_budget must be > 0, got {self.privacy_budget}")

        if self.compression_ratio < 0 or self.compression_ratio > 1:
            raise ValueError(
                f"compression_ratio must be in [0, 1], got {self.compression_ratio}"
            )

        if self.selection_strategy not in ["random", "performance", "adaptive"]:
            raise ValueError(
                f"Invalid selection_strategy: {self.selection_strategy}. "
                "Must be one of: random, performance, adaptive"
            )


@dataclass
class ClientConfig:
    """Configuration for federated learning client."""

    client_id: str
    server_address: str = "localhost:8080"

    # Local training
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32

    # Communication
    heartbeat_interval: int = 30  # seconds
    max_retries: int = 3

    # Privacy
    enable_differential_privacy: bool = True

    # Resource limits
    max_memory_mb: int = 4096
    max_cpu_percent: int = 80

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.client_id:
            raise ValueError("client_id cannot be empty")

        if self.local_epochs <= 0:
            raise ValueError(f"local_epochs must be > 0, got {self.local_epochs}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
