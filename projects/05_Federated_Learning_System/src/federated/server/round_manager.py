"""Round management for federated learning."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import mlx.core as mx

from federated.config import FederatedConfig
from federated.types import (
    ClientInfo,
    ClientStatus,
    ModelUpdate,
    RoundResults,
    RoundStatus,
)

logger = logging.getLogger(__name__)


class ClientManager:
    """Manages client registration, selection, and monitoring."""

    def __init__(self, config: FederatedConfig):
        self.config = config
        self.clients: dict[str, ClientInfo] = {}
        self._initialize_mock_clients()

    def _initialize_mock_clients(self) -> None:
        """Initialize mock clients for testing."""
        for i in range(self.config.num_clients):
            client_id = f"client_{i}"
            self.clients[client_id] = ClientInfo(
                client_id=client_id,
                status=ClientStatus.ACTIVE,
                num_samples=random.randint(100, 1000),
                last_seen=time.time(),
                performance_score=random.uniform(0.7, 1.0),
            )

    def register_client(
        self, client_id: str, num_samples: int
    ) -> ClientInfo:
        """Register a new client.

        Args:
            client_id: Unique client identifier
            num_samples: Number of training samples on client

        Returns:
            ClientInfo object
        """
        client_info = ClientInfo(
            client_id=client_id,
            status=ClientStatus.ACTIVE,
            num_samples=num_samples,
            last_seen=time.time(),
        )
        self.clients[client_id] = client_info
        logger.info(f"Registered client {client_id} with {num_samples} samples")
        return client_info

    def unregister_client(self, client_id: str) -> None:
        """Unregister a client.

        Args:
            client_id: Client identifier
        """
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Unregistered client {client_id}")

    def update_client_status(
        self, client_id: str, status: ClientStatus
    ) -> None:
        """Update client status.

        Args:
            client_id: Client identifier
            status: New status
        """
        if client_id in self.clients:
            self.clients[client_id].status = status
            self.clients[client_id].last_seen = time.time()

    def get_client_info(self, client_id: str) -> ClientInfo | None:
        """Get client information.

        Args:
            client_id: Client identifier

        Returns:
            ClientInfo if exists, None otherwise
        """
        return self.clients.get(client_id)

    def get_all_clients(self) -> list[ClientInfo]:
        """Get all registered clients.

        Returns:
            List of all clients
        """
        return list(self.clients.values())

    def get_available_clients(self) -> list[ClientInfo]:
        """Get all available clients.

        Returns:
            List of available clients
        """
        return [c for c in self.clients.values() if c.is_available()]

    def select_clients(
        self, num_clients: int, strategy: str = "random"
    ) -> list[str]:
        """Select clients for training round.

        Args:
            num_clients: Number of clients to select
            strategy: Selection strategy (random, performance, adaptive)

        Returns:
            List of selected client IDs
        """
        available = self.get_available_clients()

        if len(available) < num_clients:
            logger.warning(
                f"Only {len(available)} clients available, "
                f"requested {num_clients}"
            )
            num_clients = len(available)

        if strategy == "random":
            selected = random.sample(available, num_clients)
        elif strategy == "performance":
            # Select clients with highest performance scores
            sorted_clients = sorted(
                available, key=lambda c: c.performance_score, reverse=True
            )
            selected = sorted_clients[:num_clients]
        elif strategy == "adaptive":
            # Weight selection by both performance and data size
            weights = [
                c.performance_score * c.num_samples for c in available
            ]
            selected = random.choices(available, weights=weights, k=num_clients)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

        return [c.client_id for c in selected]


class RoundManager:
    """Manages federated learning rounds."""

    def __init__(self, config: FederatedConfig):
        self.config = config
        self.client_manager = ClientManager(config)

    def execute_round(
        self,
        round_id: int,
        global_parameters: dict[str, mx.array],
    ) -> RoundResults:
        """Execute a single federated learning round.

        Args:
            round_id: Round identifier
            global_parameters: Current global model parameters

        Returns:
            RoundResults containing aggregation results
        """
        logger.info(f"Starting round {round_id}")

        # Select clients
        selected_clients = self.client_manager.select_clients(
            self.config.clients_per_round,
            self.config.selection_strategy,
        )

        if len(selected_clients) < self.config.min_clients:
            logger.error(
                f"Insufficient clients: {len(selected_clients)} < "
                f"{self.config.min_clients}"
            )
            return RoundResults(
                round_id=round_id,
                status=RoundStatus.FAILED,
                num_clients=0,
                aggregated_loss=float("inf"),
                aggregated_metrics={},
                client_updates=[],
            )

        logger.info(f"Selected {len(selected_clients)} clients for round {round_id}")

        # Collect updates from clients
        client_updates = []
        for client_id in selected_clients:
            try:
                update = self._simulate_client_training(
                    client_id, round_id, global_parameters
                )
                client_updates.append(update)
            except Exception as e:
                logger.error(f"Client {client_id} failed: {e}")
                client_info = self.client_manager.get_client_info(client_id)
                if client_info:
                    client_info.num_failures += 1

        if not client_updates:
            logger.error("No successful client updates")
            return RoundResults(
                round_id=round_id,
                status=RoundStatus.FAILED,
                num_clients=0,
                aggregated_loss=float("inf"),
                aggregated_metrics={},
                client_updates=[],
            )

        # Compute aggregated metrics
        aggregated_loss = sum(
            u.loss * u.num_samples for u in client_updates
        ) / sum(u.num_samples for u in client_updates)

        return RoundResults(
            round_id=round_id,
            status=RoundStatus.COMPLETED,
            num_clients=len(client_updates),
            aggregated_loss=aggregated_loss,
            aggregated_metrics={},
            client_updates=client_updates,
        )

    def _simulate_client_training(
        self,
        client_id: str,
        round_id: int,
        global_parameters: dict[str, mx.array],
    ) -> ModelUpdate:
        """Simulate client training (placeholder for actual client communication).

        Args:
            client_id: Client identifier
            round_id: Round identifier
            global_parameters: Global model parameters

        Returns:
            ModelUpdate from simulated training
        """
        client_info = self.client_manager.get_client_info(client_id)
        if not client_info:
            raise ValueError(f"Client {client_id} not found")

        # Simulate training by adding small random noise to parameters
        # This will be replaced with actual client communication in FEDE-008
        updated_params = {}
        for name, param in global_parameters.items():
            noise = mx.random.normal(shape=param.shape) * 0.01
            updated_params[name] = param + noise

        # Simulate loss
        simulated_loss = random.uniform(0.5, 2.0)

        return ModelUpdate(
            client_id=client_id,
            round_id=round_id,
            parameters=updated_params,
            num_samples=client_info.num_samples,
            loss=simulated_loss,
            metrics={"accuracy": random.uniform(0.7, 0.95)},
        )
