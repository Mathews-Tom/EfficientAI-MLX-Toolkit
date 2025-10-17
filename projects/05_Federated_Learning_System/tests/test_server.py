"""Tests for federated server architecture."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx
import mlx.nn as nn
import pytest

from federated.config import FederatedConfig
from federated.server.coordinator import FederatedServer
from federated.server.round_manager import ClientManager, RoundManager
from federated.types import ClientStatus, RoundStatus


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


@pytest.fixture
def config() -> FederatedConfig:
    """Create test configuration."""
    return FederatedConfig(
        num_clients=10,
        clients_per_round=5,
        num_rounds=3,
        local_epochs=1,
        learning_rate=0.01,
        checkpoint_dir=Path("test_checkpoints"),
    )


@pytest.fixture
def model_factory():
    """Create model factory."""
    return lambda: SimpleModel()


class TestFederatedConfig:
    """Tests for FederatedConfig."""

    def test_config_validation_success(self):
        """Test successful config validation."""
        config = FederatedConfig(
            num_clients=10,
            clients_per_round=5,
            min_clients=3,
        )
        config.validate()  # Should not raise

    def test_config_validation_insufficient_clients(self):
        """Test validation fails with insufficient clients."""
        config = FederatedConfig(
            num_clients=3,
            clients_per_round=5,
        )
        with pytest.raises(ValueError, match="num_clients"):
            config.validate()

    def test_config_validation_invalid_privacy_budget(self):
        """Test validation fails with invalid privacy budget."""
        config = FederatedConfig(privacy_budget=-1.0)
        with pytest.raises(ValueError, match="privacy_budget"):
            config.validate()

    def test_config_validation_invalid_compression(self):
        """Test validation fails with invalid compression ratio."""
        config = FederatedConfig(compression_ratio=1.5)
        with pytest.raises(ValueError, match="compression_ratio"):
            config.validate()


class TestClientManager:
    """Tests for ClientManager."""

    def test_client_registration(self, config):
        """Test client registration."""
        manager = ClientManager(config)

        client_info = manager.register_client("test_client", 100)
        assert client_info.client_id == "test_client"
        assert client_info.num_samples == 100
        assert client_info.status == ClientStatus.ACTIVE

    def test_client_unregistration(self, config):
        """Test client unregistration."""
        manager = ClientManager(config)

        manager.register_client("test_client", 100)
        manager.unregister_client("test_client")

        assert manager.get_client_info("test_client") is None

    def test_client_selection_random(self, config):
        """Test random client selection."""
        manager = ClientManager(config)

        selected = manager.select_clients(5, strategy="random")
        assert len(selected) == 5
        assert len(set(selected)) == 5  # All unique

    def test_client_selection_performance(self, config):
        """Test performance-based client selection."""
        manager = ClientManager(config)

        selected = manager.select_clients(5, strategy="performance")
        assert len(selected) == 5

        # Verify selected clients have high performance scores
        selected_clients = [manager.get_client_info(cid) for cid in selected]
        avg_score = sum(c.performance_score for c in selected_clients) / len(selected_clients)
        assert avg_score > 0.7

    def test_get_available_clients(self, config):
        """Test getting available clients."""
        manager = ClientManager(config)

        # All clients should be available initially
        available = manager.get_available_clients()
        assert len(available) == config.num_clients

        # Mark one client as inactive
        clients = manager.get_all_clients()
        manager.update_client_status(clients[0].client_id, ClientStatus.INACTIVE)

        available = manager.get_available_clients()
        assert len(available) == config.num_clients - 1


class TestRoundManager:
    """Tests for RoundManager."""

    def test_round_execution_success(self, config):
        """Test successful round execution."""
        manager = RoundManager(config)

        # Create mock global parameters
        global_params = {
            "linear.weight": mx.random.normal((2, 10)),
            "linear.bias": mx.random.normal((2,)),
        }

        results = manager.execute_round(0, global_params)

        assert results.status == RoundStatus.COMPLETED
        assert results.num_clients == config.clients_per_round
        assert len(results.client_updates) == config.clients_per_round
        assert results.aggregated_loss < float("inf")

    def test_round_execution_insufficient_clients(self):
        """Test round fails with insufficient clients."""
        config = FederatedConfig(
            num_clients=2,
            clients_per_round=5,
            min_clients=5,
        )
        manager = RoundManager(config)

        global_params = {
            "linear.weight": mx.random.normal((2, 10)),
            "linear.bias": mx.random.normal((2,)),
        }

        results = manager.execute_round(0, global_params)
        assert results.status == RoundStatus.FAILED


class TestFederatedServer:
    """Tests for FederatedServer."""

    def test_server_initialization(self, config, model_factory):
        """Test server initialization."""
        server = FederatedServer(config, model_factory)

        assert server.current_round == 0
        assert len(server.training_history) == 0
        assert server.global_model is not None

    def test_get_set_global_parameters(self, config, model_factory):
        """Test getting and setting global parameters."""
        server = FederatedServer(config, model_factory)

        # Get parameters
        params = server.get_global_parameters()
        assert "linear.weight" in params
        assert "linear.bias" in params

        # Modify parameters
        new_params = {
            name: param + 0.1 for name, param in params.items()
        }

        # Set parameters
        server.set_global_parameters(new_params)

        # Verify parameters were updated
        updated_params = server.get_global_parameters()
        for name in params:
            assert not mx.array_equal(params[name], updated_params[name])

    def test_federated_training(self, config, model_factory, tmp_path):
        """Test federated training execution."""
        config.checkpoint_dir = tmp_path / "checkpoints"
        config.num_rounds = 3
        config.save_frequency = 1

        server = FederatedServer(config, model_factory)

        results = server.train_federated(num_rounds=3)

        assert "history" in results
        assert len(results["history"]) == 3
        assert results["total_rounds"] == 3
        assert results["final_loss"] is not None

    def test_checkpoint_save_load(self, config, model_factory, tmp_path):
        """Test checkpoint saving and loading."""
        config.checkpoint_dir = tmp_path / "checkpoints"
        server = FederatedServer(config, model_factory)

        # Save checkpoint
        checkpoint_path = server.save_checkpoint(0)
        assert checkpoint_path.exists()

        # Modify model
        original_params = server.get_global_parameters()
        new_params = {
            name: param + 1.0 for name, param in original_params.items()
        }
        server.set_global_parameters(new_params)

        # Load checkpoint
        server.load_checkpoint(checkpoint_path)
        loaded_params = server.get_global_parameters()

        # Verify parameters match original
        for name in original_params:
            assert mx.allclose(original_params[name], loaded_params[name])

    def test_client_info_retrieval(self, config, model_factory):
        """Test client information retrieval."""
        server = FederatedServer(config, model_factory)

        # Get all clients
        all_clients = server.get_all_clients()
        assert len(all_clients) == config.num_clients

        # Get specific client
        client_id = all_clients[0].client_id
        client_info = server.get_client_info(client_id)
        assert client_info is not None
        assert client_info.client_id == client_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
