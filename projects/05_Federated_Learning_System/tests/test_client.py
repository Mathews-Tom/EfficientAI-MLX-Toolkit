"""Tests for federated learning client."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx
import mlx.nn as nn
import pytest

from federated.client.fl_client import FederatedClient
from federated.client.local_trainer import LocalTrainer
from federated.config import ClientConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


@pytest.fixture
def client_config() -> ClientConfig:
    """Create test client configuration."""
    return ClientConfig(
        client_id="test_client",
        local_epochs=2,
        learning_rate=0.01,
        batch_size=16,
    )


@pytest.fixture
def model_factory():
    """Create model factory."""
    return lambda: SimpleModel()


@pytest.fixture
def train_data():
    """Create synthetic training data."""
    X = mx.random.normal((100, 10))
    y = mx.random.randint(0, 2, (100,))
    return X, y


@pytest.fixture
def val_data():
    """Create synthetic validation data."""
    X = mx.random.normal((30, 10))
    y = mx.random.randint(0, 2, (30,))
    return X, y


class TestClientConfig:
    """Tests for ClientConfig."""

    def test_config_validation_success(self):
        """Test successful config validation."""
        config = ClientConfig(
            client_id="client_1",
            local_epochs=5,
        )
        config.validate()  # Should not raise

    def test_config_validation_empty_client_id(self):
        """Test validation fails with empty client ID."""
        config = ClientConfig(client_id="")
        with pytest.raises(ValueError, match="client_id"):
            config.validate()

    def test_config_validation_invalid_epochs(self):
        """Test validation fails with invalid epochs."""
        config = ClientConfig(client_id="test", local_epochs=0)
        with pytest.raises(ValueError, match="local_epochs"):
            config.validate()


class TestLocalTrainer:
    """Tests for LocalTrainer."""

    def test_trainer_initialization(self, model_factory, train_data):
        """Test trainer initialization."""
        model = model_factory()
        trainer = LocalTrainer(
            model=model,
            train_data=train_data,
            learning_rate=0.01,
            batch_size=16,
        )

        assert trainer.model is not None
        assert trainer.learning_rate == 0.01
        assert trainer.batch_size == 16

    def test_local_training(self, model_factory, train_data):
        """Test local training execution."""
        model = model_factory()
        trainer = LocalTrainer(
            model=model,
            train_data=train_data,
            learning_rate=0.01,
            batch_size=16,
        )

        # Get initial parameters
        initial_params = {
            name: mx.array(param)
            for name, param in model.parameters().items()
            if not isinstance(param, dict)
        }

        # Train for 2 epochs
        metrics = trainer.train(epochs=2)

        assert "history" in metrics
        assert "final_loss" in metrics
        assert "num_epochs" in metrics
        assert metrics["num_epochs"] == 2
        assert len(metrics["history"]["loss"]) == 2

    def test_evaluation(self, model_factory, train_data, val_data):
        """Test model evaluation."""
        model = model_factory()
        trainer = LocalTrainer(
            model=model,
            train_data=train_data,
            val_data=val_data,
        )

        # Evaluate before training
        metrics = trainer.evaluate(val_data)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_accuracy_computation(self, model_factory, train_data):
        """Test accuracy computation."""
        model = model_factory()
        trainer = LocalTrainer(
            model=model,
            train_data=train_data,
        )

        X, y = train_data
        accuracy = trainer._compute_accuracy(X, y)

        assert 0 <= accuracy <= 1


class TestFederatedClient:
    """Tests for FederatedClient."""

    def test_client_initialization(
        self, client_config, model_factory, train_data
    ):
        """Test client initialization."""
        client = FederatedClient(
            config=client_config,
            model_factory=model_factory,
            train_data=train_data,
        )

        assert client.config.client_id == "test_client"
        assert client.get_num_samples() == 100
        assert client.current_round == 0

    def test_get_client_info(
        self, client_config, model_factory, train_data
    ):
        """Test getting client information."""
        client = FederatedClient(
            config=client_config,
            model_factory=model_factory,
            train_data=train_data,
        )

        info = client.get_client_info()

        assert info["client_id"] == "test_client"
        assert info["num_samples"] == 100
        assert info["current_round"] == 0

    def test_model_update(
        self, client_config, model_factory, train_data
    ):
        """Test updating client model with global parameters."""
        client = FederatedClient(
            config=client_config,
            model_factory=model_factory,
            train_data=train_data,
        )

        # Create mock global parameters
        global_params = {
            "linear.weight": mx.random.normal((2, 10)),
            "linear.bias": mx.random.normal((2,)),
        }

        # Update model
        client.update_model(global_params)

        # Verify parameters were updated
        current_params = client._get_model_parameters()
        assert "linear.weight" in current_params
        assert "linear.bias" in current_params

    def test_local_training(
        self, client_config, model_factory, train_data
    ):
        """Test local training execution."""
        client = FederatedClient(
            config=client_config,
            model_factory=model_factory,
            train_data=train_data,
        )

        # Train for one round
        update = client.train_local(round_id=0, epochs=2)

        assert update.client_id == "test_client"
        assert update.round_id == 0
        assert update.num_samples == 100
        assert "linear.weight" in update.parameters
        assert "linear.bias" in update.parameters
        assert update.loss >= 0

    def test_training_history(
        self, client_config, model_factory, train_data
    ):
        """Test training history tracking."""
        client = FederatedClient(
            config=client_config,
            model_factory=model_factory,
            train_data=train_data,
        )

        # Train for multiple rounds
        client.train_local(round_id=0, epochs=1)
        client.train_local(round_id=1, epochs=1)
        client.train_local(round_id=2, epochs=1)

        assert len(client.training_history) == 3
        assert client.current_round == 2

    def test_evaluation(
        self, client_config, model_factory, train_data, val_data
    ):
        """Test client evaluation."""
        client = FederatedClient(
            config=client_config,
            model_factory=model_factory,
            train_data=train_data,
            val_data=val_data,
        )

        metrics = client.evaluate()

        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_evaluation_no_validation_data(
        self, client_config, model_factory, train_data
    ):
        """Test evaluation without validation data."""
        client = FederatedClient(
            config=client_config,
            model_factory=model_factory,
            train_data=train_data,
            val_data=None,
        )

        metrics = client.evaluate()
        assert metrics == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
