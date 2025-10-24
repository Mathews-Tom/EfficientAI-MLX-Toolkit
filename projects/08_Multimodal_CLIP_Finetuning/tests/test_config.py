"""Tests for CLIPFinetuningConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import CLIPFinetuningConfig


class TestCLIPFinetuningConfig:
    """Test suite for CLIPFinetuningConfig."""

    def test_default_initialization(self):
        """Test config initialization with default values."""
        config = CLIPFinetuningConfig()

        assert config.model_name == "openai/clip-vit-base-patch32"
        assert config.domain == "general"
        assert config.learning_rate == 5e-5
        assert config.batch_size is None
        assert config.num_epochs == 10
        assert config.max_sequence_length == 77
        assert config.image_resolution == 224
        assert config.use_mps is True
        assert config.mixed_precision is True
        assert config.gradient_accumulation_steps == 1

    def test_custom_initialization(self):
        """Test config initialization with custom values."""
        config = CLIPFinetuningConfig(
            model_name="openai/clip-vit-large-patch14",
            domain="medical",
            learning_rate=1e-4,
            batch_size=16,
            num_epochs=5,
            use_mps=False,
        )

        assert config.model_name == "openai/clip-vit-large-patch14"
        assert config.domain == "medical"
        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
        assert config.num_epochs == 5
        assert config.use_mps is False

    def test_output_dir_creation(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "test_outputs"
        config = CLIPFinetuningConfig(output_dir=output_dir)

        assert config.output_dir.exists()
        assert config.output_dir.is_dir()

    def test_path_conversion(self, tmp_path):
        """Test that string paths are converted to Path objects."""
        output_dir = str(tmp_path / "string_path")
        config = CLIPFinetuningConfig(output_dir=output_dir)

        assert isinstance(config.output_dir, Path)

    def test_invalid_domain(self):
        """Test that invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            CLIPFinetuningConfig(domain="invalid_domain")

    def test_invalid_learning_rate(self):
        """Test that invalid learning rate raises ValueError."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            CLIPFinetuningConfig(learning_rate=-0.001)

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            CLIPFinetuningConfig(learning_rate=0)

    def test_invalid_batch_size(self):
        """Test that invalid batch size raises ValueError."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            CLIPFinetuningConfig(batch_size=-1)

        with pytest.raises(ValueError, match="Batch size must be positive"):
            CLIPFinetuningConfig(batch_size=0)

    def test_invalid_num_epochs(self):
        """Test that invalid number of epochs raises ValueError."""
        with pytest.raises(ValueError, match="Number of epochs must be positive"):
            CLIPFinetuningConfig(num_epochs=-1)

        with pytest.raises(ValueError, match="Number of epochs must be positive"):
            CLIPFinetuningConfig(num_epochs=0)

    def test_invalid_gradient_accumulation_steps(self):
        """Test that invalid gradient accumulation steps raises ValueError."""
        with pytest.raises(
            ValueError, match="Gradient accumulation steps must be positive"
        ):
            CLIPFinetuningConfig(gradient_accumulation_steps=-1)

        with pytest.raises(
            ValueError, match="Gradient accumulation steps must be positive"
        ):
            CLIPFinetuningConfig(gradient_accumulation_steps=0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = CLIPFinetuningConfig(
            model_name="test-model",
            domain="scientific",
            learning_rate=2e-5,
            batch_size=32,
        )

        config_dict = config.to_dict()

        assert config_dict["model_name"] == "test-model"
        assert config_dict["domain"] == "scientific"
        assert config_dict["learning_rate"] == 2e-5
        assert config_dict["batch_size"] == 32
        assert "output_dir" in config_dict
        assert isinstance(config_dict["output_dir"], str)

    def test_from_dict(self, tmp_path):
        """Test creation from dictionary."""
        config_dict = {
            "model_name": "test-model",
            "domain": "industrial",
            "learning_rate": 3e-5,
            "batch_size": 64,
            "num_epochs": 15,
            "output_dir": str(tmp_path / "from_dict"),
        }

        config = CLIPFinetuningConfig.from_dict(config_dict)

        assert config.model_name == "test-model"
        assert config.domain == "industrial"
        assert config.learning_rate == 3e-5
        assert config.batch_size == 64
        assert config.num_epochs == 15

    def test_all_valid_domains(self):
        """Test that all valid domains work."""
        valid_domains = ["general", "medical", "industrial", "scientific"]

        for domain in valid_domains:
            config = CLIPFinetuningConfig(domain=domain)
            assert config.domain == domain
