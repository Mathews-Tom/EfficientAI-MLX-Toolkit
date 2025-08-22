"""Test configuration classes."""

import pytest
from pathlib import Path

from src.diffusion.config import DiffusionConfig
from src.style_transfer.config import StyleTransferConfig
from src.training.config import TrainingConfig


class TestDiffusionConfig:
    """Test DiffusionConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DiffusionConfig()
        assert config.model_name == "runwayml/stable-diffusion-v1-5"
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 7.5
        assert config.use_mlx is True
        assert config.use_mps is True

    def test_config_validation_valid(self):
        """Test validation with valid config."""
        config = DiffusionConfig()
        config.validate()  # Should not raise

    def test_config_validation_invalid_steps(self):
        """Test validation with invalid steps."""
        config = DiffusionConfig(num_inference_steps=0)
        with pytest.raises(ValueError, match="num_inference_steps must be positive"):
            config.validate()

    def test_config_validation_invalid_guidance(self):
        """Test validation with invalid guidance scale."""
        config = DiffusionConfig(guidance_scale=50)
        with pytest.raises(ValueError, match="guidance_scale must be between"):
            config.validate()

    def test_config_serialization(self):
        """Test config serialization."""
        config = DiffusionConfig(num_inference_steps=30)
        config_dict = config.to_dict()

        new_config = DiffusionConfig.from_dict(config_dict)
        assert new_config.num_inference_steps == 30


class TestStyleTransferConfig:
    """Test StyleTransferConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StyleTransferConfig()
        assert config.style_strength == 0.8
        assert config.content_strength == 0.6
        assert config.output_resolution == (512, 512)
        assert config.method == "diffusion"

    def test_config_validation_valid(self):
        """Test validation with valid config."""
        config = StyleTransferConfig()
        config.validate()  # Should not raise

    def test_config_validation_invalid_strength(self):
        """Test validation with invalid strength."""
        config = StyleTransferConfig(style_strength=1.5)
        with pytest.raises(ValueError, match="style_strength must be between"):
            config.validate()

    def test_config_validation_invalid_method(self):
        """Test validation with invalid method."""
        config = StyleTransferConfig(method="invalid")
        with pytest.raises(ValueError, match="method must be"):
            config.validate()

    def test_config_serialization(self):
        """Test config serialization."""
        config = StyleTransferConfig(style_strength=0.9)
        config_dict = config.to_dict()

        new_config = StyleTransferConfig.from_dict(config_dict)
        assert new_config.style_strength == 0.9


class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.epochs == 10
        assert config.batch_size == 1
        assert config.learning_rate == 1e-4
        assert config.use_lora is True
        assert config.lora_rank == 16

    def test_config_validation_valid(self):
        """Test validation with valid config."""
        config = TrainingConfig()
        config.validate()  # Should not raise

    def test_config_validation_invalid_epochs(self):
        """Test validation with invalid epochs."""
        config = TrainingConfig(epochs=0)
        with pytest.raises(ValueError, match="epochs must be positive"):
            config.validate()

    def test_config_validation_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        config = TrainingConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

    def test_config_validation_invalid_lora_dropout(self):
        """Test validation with invalid LoRA dropout."""
        config = TrainingConfig(lora_dropout=1.5)
        with pytest.raises(ValueError, match="lora_dropout must be between"):
            config.validate()

    def test_config_serialization(self):
        """Test config serialization."""
        config = TrainingConfig(epochs=5, batch_size=2)
        config_dict = config.to_dict()

        new_config = TrainingConfig.from_dict(config_dict)
        assert new_config.epochs == 5
        assert new_config.batch_size == 2

    def test_get_output_dir(self):
        """Test output directory generation."""
        config = TrainingConfig(epochs=5, learning_rate=1e-3)
        output_dir = config.get_output_dir()
        expected_path = config.checkpoint_dir / "run_5epochs_lr0.001"
        assert output_dir == expected_path

    def test_get_output_dir_with_experiment_name(self):
        """Test output directory with experiment name."""
        config = TrainingConfig(experiment_name="test_experiment")
        output_dir = config.get_output_dir()
        expected_path = config.checkpoint_dir / "test_experiment"
        assert output_dir == expected_path