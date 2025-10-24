"""Tests for CLIPFinetuningController."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from config import CLIPFinetuningConfig
from model import CLIPFinetuningController


class TestCLIPFinetuningController:
    """Test suite for CLIPFinetuningController."""

    def test_initialization(self, sample_config):
        """Test controller initialization."""
        controller = CLIPFinetuningController(sample_config)

        assert controller.config == sample_config
        assert controller.device_manager is not None
        assert controller.model is None
        assert controller.processor is None

    def test_setup_loads_model_and_processor(self, sample_config):
        """Test that setup loads model and processor."""
        controller = CLIPFinetuningController(sample_config)

        # Mock the transformers imports to avoid downloading models
        with (
            patch("model.CLIPModel") as mock_model_class,
            patch("model.CLIPProcessor") as mock_processor_class,
        ):
            mock_model = MagicMock()
            mock_processor = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_processor_class.from_pretrained.return_value = mock_processor

            # Mock model methods - return iter for parameters()
            mock_param = torch.zeros(10)
            mock_model.to.return_value = mock_model
            mock_model.half.return_value = mock_model
            mock_model.parameters.return_value = iter([mock_param])

            controller.setup()

            # Verify model and processor were loaded
            mock_model_class.from_pretrained.assert_called_once_with(
                sample_config.model_name
            )
            mock_processor_class.from_pretrained.assert_called_once_with(
                sample_config.model_name
            )

            # Verify model was moved to device
            mock_model.to.assert_called_once()

            # Verify model was set to training mode
            mock_model.train.assert_called_once()

    def test_setup_with_mixed_precision(self, mps_config):
        """Test setup with mixed precision enabled."""
        controller = CLIPFinetuningController(mps_config)

        with (
            patch("model.CLIPModel") as mock_model_class,
            patch("model.CLIPProcessor") as mock_processor_class,
            patch.object(
                controller.device_manager, "_device", torch.device("mps"), create=True
            ),
        ):
            mock_model = MagicMock()
            mock_processor = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_processor_class.from_pretrained.return_value = mock_processor

            mock_param = torch.zeros(10)
            mock_model.to.return_value = mock_model
            mock_model.half.return_value = mock_model
            mock_model.parameters.return_value = iter([mock_param])

            controller.setup()

            # Verify half precision was enabled for MPS
            mock_model.half.assert_called_once()

    def test_encode_text_raises_without_setup(self, sample_config):
        """Test that encode_text raises error if setup not called."""
        controller = CLIPFinetuningController(sample_config)

        with pytest.raises(RuntimeError, match="Model not initialized"):
            controller.encode_text(["test text"])

    def test_encode_image_raises_without_setup(self, sample_config):
        """Test that encode_image raises error if setup not called."""
        controller = CLIPFinetuningController(sample_config)

        # Create a dummy image
        image = Image.new("RGB", (224, 224))

        with pytest.raises(RuntimeError, match="Model not initialized"):
            controller.encode_image([image])

    def test_compute_similarity_raises_without_setup(self, sample_config):
        """Test that compute_similarity raises error if setup not called."""
        controller = CLIPFinetuningController(sample_config)

        image = Image.new("RGB", (224, 224))

        with pytest.raises(RuntimeError, match="Model not initialized"):
            controller.compute_similarity(["test text"], [image])

    def test_determine_batch_size_from_config(self, sample_config):
        """Test batch size determination when specified in config."""
        controller = CLIPFinetuningController(sample_config)

        batch_size = controller.determine_batch_size()
        assert batch_size == sample_config.batch_size

    def test_determine_batch_size_auto_cpu(self, sample_config):
        """Test automatic batch size determination for CPU."""
        # Remove batch_size from config
        sample_config.batch_size = None
        controller = CLIPFinetuningController(sample_config)

        # Mock device as CPU
        with patch.object(
            controller.device_manager, "_device", torch.device("cpu"), create=True
        ):
            batch_size = controller.determine_batch_size()
            assert batch_size == 8  # CPU default

    @pytest.mark.apple_silicon
    def test_determine_batch_size_auto_mps(self, mps_config):
        """Test automatic batch size determination for MPS."""
        mps_config.batch_size = None
        controller = CLIPFinetuningController(mps_config)

        # Mock device as MPS
        with patch.object(
            controller.device_manager, "_device", torch.device("mps"), create=True
        ):
            batch_size = controller.determine_batch_size()
            assert batch_size == 16  # MPS default

    def test_get_model_state_not_initialized(self, sample_config):
        """Test get_model_state when model not initialized."""
        controller = CLIPFinetuningController(sample_config)

        state = controller.get_model_state()
        assert state["status"] == "not_initialized"

    def test_get_model_state_initialized(self, sample_config):
        """Test get_model_state when model is initialized."""
        controller = CLIPFinetuningController(sample_config)

        with (
            patch("model.CLIPModel") as mock_model_class,
            patch("model.CLIPProcessor") as mock_processor_class,
        ):
            mock_model = MagicMock()
            mock_processor = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_processor_class.from_pretrained.return_value = mock_processor

            mock_param = torch.zeros(10)
            mock_model.to.return_value = mock_model
            mock_model.half.return_value = mock_model
            mock_model.parameters.return_value = iter([mock_param])
            mock_model.training = True

            controller.setup()

            state = controller.get_model_state()

            assert state["status"] == "initialized"
            assert state["model_name"] == sample_config.model_name
            assert state["domain"] == sample_config.domain
            assert state["training_mode"] is True

    def test_encode_text_with_mocked_model(self, sample_config):
        """Test text encoding with mocked model."""
        controller = CLIPFinetuningController(sample_config)

        # Create mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Mock processor output
        mock_processor.return_value = {
            "input_ids": torch.zeros((1, 77), dtype=torch.long),
            "attention_mask": torch.ones((1, 77), dtype=torch.long),
        }

        # Mock model output
        mock_features = torch.randn(1, 512)
        mock_model.get_text_features.return_value = mock_features

        # Assign mocks
        controller.model = mock_model
        controller.processor = mock_processor

        # Test encoding
        text = ["a photo of a cat"]
        features = controller.encode_text(text)

        assert features is not None
        mock_processor.assert_called_once()
        mock_model.get_text_features.assert_called_once()

    def test_encode_image_with_mocked_model(self, sample_config):
        """Test image encoding with mocked model."""
        controller = CLIPFinetuningController(sample_config)

        # Create mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Mock processor output
        mock_processor.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        # Mock model output
        mock_features = torch.randn(1, 512)
        mock_model.get_image_features.return_value = mock_features

        # Assign mocks
        controller.model = mock_model
        controller.processor = mock_processor

        # Test encoding
        image = Image.new("RGB", (224, 224))
        features = controller.encode_image([image])

        assert features is not None
        mock_processor.assert_called_once()
        mock_model.get_image_features.assert_called_once()

    def test_compute_similarity_with_mocked_model(self, sample_config):
        """Test similarity computation with mocked model."""
        controller = CLIPFinetuningController(sample_config)

        # Create a real encode_text and encode_image that returns tensors
        def mock_encode_text(text):
            return torch.randn(len(text), 512)

        def mock_encode_image(images):
            return torch.randn(len(images), 512)

        controller.model = MagicMock()
        controller.processor = MagicMock()

        # Patch the encode methods
        with (
            patch.object(controller, "encode_text", side_effect=mock_encode_text),
            patch.object(controller, "encode_image", side_effect=mock_encode_image),
        ):
            text = ["a photo of a cat"]
            image = Image.new("RGB", (224, 224))

            similarity = controller.compute_similarity(text, [image])

            # Should return a similarity matrix
            assert similarity.shape == (1, 1)  # 1 image x 1 text
