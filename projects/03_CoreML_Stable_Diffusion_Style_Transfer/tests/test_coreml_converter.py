"""Test CoreML converter functionality."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from src.coreml.config import CoreMLConfig
from src.coreml.converter import CoreMLConverter


class TestCoreMLConverter:
    """Test CoreMLConverter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CoreMLConfig()
        self.converter = CoreMLConverter(self.config)

    def test_converter_creation(self):
        """Test converter creation."""
        assert self.converter.config == self.config

    def test_converter_creation_with_validation(self):
        """Test converter creation validates config."""
        config = CoreMLConfig(quantization_bits=0)  # Invalid config
        with pytest.raises(ValueError):
            CoreMLConverter(config)

    def test_convert_model_file_not_found(self):
        """Test convert_model with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Model not found"):
            self.converter.convert_model("nonexistent.pth")

    @patch("pathlib.Path.exists")
    @patch("torch.load")
    def test_convert_model_unsupported_format(self, mock_load, mock_exists):
        """Test convert_model with unsupported format."""
        mock_exists.return_value = True
        mock_load.side_effect = ValueError("Unsupported model format: .unsupported")

        with pytest.raises(RuntimeError, match="Failed to load PyTorch model"):
            self.converter.convert_model("model.unsupported")

    @patch("pathlib.Path.exists")
    @patch("torch.load")
    def test_convert_model_load_failure(self, mock_load, mock_exists):
        """Test convert_model with PyTorch load failure."""
        mock_exists.return_value = True
        mock_load.side_effect = RuntimeError("Load failed")

        with pytest.raises(RuntimeError, match="Failed to load PyTorch model"):
            self.converter.convert_model("model.pth")

    @patch("pathlib.Path.exists")
    @patch("torch.load")
    @patch("src.coreml.converter.CoreMLConverter._create_example_inputs")
    @patch("src.coreml.converter.CoreMLConverter._convert_to_coreml")
    def test_convert_model_success(
        self, mock_convert, mock_inputs, mock_load, mock_exists
    ):
        """Test successful model conversion."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_load.return_value = mock_model
        mock_inputs.return_value = (torch.randn(1, 3, 224, 224),)
        mock_coreml_model = Mock()
        mock_convert.return_value = mock_coreml_model

        result = self.converter.convert_model("model.pth")

        assert result == mock_coreml_model
        mock_model.eval.assert_called_once()
        mock_inputs.assert_called_once()
        mock_convert.assert_called_once()

    @patch("pathlib.Path.exists")
    @patch("torch.load")
    @patch("src.coreml.converter.CoreMLConverter._create_example_inputs")
    @patch("src.coreml.converter.CoreMLConverter._convert_to_coreml")
    @patch("src.coreml.converter.CoreMLConverter._apply_optimizations")
    @patch("src.coreml.converter.CoreMLConverter.save_model")
    def test_convert_model_with_optimizations(
        self,
        mock_save,
        mock_optimize,
        mock_convert,
        mock_inputs,
        mock_load,
        mock_exists,
    ):
        """Test model conversion with optimizations."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_load.return_value = mock_model
        mock_inputs.return_value = (torch.randn(1, 3, 224, 224),)
        mock_coreml_model = Mock()
        mock_optimized_model = Mock()
        mock_convert.return_value = mock_coreml_model
        mock_optimize.return_value = mock_optimized_model

        config = CoreMLConfig(optimize_for_apple_silicon=True)
        converter = CoreMLConverter(config)
        result = converter.convert_model("model.pth", "output.mlpackage")

        assert result == mock_optimized_model
        mock_optimize.assert_called_once_with(mock_coreml_model)
        mock_save.assert_called_once_with(mock_optimized_model, "output.mlpackage")

    def test_create_example_inputs_default(self):
        """Test example input creation with default config."""
        inputs = self.converter._create_example_inputs()

        assert isinstance(inputs, tuple)
        assert len(inputs) == len(self.config.test_input_shapes)

    def test_create_example_inputs_custom_shapes(self):
        """Test example input creation with custom shapes."""
        config = CoreMLConfig(
            test_input_shapes={
                "image": (1, 3, 256, 256),
                "prompt": (1, 77),
                "timestep": (1,),
            }
        )
        converter = CoreMLConverter(config)

        inputs = converter._create_example_inputs()

        assert len(inputs) == 3
        assert inputs[0].shape == (1, 3, 256, 256)  # image
        assert inputs[1].shape == (1, 77)  # prompt tokens
        assert inputs[2].shape == (1,)  # timestep

    def test_create_example_inputs_generic(self):
        """Test example input creation with generic input."""
        config = CoreMLConfig(test_input_shapes={"generic": (2, 4, 8)})
        converter = CoreMLConverter(config)

        inputs = converter._create_example_inputs()

        assert len(inputs) == 1
        assert inputs[0].shape == (2, 4, 8)

    @patch("torch.jit.trace")
    @patch("coremltools.convert")
    def test_convert_to_coreml_success(self, mock_ct_convert, mock_trace):
        """Test successful Core ML conversion."""
        mock_model = Mock()
        example_inputs = (torch.randn(1, 3, 224, 224),)
        mock_traced = Mock()
        mock_coreml_model = Mock()
        mock_trace.return_value = mock_traced
        mock_ct_convert.return_value = mock_coreml_model

        result = self.converter._convert_to_coreml(mock_model, example_inputs)

        assert result == mock_coreml_model
        mock_trace.assert_called_once_with(mock_model, example_inputs)
        mock_ct_convert.assert_called_once()

    @patch("torch.jit.trace")
    def test_convert_to_coreml_failure(self, mock_trace):
        """Test Core ML conversion failure."""
        mock_model = Mock()
        example_inputs = (torch.randn(1, 3, 224, 224),)
        mock_trace.side_effect = RuntimeError("Tracing failed")

        with pytest.raises(RuntimeError, match="Core ML conversion failed"):
            self.converter._convert_to_coreml(mock_model, example_inputs)

    @patch("coremltools.models.neural_network.quantization_utils.quantize_weights")
    def test_apply_optimizations_float16(self, mock_quantize):
        """Test float16 optimization."""
        config = CoreMLConfig(precision="float16", optimize_for_apple_silicon=True)
        converter = CoreMLConverter(config)
        mock_model = Mock()
        mock_optimized = Mock()
        mock_quantize.return_value = mock_optimized

        result = converter._apply_optimizations(mock_model)

        assert result == mock_optimized
        mock_quantize.assert_called_once_with(mock_model, nbits=16)

    @patch("coremltools.models.neural_network.quantization_utils.quantize_weights")
    def test_apply_optimizations_float16_failure(self, mock_quantize):
        """Test float16 optimization failure."""
        config = CoreMLConfig(precision="float16", optimize_for_apple_silicon=True)
        converter = CoreMLConverter(config)
        mock_model = Mock()
        mock_quantize.side_effect = RuntimeError("Quantization failed")

        # Should return original model on failure
        result = converter._apply_optimizations(mock_model)
        assert result == mock_model

    @patch("src.coreml.converter.CoreMLConverter._apply_palettization")
    def test_apply_optimizations_palettization(self, mock_palettize):
        """Test palettization optimization."""
        config = CoreMLConfig(use_palettization=True, optimize_for_apple_silicon=True)
        converter = CoreMLConverter(config)
        mock_model = Mock()
        mock_optimized = Mock()
        mock_palettize.return_value = mock_optimized

        result = converter._apply_optimizations(mock_model)

        assert result == mock_optimized
        mock_palettize.assert_called_once_with(mock_model)

    @patch("src.coreml.converter.CoreMLConverter._apply_palettization")
    def test_apply_optimizations_memory_reduction(self, mock_palettize):
        """Test memory reduction optimization."""
        config = CoreMLConfig(
            reduce_memory_footprint=True, optimize_for_apple_silicon=True
        )
        converter = CoreMLConverter(config)
        mock_model = Mock()

        result = converter._apply_optimizations(mock_model)

        # Should return original model (placeholder implementation)
        assert result == mock_model

    def test_apply_palettization_kmeans(self):
        """Test palettization using quantize_weights fallback."""
        config = CoreMLConfig(palettization_mode="kmeans", quantization_bits=8)
        converter = CoreMLConverter(config)
        mock_model = Mock()
        mock_optimized = Mock()

        with patch(
            "coremltools.models.neural_network.quantization_utils.quantize_weights",
            return_value=mock_optimized,
        ) as mock_quantize:
            result = converter._apply_palettization(mock_model)

            assert result == mock_optimized
            mock_quantize.assert_called_once_with(mock_model, nbits=8)

    def test_apply_palettization_uniform(self):
        """Test palettization using quantize_weights fallback."""
        config = CoreMLConfig(palettization_mode="uniform", quantization_bits=4)
        converter = CoreMLConverter(config)
        mock_model = Mock()
        mock_optimized = Mock()

        with patch(
            "coremltools.models.neural_network.quantization_utils.quantize_weights",
            return_value=mock_optimized,
        ) as mock_quantize:
            result = converter._apply_palettization(mock_model)

            assert result == mock_optimized
            mock_quantize.assert_called_once_with(mock_model, nbits=4)

    def test_save_model_mlpackage(self):
        """Test saving model in mlpackage format."""
        config = CoreMLConfig(model_format="mlpackage")
        converter = CoreMLConverter(config)
        mock_model = Mock()

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            converter.save_model(mock_model, "test_model")

            mock_mkdir.assert_called_once()
            mock_model.save.assert_called_once()
            # Check metadata was set
            assert mock_model.short_description == config.metadata_description
            assert mock_model.author == config.metadata_author

    def test_save_model_mlmodel(self):
        """Test saving model in mlmodel format."""
        config = CoreMLConfig(model_format="mlmodel")
        converter = CoreMLConverter(config)
        mock_model = Mock()

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            converter.save_model(mock_model, "test_model.mlmodel")

            mock_mkdir.assert_called_once()
            mock_model.save.assert_called_once()

    def test_validate_model_skip_validation(self):
        """Test model validation when skipped."""
        config = CoreMLConfig(skip_model_validation=True)
        converter = CoreMLConverter(config)
        mock_model = Mock()

        result = converter.validate_model(mock_model)

        assert result == {"validation_skipped": True}

    def test_validate_model_basic_info(self):
        """Test basic model validation."""
        config = CoreMLConfig(skip_model_validation=False)
        converter = CoreMLConverter(config)

        mock_model = Mock()
        mock_input = Mock()
        mock_output = Mock()
        mock_model.input_description = [mock_input]
        mock_model.output_description = [mock_output]

        result = converter.validate_model(mock_model)

        assert "model_type" in result
        assert "input_description" in result
        assert "output_description" in result
        assert result["compute_units"] == config.compute_units
        assert result["precision"] == config.precision

    def test_validate_model_with_prediction_test(self):
        """Test model validation with prediction test."""
        config = CoreMLConfig(skip_model_validation=False, generate_test_inputs=True)
        converter = CoreMLConverter(config)

        mock_model = Mock()
        mock_model.input_description = []
        mock_model.output_description = []
        mock_predictions = {"output": np.array([1, 2, 3])}
        mock_model.predict.return_value = mock_predictions

        test_inputs = {"input": np.array([1, 2, 3])}
        result = converter.validate_model(mock_model, test_inputs)

        assert result["prediction_test"] == "passed"
        assert "output_shapes" in result

    def test_validate_model_prediction_failure(self):
        """Test model validation with prediction failure."""
        config = CoreMLConfig(skip_model_validation=False, generate_test_inputs=True)
        converter = CoreMLConverter(config)

        mock_model = Mock()
        mock_model.input_description = []
        mock_model.output_description = []
        mock_model.predict.side_effect = RuntimeError("Prediction failed")

        test_inputs = {"input": np.array([1, 2, 3])}
        result = converter.validate_model(mock_model, test_inputs)

        assert "failed: Prediction failed" in result["prediction_test"]

    def test_validate_model_exception(self):
        """Test model validation with exception."""
        config = CoreMLConfig(skip_model_validation=False)
        converter = CoreMLConverter(config)

        mock_model = Mock()
        mock_model.input_description = Mock(side_effect=RuntimeError("Error"))

        result = converter.validate_model(mock_model)

        assert "validation_error" in result

    def test_benchmark_model_warmup_failure(self):
        """Test model benchmarking with warmup failure."""
        mock_model = Mock()
        mock_input_desc = Mock()
        mock_input_desc.name = "input"
        mock_input_desc.type.multiArrayType.shape = [1, 3, 224, 224]
        mock_model.input_description = [mock_input_desc]
        mock_model.predict.side_effect = RuntimeError("Warmup failed")

        result = self.converter.benchmark_model(mock_model)

        assert "benchmark_error" in result
        assert "Warmup failed" in result["benchmark_error"]

    def test_benchmark_model_prediction_failure(self):
        """Test model benchmarking with prediction failure."""
        mock_model = Mock()
        mock_input_desc = Mock()
        mock_input_desc.name = "input"
        mock_input_desc.type.multiArrayType.shape = [1, 3, 224, 224]
        mock_model.input_description = [mock_input_desc]

        # Warmup succeeds, but benchmark prediction fails
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # Warmup
                return {"output": np.array([1])}
            else:  # Benchmark runs
                raise RuntimeError("Prediction failed")

        mock_model.predict.side_effect = side_effect

        result = self.converter.benchmark_model(mock_model, num_runs=3)

        assert "benchmark_error" in result
        assert "Prediction failed" in result["benchmark_error"]

    def test_benchmark_model_success(self):
        """Test successful model benchmarking."""
        mock_model = Mock()
        mock_input_desc = Mock()
        mock_input_desc.name = "input"
        mock_input_desc.type.multiArrayType.shape = [1, 3, 224, 224]
        mock_model.input_description = [mock_input_desc]
        mock_model.predict.return_value = {"output": np.array([1])}

        with patch("time.time", side_effect=[0, 0.1, 1, 1.1, 2, 2.1]):
            result = self.converter.benchmark_model(mock_model, num_runs=2)

        assert result["num_runs"] == 2
        assert "avg_inference_time" in result
        assert "std_inference_time" in result
        assert "min_inference_time" in result
        assert "max_inference_time" in result
        assert "throughput_fps" in result
        assert result["config"] == self.config.to_dict()


class TestStableDiffusionConversion:
    """Test Stable Diffusion specific conversion methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CoreMLConfig(
            convert_unet=True, convert_vae=True, convert_text_encoder=True
        )
        self.converter = CoreMLConverter(self.config)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("pathlib.Path.mkdir")
    @patch("src.coreml.converter.CoreMLConverter._convert_unet")
    @patch("src.coreml.converter.CoreMLConverter._convert_vae_decoder")
    @patch("src.coreml.converter.CoreMLConverter._convert_text_encoder")
    def test_convert_stable_diffusion_components_success(
        self, mock_text_encoder, mock_vae, mock_unet, mock_mkdir, mock_pipeline
    ):
        """Test successful Stable Diffusion conversion."""
        # Mock pipeline
        mock_pipe = Mock()
        mock_pipe.unet = Mock()
        mock_pipe.vae = Mock()
        mock_pipe.text_encoder = Mock()
        mock_pipeline.return_value = mock_pipe

        # Mock converted models
        mock_unet_model = Mock()
        mock_vae_model = Mock()
        mock_text_model = Mock()
        mock_unet.return_value = mock_unet_model
        mock_vae.return_value = mock_vae_model
        mock_text_encoder.return_value = mock_text_model

        result = self.converter.convert_stable_diffusion_components(
            "test-model", "output_dir"
        )

        assert "unet" in result
        assert "vae_decoder" in result
        assert "text_encoder" in result

        mock_unet_model.save.assert_called_once()
        mock_vae_model.save.assert_called_once()
        mock_text_model.save.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("pathlib.Path.mkdir")
    @patch("src.coreml.converter.CoreMLConverter._convert_unet")
    def test_convert_stable_diffusion_unet_failure(
        self, mock_unet, mock_mkdir, mock_pipeline
    ):
        """Test Stable Diffusion conversion with U-Net failure."""
        config = CoreMLConfig(
            convert_unet=True, convert_vae=False, convert_text_encoder=False
        )
        converter = CoreMLConverter(config)

        mock_pipe = Mock()
        mock_pipe.unet = Mock()
        mock_pipeline.return_value = mock_pipe
        mock_unet.side_effect = RuntimeError("U-Net conversion failed")

        result = converter.convert_stable_diffusion_components("test-model")

        # Should return empty dict since conversion failed
        assert result == {}

    @patch("torch.jit.trace")
    @patch("coremltools.convert")
    def test_convert_unet(self, mock_ct_convert, mock_trace):
        """Test U-Net conversion."""
        mock_unet = Mock()
        mock_traced = Mock()
        mock_coreml = Mock()
        mock_trace.return_value = mock_traced
        mock_ct_convert.return_value = mock_coreml

        result = self.converter._convert_unet(mock_unet)

        assert result == mock_coreml
        mock_trace.assert_called_once()
        mock_ct_convert.assert_called_once()

    @patch("torch.jit.trace")
    @patch("coremltools.convert")
    def test_convert_vae_decoder(self, mock_ct_convert, mock_trace):
        """Test VAE decoder conversion."""
        mock_vae = Mock()
        mock_vae.decoder = Mock()
        mock_traced = Mock()
        mock_coreml = Mock()
        mock_trace.return_value = mock_traced
        mock_ct_convert.return_value = mock_coreml

        with patch("torch.randn") as mock_randn:
            mock_tensor = Mock()
            mock_tensor.shape = (1, 4, 64, 64)  # Proper shape tuple
            mock_randn.return_value = mock_tensor

            result = self.converter._convert_vae_decoder(mock_vae)

            assert result == mock_coreml
            mock_trace.assert_called_once_with(mock_vae.decoder, mock_tensor)
            mock_ct_convert.assert_called_once()

    @patch("torch.jit.trace")
    @patch("coremltools.convert")
    def test_convert_text_encoder(self, mock_ct_convert, mock_trace):
        """Test text encoder conversion."""
        mock_text_encoder = Mock()
        mock_traced = Mock()
        mock_coreml = Mock()
        mock_trace.return_value = mock_traced
        mock_ct_convert.return_value = mock_coreml

        result = self.converter._convert_text_encoder(mock_text_encoder)

        assert result == mock_coreml
        mock_trace.assert_called_once()
        mock_ct_convert.assert_called_once()
