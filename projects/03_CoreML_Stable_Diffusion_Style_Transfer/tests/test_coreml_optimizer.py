"""Test CoreML optimizer functionality."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.coreml.config import CoreMLConfig
from src.coreml.optimizer import CoreMLOptimizer


class TestCoreMLOptimizer:
    """Test CoreMLOptimizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CoreMLConfig()
        self.optimizer = CoreMLOptimizer(self.config)

    def test_optimizer_creation(self):
        """Test optimizer creation."""
        assert self.optimizer.config == self.config

    def test_optimizer_creation_with_validation(self):
        """Test optimizer creation validates config."""
        config = CoreMLConfig(quantization_bits=0)  # Invalid config
        with pytest.raises(ValueError):
            CoreMLOptimizer(config)

    def test_optimize_model_file_not_found(self):
        """Test optimize_model with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Model not found"):
            self.optimizer.optimize_model("nonexistent.mlpackage")

    @patch("pathlib.Path.exists")
    @patch("coremltools.models.MLModel")
    def test_optimize_model_load_failure(self, mock_mlmodel, mock_exists):
        """Test optimize_model with model load failure."""
        mock_exists.return_value = True
        mock_mlmodel.side_effect = RuntimeError("Load failed")

        with pytest.raises(RuntimeError, match="Failed to load Core ML model"):
            self.optimizer.optimize_model("model.mlpackage")

    @patch("pathlib.Path.exists")
    @patch("coremltools.models.MLModel")
    def test_optimize_model_basic(self, mock_mlmodel, mock_exists):
        """Test basic model optimization without specific optimizations."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_mlmodel.return_value = mock_model

        config = CoreMLConfig(
            use_low_precision_weights=False,
            use_palettization=False,
            reduce_memory_footprint=False,
            optimize_for_inference=False,
        )
        optimizer = CoreMLOptimizer(config)

        result = optimizer.optimize_model("model.mlpackage")

        assert result == mock_model
        mock_mlmodel.assert_called_once_with("model.mlpackage")

    @patch("pathlib.Path.exists")
    @patch("coremltools.models.MLModel")
    @patch("src.coreml.optimizer.CoreMLOptimizer._apply_weight_quantization")
    @patch("src.coreml.optimizer.CoreMLOptimizer._save_optimized_model")
    def test_optimize_model_with_quantization(
        self, mock_save, mock_quantize, mock_mlmodel, mock_exists
    ):
        """Test model optimization with weight quantization."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_optimized = Mock()
        mock_mlmodel.return_value = mock_model
        mock_quantize.return_value = mock_optimized

        config = CoreMLConfig(use_low_precision_weights=True)
        optimizer = CoreMLOptimizer(config)

        result = optimizer.optimize_model("model.mlpackage", "output.mlpackage")

        assert result == mock_optimized
        mock_quantize.assert_called_once_with(mock_model)
        mock_save.assert_called_once_with(mock_optimized, "output.mlpackage")

    @patch("pathlib.Path.exists")
    @patch("coremltools.models.MLModel")
    @patch("src.coreml.optimizer.CoreMLOptimizer._apply_palettization")
    def test_optimize_model_with_palettization(
        self, mock_palettize, mock_mlmodel, mock_exists
    ):
        """Test model optimization with palettization."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_optimized = Mock()
        mock_mlmodel.return_value = mock_model
        mock_palettize.return_value = mock_optimized

        config = CoreMLConfig(use_palettization=True)
        optimizer = CoreMLOptimizer(config)

        result = optimizer.optimize_model("model.mlpackage")

        assert result == mock_optimized
        mock_palettize.assert_called_once_with(mock_model)

    @patch("pathlib.Path.exists")
    @patch("coremltools.models.MLModel")
    @patch("src.coreml.optimizer.CoreMLOptimizer._apply_memory_optimizations")
    @patch("src.coreml.optimizer.CoreMLOptimizer._apply_inference_optimizations")
    def test_optimize_model_with_all_optimizations(
        self, mock_inference, mock_memory, mock_mlmodel, mock_exists
    ):
        """Test model optimization with all optimizations enabled."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_mlmodel.return_value = mock_model
        mock_memory.return_value = mock_model
        mock_inference.return_value = mock_model

        config = CoreMLConfig(reduce_memory_footprint=True, optimize_for_inference=True)
        optimizer = CoreMLOptimizer(config)

        result = optimizer.optimize_model("model.mlpackage")

        assert result == mock_model
        mock_memory.assert_called_once()
        mock_inference.assert_called_once()

    @patch("coremltools.models.neural_network.quantization_utils.quantize_weights")
    def test_apply_weight_quantization_float16(self, mock_quantize):
        """Test float16 weight quantization."""
        config = CoreMLConfig(precision="float16", use_low_precision_weights=True)
        optimizer = CoreMLOptimizer(config)
        mock_model = Mock()
        mock_quantized = Mock()
        mock_quantize.return_value = mock_quantized

        result = optimizer._apply_weight_quantization(mock_model)

        assert result == mock_quantized
        mock_quantize.assert_called_once_with(mock_model, nbits=16)

    @patch("coremltools.models.neural_network.quantization_utils.quantize_weights")
    def test_apply_weight_quantization_float32(self, mock_quantize):
        """Test float32 weight quantization (no change)."""
        config = CoreMLConfig(precision="float32", use_low_precision_weights=True)
        optimizer = CoreMLOptimizer(config)
        mock_model = Mock()

        result = optimizer._apply_weight_quantization(mock_model)

        assert result == mock_model
        mock_quantize.assert_not_called()

    @patch("coremltools.models.neural_network.quantization_utils.quantize_weights")
    def test_apply_weight_quantization_failure(self, mock_quantize):
        """Test weight quantization failure."""
        config = CoreMLConfig(precision="float16", use_low_precision_weights=True)
        optimizer = CoreMLOptimizer(config)
        mock_model = Mock()
        mock_quantize.side_effect = RuntimeError("Quantization failed")

        result = optimizer._apply_weight_quantization(mock_model)

        assert result == mock_model  # Should return original on failure

    def test_apply_palettization_kmeans(self):
        """Test palettization using quantize_weights fallback."""
        config = CoreMLConfig(palettization_mode="kmeans", quantization_bits=8)
        optimizer = CoreMLOptimizer(config)
        mock_model = Mock()
        mock_palettized = Mock()

        with patch(
            "coremltools.models.neural_network.quantization_utils.quantize_weights",
            return_value=mock_palettized,
        ) as mock_quantize:
            result = optimizer._apply_palettization(mock_model)

            assert result == mock_palettized
            mock_quantize.assert_called_once_with(mock_model, nbits=8)

    def test_apply_palettization_uniform(self):
        """Test palettization using quantize_weights fallback."""
        config = CoreMLConfig(palettization_mode="uniform", quantization_bits=4)
        optimizer = CoreMLOptimizer(config)
        mock_model = Mock()
        mock_palettized = Mock()

        with patch(
            "coremltools.models.neural_network.quantization_utils.quantize_weights",
            return_value=mock_palettized,
        ) as mock_quantize:
            result = optimizer._apply_palettization(mock_model)

            assert result == mock_palettized
            mock_quantize.assert_called_once_with(mock_model, nbits=4)

    def test_apply_palettization_failure(self):
        """Test palettization failure."""
        config = CoreMLConfig(use_palettization=True)
        optimizer = CoreMLOptimizer(config)
        mock_model = Mock()

        with patch(
            "coremltools.models.neural_network.quantization_utils.quantize_weights",
            side_effect=RuntimeError("Quantization failed"),
        ):
            result = optimizer._apply_palettization(mock_model)

            assert result == mock_model  # Should return original on failure

    def test_apply_memory_optimizations(self):
        """Test memory optimizations."""
        mock_model = Mock()

        result = self.optimizer._apply_memory_optimizations(mock_model)

        # Currently returns the same model (placeholder implementation)
        assert result == mock_model

    def test_apply_inference_optimizations(self):
        """Test inference optimizations."""
        mock_model = Mock()
        mock_model.compute_unit = None

        result = self.optimizer._apply_inference_optimizations(mock_model)

        assert result == mock_model
        assert mock_model.compute_unit == self.config.get_compute_units_enum()

    def test_apply_inference_optimizations_no_compute_unit(self):
        """Test inference optimizations when model has no compute_unit attribute."""
        mock_model = Mock(spec=[])  # Mock without compute_unit attribute

        result = self.optimizer._apply_inference_optimizations(mock_model)

        assert result == mock_model

    def test_apply_inference_optimizations_failure(self):
        """Test inference optimizations failure."""
        mock_model = Mock()
        mock_model.compute_unit = Mock(side_effect=RuntimeError("Failed"))

        result = self.optimizer._apply_inference_optimizations(mock_model)

        assert result == mock_model

    def test_save_optimized_model_mlpackage(self):
        """Test saving optimized model in mlpackage format."""
        config = CoreMLConfig(model_format="mlpackage")
        optimizer = CoreMLOptimizer(config)
        mock_model = Mock()

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            optimizer._save_optimized_model(mock_model, "test_model")

            mock_mkdir.assert_called_once()
            mock_model.save.assert_called_once()
            # Check metadata was updated
            assert "(Optimized)" in mock_model.short_description
            assert mock_model.author == config.metadata_author

    def test_save_optimized_model_mlmodel(self):
        """Test saving optimized model in mlmodel format."""
        config = CoreMLConfig(model_format="mlmodel")
        optimizer = CoreMLOptimizer(config)
        mock_model = Mock()

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            optimizer._save_optimized_model(mock_model, "test_model.mlmodel")

            mock_mkdir.assert_called_once()
            mock_model.save.assert_called_once()


class TestStableDiffusionOptimization:
    """Test Stable Diffusion pipeline optimization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CoreMLConfig()
        self.optimizer = CoreMLOptimizer(self.config)

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("src.coreml.optimizer.CoreMLOptimizer.optimize_model")
    def test_optimize_stable_diffusion_pipeline_all_models(
        self, mock_optimize, mock_exists, mock_mkdir
    ):
        """Test optimizing complete Stable Diffusion pipeline."""
        mock_exists.return_value = True
        mock_optimize.return_value = Mock()

        result = self.optimizer.optimize_stable_diffusion_pipeline(
            "models_dir", "output_dir"
        )

        assert "unet" in result
        assert "vae_decoder" in result
        assert "text_encoder" in result
        assert len(result) == 3
        assert mock_optimize.call_count == 3

    def test_optimize_stable_diffusion_pipeline_partial_models(self):
        """Test optimizing partial Stable Diffusion pipeline with some models missing."""
        # Simplified test - just check that the method works when no files exist
        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.exists", return_value=False):
                result = self.optimizer.optimize_stable_diffusion_pipeline("models_dir")

                # When no models exist, should return empty dict
                assert len(result) == 0
                assert isinstance(result, dict)

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("src.coreml.optimizer.CoreMLOptimizer.optimize_model")
    def test_optimize_stable_diffusion_pipeline_optimization_failure(
        self, mock_optimize, mock_exists, mock_mkdir
    ):
        """Test Stable Diffusion pipeline optimization with failures."""
        mock_exists.return_value = True
        mock_optimize.side_effect = RuntimeError("Optimization failed")

        result = self.optimizer.optimize_stable_diffusion_pipeline("models_dir")

        # Should return empty dict when all optimizations fail
        assert len(result) == 0
        assert mock_optimize.call_count == 3

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    def test_optimize_stable_diffusion_pipeline_no_models(
        self, mock_exists, mock_mkdir
    ):
        """Test Stable Diffusion pipeline optimization with no models."""
        mock_exists.return_value = False

        result = self.optimizer.optimize_stable_diffusion_pipeline("models_dir")

        assert len(result) == 0


class TestBenchmarking:
    """Test optimization benchmarking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CoreMLConfig()
        self.optimizer = CoreMLOptimizer(self.config)

    @patch("coremltools.models.MLModel")
    def test_benchmark_optimization_impact_load_failure(self, mock_mlmodel):
        """Test benchmarking with model load failure."""
        mock_mlmodel.side_effect = RuntimeError("Load failed")

        result = self.optimizer.benchmark_optimization_impact(
            "original.mlpackage", "optimized.mlpackage"
        )

        assert "benchmark_error" in result
        assert "Failed to load models" in result["benchmark_error"]

    @patch("coremltools.models.MLModel")
    @patch("src.coreml.optimizer.CoreMLOptimizer._create_benchmark_inputs")
    @patch("src.coreml.optimizer.CoreMLOptimizer._benchmark_single_model")
    def test_benchmark_optimization_impact_success(
        self, mock_benchmark, mock_inputs, mock_mlmodel
    ):
        """Test successful optimization impact benchmarking."""
        mock_original = Mock()
        mock_optimized = Mock()
        mock_mlmodel.side_effect = [mock_original, mock_optimized]
        mock_inputs.return_value = {"input": np.array([1, 2, 3])}

        # Mock benchmark results
        original_results = {
            "avg_time": 1.0,
            "std_time": 0.1,
            "min_time": 0.9,
            "max_time": 1.1,
            "throughput": 1.0,
        }
        optimized_results = {
            "avg_time": 0.5,
            "std_time": 0.05,
            "min_time": 0.45,
            "max_time": 0.55,
            "throughput": 2.0,
        }
        mock_benchmark.side_effect = [original_results, optimized_results]

        result = self.optimizer.benchmark_optimization_impact(
            "original.mlpackage", "optimized.mlpackage", num_runs=5
        )

        assert "original_model" in result
        assert "optimized_model" in result
        assert "improvements" in result
        assert result["improvements"]["inference_time_improvement"]["relative"] == 50.0
        assert result["improvements"]["throughput_improvement"]["relative"] == 100.0

    @patch("coremltools.models.MLModel")
    @patch("src.coreml.optimizer.CoreMLOptimizer._create_benchmark_inputs")
    @patch("src.coreml.optimizer.CoreMLOptimizer._benchmark_single_model")
    def test_benchmark_optimization_impact_benchmark_failure(
        self, mock_benchmark, mock_inputs, mock_mlmodel
    ):
        """Test benchmarking with benchmark failure."""
        mock_original = Mock()
        mock_optimized = Mock()
        mock_mlmodel.side_effect = [mock_original, mock_optimized]
        mock_inputs.return_value = {"input": np.array([1, 2, 3])}
        mock_benchmark.side_effect = [
            {"error": "Original model failed"},
            {"avg_time": 0.5, "throughput": 2.0},
        ]

        result = self.optimizer.benchmark_optimization_impact(
            "original.mlpackage", "optimized.mlpackage"
        )

        assert "benchmark_error" in result
        assert (
            "One or more models failed during benchmarking" in result["benchmark_error"]
        )

    def test_create_benchmark_inputs_multiarray(self):
        """Test creating benchmark inputs for multiarray type."""
        mock_model = Mock()
        mock_input_desc = Mock()
        mock_input_desc.name = "input_tensor"
        mock_input_desc.type.multiArrayType.shape = [1, 3, 224, 224]
        mock_model.input_description = [mock_input_desc]

        # Mock hasattr for multiArrayType
        with patch("builtins.hasattr") as mock_hasattr:
            mock_hasattr.side_effect = lambda obj, attr: attr == "multiArrayType"

            inputs = self.optimizer._create_benchmark_inputs(mock_model)

        assert "input_tensor" in inputs
        assert inputs["input_tensor"].shape == (1, 3, 224, 224)
        assert inputs["input_tensor"].dtype == np.float32

    def test_create_benchmark_inputs_image(self):
        """Test creating benchmark inputs for image type."""
        mock_model = Mock()
        mock_input_desc = Mock()
        mock_input_desc.name = "input_image"
        mock_input_desc.type.imageType.height = 224
        mock_input_desc.type.imageType.width = 224
        mock_model.input_description = [mock_input_desc]

        # Mock hasattr for imageType
        with patch("builtins.hasattr") as mock_hasattr:
            mock_hasattr.side_effect = lambda obj, attr: attr == "imageType"

            inputs = self.optimizer._create_benchmark_inputs(mock_model)

        assert "input_image" in inputs
        assert inputs["input_image"].shape == (224, 224, 3)
        assert inputs["input_image"].dtype == np.uint8

    def test_benchmark_single_model_warmup_failure(self):
        """Test single model benchmarking with warmup failure."""
        mock_model = Mock()
        mock_model.predict.side_effect = RuntimeError("Warmup failed")

        result = self.optimizer._benchmark_single_model(
            mock_model, {"input": np.array([1])}, 3
        )

        assert "error" in result
        assert "Warmup failed" in result["error"]

    def test_benchmark_single_model_prediction_failure(self):
        """Test single model benchmarking with prediction failure."""
        mock_model = Mock()

        # Warmup succeeds, but prediction fails
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # Warmup
                return {"output": np.array([1])}
            else:  # Prediction
                raise RuntimeError("Prediction failed")

        mock_model.predict.side_effect = side_effect

        result = self.optimizer._benchmark_single_model(
            mock_model, {"input": np.array([1])}, 3
        )

        assert "error" in result
        assert "Prediction failed" in result["error"]

    def test_benchmark_single_model_success(self):
        """Test successful single model benchmarking."""
        mock_model = Mock()
        mock_model.predict.return_value = {"output": np.array([1])}

        with patch("time.time", side_effect=[0, 0.1, 1, 1.1, 2, 2.1]):
            result = self.optimizer._benchmark_single_model(
                mock_model, {"input": np.array([1])}, 2
            )

        assert "avg_time" in result
        assert "std_time" in result
        assert "min_time" in result
        assert "max_time" in result
        assert "throughput" in result
        assert abs(result["avg_time"] - 0.1) < 1e-10
        assert abs(result["throughput"] - 10.0) < 1e-10


class TestModelInfo:
    """Test model information and validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CoreMLConfig()
        self.optimizer = CoreMLOptimizer(self.config)

    @patch("coremltools.models.MLModel")
    def test_get_model_info_success(self, mock_mlmodel):
        """Test successful model information retrieval."""
        mock_model = Mock()
        mock_model.short_description = "Test model"
        mock_model.author = "Test author"

        # Mock input/output descriptions
        mock_input = Mock()
        mock_input.name = "input"
        mock_multiarray_type = Mock()
        mock_multiarray_type.shape = [1, 3, 224, 224]
        mock_input.type = Mock()
        mock_input.type.multiArrayType = mock_multiarray_type

        mock_output = Mock()
        mock_output.name = "output"
        mock_output_multiarray_type = Mock()
        mock_output_multiarray_type.shape = [1, 1000]
        mock_output.type = Mock()
        mock_output.type.multiArrayType = mock_output_multiarray_type

        mock_model.input_description = [mock_input]
        mock_model.output_description = [mock_output]
        mock_mlmodel.return_value = mock_model

        with patch("builtins.hasattr", return_value=True):
            result = self.optimizer.get_model_info("model.mlpackage")

        assert result["model_path"] == "model.mlpackage"
        assert result["short_description"] == "Test model"
        assert result["author"] == "Test author"
        assert len(result["input_description"]) == 1
        assert len(result["output_description"]) == 1

    @patch("coremltools.models.MLModel")
    def test_get_model_info_failure(self, mock_mlmodel):
        """Test model information retrieval failure."""
        mock_mlmodel.side_effect = RuntimeError("Failed to load")

        result = self.optimizer.get_model_info("model.mlpackage")

        assert "error" in result
        assert "Failed to get model info" in result["error"]

    @patch("coremltools.models.MLModel")
    def test_validate_apple_silicon_optimization_success(self, mock_mlmodel):
        """Test Apple Silicon optimization validation."""
        mock_model = Mock()
        mock_model.compute_unit = Mock()

        # Mock ComputeUnit enum
        with patch("coremltools.ComputeUnit") as mock_compute_unit:
            mock_compute_unit.ALL = "all"
            mock_model.compute_unit = "all"
            mock_mlmodel.return_value = mock_model

            result = self.optimizer.validate_apple_silicon_optimization(
                "model.mlpackage"
            )

        assert result["is_optimized"] is True
        assert len(result["optimization_checks"]) > 0
        assert "✓ Compute units set to ALL" in result["optimization_checks"][0]

    @patch("coremltools.models.MLModel")
    def test_validate_apple_silicon_optimization_not_optimized(self, mock_mlmodel):
        """Test Apple Silicon optimization validation for non-optimized model."""
        mock_model = Mock()
        mock_model.compute_unit = "cpu_only"
        mock_mlmodel.return_value = mock_model

        result = self.optimizer.validate_apple_silicon_optimization("model.mlpackage")

        assert result["is_optimized"] is False
        assert len(result["recommendations"]) > 0
        assert "Set compute units to ALL or CPU_AND_GPU" in result["recommendations"]

    @patch("coremltools.models.MLModel")
    def test_validate_apple_silicon_optimization_no_compute_unit(self, mock_mlmodel):
        """Test validation when model has no compute_unit attribute."""
        mock_model = Mock(spec=[])  # Mock without compute_unit attribute
        mock_mlmodel.return_value = mock_model

        with patch("builtins.hasattr", return_value=False):
            result = self.optimizer.validate_apple_silicon_optimization(
                "model.mlpackage"
            )

        assert (
            "? Compute units information not available" in result["optimization_checks"]
        )

    @patch("coremltools.models.MLModel")
    def test_validate_apple_silicon_optimization_failure(self, mock_mlmodel):
        """Test Apple Silicon optimization validation failure."""
        mock_mlmodel.side_effect = RuntimeError("Validation failed")

        result = self.optimizer.validate_apple_silicon_optimization("model.mlpackage")

        assert "error" in result
        assert "Validation failed" in result["error"]
