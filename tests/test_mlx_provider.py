"""
Unit tests for MLX LLM Provider.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from dspy_toolkit.exceptions import MLXProviderError
from dspy_toolkit.providers.mlx_provider import (
    MLXLLMProvider,
    setup_mlx_provider_for_dspy,
)
from dspy_toolkit.types import DSPyConfig, HardwareInfo


class TestMLXLLMProvider:
    """Test cases for MLX LLM Provider."""

    @pytest.fixture
    def mock_mlx_modules(self):
        """Mock MLX modules for testing."""
        with patch.dict(
            "sys.modules",
            {
                "mlx": MagicMock(),
                "mlx.core": MagicMock(),
                "mlx.nn": MagicMock(),
                "mlx_lm": MagicMock(),
            },
        ):
            # Mock MLX core functions
            mock_mx = MagicMock()
            mock_mx.metal.is_available.return_value = True
            mock_mx.metal.get_memory_limit.return_value = 16 * 1024**3
            mock_mx.metal.set_memory_limit = MagicMock()

            # Mock MLX-LM functions
            mock_mlx_lm = MagicMock()
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # Mock token IDs
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            mock_mlx_lm.generate.return_value = "Test response"

            with (
                patch("dspy_toolkit.providers.mlx_provider.mx", mock_mx),
                patch("dspy_toolkit.providers.mlx_provider.load", mock_mlx_lm.load),
                patch(
                    "dspy_toolkit.providers.mlx_provider.generate",
                    mock_mlx_lm.generate,
                ),
                patch("dspy_toolkit.providers.mlx_provider.MLX_AVAILABLE", True),
                patch(
                    "dspy_toolkit.providers.mlx_provider.LITELLM_AVAILABLE", True
                ),
            ):

                yield {
                    "mx": mock_mx,
                    "mlx_lm": mock_mlx_lm,
                    "model": mock_model,
                    "tokenizer": mock_tokenizer,
                }

    @pytest.fixture
    def test_config(self, tmp_path):
        """Test configuration for MLX provider."""
        return DSPyConfig(
            model_provider="mlx",
            model_name="mlx/test-model",
            optimization_level=2,
            cache_dir=tmp_path / "cache",
        )

    def test_provider_initialization(self, mock_mlx_modules, test_config):
        """Test MLX provider initialization."""
        provider = MLXLLMProvider(test_config)

        assert provider.config == test_config
        assert provider.model is not None
        assert provider.tokenizer is not None
        assert provider.is_available() == True

    def test_hardware_detection(self, mock_mlx_modules, test_config):
        """Test Apple Silicon hardware detection."""
        provider = MLXLLMProvider(test_config)
        hardware_info = provider.hardware_info

        assert isinstance(hardware_info, HardwareInfo)
        assert hardware_info.metal_available == True
        assert hardware_info.device_type in ["apple_silicon", "m1"]
        assert hardware_info.total_memory > 0
        assert hardware_info.optimization_level == test_config.optimization_level

    def test_completion_generation(self, mock_mlx_modules, test_config):
        """Test completion generation."""
        provider = MLXLLMProvider(test_config)

        # Mock LiteLLM classes
        with (
            patch(
                "dspy_toolkit.providers.mlx_provider.ModelResponse"
            ) as mock_response,
            patch("dspy_toolkit.providers.mlx_provider.Choices") as mock_choices,
            patch("dspy_toolkit.providers.mlx_provider.Message") as mock_message,
            patch("dspy_toolkit.providers.mlx_provider.Usage") as mock_usage,
        ):

            mock_response.return_value = MagicMock()

            response = provider.completion(
                messages=[{"content": "Hello, world!"}], max_tokens=50, temperature=0.7
            )

            assert response is not None
            mock_response.assert_called_once()

    def test_model_info_retrieval(self, mock_mlx_modules, test_config):
        """Test model information retrieval."""
        provider = MLXLLMProvider(test_config)
        model_info = provider.get_model_info()

        assert isinstance(model_info, dict)
        assert "model_name" in model_info
        assert model_info["model_name"] == test_config.model_name
        assert "hardware_optimized" in model_info

    def test_performance_benchmarking(self, mock_mlx_modules, test_config):
        """Test performance benchmarking."""
        provider = MLXLLMProvider(test_config)

        with patch.object(provider, "completion") as mock_completion:
            mock_completion.return_value = MagicMock()

            benchmark_results = provider.benchmark_performance()

            assert isinstance(benchmark_results, dict)
            assert "average_time" in benchmark_results
            assert "tokens_per_second" in benchmark_results
            assert "hardware_type" in benchmark_results
            assert (
                benchmark_results["optimization_level"]
                == test_config.optimization_level
            )

    def test_mlx_unavailable_error(self, test_config):
        """Test error handling when MLX is unavailable."""
        with patch("dspy_toolkit.providers.mlx_provider.MLX_AVAILABLE", False):
            with pytest.raises(MLXProviderError, match="MLX is not available"):
                MLXLLMProvider(test_config)

    def test_litellm_unavailable_error(self, test_config):
        """Test error handling when LiteLLM is unavailable."""
        with (
            patch("dspy_toolkit.providers.mlx_provider.MLX_AVAILABLE", True),
            patch("dspy_toolkit.providers.mlx_provider.LITELLM_AVAILABLE", False),
        ):
            with pytest.raises(MLXProviderError, match="LiteLLM is not available"):
                MLXLLMProvider(test_config)

    def test_model_loading_failure(self, test_config):
        """Test error handling for model loading failure."""
        with (
            patch("dspy_toolkit.providers.mlx_provider.MLX_AVAILABLE", True),
            patch("dspy_toolkit.providers.mlx_provider.LITELLM_AVAILABLE", True),
            patch(
                "dspy_toolkit.providers.mlx_provider.load",
                side_effect=Exception("Model not found"),
            ),
        ):

            with pytest.raises(MLXProviderError, match="Failed to load MLX model"):
                MLXLLMProvider(test_config)

    def test_completion_error_handling(self, mock_mlx_modules, test_config):
        """Test error handling during completion generation."""
        provider = MLXLLMProvider(test_config)

        with patch(
            "dspy_toolkit.providers.mlx_provider.generate",
            side_effect=Exception("Generation failed"),
        ):
            with pytest.raises(MLXProviderError, match="Completion generation failed"):
                provider.completion(messages=[{"content": "test"}])

    @pytest.mark.benchmark
    def test_provider_performance(self, mock_mlx_modules, test_config, benchmark):
        """Benchmark provider initialization performance."""

        def create_provider():
            return MLXLLMProvider(test_config)

        result = benchmark(create_provider)
        assert result is not None


class TestMLXProviderSetup:
    """Test cases for MLX provider setup functions."""

    @pytest.fixture
    def test_config(self):
        """Test configuration."""
        return DSPyConfig(model_provider="mlx", model_name="mlx/test-model")

    def test_dspy_setup_success(self, test_config):
        """Test successful DSPy setup with MLX provider."""
        with (
            patch(
                "dspy_toolkit.providers.mlx_provider.MLXLLMProvider"
            ) as mock_provider,
            patch("dspy_toolkit.providers.mlx_provider.LITELLM_AVAILABLE", True),
            patch("dspy_toolkit.providers.mlx_provider.litellm") as mock_litellm,
            patch("dspy_toolkit.providers.mlx_provider.dspy") as mock_dspy,
        ):

            mock_provider_instance = MagicMock()
            mock_provider.return_value = mock_provider_instance

            setup_mlx_provider_for_dspy(test_config)

            mock_provider.assert_called_once_with(test_config)
            mock_dspy.configure.assert_called_once()

    def test_dspy_setup_litellm_unavailable(self, test_config):
        """Test DSPy setup failure when LiteLLM is unavailable."""
        with patch("dspy_toolkit.providers.mlx_provider.LITELLM_AVAILABLE", False):
            with pytest.raises(MLXProviderError, match="LiteLLM not available"):
                setup_mlx_provider_for_dspy(test_config)

    def test_dspy_setup_provider_failure(self, test_config):
        """Test DSPy setup failure when provider creation fails."""
        with (
            patch(
                "dspy_toolkit.providers.mlx_provider.MLXLLMProvider",
                side_effect=Exception("Provider failed"),
            ),
            patch("dspy_toolkit.providers.mlx_provider.LITELLM_AVAILABLE", True),
        ):

            with pytest.raises(MLXProviderError, match="DSPy MLX setup failed"):
                setup_mlx_provider_for_dspy(test_config)


@pytest.mark.integration
class TestMLXProviderIntegration:
    """Integration tests for MLX provider."""

    @pytest.mark.apple_silicon
    def test_real_hardware_detection(self):
        """Test hardware detection on real Apple Silicon (if available)."""
        config = DSPyConfig(model_name="test")

        try:
            # This will only work on actual Apple Silicon with MLX installed
            provider = MLXLLMProvider(config)
            hardware_info = provider.hardware_info

            # Basic validation
            assert isinstance(hardware_info, HardwareInfo)
            assert hardware_info.device_type in [
                "apple_silicon",
                "m1",
                "m2",
                "m3",
                "cpu",
            ]

        except MLXProviderError:
            # Expected on non-Apple Silicon or when MLX is not installed
            pytest.skip("MLX not available or not on Apple Silicon")

    @pytest.mark.slow
    def test_end_to_end_completion(self):
        """Test end-to-end completion generation (requires real model)."""
        # This test would require a real model and is marked as slow
        pytest.skip("Requires real model download - run manually for full testing")
