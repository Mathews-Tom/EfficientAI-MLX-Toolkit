"""
Tests for quantization functionality.
"""

import pytest
from pathlib import Path

from src.quantization.config import QuantizationConfig, QuantizationMethod, CalibrationMethod
from src.quantization.quantizer import MLXQuantizer


class TestQuantizationConfig:
    """Test quantization configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = QuantizationConfig()
        assert config.target_bits == 4
        assert config.method == QuantizationMethod.POST_TRAINING
        assert config.calibration_method == CalibrationMethod.MINMAX
        assert config.use_mlx_quantization is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = QuantizationConfig(target_bits=8, calibration_samples=100)
        assert config.target_bits == 8

        # Invalid target_bits
        with pytest.raises(ValueError):
            QuantizationConfig(target_bits=3)

        # Invalid calibration_samples
        with pytest.raises(ValueError):
            QuantizationConfig(calibration_samples=-1)

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "target_bits": 8,
            "method": "quantization_aware",
            "calibration_method": "entropy",
            "symmetric": True,
        }

        config = QuantizationConfig.from_dict(config_dict)
        assert config.target_bits == 8
        assert config.method == QuantizationMethod.QUANTIZATION_AWARE
        assert config.calibration_method == CalibrationMethod.ENTROPY
        assert config.symmetric is True

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        config = QuantizationConfig(target_bits=4)
        ratio = config.get_compression_ratio()
        assert ratio == 4.0  # 16-bit to 4-bit = 4x compression


class TestMLXQuantizer:
    """Test MLX quantizer."""

    def test_quantizer_initialization(self):
        """Test quantizer initialization."""
        config = QuantizationConfig()
        quantizer = MLXQuantizer(config)

        assert quantizer.config == config
        assert quantizer.model is None
        assert quantizer.quantized_model is None

    def test_quantizer_info(self):
        """Test quantizer info retrieval."""
        config = QuantizationConfig(target_bits=8)
        quantizer = MLXQuantizer(config)

        info = quantizer.get_quantization_info()
        assert "config" in info
        assert info["config"]["target_bits"] == 8


@pytest.mark.apple_silicon
class TestMLXQuantizerWithMLX:
    """Test MLX quantizer with MLX available."""

    @pytest.mark.skipif(not pytest.importorskip("mlx"), reason="MLX not available")
    def test_quantizer_with_mlx(self):
        """Test quantizer when MLX is available."""
        config = QuantizationConfig()
        quantizer = MLXQuantizer(config)

        # This would test actual MLX functionality
        # For now, just verify it doesn't crash
        info = quantizer.get_quantization_info()
        assert "config" in info


@pytest.mark.benchmark
class TestQuantizationBenchmarks:
    """Benchmarking tests for quantization."""

    def test_config_performance(self):
        """Test configuration creation performance."""
        import time

        start_time = time.time()
        for _ in range(1000):
            config = QuantizationConfig()
        end_time = time.time()

        # Should be very fast
        assert (end_time - start_time) < 1.0