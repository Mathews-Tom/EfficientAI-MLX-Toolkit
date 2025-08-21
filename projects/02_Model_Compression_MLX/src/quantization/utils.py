"""
Utility functions for quantization operations.
"""

import math
from typing import Any, Dict, List, Tuple, Optional
import logging

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    np = None

from .config import QuantizationConfig

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def calculate_quantization_error(
    original_tensor: Any,
    quantized_tensor: Any,
    metric: str = "mse"
) -> float:
    """
    Calculate quantization error between original and quantized tensors.
    
    Args:
        original_tensor: Original tensor
        quantized_tensor: Quantized tensor
        metric: Error metric ("mse", "mae", "snr")
        
    Returns:
        Quantization error value
    """
    if not MLX_AVAILABLE:
        logger.warning("MLX not available, returning dummy error")
        return 0.0
    
    try:
        # Convert to numpy arrays for calculation
        if hasattr(original_tensor, 'numpy'):
            orig = original_tensor.numpy()
        else:
            orig = np.array(original_tensor)
        
        if hasattr(quantized_tensor, 'numpy'):
            quant = quantized_tensor.numpy()
        else:
            quant = np.array(quantized_tensor)
        
        if metric == "mse":
            # Mean Squared Error
            error = np.mean((orig - quant) ** 2)
        elif metric == "mae":
            # Mean Absolute Error
            error = np.mean(np.abs(orig - quant))
        elif metric == "snr":
            # Signal-to-Noise Ratio
            signal_power = np.mean(orig ** 2)
            noise_power = np.mean((orig - quant) ** 2)
            if noise_power == 0:
                error = float('inf')
            else:
                error = 10 * np.log10(signal_power / noise_power)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return float(error)
        
    except Exception as e:
        logger.error(f"Failed to calculate quantization error: {e}")
        return 0.0


def estimate_compression_ratio(
    config: QuantizationConfig,
    model_info: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Estimate compression ratio based on quantization configuration.
    
    Args:
        config: Quantization configuration
        model_info: Optional model information
        
    Returns:
        Dictionary with compression estimates
    """
    # Theoretical compression ratios
    weight_compression = 32 / config.weight_bits  # Assuming FP32 original
    activation_compression = 32 / config.activation_bits
    
    # Overall compression (weighted by typical parameter distribution)
    # Weights typically make up ~95% of model size
    overall_compression = 0.95 * weight_compression + 0.05 * activation_compression
    
    estimates = {
        "weight_compression_ratio": weight_compression,
        "activation_compression_ratio": activation_compression,
        "overall_compression_ratio": overall_compression,
        "theoretical_size_reduction": (1 - 1/overall_compression) * 100,
    }
    
    # Add model-specific estimates if available
    if model_info:
        original_size_mb = model_info.get("size_mb", 0)
        if original_size_mb > 0:
            compressed_size_mb = original_size_mb / overall_compression
            estimates.update({
                "original_size_mb": original_size_mb,
                "compressed_size_mb": compressed_size_mb,
                "size_reduction_mb": original_size_mb - compressed_size_mb,
            })
    
    return estimates


def validate_quantization_config(config: QuantizationConfig) -> List[str]:
    """
    Validate quantization configuration and return warnings/errors.
    
    Args:
        config: Quantization configuration
        
    Returns:
        List of validation messages
    """
    messages = []
    
    # Check bit widths
    if config.target_bits < 1 or config.target_bits > 32:
        messages.append(f"Warning: Unusual target_bits value: {config.target_bits}")
    
    if config.weight_bits < 1 or config.weight_bits > 32:
        messages.append(f"Warning: Unusual weight_bits value: {config.weight_bits}")
    
    if config.activation_bits < 4 or config.activation_bits > 32:
        messages.append(f"Warning: Unusual activation_bits value: {config.activation_bits}")
    
    # Check for extremely aggressive quantization
    if config.weight_bits <= 2:
        messages.append("Warning: Very aggressive weight quantization may cause significant accuracy loss")
    
    if config.activation_bits <= 4:
        messages.append("Warning: Very aggressive activation quantization may cause accuracy loss")
    
    # Check calibration settings
    if config.calibration_samples < 100:
        messages.append("Warning: Low calibration sample count may affect quantization quality")
    
    if config.calibration_samples > 10000:
        messages.append("Warning: High calibration sample count may slow down quantization")
    
    # Check MLX settings
    if config.use_mlx_quantization and not MLX_AVAILABLE:
        messages.append("Error: MLX quantization requested but MLX is not available")
    
    # Check group size
    if config.mlx_group_size < 1 or config.mlx_group_size > 1024:
        messages.append(f"Warning: Unusual MLX group size: {config.mlx_group_size}")
    
    return messages


def calculate_quantization_scale_and_zero_point(
    min_val: float,
    max_val: float,
    num_bits: int,
    symmetric: bool = False
) -> Tuple[float, int]:
    """
    Calculate quantization scale and zero point.
    
    Args:
        min_val: Minimum value in the tensor
        max_val: Maximum value in the tensor
        num_bits: Number of quantization bits
        symmetric: Use symmetric quantization
        
    Returns:
        Tuple of (scale, zero_point)
    """
    if num_bits <= 0 or num_bits > 32:
        raise ValueError(f"Invalid num_bits: {num_bits}")
    
    qmin = 0
    qmax = (2 ** num_bits) - 1
    
    if symmetric:
        # Symmetric quantization
        max_range = max(abs(min_val), abs(max_val))
        scale = (2 * max_range) / (qmax - qmin)
        zero_point = (qmax + qmin) // 2
    else:
        # Asymmetric quantization
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - round(min_val / scale)
        zero_point = max(qmin, min(qmax, zero_point))
    
    # Ensure scale is not zero
    if scale == 0:
        scale = 1.0
    
    return float(scale), int(zero_point)


def quantize_weights(
    weights: Any,
    num_bits: int = 8,
    symmetric: bool = True,
    per_channel: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Quantize weight tensor.
    
    Args:
        weights: Weight tensor to quantize
        num_bits: Number of quantization bits
        symmetric: Use symmetric quantization
        per_channel: Use per-channel quantization
        
    Returns:
        Tuple of (quantized_weights, quantization_params)
    """
    if not MLX_AVAILABLE:
        logger.warning("MLX not available, returning original weights")
        return weights, {}
    
    try:
        # Convert to numpy for processing
        if hasattr(weights, 'numpy'):
            w_np = weights.numpy()
        else:
            w_np = np.array(weights)
        
        if per_channel and len(w_np.shape) > 1:
            # Per-channel quantization
            quantized_weights = np.zeros_like(w_np)
            scales = []
            zero_points = []
            
            for i in range(w_np.shape[0]):
                channel = w_np[i]
                min_val = np.min(channel)
                max_val = np.max(channel)
                
                scale, zero_point = calculate_quantization_scale_and_zero_point(
                    min_val, max_val, num_bits, symmetric
                )
                
                # Quantize channel
                quantized_channel = np.round(channel / scale + zero_point)
                quantized_channel = np.clip(quantized_channel, 0, (2**num_bits) - 1)
                
                # Dequantize for storage
                quantized_weights[i] = (quantized_channel - zero_point) * scale
                
                scales.append(scale)
                zero_points.append(zero_point)
            
            quantization_params = {
                "scales": scales,
                "zero_points": zero_points,
                "per_channel": True,
                "num_bits": num_bits,
                "symmetric": symmetric,
            }
            
        else:
            # Per-tensor quantization
            min_val = np.min(w_np)
            max_val = np.max(w_np)
            
            scale, zero_point = calculate_quantization_scale_and_zero_point(
                min_val, max_val, num_bits, symmetric
            )
            
            # Quantize
            quantized = np.round(w_np / scale + zero_point)
            quantized = np.clip(quantized, 0, (2**num_bits) - 1)
            
            # Dequantize for storage
            quantized_weights = (quantized - zero_point) * scale
            
            quantization_params = {
                "scale": scale,
                "zero_point": zero_point,
                "per_channel": False,
                "num_bits": num_bits,
                "symmetric": symmetric,
            }
        
        # Convert back to original tensor type
        if hasattr(weights, 'numpy'):
            # Assume MLX tensor
            quantized_weights = mx.array(quantized_weights)
        
        return quantized_weights, quantization_params
        
    except Exception as e:
        logger.error(f"Failed to quantize weights: {e}")
        return weights, {}


def dequantize_weights(
    quantized_weights: Any,
    quantization_params: Dict[str, Any]
) -> Any:
    """
    Dequantize weight tensor.
    
    Args:
        quantized_weights: Quantized weight tensor
        quantization_params: Quantization parameters
        
    Returns:
        Dequantized weights
    """
    if not quantization_params:
        return quantized_weights
    
    try:
        # Convert to numpy for processing
        if hasattr(quantized_weights, 'numpy'):
            q_np = quantized_weights.numpy()
        else:
            q_np = np.array(quantized_weights)
        
        if quantization_params.get("per_channel", False):
            # Per-channel dequantization
            scales = quantization_params["scales"]
            zero_points = quantization_params["zero_points"]
            
            dequantized = np.zeros_like(q_np)
            for i in range(len(scales)):
                dequantized[i] = (q_np[i] - zero_points[i]) * scales[i]
        else:
            # Per-tensor dequantization
            scale = quantization_params["scale"]
            zero_point = quantization_params["zero_point"]
            dequantized = (q_np - zero_point) * scale
        
        # Convert back to original tensor type
        if hasattr(quantized_weights, 'numpy'):
            dequantized = mx.array(dequantized)
        
        return dequantized
        
    except Exception as e:
        logger.error(f"Failed to dequantize weights: {e}")
        return quantized_weights


def analyze_quantization_sensitivity(
    model: Any,
    calibration_data: Any,
    layer_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Analyze quantization sensitivity of different layers.
    
    Args:
        model: Model to analyze
        calibration_data: Calibration dataset
        layer_names: Specific layers to analyze
        
    Returns:
        Dictionary of layer sensitivity scores
    """
    logger.info("Analyzing quantization sensitivity")
    
    if not MLX_AVAILABLE:
        logger.warning("MLX not available, returning dummy sensitivity scores")
        return {"dummy_layer": 0.5}
    
    sensitivity_scores = {}
    
    try:
        # This is a simplified implementation
        # In practice, you would:
        # 1. Run model on calibration data to get baseline accuracy
        # 2. Quantize each layer individually and measure accuracy drop
        # 3. Rank layers by sensitivity
        
        if layer_names:
            for layer_name in layer_names:
                # Dummy sensitivity score
                sensitivity_scores[layer_name] = 0.5
        else:
            # Generate dummy scores for common layer types
            common_layers = ["embedding", "attention", "mlp", "output"]
            for layer in common_layers:
                sensitivity_scores[layer] = 0.3 + (hash(layer) % 100) / 200.0
        
        logger.info(f"Analyzed sensitivity for {len(sensitivity_scores)} layers")
        return sensitivity_scores
        
    except Exception as e:
        logger.error(f"Failed to analyze quantization sensitivity: {e}")
        return {}


def create_quantization_report(
    config: QuantizationConfig,
    quantization_stats: Dict[str, Any],
    performance_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive quantization report.
    
    Args:
        config: Quantization configuration
        quantization_stats: Quantization statistics
        performance_metrics: Optional performance metrics
        
    Returns:
        Comprehensive quantization report
    """
    report = {
        "configuration": config.to_dict(),
        "statistics": quantization_stats,
        "compression_analysis": estimate_compression_ratio(config),
        "validation_messages": validate_quantization_config(config),
    }
    
    if performance_metrics:
        report["performance"] = performance_metrics
    
    # Add summary
    compression_ratio = quantization_stats.get("actual_compression_ratio", 
                                               config.get_compression_ratio())
    
    report["summary"] = {
        "method": config.method.value,
        "target_bits": config.target_bits,
        "compression_ratio": compression_ratio,
        "estimated_speedup": math.sqrt(compression_ratio),  # Rough estimate
        "memory_savings_percent": ((compression_ratio - 1) / compression_ratio) * 100,
    }
    
    return report