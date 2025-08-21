"""
Calibration utilities for quantization.
"""

from typing import Any, List, Dict, Optional, Iterator
from pathlib import Path
import logging
import json

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    np = None

from .config import CalibrationMethod

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


class CalibrationDataLoader:
    """
    Data loader for calibration data during quantization.
    """
    
    def __init__(
        self,
        dataset_path: Path,
        max_samples: int = 512,
        sequence_length: int = 512,
    ):
        """
        Initialize calibration data loader.
        
        Args:
            dataset_path: Path to calibration dataset
            max_samples: Maximum number of calibration samples
            sequence_length: Maximum sequence length
        """
        self.dataset_path = Path(dataset_path)
        self.max_samples = max_samples
        self.sequence_length = sequence_length
        self.samples = []
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load calibration data from file."""
        logger.info(f"Loading calibration data from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            logger.warning(f"Calibration dataset not found: {self.dataset_path}")
            self._create_dummy_data()
            return
        
        try:
            if self.dataset_path.suffix == '.jsonl':
                self._load_jsonl()
            elif self.dataset_path.suffix == '.json':
                self._load_json()
            elif self.dataset_path.suffix == '.txt':
                self._load_text()
            else:
                logger.warning(f"Unsupported format: {self.dataset_path.suffix}")
                self._create_dummy_data()
                
        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")
            self._create_dummy_data()
    
    def _load_jsonl(self) -> None:
        """Load data from JSONL file."""
        with open(self.dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_samples:
                    break
                
                try:
                    data = json.loads(line.strip())
                    if 'text' in data:
                        self.samples.append(data['text'])
                    elif 'prompt' in data:
                        self.samples.append(data['prompt'])
                    elif isinstance(data, str):
                        self.samples.append(data)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(self.samples)} samples from JSONL")
    
    def _load_json(self) -> None:
        """Load data from JSON file."""
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data[:self.max_samples]:
                if isinstance(item, str):
                    self.samples.append(item)
                elif isinstance(item, dict) and 'text' in item:
                    self.samples.append(item['text'])
        elif isinstance(data, dict):
            if 'data' in data:
                for item in data['data'][:self.max_samples]:
                    if isinstance(item, str):
                        self.samples.append(item)
        
        logger.info(f"Loaded {len(self.samples)} samples from JSON")
    
    def _load_text(self) -> None:
        """Load data from text file."""
        with open(self.dataset_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines[:self.max_samples]:
            line = line.strip()
            if line:
                self.samples.append(line)
        
        logger.info(f"Loaded {len(self.samples)} samples from text file")
    
    def _create_dummy_data(self) -> None:
        """Create dummy calibration data."""
        logger.info("Creating dummy calibration data")
        
        dummy_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming various industries.",
            "Apple Silicon provides excellent performance for AI workloads.",
            "Quantization reduces model size while maintaining accuracy.",
            "MLX framework enables efficient computation on Apple hardware.",
        ]
        
        # Extend dummy data to reach desired sample count
        while len(self.samples) < min(self.max_samples, 100):
            for text in dummy_texts:
                if len(self.samples) >= self.max_samples:
                    break
                self.samples.append(text)
        
        logger.info(f"Created {len(self.samples)} dummy calibration samples")
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over calibration samples."""
        return iter(self.samples)
    
    def __len__(self) -> int:
        """Get number of calibration samples."""
        return len(self.samples)
    
    def get_batch(self, batch_size: int = 1) -> List[str]:
        """Get a batch of calibration samples."""
        import random
        return random.sample(self.samples, min(batch_size, len(self.samples)))


class CalibrationStrategy:
    """
    Different calibration strategies for quantization.
    """
    
    def __init__(self, method: CalibrationMethod):
        """Initialize calibration strategy."""
        self.method = method
        self.activation_stats = {}
    
    def collect_activations(
        self,
        model: Any,
        calibration_data: CalibrationDataLoader,
        tokenizer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Collect activation statistics from calibration data.
        
        Args:
            model: Model to analyze
            calibration_data: Calibration dataset
            tokenizer: Tokenizer for text processing
            
        Returns:
            Activation statistics
        """
        logger.info(f"Collecting activations using {self.method} method")
        
        if not MLX_AVAILABLE:
            logger.warning("MLX not available, returning dummy stats")
            return {"dummy": True}
        
        activation_stats = {}
        
        try:
            # Hook into model layers to collect statistics
            hooks = []
            layer_stats = {}
            
            def create_hook(layer_name: str):
                def hook_fn(module, input, output):
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = []
                    
                    # Convert to numpy for statistics calculation
                    if hasattr(output, 'numpy'):
                        output_np = output.numpy()
                    else:
                        output_np = np.array(output)
                    
                    layer_stats[layer_name].append(output_np)
                
                return hook_fn
            
            # Register hooks (simplified implementation)
            # In practice, you would iterate through model layers
            
            # Process calibration samples
            for i, sample in enumerate(calibration_data):
                if i >= 100:  # Limit calibration samples
                    break
                
                # In practice, you would tokenize and run forward pass
                # This is a simplified version
                pass
            
            # Calculate statistics based on method
            for layer_name, activations in layer_stats.items():
                if self.method == CalibrationMethod.MINMAX:
                    stats = self._calculate_minmax_stats(activations)
                elif self.method == CalibrationMethod.ENTROPY:
                    stats = self._calculate_entropy_stats(activations)
                elif self.method == CalibrationMethod.PERCENTILE:
                    stats = self._calculate_percentile_stats(activations)
                else:
                    stats = self._calculate_minmax_stats(activations)
                
                activation_stats[layer_name] = stats
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            self.activation_stats = activation_stats
            logger.info(f"Collected statistics for {len(activation_stats)} layers")
            
            return activation_stats
            
        except Exception as e:
            logger.error(f"Failed to collect activations: {e}")
            return {}
    
    def _calculate_minmax_stats(self, activations: List[Any]) -> Dict[str, float]:
        """Calculate min-max statistics."""
        if not activations:
            return {}
        
        try:
            all_activations = np.concatenate([act.flatten() for act in activations])
            return {
                "min": float(np.min(all_activations)),
                "max": float(np.max(all_activations)),
                "mean": float(np.mean(all_activations)),
                "std": float(np.std(all_activations)),
            }
        except Exception as e:
            logger.warning(f"Failed to calculate minmax stats: {e}")
            return {}
    
    def _calculate_entropy_stats(self, activations: List[Any]) -> Dict[str, float]:
        """Calculate entropy-based statistics."""
        if not activations:
            return {}
        
        try:
            all_activations = np.concatenate([act.flatten() for act in activations])
            
            # Calculate histogram for entropy
            hist, bin_edges = np.histogram(all_activations, bins=256)
            hist = hist + 1e-12  # Avoid log(0)
            hist = hist / np.sum(hist)
            
            entropy = -np.sum(hist * np.log2(hist))
            
            return {
                "entropy": float(entropy),
                "min": float(np.min(all_activations)),
                "max": float(np.max(all_activations)),
                "mean": float(np.mean(all_activations)),
            }
        except Exception as e:
            logger.warning(f"Failed to calculate entropy stats: {e}")
            return {}
    
    def _calculate_percentile_stats(self, activations: List[Any]) -> Dict[str, float]:
        """Calculate percentile-based statistics."""
        if not activations:
            return {}
        
        try:
            all_activations = np.concatenate([act.flatten() for act in activations])
            
            return {
                "p1": float(np.percentile(all_activations, 1)),
                "p5": float(np.percentile(all_activations, 5)),
                "p95": float(np.percentile(all_activations, 95)),
                "p99": float(np.percentile(all_activations, 99)),
                "median": float(np.percentile(all_activations, 50)),
                "mean": float(np.mean(all_activations)),
            }
        except Exception as e:
            logger.warning(f"Failed to calculate percentile stats: {e}")
            return {}
    
    def get_quantization_range(self, layer_name: str) -> tuple:
        """
        Get quantization range for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Tuple of (min_val, max_val)
        """
        if layer_name not in self.activation_stats:
            return (-1.0, 1.0)  # Default range
        
        stats = self.activation_stats[layer_name]
        
        if self.method == CalibrationMethod.MINMAX:
            return (stats.get("min", -1.0), stats.get("max", 1.0))
        elif self.method == CalibrationMethod.PERCENTILE:
            return (stats.get("p1", -1.0), stats.get("p99", 1.0))
        else:
            # Default to mean ± 3*std
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            return (mean - 3*std, mean + 3*std)