"""
Unified memory management for Apple Silicon optimization.
"""

# Standard library imports
import logging
import time
from dataclasses import dataclass
from datetime import datetime

# Optional third-party imports
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

# Local imports
from ..exceptions import DSPyIntegrationError
from ..types import HardwareInfo

logger = logging.getLogger(__name__)


@dataclass
class MemoryProfile:
    """Memory profiling information for Apple Silicon."""

    peak_memory_mb: float
    average_memory_mb: float
    memory_efficiency: float
    recommended_batch_size: int
    memory_pressure: str  # "low", "medium", "high"
    optimization_suggestions: list[str]
    timestamp: str


class UnifiedMemoryManager:
    """Manages unified memory architecture on Apple Silicon."""

    def __init__(self, hardware_info: HardwareInfo):
        """Initialize unified memory manager."""
        self.hardware_info = hardware_info
        self.memory_profiles: list[MemoryProfile] = []
        self.memory_limit_gb = hardware_info.total_memory * 0.8  # Use 80% of total memory

        if MLX_AVAILABLE:
            self._configure_mlx_memory()
        else:
            logger.warning("MLX not available, memory management will use fallback strategies")

    def _configure_mlx_memory(self) -> None:
        """Configure MLX memory settings for optimal performance."""
        try:
            # Set memory limit based on available memory
            memory_limit_bytes = int(self.memory_limit_gb * 1024**3)
            mx.metal.set_memory_limit(memory_limit_bytes)

            # Set cache limit to 25% of memory limit
            cache_limit_bytes = memory_limit_bytes // 4
            mx.metal.set_cache_limit(cache_limit_bytes)

            logger.info(
                "Configured MLX memory - Limit: %.1fGB, Cache: %.1fGB",
                self.memory_limit_gb,
                cache_limit_bytes / (1024**3),
            )

        except Exception as e:
            logger.error("Failed to configure MLX memory: %s", e)
            raise DSPyIntegrationError("Memory configuration failed") from e

    def optimize_for_model_size(self, model_size_gb: float) -> dict[str, int | float]:
        """Optimize memory settings for specific model size."""
        try:
            # Calculate optimal settings based on model size
            available_memory = self.memory_limit_gb - model_size_gb

            if available_memory < 1.0:
                # Very tight memory
                batch_size = 1
                cache_size_gb = 0.5
                memory_pressure = "high"
                suggestions = [
                    "Consider using gradient checkpointing",
                    "Reduce model precision to float16",
                    "Use smaller batch sizes",
                    "Enable memory-efficient attention",
                ]
            elif available_memory < 4.0:
                # Moderate memory
                batch_size = 4
                cache_size_gb = 1.0
                memory_pressure = "medium"
                suggestions = [
                    "Monitor memory usage during training",
                    "Consider mixed precision training",
                    "Use gradient accumulation for larger effective batch sizes",
                ]
            else:
                # Plenty of memory
                batch_size = min(16, int(available_memory))
                cache_size_gb = min(2.0, available_memory * 0.25)
                memory_pressure = "low"

            # Apply MLX settings
            if MLX_AVAILABLE:
                cache_limit_bytes = int(cache_size_gb * 1024**3)
                mx.metal.set_cache_limit(cache_limit_bytes)

            optimization_config = {
                "recommended_batch_size": batch_size,
                "cache_size_gb": cache_size_gb,
                "memory_pressure": memory_pressure,
                "available_memory_gb": available_memory,
                "model_size_gb": model_size_gb,
            }

            logger.info(
                "Memory optimized for %.1fGB model - Batch size: %d, Cache: %.1fGB",
                model_size_gb,
                batch_size,
                cache_size_gb,
            )

            return optimization_config

        except Exception as e:
            logger.error("Memory optimization failed: %s", e)
            raise DSPyIntegrationError("Memory optimization failed") from e

    def profile_memory_usage(
        self,
        module,  # Any callable module
        test_inputs: list[dict[str, str | int | float]],
        duration_seconds: int = 60,
    ) -> MemoryProfile:
        """Profile memory usage over time."""
        try:
            memory_samples = []
            start_time = time.time()

            logger.info("Starting memory profiling for %d seconds", duration_seconds)

            # Run continuous inference and monitor memory
            while time.time() - start_time < duration_seconds:
                try:
                    # Random input selection
                    import random

                    test_input = random.choice(test_inputs)

                    # Run inference and record memory
                    _ = module(**test_input)
                    memory_after = self._get_current_memory()
                    memory_samples.append(memory_after)

                    # Small delay to avoid overwhelming the system
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning("Memory profiling sample failed: %s", e)
                    continue

            # Calculate statistics
            if not memory_samples:
                raise DSPyIntegrationError("No memory samples collected")

            peak_memory = max(memory_samples)
            avg_memory = sum(memory_samples) / len(memory_samples)
            memory_efficiency = len(memory_samples) / peak_memory  # samples per MB

            # Determine memory pressure
            memory_usage_ratio = peak_memory / (self.memory_limit_gb * 1024)  # MB to GB conversion
            if memory_usage_ratio > 0.9:
                pressure = "high"
                batch_size = 1
                suggestions = [
                    "Memory usage is very high",
                    "Consider reducing model size or batch size",
                    "Enable gradient checkpointing",
                    "Use memory-efficient optimizers",
                ]
            elif memory_usage_ratio > 0.7:
                pressure = "medium"
                batch_size = 4
                suggestions = [
                    "Memory usage is moderate",
                    "Monitor for memory spikes during training",
                    "Consider mixed precision training",
                ]
            else:
                pressure = "low"
                batch_size = 8
                suggestions = [
                    "Memory usage is optimal",
                    "Can increase batch size for better throughput",
                ]

            profile = MemoryProfile(
                peak_memory_mb=peak_memory,
                average_memory_mb=avg_memory,
                memory_efficiency=memory_efficiency,
                recommended_batch_size=batch_size,
                memory_pressure=pressure,
                optimization_suggestions=suggestions,
                timestamp=datetime.now().isoformat(),
            )

            self.memory_profiles.append(profile)

            logger.info(
                "Memory profiling completed - Peak: %.2fMB, Avg: %.2fMB, Pressure: %s",
                peak_memory,
                avg_memory,
                pressure,
            )

            return profile

        except Exception as e:
            logger.error("Memory profiling failed: %s", e)
            raise DSPyIntegrationError("Memory profiling failed") from e

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        if MLX_AVAILABLE:
            return mx.metal.get_active_memory() / (1024**2)

        # Fallback to system memory
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024**2)
        except ImportError:
            return 0.0

    def clear_memory_cache(self) -> None:
        """Clear memory caches to free up memory."""
        try:
            if MLX_AVAILABLE:
                mx.metal.clear_cache()
                logger.info("Cleared MLX memory cache")
            else:
                # Fallback cache clearing
                import gc

                gc.collect()
                logger.info("Performed garbage collection")

        except Exception as e:
            logger.error("Failed to clear memory cache: %s", e)

    def get_memory_recommendations(self) -> dict[str, str | int | float | list[str]]:
        """Get memory optimization recommendations."""
        try:
            current_memory = self._get_current_memory()
            memory_usage_ratio = current_memory / (self.memory_limit_gb * 1024)

            recommendations = {
                "current_memory_mb": current_memory,
                "memory_limit_gb": self.memory_limit_gb,
                "usage_ratio": memory_usage_ratio,
                "recommendations": [],
            }

            if memory_usage_ratio > 0.9:
                recommendations["recommendations"].extend(
                    [
                        "Critical: Memory usage is very high",
                        "Reduce batch size immediately",
                        "Clear memory caches",
                        "Consider model quantization",
                    ]
                )
            elif memory_usage_ratio > 0.7:
                recommendations["recommendations"].extend(
                    [
                        "Warning: Memory usage is high",
                        "Monitor memory during training",
                        "Consider gradient checkpointing",
                        "Use mixed precision training",
                    ]
                )
            else:
                recommendations["recommendations"].extend(
                    [
                        "Memory usage is optimal",
                        "Can increase batch size for better performance",
                        "Current settings are well-tuned",
                    ]
                )

            return recommendations

        except Exception as e:
            logger.error("Failed to get memory recommendations: %s", e)
            return {"error": str(e), "recommendations": []}
