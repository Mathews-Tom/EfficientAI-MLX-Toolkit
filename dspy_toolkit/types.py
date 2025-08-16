"""
Type definitions and data models for DSPy Integration Framework.
"""

# Standard library imports
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Third-party imports
import dspy


class OptimizerType(Enum):
    """Available DSPy optimizer types."""

    BOOTSTRAP = "bootstrap"
    MIPRO = "mipro"
    GEPA = "gepa"
    AUTO = "auto"


@dataclass
class DSPyConfig:
    """Configuration for DSPy integration."""

    model_provider: str = "mlx"
    model_name: str = "mlx/mlx-7b"
    optimization_level: int = 2
    cache_dir: Path = Path(".dspy_cache")
    enable_tracing: bool = True
    max_retries: int = 3
    api_key: str | None = None
    base_url: str | None = None


@dataclass
class AppleSiliconConfig:
    """Apple Silicon specific configuration."""

    chip_type: str  # "m1", "m2", "m3"
    total_memory: int  # in GB
    metal_available: bool
    mps_available: bool
    unified_memory: bool
    optimization_level: int


@dataclass
class LLMProviderConfig:
    """LLM provider configuration."""

    provider_type: str  # "mlx", "openai", "anthropic"
    model_name: str
    api_key: str | None
    base_url: str | None
    hardware_config: AppleSiliconConfig | None


@dataclass
class ProjectSignatureConfig:
    """Configuration for project-specific signatures."""

    project_name: str
    signature_classes: dict[str, type]
    default_optimizer: OptimizerType
    optimization_metrics: list[str]


@dataclass
class OptimizationResult:
    """Results from DSPy optimization."""

    optimizer_used: str
    original_performance: dict[str, float]
    optimized_performance: dict[str, float]
    optimization_time: float
    num_examples_used: int
    metadata: dict[str, str | int | float | bool]


@dataclass
class ModuleRegistry:
    """Registry of optimized DSPy modules."""

    modules: dict[str, dspy.Module]
    optimization_results: dict[str, OptimizationResult]
    creation_timestamps: dict[str, str]
    version_info: dict[str, str]


@dataclass
class HardwareInfo:
    """Hardware information for optimization."""

    device_type: str  # "m1", "m2", "cpu"
    total_memory: int  # in GB
    available_memory: int  # in GB
    metal_available: bool
    mps_available: bool
    optimization_level: int


@dataclass
class MemoryProfile:
    """Memory profiling information."""

    peak_memory: float
    average_memory: float
    memory_efficiency: float
    batch_size_recommendation: int
