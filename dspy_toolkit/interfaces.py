"""
Core interfaces and abstract base classes for DSPy Integration Framework.
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Any

# Third-party imports
import dspy

# Local imports
from .types import DSPyConfig, HardwareInfo, OptimizationResult


class LLMProviderInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def __init__(self, config: DSPyConfig):
        """Initialize the LLM provider with configuration."""

    @abstractmethod
    def completion(self, *args, **kwargs) -> Any:
        """Generate completion using the LLM."""

    @abstractmethod
    def detect_hardware(self) -> HardwareInfo:
        """Detect available hardware capabilities."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and functional."""


class SignatureRegistryInterface(ABC):
    """Abstract interface for signature registry."""

    @abstractmethod
    def register_project(self, project_name: str, signatures: dict[str, type]) -> None:
        """Register signatures for a project."""

    @abstractmethod
    def get_project_signatures(self, project_name: str) -> dict[str, type]:
        """Get signatures for a specific project."""

    @abstractmethod
    def get_all_signatures(self) -> dict[str, dict[str, type]]:
        """Get all registered signatures."""

    @abstractmethod
    def validate_signature(self, signature: type) -> bool:
        """Validate a DSPy signature."""


class OptimizerEngineInterface(ABC):
    """Abstract interface for optimizer engine."""

    @abstractmethod
    def select_optimizer(self, task_type: str, dataset_size: int, complexity: str) -> str:
        """Select the best optimizer for given task characteristics."""

    @abstractmethod
    def optimize(
        self,
        module: dspy.Module,
        dataset: list[dict],
        metrics: list[str],
        task_type: str = "general",
    ) -> dspy.Module:
        """Optimize a DSPy module."""

    @abstractmethod
    def get_optimization_history(self) -> list[OptimizationResult]:
        """Get history of optimization results."""


class ModuleManagerInterface(ABC):
    """Abstract interface for module manager."""

    @abstractmethod
    def register_module(
        self,
        name: str,
        module: dspy.Module,
        metadata: dict[str, str | int | float | bool],
    ) -> None:
        """Register a DSPy module."""

    @abstractmethod
    def get_module(self, name: str) -> dspy.Module | None:
        """Get a registered module by name."""

    @abstractmethod
    def list_modules(self) -> list[str]:
        """List all registered module names."""

    @abstractmethod
    def save_module(self, name: str, path: str) -> None:
        """Save a module to disk."""

    @abstractmethod
    def load_module(self, name: str, path: str) -> dspy.Module:
        """Load a module from disk."""


class DSPyFrameworkInterface(ABC):
    """Abstract interface for the main DSPy framework."""

    @abstractmethod
    def __init__(self, config: DSPyConfig):
        """Initialize the framework with configuration."""

    @abstractmethod
    def register_project_signatures(self, project_name: str, signatures: dict[str, type]) -> None:
        """Register project-specific signatures."""

    @abstractmethod
    def optimize_module(
        self, module: dspy.Module, dataset: list[dict], metrics: list[str]
    ) -> dspy.Module:
        """Optimize a DSPy module using appropriate optimizer."""

    @abstractmethod
    def get_project_module(self, project_name: str, module_name: str) -> dspy.Module | None:
        """Get a project-specific module."""

    @abstractmethod
    def setup_llm_provider(self) -> None:
        """Setup and configure the LLM provider."""
