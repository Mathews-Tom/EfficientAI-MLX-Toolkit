"""
Base LLM provider implementation for DSPy Integration Framework.
"""

# Standard library imports
import logging
from abc import ABC, abstractmethod
from typing import Any

# Local imports
from ..exceptions import DSPyIntegrationError
from ..interfaces import LLMProviderInterface
from ..types import DSPyConfig, HardwareInfo

logger = logging.getLogger(__name__)


class BaseLLMProvider(LLMProviderInterface, ABC):
    """Base implementation for LLM providers."""

    def __init__(self, config: DSPyConfig):
        """Initialize the base LLM provider."""
        self.config = config
        self.hardware_info = self.detect_hardware()
        self._initialized = False

    @abstractmethod
    def completion(self, *args, **kwargs) -> Any:
        """Generate completion using the LLM."""

    @abstractmethod
    def detect_hardware(self) -> HardwareInfo:
        """Detect available hardware capabilities."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and functional."""

    def _validate_config(self) -> None:
        """Validate the provider configuration."""
        if not self.config.model_name:
            raise DSPyIntegrationError("Model name is required")

        if self.config.optimization_level < 0 or self.config.optimization_level > 3:
            raise DSPyIntegrationError("Optimization level must be between 0 and 3")

    def _setup_logging(self) -> None:
        """Setup provider-specific logging."""
        if self.config.enable_tracing:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
        else:
            logging.getLogger(__name__).setLevel(logging.INFO)

    def initialize(self) -> None:
        """Initialize the provider."""
        if self._initialized:
            return

        self._validate_config()
        self._setup_logging()
        self._initialized = True

        logger.info("Initialized %s with model %s", self.__class__.__name__, self.config.model_name)

    def get_provider_info(self) -> dict[str, str | int | float | bool]:
        """Get information about the provider."""
        return {
            "provider_type": self.__class__.__name__,
            "model_name": self.config.model_name,
            "hardware_info": self.hardware_info.__dict__,
            "optimization_level": self.config.optimization_level,
            "initialized": self._initialized,
        }
