"""
Core DSPy Framework manager for EfficientAI-MLX-Toolkit.
"""

# Standard library imports
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# Third-party imports
import dspy

# Local imports
from .exceptions import DSPyIntegrationError, MLXProviderError, handle_dspy_errors
from .interfaces import DSPyFrameworkInterface
from .manager import ModuleManager
from .optimizer import OptimizerEngine
from .providers import BaseLLMProvider, MLXLLMProvider
from .providers.mlx_provider import setup_mlx_provider_for_dspy
from .registry import SignatureRegistry
from .types import DSPyConfig, HardwareInfo

logger = logging.getLogger(__name__)


class DSPyFramework(DSPyFrameworkInterface):
    """Central DSPy framework manager for intelligent automation."""

    def __init__(self, config: DSPyConfig):
        """Initialize the DSPy framework."""
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize core components
        self.signature_registry = SignatureRegistry(
            self.config.cache_dir / "signatures"
        )
        self.module_manager = ModuleManager(self.config.cache_dir / "modules")
        self.optimizer_engine = OptimizerEngine()

        # LLM provider will be set up later
        self.llm_provider: BaseLLMProvider | None = None
        self.hardware_info: HardwareInfo | None = None

        # Setup logging
        self._setup_logging()

        # Initialize the framework
        self._initialize()

    def _setup_logging(self) -> None:
        """Setup framework logging."""
        log_level = logging.DEBUG if self.config.enable_tracing else logging.INFO
        logging.getLogger(__name__).setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Setup file handler if cache directory exists
        try:
            log_file = self.config.cache_dir / "dspy_framework.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logging.getLogger(__name__).addHandler(file_handler)
        except Exception as e:
            logger.warning("Failed to setup file logging: %s", e)

    def _initialize(self) -> None:
        """Initialize the framework components."""
        try:
            # Setup LLM provider
            self.setup_llm_provider()

            # Configure DSPy with the provider
            if self.llm_provider and self.llm_provider.is_available():
                dspy.configure(lm=dspy.LM(model=self.config.model_name))
                logger.info("DSPy configured with LLM provider")
            else:
                logger.warning("LLM provider not available, DSPy not configured")

            logger.info("DSPy Framework initialized successfully")

        except Exception as e:
            logger.error("Framework initialization failed: %s", e)
            raise DSPyIntegrationError("Framework initialization failed") from e

    @handle_dspy_errors()
    def setup_llm_provider(self) -> None:
        """Setup and configure the LLM provider."""
        try:
            if self.config.model_provider == "mlx":
                self.llm_provider = MLXLLMProvider(self.config)
                self.hardware_info = self.llm_provider.hardware_info

                # Setup MLX provider for DSPy integration
                setup_mlx_provider_for_dspy(self.config)

                logger.info("MLX LLM provider setup completed")

            else:
                # For other providers (OpenAI, Anthropic, etc.)
                logger.info("Setting up %s provider", self.config.model_provider)
                # This would be implemented for other providers
                raise DSPyIntegrationError(
                    f"Provider {self.config.model_provider} not yet implemented"
                )

        except Exception as e:
            logger.error("LLM provider setup failed: %s", e)
            raise MLXProviderError("Provider setup failed") from e

    def register_project_signatures(
        self, project_name: str, signatures: dict[str, type]
    ) -> None:
        """Register project-specific signatures."""
        try:
            self.signature_registry.register_project(project_name, signatures)
            logger.info(
                "Registered %d signatures for project %s",
                len(signatures),
                project_name,
            )
        except Exception as e:
            logger.error(
                "Failed to register signatures for project %s: %s", project_name, e
            )
            raise DSPyIntegrationError("Signature registration failed") from e

    @handle_dspy_errors()
    def optimize_module(
        self, module: dspy.Module, dataset: list[dict], metrics: list[str]
    ) -> dspy.Module:
        """Optimize a DSPy module using appropriate optimizer."""
        try:
            if not self.llm_provider or not self.llm_provider.is_available():
                logger.warning(
                    "LLM provider not available, returning unoptimized module"
                )
                return module

            # Use optimizer engine to optimize the module
            optimized_module = self.optimizer_engine.optimize(
                module=module, dataset=dataset, metrics=metrics, task_type="general"
            )

            # Register the optimized module
            module_name = f"optimized_{type(module).__name__}_{len(dataset)}"
            metadata = {
                "original_module": type(module).__name__,
                "dataset_size": len(dataset),
                "metrics": metrics,
                "optimization_method": "auto",
            }

            self.module_manager.register_module(module_name, optimized_module, metadata)

            logger.info("Module optimized and registered as %s", module_name)
            return optimized_module

        except Exception as e:
            logger.error("Module optimization failed: %s", e)
            raise DSPyIntegrationError("Module optimization failed") from e

    def get_project_module(
        self, project_name: str, module_name: str
    ) -> dspy.Module | None:
        """Get a project-specific module."""
        full_module_name = f"{project_name}_{module_name}"
        return self.module_manager.get_module(full_module_name)

    def create_project_module(
        self,
        project_name: str,
        module_name: str,
        signature_name: str,
        module_type: str = "ChainOfThought",
    ) -> dspy.Module:
        """Create a new DSPy module for a project."""
        try:
            # Get the signature for the project
            project_signatures = self.signature_registry.get_project_signatures(
                project_name
            )
            if signature_name not in project_signatures:
                raise DSPyIntegrationError(
                    f"Signature {signature_name} not found for project {project_name}"
                )

            signature = project_signatures[signature_name]

            # Create the module based on type
            if module_type == "ChainOfThought":
                module = dspy.ChainOfThought(signature)
            elif module_type == "ReAct":
                module = dspy.ReAct(signature)
            elif module_type == "ProgramOfThought":
                module = dspy.ProgramOfThought(signature)
            else:
                raise DSPyIntegrationError(f"Unknown module type: {module_type}")

            # Register the module
            full_module_name = f"{project_name}_{module_name}"
            metadata = {
                "project": project_name,
                "signature": signature_name,
                "module_type": module_type,
                "created_by": "framework",
            }

            self.module_manager.register_module(full_module_name, module, metadata)

            logger.info(
                "Created module %s} with signature %s", full_module_name, signature_name
            )
            return module

        except Exception as e:
            logger.error(
                "Failed to create module %s for project %s: %s",
                module_name,
                project_name,
                e,
            )
            raise DSPyIntegrationError("Module creation failed") from e

    def get_framework_stats(self) -> dict[str, str | int | float]:
        """Get comprehensive framework statistics."""
        try:
            stats = {
                "framework": {
                    "config": {
                        "model_provider": self.config.model_provider,
                        "model_name": self.config.model_name,
                        "optimization_level": self.config.optimization_level,
                        "cache_dir": str(self.config.cache_dir),
                    },
                    "llm_provider_available": self.llm_provider is not None
                    and self.llm_provider.is_available(),
                    "hardware_info": (
                        self.hardware_info.__dict__ if self.hardware_info else None
                    ),
                },
                "signatures": self.signature_registry.get_registry_stats(),
                "modules": self.module_manager.get_manager_stats(),
                "optimizer": self.optimizer_engine.get_optimizer_stats(),
            }

            # Add provider-specific stats
            if self.llm_provider:
                stats["provider"] = self.llm_provider.get_provider_info()

                # Add MLX-specific stats if available
                if hasattr(self.llm_provider, "get_model_info"):
                    stats["model"] = self.llm_provider.get_model_info()

            return stats

        except Exception as e:
            logger.error("Failed to get framework stats: %s", e)
            return {"error": str(e)}

    def benchmark_framework(self) -> dict[str, str | int | float]:
        """Benchmark the framework performance."""
        try:
            results = {}

            # Benchmark LLM provider if available
            if self.llm_provider and hasattr(
                self.llm_provider, "benchmark_performance"
            ):
                results["llm_provider"] = self.llm_provider.benchmark_performance()

            # Benchmark signature registry
            start_time = time.time()
            self.signature_registry.get_registry_stats()
            results["signature_registry_time"] = time.time() - start_time

            # Benchmark module manager
            start_time = time.time()
            self.module_manager.get_manager_stats()
            results["module_manager_time"] = time.time() - start_time

            return results

        except Exception as e:
            logger.error("Framework benchmarking failed: %s", e)
            return {"error": str(e)}

    def export_framework_state(self, export_dir: Path) -> None:
        """Export the entire framework state."""
        try:
            export_dir.mkdir(parents=True, exist_ok=True)

            # Export signatures
            signatures_dir = export_dir / "signatures"
            self.signature_registry.export_registry(signatures_dir / "registry.json")

            # Export modules
            modules_dir = export_dir / "modules"
            self.module_manager.export_modules(modules_dir)

            # Export framework configuration and stats
            framework_info = {
                "config": {
                    "model_provider": self.config.model_provider,
                    "model_name": self.config.model_name,
                    "optimization_level": self.config.optimization_level,
                },
                "stats": self.get_framework_stats(),
                "exported_at": str(datetime.now()),
            }

            with open(export_dir / "framework_info.json", "w") as f:
                json.dump(framework_info, f, indent=2)

            logger.info("Framework state exported to %s", export_dir)

        except Exception as e:
            logger.error("Failed to export framework state: %s", e)
            raise DSPyIntegrationError("Framework export failed") from e

    def clear_all_caches(self) -> None:
        """Clear all framework caches."""
        try:
            self.signature_registry.clear_cache()
            self.module_manager.clear_cache()

            # Clear main cache directory
            for cache_file in self.config.cache_dir.glob("*.log"):
                cache_file.unlink()

            logger.info("All framework caches cleared")

        except Exception as e:
            logger.error("Failed to clear caches: %s", e)

    def health_check(self) -> dict[str, str | bool | int | float]:
        """Perform a comprehensive health check of the framework."""
        health = {
            "overall_status": "healthy",
            "components": {},
            "issues": [],
        }

        try:
            # Check LLM provider
            if self.llm_provider and self.llm_provider.is_available():
                health["components"]["llm_provider"] = "healthy"
            else:
                health["components"]["llm_provider"] = "unhealthy"
                health["issues"].append("LLM provider not available")

            # Check signature registry
            try:
                self.signature_registry.get_registry_stats()
                health["components"]["signature_registry"] = "healthy"
            except Exception as e:
                health["components"]["signature_registry"] = "unhealthy"
                health["issues"].append(f"Signature registry error: {e}")

            # Check module manager
            try:
                self.module_manager.get_manager_stats()
                health["components"]["module_manager"] = "healthy"
            except Exception as e:
                health["components"]["module_manager"] = "unhealthy"
                health["issues"].append(f"Module manager error: {e}")

            # Check optimizer engine
            try:
                self.optimizer_engine.get_optimizer_stats()
                health["components"]["optimizer_engine"] = "healthy"
            except Exception as e:
                health["components"]["optimizer_engine"] = "unhealthy"
                health["issues"].append(f"Optimizer engine error: {e}")

            # Determine overall status
            if health["issues"]:
                health["overall_status"] = (
                    "degraded" if len(health["issues"]) < 3 else "unhealthy"
                )

            return health

        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "components": {},
                "issues": [f"Health check failed: {e}"],
            }
