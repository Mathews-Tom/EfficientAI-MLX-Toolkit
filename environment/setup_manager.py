"""
Environment setup and management for Apple Silicon optimization.

This module provides automated environment setup, dependency management,
and Apple Silicon hardware detection and configuration.
"""

import logging
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


class EnvironmentError(Exception):
    """Raised when environment setup operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        details: dict[str, str | int | float] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.details = dict(details or {})


def detect_apple_silicon() -> dict[str, str | bool]:
    """
    Detect Apple Silicon hardware and capabilities.

    Returns:
        Dictionary containing hardware detection results

    Example:
        >>> hardware_info = detect_apple_silicon()
        >>> if hardware_info["is_apple_silicon"]:
        ...     print("Running on Apple Silicon")
    """
    try:
        system = platform.system()
        machine = platform.machine()
        processor = platform.processor()

        # Check for Apple Silicon
        is_apple_silicon = system == "Darwin" and machine == "arm64" and "arm" in processor.lower()

        # Check for MLX availability
        mlx_available = False
        try:
            import mlx.core as mx

            mlx_available = True
            logger.info("MLX framework detected and available")
        except ImportError:
            logger.info("MLX framework not available")

        # Check for MPS availability
        mps_available = False
        try:
            import torch

            mps_available = torch.backends.mps.is_available()
            logger.info("MPS backend available: %s", mps_available)
        except ImportError:
            logger.info("PyTorch not available for MPS detection")

        hardware_info = {
            "system": system,
            "machine": machine,
            "processor": processor,
            "is_apple_silicon": is_apple_silicon,
            "mlx_available": mlx_available,
            "mps_available": mps_available,
            "python_version": platform.python_version(),
        }

        logger.info("Hardware detection completed: Apple Silicon=%s", is_apple_silicon)
        return hardware_info

    except Exception as e:
        logger.error("Hardware detection failed: %s", e)
        return {
            "system": "unknown",
            "machine": "unknown",
            "processor": "unknown",
            "is_apple_silicon": False,
            "mlx_available": False,
            "mps_available": False,
            "python_version": platform.python_version(),
        }


def setup_mlx_optimization() -> bool:
    """
    Set up MLX optimization for Apple Silicon.

    Returns:
        True if MLX optimization was successfully configured

    Raises:
        EnvironmentError: If MLX setup fails on Apple Silicon hardware
    """
    hardware_info = detect_apple_silicon()

    if not hardware_info["is_apple_silicon"]:
        logger.info("Not running on Apple Silicon, skipping MLX optimization")
        return False

    if not hardware_info["mlx_available"]:
        raise EnvironmentError(
            "MLX not available on Apple Silicon hardware. Install with: uv add mlx",
            operation="mlx_setup",
            details=hardware_info,
        )

    try:
        import mlx.core as mx

        # Configure MLX memory settings for optimal performance
        # Set memory limit to 80% of available memory (conservative)
        try:
            # Get available memory (this is a simplified approach)
            memory_limit = 16 * 1024**3  # 16GB default, adjust based on system
            mx.metal.set_memory_limit(memory_limit)
            logger.info("MLX memory limit set to %d GB", memory_limit // (1024**3))
        except Exception as e:
            logger.warning("Failed to set MLX memory limit: %s", e)

        # Test MLX functionality
        test_array = mx.array([1.0, 2.0, 3.0])
        result = mx.sum(test_array)

        logger.info("MLX optimization configured successfully")
        return True

    except Exception as e:
        raise EnvironmentError(
            f"Failed to configure MLX optimization: {e}",
            operation="mlx_setup",
            details=hardware_info,
        ) from e


class EnvironmentSetup:
    """
    Manages environment setup and configuration for the EfficientAI-MLX-Toolkit.

    This class provides automated environment setup, dependency validation,
    and Apple Silicon optimization configuration.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        """
        Initialize environment setup manager.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.hardware_info = detect_apple_silicon()

        logger.info("Environment setup initialized for project: %s", self.project_root)

    def validate_python_version(self, min_version: str = "3.11") -> bool:
        """
        Validate Python version meets requirements.

        Args:
            min_version: Minimum required Python version

        Returns:
            True if Python version is sufficient

        Raises:
            EnvironmentError: If Python version is insufficient
        """
        current_version = platform.python_version()

        # Simple version comparison (works for major.minor format)
        current_parts = [int(x) for x in current_version.split(".")]
        min_parts = [int(x) for x in min_version.split(".")]

        if current_parts < min_parts:
            raise EnvironmentError(
                f"Python {min_version}+ required, found {current_version}",
                operation="python_validation",
                details={"current": current_version, "required": min_version},
            )

        logger.info("Python version validation passed: %s", current_version)
        return True

    def check_uv_installation(self) -> bool:
        """
        Check if uv package manager is installed.

        Returns:
            True if uv is available

        Raises:
            EnvironmentError: If uv is not installed
        """
        try:
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)

            logger.info("UV package manager found: %s", result.stdout.strip())
            return True

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise EnvironmentError(
                "UV package manager not found. Install with: pip install uv",
                operation="uv_check",
                details={"error": str(e)},
            ) from e

    def install_dependencies(self, extra_groups: Sequence[str] | None = None) -> bool:
        """
        Install project dependencies using uv.

        Args:
            extra_groups: Optional dependency groups to install

        Returns:
            True if installation succeeded

        Raises:
            EnvironmentError: If dependency installation fails
        """
        try:
            # Check if pyproject.toml exists
            pyproject_path = self.project_root / "pyproject.toml"
            if not pyproject_path.exists():
                raise EnvironmentError(
                    f"pyproject.toml not found in {self.project_root}",
                    operation="dependency_install",
                )

            # Build uv sync command
            cmd = ["uv", "sync"]

            if extra_groups:
                for group in extra_groups:
                    cmd.extend(["--extra", group])

            logger.info("Installing dependencies with command: %s", " ".join(cmd))

            # Run uv sync
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True, check=True
            )

            logger.info("Dependencies installed successfully")
            logger.debug("UV sync output: %s", result.stdout)

            return True

        except subprocess.CalledProcessError as e:
            raise EnvironmentError(
                f"Failed to install dependencies: {e}",
                operation="dependency_install",
                details={"stdout": e.stdout, "stderr": e.stderr},
            ) from e

    def setup_apple_silicon_optimizations(self) -> bool:
        """
        Set up Apple Silicon specific optimizations.

        Returns:
            True if optimizations were configured
        """
        if not self.hardware_info["is_apple_silicon"]:
            logger.info("Not on Apple Silicon, skipping optimizations")
            return False

        try:
            # Install Apple Silicon specific dependencies
            extra_groups = ["apple-silicon"]
            self.install_dependencies(extra_groups)

            # Configure MLX optimization
            mlx_configured = setup_mlx_optimization()

            if mlx_configured:
                logger.info("Apple Silicon optimizations configured successfully")

            return mlx_configured

        except Exception as e:
            logger.error("Failed to setup Apple Silicon optimizations: %s", e)
            return False

    def run_full_setup(self) -> dict[str, bool]:
        """
        Run complete environment setup process.

        Returns:
            Dictionary with setup results for each step

        Raises:
            EnvironmentError: If critical setup steps fail
        """
        results = {
            "python_validation": False,
            "uv_check": False,
            "dependency_install": False,
            "apple_silicon_setup": False,
        }

        try:
            # Validate Python version
            results["python_validation"] = self.validate_python_version()

            # Check uv installation
            results["uv_check"] = self.check_uv_installation()

            # Install dependencies
            extra_groups = ["dev"]
            if self.hardware_info["is_apple_silicon"]:
                extra_groups.append("apple-silicon")

            results["dependency_install"] = self.install_dependencies(extra_groups)

            # Setup Apple Silicon optimizations if applicable
            if self.hardware_info["is_apple_silicon"]:
                results["apple_silicon_setup"] = self.setup_apple_silicon_optimizations()
            else:
                results["apple_silicon_setup"] = True  # N/A but successful

            logger.info("Full environment setup completed successfully")
            return results

        except Exception as e:
            logger.error("Environment setup failed: %s", e)
            raise EnvironmentError(
                f"Full environment setup failed: {e}", operation="full_setup", details=results
            ) from e
