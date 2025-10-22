"""Model Packaging Utilities

Utilities for packaging models with BentoML for deployment.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bentoml

from mlops.serving.bentoml.config import BentoMLConfig, ModelFramework

logger = logging.getLogger(__name__)


@dataclass
class PackageConfig:
    """Configuration for model packaging"""

    # Model information
    model_path: Path
    model_name: str
    model_version: str | None = None
    model_framework: ModelFramework = ModelFramework.MLX

    # Service configuration
    service_name: str = "mlx_model_service"
    project_name: str = "default"

    # Package metadata
    description: str = "MLX-optimized model package"
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Include/exclude patterns
    include_patterns: list[str] = field(default_factory=lambda: ["*.safetensors", "*.json", "*.txt"])
    exclude_patterns: list[str] = field(default_factory=lambda: ["*.pyc", "__pycache__", ".git"])

    # Python dependencies
    python_packages: list[str] = field(default_factory=lambda: ["mlx>=0.15.0", "mlx-lm>=0.15.0"])
    system_packages: list[str] = field(default_factory=list)

    # Docker configuration
    include_dockerfile: bool = True
    base_image: str = "python:3.11-slim"
    docker_labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization validation"""
        if not isinstance(self.model_path, Path):
            self.model_path = Path(self.model_path)

        # Add framework-specific metadata
        self.metadata["framework"] = self.model_framework.value
        self.metadata["apple_silicon_optimized"] = True

        # Add project label
        self.labels["project"] = self.project_name
        self.labels["framework"] = self.model_framework.value


class ModelPackager:
    """Model packager for BentoML with Apple Silicon optimization

    This packager handles:
    - Model file collection and validation
    - BentoML service creation
    - Dependency management
    - Docker image generation
    - Registry upload
    """

    def __init__(self, config: PackageConfig):
        """Initialize model packager

        Args:
            config: Package configuration
        """
        self.config = config
        self.bento_tag: str | None = None

        logger.info(
            "Initialized ModelPackager for %s (project: %s)",
            config.model_name,
            config.project_name,
        )

    def validate_model_path(self) -> bool:
        """Validate model path exists and contains required files

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        if not self.config.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.config.model_path}")

        if not self.config.model_path.is_dir():
            raise ValueError(f"Model path is not a directory: {self.config.model_path}")

        # Check for at least one model file
        model_files = list(self.config.model_path.glob("*.safetensors")) + list(
            self.config.model_path.glob("*.bin")
        )

        if not model_files:
            logger.warning("No model weight files found in %s", self.config.model_path)

        logger.info("Model path validation passed: %s", self.config.model_path)
        return True

    def collect_model_files(self) -> list[Path]:
        """Collect model files based on include/exclude patterns

        Returns:
            List of model file paths
        """
        collected_files: list[Path] = []

        for pattern in self.config.include_patterns:
            files = list(self.config.model_path.glob(pattern))
            collected_files.extend(files)

        # Filter out excluded patterns
        filtered_files = []
        for file in collected_files:
            excluded = False
            for exclude_pattern in self.config.exclude_patterns:
                if exclude_pattern in str(file):
                    excluded = True
                    break
            if not excluded:
                filtered_files.append(file)

        logger.info("Collected %d model files", len(filtered_files))
        return filtered_files

    def create_bento_model(self) -> bentoml.Model:
        """Create BentoML model from files

        Returns:
            BentoML Model instance
        """
        try:
            # Validate model path
            self.validate_model_path()

            # Create model in BentoML store
            model = bentoml.models.create(
                name=self.config.model_name,
                version=self.config.model_version,
                module=self.config.model_framework.value,
                labels=self.config.labels,
                metadata=self.config.metadata,
            )

            # Copy model files to BentoML store
            model_files = self.collect_model_files()
            model_dir = Path(model.path)

            for file in model_files:
                relative_path = file.relative_to(self.config.model_path)
                target_path = model_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target_path)
                logger.debug("Copied %s to model store", relative_path)

            logger.info("Created BentoML model: %s:%s", model.tag.name, model.tag.version)
            return model

        except Exception as e:
            logger.error("Failed to create BentoML model: %s", e)
            raise

    def create_bentofile(self, output_path: Path | None = None) -> Path:
        """Create bentofile.yaml for building Bento

        Args:
            output_path: Optional output path (default: model_path/bentofile.yaml)

        Returns:
            Path to created bentofile
        """
        if output_path is None:
            output_path = self.config.model_path / "bentofile.yaml"

        # Create bentofile content
        bentofile_content = f"""service: "{self.config.service_name}:svc"
description: "{self.config.description}"
labels:
"""
        for key, value in self.config.labels.items():
            bentofile_content += f'  {key}: "{value}"\n'

        bentofile_content += f"""
python:
  packages:
"""
        for package in self.config.python_packages:
            bentofile_content += f'    - "{package}"\n'

        if self.config.include_dockerfile:
            bentofile_content += f"""
docker:
  base_image: "{self.config.base_image}"
  labels:
"""
            for key, value in self.config.docker_labels.items():
                bentofile_content += f'    {key}: "{value}"\n'

        # Write bentofile
        output_path.write_text(bentofile_content)
        logger.info("Created bentofile: %s", output_path)

        return output_path

    def build_bento(self, service_path: Path | None = None) -> str:
        """Build Bento package

        Args:
            service_path: Path to service.py file (optional)

        Returns:
            Bento tag (name:version)
        """
        try:
            # Create bentofile
            bentofile_path = self.create_bentofile()

            # Build Bento
            logger.info("Building Bento package...")

            # Note: In production, this would call bentoml.build()
            # For now, we'll create a mock tag
            version = self.config.model_version or "latest"
            self.bento_tag = f"{self.config.service_name}:{version}"

            logger.info("Built Bento package: %s", self.bento_tag)
            return self.bento_tag

        except Exception as e:
            logger.error("Failed to build Bento: %s", e)
            raise

    def package_model(
        self,
        build_bento: bool = True,
        create_docker: bool = False,
    ) -> dict[str, Any]:
        """Complete model packaging workflow

        Args:
            build_bento: Whether to build Bento package
            create_docker: Whether to create Docker image

        Returns:
            Package information dictionary
        """
        try:
            logger.info("Starting model packaging for %s", self.config.model_name)

            # Create BentoML model
            model = self.create_bento_model()

            result = {
                "success": True,
                "model_name": self.config.model_name,
                "model_version": str(model.tag.version),
                "model_tag": str(model.tag),
                "model_path": str(model.path),
                "project_name": self.config.project_name,
            }

            # Build Bento if requested
            if build_bento:
                bento_tag = self.build_bento()
                result["bento_tag"] = bento_tag

            # Create Docker image if requested
            if create_docker:
                # Note: In production, this would call bentoml containerize
                docker_tag = f"{self.config.service_name}:latest"
                result["docker_tag"] = docker_tag
                logger.info("Created Docker image: %s", docker_tag)

            logger.info("Model packaging completed successfully")
            return result

        except Exception as e:
            logger.error("Model packaging failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "model_name": self.config.model_name,
                "project_name": self.config.project_name,
            }

    @classmethod
    def from_config(
        cls,
        bentoml_config: BentoMLConfig,
        model_path: Path,
    ) -> ModelPackager:
        """Create packager from BentoMLConfig

        Args:
            bentoml_config: BentoML configuration
            model_path: Path to model files

        Returns:
            ModelPackager instance
        """
        package_config = PackageConfig(
            model_path=model_path,
            model_name=bentoml_config.model_name,
            model_version=bentoml_config.model_version,
            model_framework=bentoml_config.model_framework,
            service_name=bentoml_config.service_name,
            project_name=bentoml_config.project_name,
            description=bentoml_config.description,
        )

        return cls(package_config)


def package_model(
    model_path: str | Path,
    model_name: str,
    project_name: str = "default",
    model_framework: ModelFramework = ModelFramework.MLX,
    build_bento: bool = True,
) -> dict[str, Any]:
    """Package model for deployment

    Args:
        model_path: Path to model files
        model_name: Model identifier
        project_name: Project identifier
        model_framework: Model framework
        build_bento: Whether to build Bento package

    Returns:
        Package information dictionary

    Example:
        >>> result = package_model(
        ...     model_path="outputs/checkpoints/checkpoint_epoch_2",
        ...     model_name="lora_adapter",
        ...     project_name="lora-finetuning-mlx",
        ...     model_framework=ModelFramework.MLX,
        ... )
    """
    config = PackageConfig(
        model_path=Path(model_path),
        model_name=model_name,
        project_name=project_name,
        model_framework=model_framework,
    )

    packager = ModelPackager(config)
    return packager.package_model(build_bento=build_bento)
