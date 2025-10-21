"""Ray Serve Cluster Manager

This module provides the shared Ray Serve cluster manager for multi-project
model serving with Apple Silicon optimization and auto-scaling support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mlops.config.ray_config import RayServeConfig

logger = logging.getLogger(__name__)


class RayServeError(Exception):
    """Raised when Ray Serve operations fail"""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.details = dict(details or {})


class SharedRayCluster:
    """Shared Ray Serve cluster for multi-project model serving

    This class manages a shared Ray cluster that can serve models from multiple
    projects with Apple Silicon optimization, auto-scaling, and load balancing.

    Attributes:
        config: Ray Serve configuration
        cluster_config: Ray cluster configuration dict
        model_deployments: Dictionary tracking deployed models by project
    """

    def __init__(self, config: RayServeConfig | None = None):
        """Initialize shared Ray cluster

        Args:
            config: Ray Serve configuration (uses default if None)
        """
        from mlops.config import get_ray_serve_config

        self.config = config or get_ray_serve_config()
        self.cluster_config = self.config.to_ray_init_config()
        self.model_deployments: dict[str, dict[str, Any]] = {}
        self._ray_initialized = False
        self._serve_started = False

        # Check Ray availability
        self._check_ray_available()

    def _check_ray_available(self) -> None:
        """Check if Ray is available and importable"""
        try:
            import ray
            from ray import serve
            self._ray_available = True
        except ImportError as e:
            raise RayServeError(
                "Ray not available. Install with: uv add 'ray[serve]'",
                operation="check_ray_available",
                details={"error": str(e)},
            ) from e

    def initialize_cluster(self) -> None:
        """Initialize Ray cluster with configuration

        Raises:
            RayServeError: If cluster initialization fails
        """
        if self._ray_initialized:
            logger.debug("Ray cluster already initialized")
            return

        try:
            import ray

            logger.info("Initializing Ray cluster with config: %s", self.cluster_config)

            # Initialize Ray
            ray.init(**self.cluster_config)

            self._ray_initialized = True

            logger.info("Ray cluster initialized successfully")

            # Log Apple Silicon info if available
            if self.config.apple_silicon.chip_type:
                logger.info(
                    "Apple Silicon detected: %s with %d cores, %.1f GB memory",
                    self.config.apple_silicon.chip_type,
                    self.config.apple_silicon.cores,
                    self.config.apple_silicon.memory_gb,
                )

        except Exception as e:
            raise RayServeError(
                f"Failed to initialize Ray cluster: {e}",
                operation="initialize_cluster",
                details=self.cluster_config,
            ) from e

    def start_serve(self) -> None:
        """Start Ray Serve with configuration

        Raises:
            RayServeError: If Serve startup fails
        """
        if not self._ray_initialized:
            self.initialize_cluster()

        if self._serve_started:
            logger.debug("Ray Serve already started")
            return

        try:
            from ray import serve

            logger.info("Starting Ray Serve")

            # Start Serve with configuration
            serve_config = self.config.to_serve_config()
            serve.start(**serve_config)

            self._serve_started = True

            logger.info("Ray Serve started successfully on %s:%d",
                       self.config.host, self.config.port)

        except Exception as e:
            raise RayServeError(
                f"Failed to start Ray Serve: {e}",
                operation="start_serve",
            ) from e

    def deploy_project_model(
        self,
        project_name: str,
        model_name: str,
        model_wrapper: Any,
        deployment_config: dict[str, Any] | None = None,
    ) -> str:
        """Deploy model from an individual project to shared Ray cluster

        Args:
            project_name: Project identifier
            model_name: Model identifier within project
            model_wrapper: Model wrapper instance (callable)
            deployment_config: Optional deployment configuration override

        Returns:
            Deployment name

        Raises:
            RayServeError: If deployment fails
        """
        if not self._serve_started:
            self.start_serve()

        try:
            from ray import serve

            # Generate deployment name
            deployment_name = f"{project_name}_{model_name}"

            logger.info("Deploying model: %s (project: %s)", model_name, project_name)

            # Get deployment config
            config = deployment_config or self.config.to_deployment_config()

            # Update config with project context
            if "user_config" not in config:
                config["user_config"] = {}
            config["user_config"]["project_name"] = project_name
            config["user_config"]["model_name"] = model_name

            # Create and deploy
            @serve.deployment(name=deployment_name, **config)
            class ModelDeployment:
                def __init__(self):
                    self.wrapper = model_wrapper
                    if hasattr(self.wrapper, 'load_model'):
                        self.wrapper.load_model()

                async def __call__(self, request):
                    # Handle inference request
                    if hasattr(self.wrapper, 'predict'):
                        return self.wrapper.predict(request)
                    return {"error": "Model wrapper has no predict method"}

            # Deploy the model
            ModelDeployment.deploy()

            # Track deployment
            if project_name not in self.model_deployments:
                self.model_deployments[project_name] = {}

            self.model_deployments[project_name][model_name] = {
                "deployment_name": deployment_name,
                "config": config,
                "status": "deployed",
            }

            logger.info("Model deployed successfully: %s", deployment_name)

            return deployment_name

        except Exception as e:
            raise RayServeError(
                f"Failed to deploy model: {e}",
                operation="deploy_project_model",
                details={
                    "project_name": project_name,
                    "model_name": model_name,
                },
            ) from e

    def scale_deployment(
        self,
        project_name: str,
        model_name: str,
        num_replicas: int,
    ) -> None:
        """Scale specific project model deployment

        Args:
            project_name: Project identifier
            model_name: Model identifier
            num_replicas: Target number of replicas

        Raises:
            RayServeError: If scaling fails
        """
        try:
            from ray import serve

            # Check deployment exists
            if project_name not in self.model_deployments:
                raise RayServeError(
                    f"Project not found: {project_name}",
                    operation="scale_deployment",
                )

            if model_name not in self.model_deployments[project_name]:
                raise RayServeError(
                    f"Model not found: {model_name}",
                    operation="scale_deployment",
                )

            deployment_name = self.model_deployments[project_name][model_name]["deployment_name"]

            logger.info("Scaling deployment %s to %d replicas", deployment_name, num_replicas)

            # Get deployment and scale
            deployment = serve.get_deployment(deployment_name)
            deployment.options(num_replicas=num_replicas).deploy()

            # Update tracking
            self.model_deployments[project_name][model_name]["config"]["num_replicas"] = num_replicas

            logger.info("Deployment scaled successfully")

        except Exception as e:
            raise RayServeError(
                f"Failed to scale deployment: {e}",
                operation="scale_deployment",
                details={
                    "project_name": project_name,
                    "model_name": model_name,
                    "num_replicas": num_replicas,
                },
            ) from e

    def get_cluster_resource_usage(self) -> dict[str, Any]:
        """Get resource usage across all deployed models

        Returns:
            Dictionary with cluster resource usage metrics
        """
        if not self._ray_initialized:
            return {
                "initialized": False,
                "total_deployments": 0,
            }

        try:
            import ray

            # Get cluster resources
            resources = ray.cluster_resources()
            available = ray.available_resources()

            # Calculate usage
            total_cpus = resources.get("CPU", 0)
            available_cpus = available.get("CPU", 0)
            used_cpus = total_cpus - available_cpus

            total_memory = resources.get("memory", 0)
            available_memory = available.get("memory", 0)
            used_memory = total_memory - available_memory

            # Count deployments
            total_deployments = sum(
                len(models) for models in self.model_deployments.values()
            )

            usage = {
                "initialized": True,
                "total_deployments": total_deployments,
                "projects": list(self.model_deployments.keys()),
                "cpu": {
                    "total": total_cpus,
                    "used": used_cpus,
                    "available": available_cpus,
                    "utilization_pct": (used_cpus / total_cpus * 100) if total_cpus > 0 else 0,
                },
                "memory": {
                    "total_bytes": total_memory,
                    "used_bytes": used_memory,
                    "available_bytes": available_memory,
                    "utilization_pct": (used_memory / total_memory * 100) if total_memory > 0 else 0,
                },
            }

            # Add Apple Silicon info if available
            if self.config.apple_silicon.chip_type:
                usage["apple_silicon"] = {
                    "chip_type": self.config.apple_silicon.chip_type,
                    "cores": self.config.apple_silicon.cores,
                    "memory_gb": self.config.apple_silicon.memory_gb,
                    "thermal_aware": self.config.apple_silicon.thermal_aware,
                }

            return usage

        except Exception as e:
            logger.error("Failed to get cluster resource usage: %s", e)
            return {
                "initialized": True,
                "error": str(e),
                "total_deployments": len(self.model_deployments),
            }

    def list_deployments(
        self,
        project_name: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """List all deployments, optionally filtered by project

        Args:
            project_name: Optional project filter

        Returns:
            Dictionary mapping project names to lists of deployment info
        """
        if project_name:
            if project_name not in self.model_deployments:
                return {}

            return {
                project_name: [
                    {
                        "model_name": model_name,
                        "deployment_name": info["deployment_name"],
                        "status": info["status"],
                    }
                    for model_name, info in self.model_deployments[project_name].items()
                ]
            }

        # Return all deployments
        return {
            proj_name: [
                {
                    "model_name": model_name,
                    "deployment_name": info["deployment_name"],
                    "status": info["status"],
                }
                for model_name, info in models.items()
            ]
            for proj_name, models in self.model_deployments.items()
        }

    def shutdown(self) -> None:
        """Shutdown Ray Serve and cluster

        Raises:
            RayServeError: If shutdown fails
        """
        try:
            if self._serve_started:
                from ray import serve

                logger.info("Shutting down Ray Serve")
                serve.shutdown()
                self._serve_started = False

            if self._ray_initialized:
                import ray

                logger.info("Shutting down Ray cluster")
                ray.shutdown()
                self._ray_initialized = False

            logger.info("Ray cluster shutdown complete")

        except Exception as e:
            raise RayServeError(
                f"Failed to shutdown cluster: {e}",
                operation="shutdown",
            ) from e


def create_shared_cluster(config: RayServeConfig | None = None) -> SharedRayCluster:
    """Create a new shared Ray cluster

    Args:
        config: Optional Ray Serve configuration

    Returns:
        Initialized SharedRayCluster instance

    Example:
        >>> cluster = create_shared_cluster()
        >>> cluster.initialize_cluster()
        >>> cluster.start_serve()
    """
    return SharedRayCluster(config=config)
