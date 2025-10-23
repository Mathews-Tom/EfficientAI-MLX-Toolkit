"""Evidently Dashboard Configuration

Configuration and setup for Evidently monitoring dashboard with
support for multiple projects and Apple Silicon metrics visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DashboardConfig:
    """Evidently dashboard configuration

    Provides configuration for running Evidently dashboard server
    with support for multiple projects and custom metrics.
    """

    def __init__(
        self,
        workspace_path: Path | str = "mlops/monitoring/workspace",
        host: str = "0.0.0.0",
        port: int = 8000,
        projects: list[str] | None = None,
    ):
        """Initialize dashboard configuration

        Args:
            workspace_path: Path to monitoring workspace
            host: Dashboard host address
            port: Dashboard port
            projects: List of project names to include in dashboard
        """
        self.workspace_path = Path(workspace_path)
        self.host = host
        self.port = port
        self.projects = projects or []

        logger.info(
            "Initialized DashboardConfig: workspace=%s, host=%s, port=%d, projects=%d",
            self.workspace_path,
            self.host,
            self.port,
            len(self.projects),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary

        Returns:
            Dictionary representation of dashboard configuration
        """
        return {
            "workspace_path": str(self.workspace_path),
            "host": self.host,
            "port": self.port,
            "projects": self.projects,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> DashboardConfig:
        """Create configuration from dictionary

        Args:
            config: Configuration dictionary

        Returns:
            DashboardConfig instance
        """
        return cls(**config)


def start_dashboard(
    workspace_path: Path | str = "mlops/monitoring/workspace",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Start Evidently monitoring dashboard

    Note: This function provides a placeholder for dashboard startup.
    In production, use Evidently's built-in dashboard or integrate
    with a custom web framework.

    Args:
        workspace_path: Path to monitoring workspace
        host: Dashboard host address
        port: Dashboard port

    Example:
        >>> start_dashboard(workspace_path="mlops/monitoring/workspace", port=8000)
    """
    workspace = Path(workspace_path)

    if not workspace.exists():
        logger.warning("Workspace does not exist: %s", workspace)
        workspace.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting Evidently dashboard at http://%s:%d (workspace: %s)",
        host,
        port,
        workspace,
    )

    # Note: Evidently's dashboard server can be started using:
    # evidently ui --workspace <workspace_path> --host <host> --port <port>
    #
    # For programmatic access, we recommend:
    # 1. Generate reports and save as HTML
    # 2. Serve via FastAPI/Flask
    # 3. Use Evidently Cloud for production deployments

    logger.info(
        "To start dashboard, run: evidently ui --workspace %s --host %s --port %d",
        workspace,
        host,
        port,
    )


def get_dashboard_url(host: str = "localhost", port: int = 8000) -> str:
    """Get dashboard URL

    Args:
        host: Dashboard host
        port: Dashboard port

    Returns:
        Dashboard URL
    """
    return f"http://{host}:{port}"
