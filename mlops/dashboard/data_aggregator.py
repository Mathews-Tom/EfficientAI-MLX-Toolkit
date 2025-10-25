"""Dashboard Data Aggregator

Aggregates data from MLFlow, Evidently, workspaces, and Apple Silicon metrics
for unified dashboard visualization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow

from mlops.client.mlflow_client import MLFlowClient, MLFlowClientError
from mlops.monitoring.evidently.alert_manager import AlertManager
from mlops.monitoring.evidently.monitor import EvidentlyMonitor
from mlops.workspace.manager import WorkspaceManager, WorkspaceError

logger = logging.getLogger(__name__)


@dataclass
class DashboardData:
    """Aggregated dashboard data

    Attributes:
        workspaces: List of workspace information
        experiments: List of experiments grouped by project
        models: List of models from registry
        alerts: List of active alerts
        monitoring_status: Monitoring status by project
        apple_silicon_metrics: Apple Silicon metrics by project
        cross_project_stats: Cross-project statistics
    """

    workspaces: list[dict[str, Any]]
    experiments: dict[str, list[dict[str, Any]]]
    models: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    monitoring_status: dict[str, Any]
    apple_silicon_metrics: dict[str, Any]
    cross_project_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workspaces": self.workspaces,
            "experiments": self.experiments,
            "models": self.models,
            "alerts": self.alerts,
            "monitoring_status": self.monitoring_status,
            "apple_silicon_metrics": self.apple_silicon_metrics,
            "cross_project_stats": self.cross_project_stats,
        }


class DashboardDataAggregator:
    """Aggregates data from all MLOps components for dashboard

    Provides unified data access layer for dashboard visualization,
    pulling data from MLFlow, Evidently, workspaces, and Apple Silicon metrics.
    """

    def __init__(
        self,
        repo_root: str | Path | None = None,
        workspace_manager: WorkspaceManager | None = None,
    ):
        """Initialize data aggregator

        Args:
            repo_root: Repository root directory
            workspace_manager: Optional workspace manager instance
        """
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.workspace_manager = workspace_manager or WorkspaceManager(repo_root=self.repo_root)

        logger.info("Initialized DashboardDataAggregator")

    def get_all_data(self) -> DashboardData:
        """Get all dashboard data

        Returns:
            DashboardData with aggregated information from all sources
        """
        logger.info("Aggregating all dashboard data")

        # Get workspaces
        workspaces = self._get_workspaces()

        # Get experiments grouped by project
        experiments = self._get_experiments_by_project(workspaces)

        # Get models
        models = self._get_models(workspaces)

        # Get alerts
        alerts = self._get_alerts(workspaces)

        # Get monitoring status
        monitoring_status = self._get_monitoring_status(workspaces)

        # Get Apple Silicon metrics
        apple_silicon_metrics = self._get_apple_silicon_metrics(workspaces)

        # Calculate cross-project stats
        cross_project_stats = self._calculate_cross_project_stats(
            workspaces, experiments, models, alerts
        )

        return DashboardData(
            workspaces=workspaces,
            experiments=experiments,
            models=models,
            alerts=alerts,
            monitoring_status=monitoring_status,
            apple_silicon_metrics=apple_silicon_metrics,
            cross_project_stats=cross_project_stats,
        )

    def _get_workspaces(self) -> list[dict[str, Any]]:
        """Get all workspace information"""
        try:
            workspaces = self.workspace_manager.list_workspaces()
            return [ws.to_dict() for ws in workspaces]
        except Exception as e:
            logger.error("Failed to get workspaces: %s", e)
            return []

    def _get_experiments_by_project(
        self,
        workspaces: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Get experiments grouped by project

        Args:
            workspaces: List of workspace information

        Returns:
            Dictionary mapping project names to experiment lists
        """
        experiments_by_project: dict[str, list[dict[str, Any]]] = {}

        for ws in workspaces:
            project_name = ws["project_name"]
            experiments_by_project[project_name] = []

            try:
                # Get experiments from MLFlow
                tracking_uri = ws.get("mlflow_tracking_uri")
                if not tracking_uri:
                    continue

                mlflow.set_tracking_uri(tracking_uri)

                # Search experiments with project tag
                experiments = mlflow.search_experiments(
                    filter_string=f"tags.project = '{project_name}'"
                )

                for exp in experiments:
                    # Get runs for this experiment
                    runs = mlflow.search_runs(
                        experiment_ids=[exp.experiment_id],
                        max_results=10,
                        order_by=["start_time DESC"],
                    )

                    exp_data = {
                        "experiment_id": exp.experiment_id,
                        "experiment_name": exp.name,
                        "artifact_location": exp.artifact_location,
                        "lifecycle_stage": exp.lifecycle_stage,
                        "tags": dict(exp.tags) if exp.tags else {},
                        "runs": runs.to_dict("records") if not runs.empty else [],
                    }

                    experiments_by_project[project_name].append(exp_data)

            except Exception as e:
                logger.error("Failed to get experiments for %s: %s", project_name, e)
                continue

        return experiments_by_project

    def _get_models(self, workspaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Get all models from registry

        Args:
            workspaces: List of workspace information

        Returns:
            List of model information
        """
        models: list[dict[str, Any]] = []

        for ws in workspaces:
            project_name = ws["project_name"]
            models_path = Path(ws["models_path"])

            if not models_path.exists():
                continue

            try:
                # List model directories
                for model_dir in models_path.iterdir():
                    if not model_dir.is_dir():
                        continue

                    model_info = {
                        "project_name": project_name,
                        "model_name": model_dir.name,
                        "model_path": str(model_dir),
                        "created_at": datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat(),
                        "size_mb": sum(
                            f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                        ) / (1024 * 1024),
                    }

                    models.append(model_info)

            except Exception as e:
                logger.error("Failed to get models for %s: %s", project_name, e)
                continue

        return models

    def _get_alerts(self, workspaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Get all alerts

        Args:
            workspaces: List of workspace information

        Returns:
            List of alert information
        """
        all_alerts: list[dict[str, Any]] = []

        for ws in workspaces:
            project_name = ws["project_name"]
            monitoring_path = Path(ws["monitoring_path"])

            if not monitoring_path.exists():
                continue

            try:
                # Create alert manager for this project
                alert_manager = AlertManager(
                    project_name=project_name,
                    workspace_path=monitoring_path,
                )

                # Get unresolved alerts
                alerts = alert_manager.get_all_alerts(unresolved_only=True)

                for alert in alerts:
                    alert_dict = alert.to_dict()
                    alert_dict["project_name"] = project_name
                    all_alerts.append(alert_dict)

            except Exception as e:
                logger.debug("No alerts for %s: %s", project_name, e)
                continue

        # Sort by timestamp (newest first)
        all_alerts.sort(key=lambda x: x["timestamp"], reverse=True)

        return all_alerts

    def _get_monitoring_status(
        self,
        workspaces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Get monitoring status for all projects

        Args:
            workspaces: List of workspace information

        Returns:
            Dictionary mapping project names to monitoring status
        """
        monitoring_status: dict[str, Any] = {}

        for ws in workspaces:
            project_name = ws["project_name"]
            monitoring_path = Path(ws["monitoring_path"])

            if not monitoring_path.exists():
                monitoring_status[project_name] = {"enabled": False}
                continue

            try:
                # Create monitor for this project
                from mlops.monitoring.evidently.monitor import create_monitor

                monitor = create_monitor(
                    project_name=project_name,
                    workspace_path=monitoring_path,
                )

                status = monitor.get_monitoring_status()
                monitoring_status[project_name] = status

            except Exception as e:
                logger.debug("No monitoring for %s: %s", project_name, e)
                monitoring_status[project_name] = {"enabled": False, "error": str(e)}

        return monitoring_status

    def _get_apple_silicon_metrics(
        self,
        workspaces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Get Apple Silicon metrics

        Args:
            workspaces: List of workspace information

        Returns:
            Dictionary mapping project names to Apple Silicon metrics
        """
        metrics_by_project: dict[str, Any] = {}

        # Try to get current Apple Silicon metrics
        try:
            from mlops.monitoring.evidently.apple_silicon_metrics import (
                AppleSiliconMetricsCollector,
            )

            collector = AppleSiliconMetricsCollector(project_name="dashboard")

            if collector.is_apple_silicon():
                current_metrics = collector.collect()

                # Assign to each project (they all share the same hardware)
                for ws in workspaces:
                    project_name = ws["project_name"]
                    metrics_by_project[project_name] = current_metrics.to_dict()
            else:
                logger.info("Not running on Apple Silicon, skipping metrics collection")

        except Exception as e:
            logger.debug("Failed to collect Apple Silicon metrics: %s", e)

        return metrics_by_project

    def _calculate_cross_project_stats(
        self,
        workspaces: list[dict[str, Any]],
        experiments: dict[str, list[dict[str, Any]]],
        models: list[dict[str, Any]],
        alerts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate cross-project statistics

        Args:
            workspaces: List of workspace information
            experiments: Experiments grouped by project
            models: List of models
            alerts: List of alerts

        Returns:
            Dictionary with cross-project statistics
        """
        # Count total runs across all experiments
        total_runs = 0
        for project_exps in experiments.values():
            for exp in project_exps:
                total_runs += len(exp.get("runs", []))

        # Count alerts by severity
        alerts_by_severity: dict[str, int] = {
            "info": 0,
            "warning": 0,
            "error": 0,
            "critical": 0,
        }

        for alert in alerts:
            severity = alert.get("severity", "info")
            alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1

        # Count active projects (those with recent activity)
        active_projects = sum(
            1 for ws in workspaces
            if experiments.get(ws["project_name"], [])
        )

        return {
            "total_projects": len(workspaces),
            "active_projects": active_projects,
            "total_experiments": sum(len(exps) for exps in experiments.values()),
            "total_runs": total_runs,
            "total_models": len(models),
            "total_alerts": len(alerts),
            "alerts_by_severity": alerts_by_severity,
            "total_models_size_mb": sum(m.get("size_mb", 0) for m in models),
        }

    def get_project_data(self, project_name: str) -> dict[str, Any]:
        """Get data for a specific project

        Args:
            project_name: Project namespace identifier

        Returns:
            Dictionary with project-specific data

        Raises:
            ValueError: If project not found
        """
        try:
            workspace = self.workspace_manager.get_workspace(project_name)
            ws_dict = workspace.to_dict()

            # Get all data and filter for this project
            all_data = self.get_all_data()

            return {
                "workspace": ws_dict,
                "experiments": all_data.experiments.get(project_name, []),
                "models": [m for m in all_data.models if m["project_name"] == project_name],
                "alerts": [a for a in all_data.alerts if a["project_name"] == project_name],
                "monitoring_status": all_data.monitoring_status.get(project_name, {}),
                "apple_silicon_metrics": all_data.apple_silicon_metrics.get(project_name, {}),
            }

        except WorkspaceError as e:
            raise ValueError(f"Project not found: {project_name}") from e
