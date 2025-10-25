"""Tests for Dashboard Components"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops.dashboard import DashboardDataAggregator, DashboardServer, create_dashboard_app
from mlops.workspace.manager import ProjectWorkspace


@pytest.fixture
def mock_workspace():
    """Create a mock workspace"""
    workspace = Mock(spec=ProjectWorkspace)
    workspace.project_name = "test-project"
    workspace.root_path = Path("/tmp/test-workspace")
    workspace.mlflow_path = Path("/tmp/test-workspace/mlflow")
    workspace.dvc_path = Path("/tmp/test-workspace/dvc")
    workspace.monitoring_path = Path("/tmp/test-workspace/monitoring")
    workspace.models_path = Path("/tmp/test-workspace/models")
    workspace.outputs_path = Path("/tmp/test-workspace/outputs")
    workspace.mlflow_experiment_id = "test-exp-123"
    workspace.mlflow_tracking_uri = "file:///tmp/mlflow"
    workspace.dvc_remote_path = "/tmp/dvc/remote"
    workspace.bentoml_tag_prefix = "test-project:"
    workspace.metadata = {}
    workspace.created_at = datetime.now()
    workspace.updated_at = datetime.now()
    workspace.to_dict = Mock(
        return_value={
            "project_name": "test-project",
            "root_path": "/tmp/test-workspace",
            "mlflow_path": "/tmp/test-workspace/mlflow",
            "dvc_path": "/tmp/test-workspace/dvc",
            "monitoring_path": "/tmp/test-workspace/monitoring",
            "models_path": "/tmp/test-workspace/models",
            "outputs_path": "/tmp/test-workspace/outputs",
            "mlflow_experiment_id": "test-exp-123",
            "mlflow_tracking_uri": "file:///tmp/mlflow",
            "dvc_remote_path": "/tmp/dvc/remote",
            "bentoml_tag_prefix": "test-project:",
            "metadata": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    )
    return workspace


@pytest.fixture
def mock_workspace_manager(mock_workspace):
    """Create a mock workspace manager"""
    manager = MagicMock()
    manager.list_workspaces.return_value = [mock_workspace]
    manager.get_workspace.return_value = mock_workspace
    return manager


@pytest.fixture
def data_aggregator(mock_workspace_manager, tmp_path):
    """Create a data aggregator with mocked dependencies"""
    with patch("mlops.dashboard.data_aggregator.WorkspaceManager", return_value=mock_workspace_manager):
        aggregator = DashboardDataAggregator(repo_root=tmp_path)
        aggregator.workspace_manager = mock_workspace_manager
        return aggregator


class TestDashboardDataAggregator:
    """Tests for DashboardDataAggregator"""

    def test_initialization(self, tmp_path):
        """Test data aggregator initialization"""
        aggregator = DashboardDataAggregator(repo_root=tmp_path)

        assert aggregator.repo_root == tmp_path
        assert aggregator.workspace_manager is not None

    def test_get_workspaces(self, data_aggregator, mock_workspace):
        """Test getting workspaces"""
        workspaces = data_aggregator._get_workspaces()

        assert len(workspaces) == 1
        assert workspaces[0]["project_name"] == "test-project"

    def test_get_workspaces_error_handling(self, data_aggregator, mock_workspace_manager):
        """Test workspace retrieval error handling"""
        mock_workspace_manager.list_workspaces.side_effect = Exception("Test error")

        workspaces = data_aggregator._get_workspaces()

        assert workspaces == []

    @patch("mlops.dashboard.data_aggregator.mlflow")
    def test_get_experiments_by_project(self, mock_mlflow, data_aggregator, mock_workspace):
        """Test getting experiments grouped by project"""
        # Mock MLFlow experiments
        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp-123"
        mock_exp.name = "test-experiment"
        mock_exp.artifact_location = "/tmp/artifacts"
        mock_exp.lifecycle_stage = "active"
        mock_exp.tags = {"project": "test-project"}

        mock_mlflow.search_experiments.return_value = [mock_exp]
        mock_mlflow.search_runs.return_value = pd.DataFrame()

        workspaces = [mock_workspace.to_dict()]
        experiments = data_aggregator._get_experiments_by_project(workspaces)

        assert "test-project" in experiments
        assert len(experiments["test-project"]) == 1
        assert experiments["test-project"][0]["experiment_id"] == "exp-123"

    def test_get_models(self, data_aggregator, mock_workspace, tmp_path):
        """Test getting models"""
        # Create a mock model directory
        models_path = tmp_path / "models"
        models_path.mkdir()
        model_dir = models_path / "test-model"
        model_dir.mkdir()
        (model_dir / "model.bin").write_text("test")

        # Update mock workspace
        mock_workspace.to_dict.return_value["models_path"] = str(models_path)

        workspaces = [mock_workspace.to_dict()]
        models = data_aggregator._get_models(workspaces)

        assert len(models) == 1
        assert models[0]["model_name"] == "test-model"
        assert models[0]["project_name"] == "test-project"

    @patch("mlops.dashboard.data_aggregator.AlertManager")
    def test_get_alerts(self, mock_alert_manager_class, data_aggregator, mock_workspace, tmp_path):
        """Test getting alerts"""
        # Create monitoring directory
        monitoring_path = tmp_path / "monitoring"
        monitoring_path.mkdir()

        # Mock alert manager
        mock_alert = MagicMock()
        mock_alert.to_dict.return_value = {
            "alert_id": "alert-123",
            "timestamp": datetime.now().isoformat(),
            "alert_type": "drift_detected",
            "severity": "warning",
            "title": "Test Alert",
            "message": "Test message",
        }

        mock_alert_manager = MagicMock()
        mock_alert_manager.get_all_alerts.return_value = [mock_alert]
        mock_alert_manager_class.return_value = mock_alert_manager

        # Update mock workspace
        mock_workspace.to_dict.return_value["monitoring_path"] = str(monitoring_path)

        workspaces = [mock_workspace.to_dict()]
        alerts = data_aggregator._get_alerts(workspaces)

        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "alert-123"
        assert alerts[0]["project_name"] == "test-project"

    @patch("mlops.monitoring.evidently.monitor.create_monitor")
    def test_get_monitoring_status(self, mock_create_monitor, data_aggregator, mock_workspace, tmp_path):
        """Test getting monitoring status"""
        # Create monitoring directory
        monitoring_path = tmp_path / "monitoring"
        monitoring_path.mkdir()

        # Mock monitor
        mock_monitor = MagicMock()
        mock_monitor.get_monitoring_status.return_value = {
            "enabled": True,
            "drift_detection_enabled": True,
        }
        mock_create_monitor.return_value = mock_monitor

        # Update mock workspace
        mock_workspace.to_dict.return_value["monitoring_path"] = str(monitoring_path)

        workspaces = [mock_workspace.to_dict()]
        status = data_aggregator._get_monitoring_status(workspaces)

        assert "test-project" in status
        assert status["test-project"]["enabled"] is True

    @patch("mlops.monitoring.evidently.apple_silicon_metrics.AppleSiliconMetricsCollector")
    def test_get_apple_silicon_metrics(self, mock_collector_class, data_aggregator, mock_workspace):
        """Test getting Apple Silicon metrics"""
        # Mock metrics collector
        mock_metrics = MagicMock()
        mock_metrics.to_dict.return_value = {
            "chip_type": "M1",
            "memory_total_gb": 16.0,
            "memory_utilization_percent": 50.0,
        }

        mock_collector = MagicMock()
        mock_collector.is_apple_silicon.return_value = True
        mock_collector.collect.return_value = mock_metrics
        mock_collector_class.return_value = mock_collector

        workspaces = [mock_workspace.to_dict()]
        metrics = data_aggregator._get_apple_silicon_metrics(workspaces)

        assert "test-project" in metrics
        assert metrics["test-project"]["chip_type"] == "M1"

    def test_calculate_cross_project_stats(self, data_aggregator):
        """Test calculating cross-project statistics"""
        workspaces = [{"project_name": "project1"}, {"project_name": "project2"}]
        experiments = {
            "project1": [{"runs": [1, 2, 3]}],
            "project2": [{"runs": [4, 5]}],
        }
        models = [
            {"size_mb": 10.0},
            {"size_mb": 20.0},
        ]
        alerts = [
            {"severity": "warning"},
            {"severity": "critical"},
        ]

        stats = data_aggregator._calculate_cross_project_stats(
            workspaces, experiments, models, alerts
        )

        assert stats["total_projects"] == 2
        assert stats["total_experiments"] == 2
        assert stats["total_runs"] == 5
        assert stats["total_models"] == 2
        assert stats["total_alerts"] == 2
        assert stats["alerts_by_severity"]["warning"] == 1
        assert stats["alerts_by_severity"]["critical"] == 1
        assert stats["total_models_size_mb"] == 30.0

    @patch("mlops.dashboard.data_aggregator.mlflow")
    @patch("mlops.monitoring.evidently.monitor.create_monitor")
    @patch("mlops.monitoring.evidently.apple_silicon_metrics.AppleSiliconMetricsCollector")
    def test_get_all_data(
        self,
        mock_collector_class,
        mock_create_monitor,
        mock_mlflow,
        data_aggregator,
        mock_workspace,
        tmp_path,
    ):
        """Test getting all dashboard data"""
        # Setup mocks
        mock_mlflow.search_experiments.return_value = []
        mock_create_monitor.return_value.get_monitoring_status.return_value = {"enabled": False}

        mock_collector = MagicMock()
        mock_collector.is_apple_silicon.return_value = False
        mock_collector_class.return_value = mock_collector

        # Get all data
        data = data_aggregator.get_all_data()

        assert data.workspaces is not None
        assert data.experiments is not None
        assert data.models is not None
        assert data.alerts is not None
        assert data.monitoring_status is not None
        assert data.apple_silicon_metrics is not None
        assert data.cross_project_stats is not None

        # Test to_dict
        data_dict = data.to_dict()
        assert "workspaces" in data_dict
        assert "experiments" in data_dict

    def test_get_project_data(self, data_aggregator, mock_workspace_manager):
        """Test getting project-specific data"""
        with patch.object(data_aggregator, "get_all_data") as mock_get_all:
            mock_data = MagicMock()
            mock_data.experiments = {"test-project": []}
            mock_data.models = [{"project_name": "test-project"}]
            mock_data.alerts = [{"project_name": "test-project"}]
            mock_data.monitoring_status = {"test-project": {}}
            mock_data.apple_silicon_metrics = {"test-project": {}}
            mock_get_all.return_value = mock_data

            project_data = data_aggregator.get_project_data("test-project")

            assert "workspace" in project_data
            assert "experiments" in project_data
            assert "models" in project_data

    def test_get_project_data_not_found(self, data_aggregator, mock_workspace_manager):
        """Test getting data for non-existent project"""
        from mlops.workspace.manager import WorkspaceError

        mock_workspace_manager.get_workspace.side_effect = WorkspaceError(
            "Not found", workspace="test"
        )

        with pytest.raises(ValueError, match="Project not found"):
            data_aggregator.get_project_data("nonexistent")


class TestDashboardServer:
    """Tests for DashboardServer"""

    @pytest.fixture
    def app(self, tmp_path):
        """Create test FastAPI app"""
        return create_dashboard_app(repo_root=tmp_path)

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)

    def test_server_initialization(self, tmp_path):
        """Test dashboard server initialization"""
        server = DashboardServer(repo_root=tmp_path, host="127.0.0.1", port=8080)

        assert server.repo_root == tmp_path
        assert server.host == "127.0.0.1"
        assert server.port == 8080
        assert server.app is not None

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @patch("mlops.dashboard.server.DashboardDataAggregator")
    def test_home_page(self, mock_aggregator_class, client, mock_workspace):
        """Test dashboard home page"""
        # Mock data aggregator
        mock_data = MagicMock()
        mock_data.workspaces = [mock_workspace.to_dict()]
        mock_data.cross_project_stats = {
            "total_projects": 1,
            "active_projects": 1,
            "total_experiments": 0,
            "total_runs": 0,
            "total_models": 0,
            "total_alerts": 0,
            "alerts_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0},
            "total_models_size_mb": 0.0,
        }

        mock_aggregator = MagicMock()
        mock_aggregator.get_all_data.return_value = mock_data
        mock_aggregator_class.return_value = mock_aggregator

        response = client.get("/")

        assert response.status_code == 200
        assert "EfficientAI MLOps Dashboard" in response.text

    @patch("mlops.dashboard.server.DashboardDataAggregator")
    def test_api_data_endpoint(self, mock_aggregator_class, client):
        """Test API data endpoint"""
        # Mock data
        mock_data = MagicMock()
        mock_data.to_dict.return_value = {"workspaces": [], "experiments": {}}

        mock_aggregator = MagicMock()
        mock_aggregator.get_all_data.return_value = mock_data
        mock_aggregator_class.return_value = mock_aggregator

        response = client.get("/api/data")

        assert response.status_code == 200
        data = response.json()
        assert "workspaces" in data
        assert "experiments" in data

    @patch("mlops.dashboard.server.DashboardDataAggregator")
    def test_experiments_page(self, mock_aggregator_class, client):
        """Test experiments page"""
        mock_data = MagicMock()
        mock_data.workspaces = []
        mock_data.experiments = {}

        mock_aggregator = MagicMock()
        mock_aggregator.get_all_data.return_value = mock_data
        mock_aggregator_class.return_value = mock_aggregator

        response = client.get("/experiments")

        assert response.status_code == 200
        assert "Experiments" in response.text

    @patch("mlops.dashboard.server.DashboardDataAggregator")
    def test_experiments_with_project_filter(self, mock_aggregator_class, client):
        """Test experiments page with project filter"""
        mock_data = MagicMock()
        mock_data.workspaces = []
        mock_data.experiments = {"test-project": []}

        mock_aggregator = MagicMock()
        mock_aggregator.get_all_data.return_value = mock_data
        mock_aggregator_class.return_value = mock_aggregator

        response = client.get("/experiments?project=test-project")

        assert response.status_code == 200

    @patch("mlops.dashboard.server.DashboardDataAggregator")
    def test_models_page(self, mock_aggregator_class, client):
        """Test models page"""
        mock_data = MagicMock()
        mock_data.workspaces = []
        mock_data.models = []

        mock_aggregator = MagicMock()
        mock_aggregator.get_all_data.return_value = mock_data
        mock_aggregator_class.return_value = mock_aggregator

        response = client.get("/models")

        assert response.status_code == 200
        assert "Model Registry" in response.text

    @patch("mlops.dashboard.server.DashboardDataAggregator")
    def test_monitoring_page(self, mock_aggregator_class, client):
        """Test monitoring page"""
        mock_data = MagicMock()
        mock_data.workspaces = []
        mock_data.monitoring_status = {}

        mock_aggregator = MagicMock()
        mock_aggregator.get_all_data.return_value = mock_data
        mock_aggregator_class.return_value = mock_aggregator

        response = client.get("/monitoring")

        assert response.status_code == 200
        assert "Monitoring Dashboard" in response.text

    @patch("mlops.dashboard.server.DashboardDataAggregator")
    def test_alerts_page(self, mock_aggregator_class, client, mock_workspace):
        """Test alerts page"""
        mock_data = MagicMock()
        mock_data.workspaces = []
        mock_data.alerts = []
        mock_data.cross_project_stats = {
            "alerts_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0}
        }

        mock_aggregator = MagicMock()
        mock_aggregator.get_all_data.return_value = mock_data
        mock_aggregator_class.return_value = mock_aggregator

        response = client.get("/alerts")

        assert response.status_code == 200
        assert "Alerts Management" in response.text

    @patch("mlops.dashboard.server.DashboardDataAggregator")
    def test_hardware_page(self, mock_aggregator_class, client):
        """Test hardware page"""
        mock_data = MagicMock()
        mock_data.workspaces = []
        mock_data.apple_silicon_metrics = {}

        mock_aggregator = MagicMock()
        mock_aggregator.get_all_data.return_value = mock_data
        mock_aggregator_class.return_value = mock_aggregator

        response = client.get("/hardware")

        assert response.status_code == 200
        assert "Apple Silicon Hardware Metrics" in response.text

    def test_project_detail_page(self, tmp_path, mock_workspace):
        """Test project detail page"""
        # Create a fresh app with mocked aggregator
        with patch("mlops.dashboard.server.DashboardDataAggregator") as mock_aggregator_class:
            mock_aggregator = MagicMock()
            mock_aggregator.get_project_data.return_value = {
                "workspace": mock_workspace.to_dict(),
                "experiments": [],
                "models": [],
                "alerts": [],
                "monitoring_status": {},
                "apple_silicon_metrics": {},
            }
            mock_aggregator_class.return_value = mock_aggregator

            app = create_dashboard_app(repo_root=tmp_path)
            client = TestClient(app)

            response = client.get("/project/test-project")

            assert response.status_code == 200
            assert "test-project" in response.text

    @patch("mlops.dashboard.server.DashboardDataAggregator")
    def test_project_detail_not_found(self, mock_aggregator_class, client):
        """Test project detail page for non-existent project"""
        mock_aggregator = MagicMock()
        mock_aggregator.get_project_data.side_effect = ValueError("Project not found")
        mock_aggregator_class.return_value = mock_aggregator

        response = client.get("/project/nonexistent")

        assert response.status_code == 404

    @patch("mlops.dashboard.server.DashboardDataAggregator")
    def test_api_endpoints(self, mock_aggregator_class, client):
        """Test all API endpoints"""
        mock_data = MagicMock()
        mock_data.experiments = {}
        mock_data.models = []
        mock_data.monitoring_status = {}
        mock_data.alerts = []
        mock_data.apple_silicon_metrics = {}

        mock_aggregator = MagicMock()
        mock_aggregator.get_all_data.return_value = mock_data
        mock_aggregator_class.return_value = mock_aggregator

        # Test all API endpoints
        endpoints = [
            "/api/experiments",
            "/api/models",
            "/api/monitoring",
            "/api/alerts",
            "/api/hardware",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            assert isinstance(response.json(), dict)
