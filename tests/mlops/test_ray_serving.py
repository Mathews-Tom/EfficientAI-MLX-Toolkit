"""Tests for Ray Serve Cluster Manager

This module tests the SharedRayCluster with mocked Ray infrastructure.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock

# Skip all tests if Ray not available
pytest.importorskip("ray", reason="Ray not available - install with: uv add 'ray[serve]'")

from mlops.config.ray_config import RayServeConfig, DeploymentMode
from mlops.serving.ray_serve import SharedRayCluster, RayServeError, create_shared_cluster
from mlops.serving.model_wrapper import MLXModelWrapper


class TestSharedRayCluster:
    """Tests for SharedRayCluster"""

    @patch("mlops.serving.ray_serve.logger")
    def test_init_without_ray(self, mock_logger):
        """Test initialization fails gracefully when Ray not available"""
        with patch.dict("sys.modules", {"ray": None, "ray.serve": None}):
            with pytest.raises(RayServeError) as exc_info:
                cluster = SharedRayCluster()

            assert "Ray not available" in str(exc_info.value)
            assert exc_info.value.operation == "check_ray_available"

    @patch("ray.serve")
    @patch("ray.init")
    def test_init_with_ray(self, mock_ray_init, mock_serve):
        """Test successful initialization with Ray available"""
        config = RayServeConfig()
        cluster = SharedRayCluster(config=config)

        assert cluster.config == config
        assert cluster.model_deployments == {}
        assert cluster._ray_initialized is False
        assert cluster._serve_started is False

    @patch("ray.serve")
    @patch("ray.init")
    def test_initialize_cluster(self, mock_ray_init, mock_serve):
        """Test Ray cluster initialization"""
        cluster = SharedRayCluster()
        cluster.initialize_cluster()

        assert cluster._ray_initialized is True
        mock_ray_init.assert_called_once()

    @patch("ray.serve")
    @patch("ray.init")
    def test_initialize_cluster_already_initialized(self, mock_ray_init, mock_serve):
        """Test initialization when already initialized"""
        cluster = SharedRayCluster()
        cluster._ray_initialized = True

        cluster.initialize_cluster()

        # Should not call ray.init again
        mock_ray_init.assert_not_called()

    @patch("ray.serve.start")
    @patch("ray.init")
    def test_start_serve(self, mock_ray_init, mock_serve_start):
        """Test starting Ray Serve"""
        cluster = SharedRayCluster()
        cluster._ray_initialized = True

        cluster.start_serve()

        assert cluster._serve_started is True
        mock_serve_start.assert_called_once()

    @patch("ray.serve.start")
    @patch("ray.init")
    def test_start_serve_initializes_ray(self, mock_ray_init, mock_serve_start):
        """Test starting Serve initializes Ray if needed"""
        cluster = SharedRayCluster()

        cluster.start_serve()

        assert cluster._ray_initialized is True
        assert cluster._serve_started is True
        mock_ray_init.assert_called_once()
        mock_serve_start.assert_called_once()

    @patch("ray.serve.deployment")
    @patch("ray.serve.start")
    @patch("ray.init")
    def test_deploy_project_model(self, mock_ray_init, mock_serve_start, mock_deployment):
        """Test deploying a model from a project"""
        cluster = SharedRayCluster()
        cluster._serve_started = True

        # Mock model wrapper
        model_wrapper = Mock()
        model_wrapper.load_model = Mock()
        model_wrapper.predict = Mock(return_value={"output": "test"})

        # Mock deployment decorator
        mock_deployment_class = Mock()
        mock_deployment_instance = Mock()
        mock_deployment_instance.deploy = Mock()
        mock_deployment_class.return_value = mock_deployment_instance
        mock_serve.deployment.return_value = lambda cls: cls

        deployment_name = cluster.deploy_project_model(
            project_name="test-project",
            model_name="test-model",
            model_wrapper=model_wrapper,
        )

        assert deployment_name == "test-project_test-model"
        assert "test-project" in cluster.model_deployments
        assert "test-model" in cluster.model_deployments["test-project"]
        assert cluster.model_deployments["test-project"]["test-model"]["status"] == "deployed"

    @patch("mlops.serving.ray_serve.ray")
    @patch("mlops.serving.ray_serve.serve")
    def test_scale_deployment(self, mock_serve, mock_ray):
        """Test scaling a deployment"""
        cluster = SharedRayCluster()
        cluster._serve_started = True

        # Setup existing deployment
        cluster.model_deployments = {
            "test-project": {
                "test-model": {
                    "deployment_name": "test-project_test-model",
                    "config": {"num_replicas": 1},
                    "status": "deployed",
                }
            }
        }

        # Mock get_deployment
        mock_deployment = Mock()
        mock_deployment.options = Mock(return_value=mock_deployment)
        mock_deployment.deploy = Mock()
        mock_serve.get_deployment.return_value = mock_deployment

        cluster.scale_deployment(
            project_name="test-project",
            model_name="test-model",
            num_replicas=3,
        )

        mock_serve.get_deployment.assert_called_once_with("test-project_test-model")
        mock_deployment.options.assert_called_once_with(num_replicas=3)
        assert cluster.model_deployments["test-project"]["test-model"]["config"]["num_replicas"] == 3

    @patch("mlops.serving.ray_serve.ray")
    @patch("mlops.serving.ray_serve.serve")
    def test_scale_deployment_not_found(self, mock_serve, mock_ray):
        """Test scaling non-existent deployment raises error"""
        cluster = SharedRayCluster()
        cluster._serve_started = True

        with pytest.raises(RayServeError) as exc_info:
            cluster.scale_deployment(
                project_name="nonexistent",
                model_name="model",
                num_replicas=2,
            )

        assert "Project not found" in str(exc_info.value)

    @patch("mlops.serving.ray_serve.ray")
    @patch("mlops.serving.ray_serve.serve")
    def test_get_cluster_resource_usage(self, mock_serve, mock_ray):
        """Test getting cluster resource usage"""
        cluster = SharedRayCluster()
        cluster._ray_initialized = True

        # Mock Ray cluster resources
        mock_ray.cluster_resources.return_value = {
            "CPU": 8.0,
            "memory": 16 * 1024 * 1024 * 1024,  # 16GB in bytes
        }
        mock_ray.available_resources.return_value = {
            "CPU": 4.0,
            "memory": 8 * 1024 * 1024 * 1024,  # 8GB in bytes
        }

        # Add some deployments
        cluster.model_deployments = {
            "proj1": {"model1": {}},
            "proj2": {"model2": {}, "model3": {}},
        }

        usage = cluster.get_cluster_resource_usage()

        assert usage["initialized"] is True
        assert usage["total_deployments"] == 3
        assert usage["projects"] == ["proj1", "proj2"]
        assert usage["cpu"]["total"] == 8.0
        assert usage["cpu"]["used"] == 4.0
        assert usage["cpu"]["available"] == 4.0
        assert usage["cpu"]["utilization_pct"] == 50.0

    @patch("mlops.serving.ray_serve.ray")
    @patch("mlops.serving.ray_serve.serve")
    def test_list_deployments_all(self, mock_serve, mock_ray):
        """Test listing all deployments"""
        cluster = SharedRayCluster()
        cluster.model_deployments = {
            "proj1": {
                "model1": {
                    "deployment_name": "proj1_model1",
                    "status": "deployed",
                }
            },
            "proj2": {
                "model2": {
                    "deployment_name": "proj2_model2",
                    "status": "deployed",
                }
            }
        }

        deployments = cluster.list_deployments()

        assert "proj1" in deployments
        assert "proj2" in deployments
        assert len(deployments["proj1"]) == 1
        assert deployments["proj1"][0]["model_name"] == "model1"

    @patch("mlops.serving.ray_serve.ray")
    @patch("mlops.serving.ray_serve.serve")
    def test_list_deployments_filtered(self, mock_serve, mock_ray):
        """Test listing deployments for specific project"""
        cluster = SharedRayCluster()
        cluster.model_deployments = {
            "proj1": {
                "model1": {
                    "deployment_name": "proj1_model1",
                    "status": "deployed",
                }
            },
        }

        deployments = cluster.list_deployments(project_name="proj1")

        assert "proj1" in deployments
        assert "proj2" not in deployments

    @patch("mlops.serving.ray_serve.ray")
    @patch("mlops.serving.ray_serve.serve")
    def test_shutdown(self, mock_serve, mock_ray):
        """Test cluster shutdown"""
        cluster = SharedRayCluster()
        cluster._ray_initialized = True
        cluster._serve_started = True

        cluster.shutdown()

        assert cluster._ray_initialized is False
        assert cluster._serve_started is False
        mock_serve.shutdown.assert_called_once()
        mock_ray.shutdown.assert_called_once()


class TestCreateSharedCluster:
    """Tests for cluster factory function"""

    @patch("mlops.serving.ray_serve.ray")
    @patch("mlops.serving.ray_serve.serve")
    def test_create_shared_cluster(self, mock_serve, mock_ray):
        """Test creating shared cluster"""
        cluster = create_shared_cluster()

        assert isinstance(cluster, SharedRayCluster)
        assert cluster.config is not None

    @patch("mlops.serving.ray_serve.ray")
    @patch("mlops.serving.ray_serve.serve")
    def test_create_shared_cluster_with_config(self, mock_serve, mock_ray):
        """Test creating shared cluster with custom config"""
        config = RayServeConfig(deployment_mode=DeploymentMode.CLUSTER)
        cluster = create_shared_cluster(config=config)

        assert cluster.config == config
        assert cluster.config.deployment_mode == DeploymentMode.CLUSTER
