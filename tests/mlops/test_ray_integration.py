"""Integration Tests for Ray Serve

This module tests end-to-end workflows with mocked Ray cluster.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

# Skip all tests if Ray not available
pytest.importorskip("ray", reason="Ray not available - install with: uv add 'ray[serve]'")

from mlops.config.ray_config import RayServeConfig, DeploymentMode, ScalingMode
from mlops.serving.ray_serve import SharedRayCluster
from mlops.serving.model_wrapper import MLXModelWrapper, create_model_wrapper
from mlops.serving.scaling_manager import ScalingManager, ScalingConfig


@pytest.fixture
def mock_ray_environment():
    """Fixture to mock Ray environment"""
    with patch("mlops.serving.ray_serve.ray") as mock_ray, \
         patch("mlops.serving.ray_serve.serve") as mock_serve:

        # Setup mock Ray cluster resources
        mock_ray.cluster_resources.return_value = {
            "CPU": 8.0,
            "memory": 16 * 1024 * 1024 * 1024,
        }
        mock_ray.available_resources.return_value = {
            "CPU": 4.0,
            "memory": 8 * 1024 * 1024 * 1024,
        }

        # Setup mock deployment
        mock_deployment = Mock()
        mock_deployment.options = Mock(return_value=mock_deployment)
        mock_deployment.deploy = Mock()
        mock_serve.get_deployment.return_value = mock_deployment
        mock_serve.deployment = lambda **kwargs: lambda cls: cls

        yield {
            "ray": mock_ray,
            "serve": mock_serve,
            "deployment": mock_deployment,
        }


class TestEndToEndDeployment:
    """End-to-end deployment tests"""

    def test_full_deployment_workflow(self, mock_ray_environment):
        """Test complete deployment workflow"""
        # 1. Create cluster with configuration
        config = RayServeConfig(
            deployment_mode=DeploymentMode.LOCAL,
            scaling_mode=ScalingMode.AUTO,
            min_replicas=1,
            max_replicas=4,
        )

        cluster = SharedRayCluster(config=config)

        # 2. Initialize cluster
        cluster.initialize_cluster()
        assert cluster._ray_initialized is True

        # 3. Start Ray Serve
        cluster.start_serve()
        assert cluster._serve_started is True

        # 4. Create model wrapper
        model_wrapper = Mock()
        model_wrapper.load_model = Mock()
        model_wrapper.predict = Mock(return_value={"result": "success"})

        # 5. Deploy model
        deployment_name = cluster.deploy_project_model(
            project_name="test-project",
            model_name="test-model",
            model_wrapper=model_wrapper,
        )

        assert deployment_name == "test-project_test-model"
        assert "test-project" in cluster.model_deployments

        # 6. Verify deployment
        deployments = cluster.list_deployments()
        assert "test-project" in deployments
        assert len(deployments["test-project"]) == 1

        # 7. Get resource usage
        usage = cluster.get_cluster_resource_usage()
        assert usage["initialized"] is True
        assert usage["total_deployments"] == 1

        # 8. Shutdown
        cluster.shutdown()
        assert cluster._ray_initialized is False
        assert cluster._serve_started is False

    def test_multi_project_deployment(self, mock_ray_environment):
        """Test deploying models from multiple projects"""
        cluster = SharedRayCluster()
        cluster.initialize_cluster()
        cluster.start_serve()

        # Deploy from multiple projects
        projects = ["proj1", "proj2", "proj3"]
        for proj in projects:
            model_wrapper = Mock()
            model_wrapper.load_model = Mock()

            cluster.deploy_project_model(
                project_name=proj,
                model_name=f"{proj}-model",
                model_wrapper=model_wrapper,
            )

        # Verify all deployed
        deployments = cluster.list_deployments()
        assert len(deployments) == 3
        for proj in projects:
            assert proj in deployments

        # Verify resource tracking
        usage = cluster.get_cluster_resource_usage()
        assert usage["total_deployments"] == 3
        assert set(usage["projects"]) == set(projects)


class TestScalingIntegration:
    """Integration tests for auto-scaling"""

    def test_auto_scaling_workflow(self, mock_ray_environment):
        """Test auto-scaling in response to load"""
        # Setup cluster and scaling manager
        cluster = SharedRayCluster()
        cluster.initialize_cluster()
        cluster.start_serve()

        scaling_config = ScalingConfig(
            min_replicas=1,
            max_replicas=5,
            target_requests_per_replica=10,
            cooldown_period_s=0,  # Disable cooldown for testing
        )
        scaling_manager = ScalingManager(
            cluster=cluster,
            scaling_config=scaling_config,
        )

        # Deploy model
        model_wrapper = Mock()
        cluster.deploy_project_model(
            project_name="test-project",
            model_name="test-model",
            model_wrapper=model_wrapper,
        )

        # Simulate high load
        high_load_metrics = {
            "num_replicas": 1,
            "ongoing_requests": 20,  # 20 req/replica, should scale up
            "cpu_utilization_pct": 80.0,
            "memory_utilization_pct": 70.0,
        }

        # Evaluate scaling
        metrics = scaling_manager.evaluate_scaling(
            "test-project",
            "test-model",
            high_load_metrics,
        )

        assert metrics.target_replicas > metrics.current_replicas

        # Apply scaling
        result = scaling_manager.apply_scaling(
            "test-project",
            "test-model",
            metrics,
        )

        assert result is True

        # Verify scaling was applied
        mock_ray_environment["serve"].get_deployment.assert_called()

    def test_thermal_aware_scaling(self, mock_ray_environment):
        """Test scaling respects thermal constraints"""
        cluster = SharedRayCluster()
        cluster.initialize_cluster()
        cluster.start_serve()

        scaling_config = ScalingConfig(
            thermal_aware=True,
            cooldown_period_s=0,
        )

        with patch("mlops.serving.scaling_manager.ThermalMonitor") as mock_thermal_class:
            # Mock critical thermal state
            mock_thermal = Mock()
            mock_thermal.get_thermal_state.return_value = "critical"
            mock_thermal_class.return_value = mock_thermal

            scaling_manager = ScalingManager(
                cluster=cluster,
                scaling_config=scaling_config,
            )

            # Deploy model
            model_wrapper = Mock()
            cluster.deploy_project_model(
                project_name="test-project",
                model_name="test-model",
                model_wrapper=model_wrapper,
            )

            # High load but critical thermal state
            metrics_data = {
                "num_replicas": 3,
                "ongoing_requests": 50,
                "cpu_utilization_pct": 90.0,
                "memory_utilization_pct": 80.0,
            }

            metrics = scaling_manager.evaluate_scaling(
                "test-project",
                "test-model",
                metrics_data,
            )

            # Should scale down due to critical thermal state
            assert metrics.target_replicas < metrics.current_replicas


class TestModelWrapperIntegration:
    """Integration tests for model wrappers"""

    @patch("mlops.serving.model_wrapper.mlx")
    def test_mlx_model_wrapper_lifecycle(self, mock_mlx):
        """Test MLX model wrapper load, predict, unload"""
        # Setup mock MLX
        mock_mlx.core = Mock()
        mock_mlx.core.mx = Mock()
        mock_mlx.metal = Mock()
        mock_mlx.metal.clear_cache = Mock()

        # Create wrapper
        wrapper = MLXModelWrapper(
            model_path=Path("/fake/model/path"),
            use_mps=True,
            use_unified_memory=True,
        )

        # Mock model loading
        with patch.object(wrapper, '_load_mlx_model', return_value={"weights": "data"}):
            wrapper.load_model()

        assert wrapper.is_loaded is True

        # Mock prediction
        with patch.object(wrapper, '_mlx_predict', return_value={"output": "result"}):
            result = wrapper.predict({"input": "data"})

        assert "output" in result

        # Unload
        wrapper.unload_model()
        assert wrapper.is_loaded is False
        mock_mlx.metal.clear_cache.assert_called_once()

    def test_model_wrapper_factory(self):
        """Test model wrapper factory function"""
        # MLX wrapper
        mlx_wrapper = create_model_wrapper(
            model_path=Path("/fake/model"),
            model_type="mlx",
        )

        assert isinstance(mlx_wrapper, MLXModelWrapper)

        # PyTorch wrapper
        pytorch_wrapper = create_model_wrapper(
            model_path=Path("/fake/model"),
            model_type="pytorch",
        )

        from mlops.serving.model_wrapper import PyTorchModelWrapper
        assert isinstance(pytorch_wrapper, PyTorchModelWrapper)


class TestConfigurationIntegration:
    """Integration tests for configuration"""

    def test_apple_silicon_optimization_flow(self, mock_ray_environment):
        """Test Apple Silicon optimization throughout the stack"""
        # Create config with Apple Silicon
        from mlops.config.ray_config import AppleSiliconConfig

        apple_silicon = AppleSiliconConfig(
            chip_type="M2",
            cores=8,
            memory_gb=24.0,
            thermal_aware=True,
        )

        config = RayServeConfig(
            apple_silicon=apple_silicon,
            enable_apple_silicon_optimization=True,
            scaling_mode=ScalingMode.AUTO,
        )

        # Verify optimizations applied
        assert config.num_cpus == 8
        assert config.max_replicas <= apple_silicon.max_replicas
        assert config.object_store_memory is not None

        # Create cluster with optimized config
        cluster = SharedRayCluster(config=config)
        cluster.initialize_cluster()

        # Verify Ray init config includes Apple Silicon settings
        ray_config = config.to_ray_init_config()
        assert ray_config["num_cpus"] == 8

        # Verify serve config includes metadata
        serve_config = config.to_serve_config()
        assert "apple_silicon_metadata" in serve_config

    def test_multi_environment_configuration(self, mock_ray_environment):
        """Test configuration for different deployment environments"""
        # Local development
        local_config = RayServeConfig.for_deployment_mode(DeploymentMode.LOCAL)
        assert local_config.num_replicas == 1
        assert local_config.max_replicas == 2

        # Cluster deployment
        cluster_config = RayServeConfig.for_deployment_mode(DeploymentMode.CLUSTER)
        assert cluster_config.num_replicas == 2
        assert cluster_config.max_replicas == 10
        assert cluster_config.ray_address == "auto"

        # Verify both can be used
        local_cluster = SharedRayCluster(config=local_config)
        cluster_cluster = SharedRayCluster(config=cluster_config)

        assert local_cluster.config.deployment_mode == DeploymentMode.LOCAL
        assert cluster_cluster.config.deployment_mode == DeploymentMode.CLUSTER
