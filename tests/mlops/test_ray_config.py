"""Tests for Ray Serve Configuration

This module tests the Ray Serve configuration with Apple Silicon detection
and optimization settings.
"""

import pytest
from unittest.mock import patch, MagicMock

from mlops.config.ray_config import (
    RayServeConfig,
    AppleSiliconConfig,
    DeploymentMode,
    ScalingMode,
    get_ray_serve_config,
)


class TestAppleSiliconConfig:
    """Tests for Apple Silicon configuration detection"""

    def test_detect_non_apple_silicon(self):
        """Test detection on non-Apple Silicon hardware"""
        with patch("platform.processor", return_value="x86_64"):
            config = AppleSiliconConfig.detect()

            assert config.chip_type is None
            assert config.thermal_aware is False
            assert config.unified_memory is False
            assert config.mps_available is False

    def test_detect_apple_silicon_m1(self):
        """Test detection of M1 chip"""
        with patch("platform.processor", return_value="arm"), \
             patch("platform.system", return_value="Darwin"), \
             patch("subprocess.run") as mock_run:

            mock_run.return_value = MagicMock(
                stdout="Apple M1 Pro",
                returncode=0
            )

            config = AppleSiliconConfig.detect()

            assert config.chip_type == "M1"
            assert config.cores == 8
            assert config.memory_gb == 16.0
            assert config.thermal_aware is True
            assert config.unified_memory is True
            assert config.mps_available is True
            assert config.max_replicas == 4

    def test_detect_apple_silicon_m2(self):
        """Test detection of M2 chip"""
        with patch("platform.processor", return_value="arm"), \
             patch("platform.system", return_value="Darwin"), \
             patch("subprocess.run") as mock_run:

            mock_run.return_value = MagicMock(
                stdout="Apple M2 Max",
                returncode=0
            )

            config = AppleSiliconConfig.detect()

            assert config.chip_type == "M2"
            assert config.cores == 8
            assert config.memory_gb == 24.0
            assert config.max_replicas == 6

    def test_detect_apple_silicon_m3(self):
        """Test detection of M3 chip"""
        with patch("platform.processor", return_value="arm"), \
             patch("platform.system", return_value="Darwin"), \
             patch("subprocess.run") as mock_run:

            mock_run.return_value = MagicMock(
                stdout="Apple M3 Ultra",
                returncode=0
            )

            config = AppleSiliconConfig.detect()

            assert config.chip_type == "M3"
            assert config.cores == 12
            assert config.memory_gb == 36.0
            assert config.max_replicas == 8


class TestRayServeConfig:
    """Tests for Ray Serve configuration"""

    def test_default_config(self):
        """Test default configuration"""
        config = RayServeConfig()

        assert config.deployment_mode == DeploymentMode.LOCAL
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.scaling_mode == ScalingMode.AUTO
        assert config.num_replicas == 1
        assert config.enable_apple_silicon_optimization is True

    def test_config_with_apple_silicon(self):
        """Test configuration with Apple Silicon"""
        apple_silicon = AppleSiliconConfig(
            chip_type="M1",
            cores=8,
            memory_gb=16.0,
            thermal_aware=True,
        )

        config = RayServeConfig(
            apple_silicon=apple_silicon,
            enable_apple_silicon_optimization=True
        )

        # Should auto-configure based on Apple Silicon
        assert config.num_cpus == 8
        assert config.max_replicas <= 4  # Thermal aware limit
        assert config.object_store_memory is not None

    def test_to_ray_init_config(self):
        """Test conversion to Ray init configuration"""
        config = RayServeConfig(num_cpus=4, num_gpus=0)
        ray_config = config.to_ray_init_config()

        assert "num_cpus" in ray_config
        assert ray_config["num_cpus"] == 4
        assert "logging_level" in ray_config

    def test_to_serve_config(self):
        """Test conversion to Ray Serve configuration"""
        config = RayServeConfig()
        serve_config = config.to_serve_config()

        assert "http_options" in serve_config

    def test_to_deployment_config(self):
        """Test conversion to deployment configuration"""
        config = RayServeConfig(
            num_replicas=2,
            scaling_mode=ScalingMode.AUTO,
            min_replicas=1,
            max_replicas=4,
        )

        deployment_config = config.to_deployment_config()

        assert deployment_config["num_replicas"] == 2
        assert "autoscaling_config" in deployment_config
        assert deployment_config["autoscaling_config"]["min_replicas"] == 1
        assert deployment_config["autoscaling_config"]["max_replicas"] == 4

    def test_for_deployment_mode_local(self):
        """Test configuration for local deployment"""
        config = RayServeConfig.for_deployment_mode(DeploymentMode.LOCAL)

        assert config.deployment_mode == DeploymentMode.LOCAL
        assert config.ray_address is None
        assert config.num_replicas == 1
        assert config.max_replicas == 2

    def test_for_deployment_mode_cluster(self):
        """Test configuration for cluster deployment"""
        config = RayServeConfig.for_deployment_mode(DeploymentMode.CLUSTER)

        assert config.deployment_mode == DeploymentMode.CLUSTER
        assert config.ray_address == "auto"
        assert config.num_replicas == 2
        assert config.max_replicas == 10

    def test_thermal_aware_optimization(self):
        """Test thermal-aware replica limiting"""
        apple_silicon = AppleSiliconConfig(
            chip_type="M1",
            cores=8,
            thermal_aware=True,
        )

        config = RayServeConfig(
            apple_silicon=apple_silicon,
            enable_apple_silicon_optimization=True,
            max_replicas=20,  # Request high replica count
        )

        # Should be limited by thermal awareness
        assert config.max_replicas <= 4  # cores // 2

    def test_memory_optimization(self):
        """Test memory per replica calculation"""
        apple_silicon = AppleSiliconConfig(
            chip_type="M2",
            cores=8,
            memory_gb=24.0,
        )

        config = RayServeConfig(
            apple_silicon=apple_silicon,
            enable_apple_silicon_optimization=True,
            max_replicas=4,
        )

        # Should reserve memory appropriately
        # (24GB - 4GB system) / 4 replicas = 5GB per replica
        expected_max_memory_mb = (24 * 1024 - 4096) // 4
        assert config.memory_per_replica_mb <= expected_max_memory_mb


class TestGetRayServeConfig:
    """Tests for config factory function"""

    def test_get_config_default(self):
        """Test getting default config"""
        with patch.dict("os.environ", {"RAY_SERVE_MODE": "local"}):
            config = get_ray_serve_config()

            assert config.deployment_mode == DeploymentMode.LOCAL
            assert config.project_name == "default"

    def test_get_config_with_project_name(self):
        """Test getting config with project name"""
        config = get_ray_serve_config(project_name="my-project")

        assert config.project_name == "my-project"

    def test_get_config_cluster_mode(self):
        """Test getting config for cluster mode"""
        config = get_ray_serve_config(deployment_mode=DeploymentMode.CLUSTER)

        assert config.deployment_mode == DeploymentMode.CLUSTER
        assert config.ray_address == "auto"
