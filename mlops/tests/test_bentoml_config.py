"""Tests for BentoML Configuration"""

from __future__ import annotations

import platform
import pytest

from mlops.serving.bentoml.config import (
    AppleSiliconOptimization,
    BentoMLConfig,
    ModelFramework,
    ServingBackend,
    get_bentoml_config,
)


class TestAppleSiliconOptimization:
    """Tests for Apple Silicon optimization configuration"""

    def test_default_initialization(self):
        """Test default AppleSiliconOptimization creation"""
        config = AppleSiliconOptimization()

        assert config.enable_mps is True
        assert config.enable_mlx is True
        assert config.enable_unified_memory is True
        assert config.enable_ane is False
        assert config.thermal_aware is True
        assert config.max_batch_size == 32
        assert config.prefetch_batches == 2

    def test_custom_initialization(self):
        """Test custom AppleSiliconOptimization configuration"""
        config = AppleSiliconOptimization(
            enable_mps=False,
            enable_mlx=True,
            max_batch_size=64,
        )

        assert config.enable_mps is False
        assert config.enable_mlx is True
        assert config.max_batch_size == 64


class TestBentoMLConfig:
    """Tests for BentoML configuration"""

    def test_default_initialization(self):
        """Test default BentoMLConfig creation"""
        config = BentoMLConfig()

        assert config.service_name == "mlx_model_service"
        assert config.service_version == "v1.0"
        assert config.project_name == "default"
        assert config.model_framework == ModelFramework.MLX
        assert config.serving_backend == ServingBackend.HYBRID
        assert config.workers == 1
        assert config.enable_apple_silicon_optimization is True

    def test_custom_initialization(self):
        """Test custom BentoMLConfig creation"""
        config = BentoMLConfig(
            service_name="custom_service",
            project_name="test_project",
            workers=4,
            model_framework=ModelFramework.PYTORCH,
            enable_apple_silicon_optimization=False,  # Disable to preserve worker count
        )

        assert config.service_name == "custom_service"
        assert config.project_name == "test_project"
        assert config.workers == 4
        assert config.model_framework == ModelFramework.PYTORCH

    def test_apple_silicon_optimization_applied(self):
        """Test Apple Silicon optimizations are applied"""
        config = BentoMLConfig(
            enable_apple_silicon_optimization=True,
            workers=10,  # High worker count
        )

        # Check that thermal-aware limits are applied on Apple Silicon
        system = platform.system()
        machine = platform.machine()
        is_apple_silicon = system == "Darwin" and machine == "arm64"

        if is_apple_silicon:
            # Workers should be limited for thermal management
            assert config.workers <= 2
        else:
            # No optimization on non-Apple Silicon
            assert config.workers == 10

    def test_ray_deployment_name_generation(self):
        """Test automatic Ray deployment name generation"""
        config = BentoMLConfig(
            service_name="test_service",
            project_name="test_project",
        )

        assert config.ray_deployment_name == "test_project_test_service"

    def test_to_bentoml_config(self):
        """Test conversion to BentoML configuration dict"""
        config = BentoMLConfig(
            service_name="test_service",
            workers=2,
            timeout=600,
        )

        bento_config = config.to_bentoml_config()

        assert bento_config["service"]["name"] == "test_service"
        assert bento_config["api_server"]["workers"] == 2
        assert bento_config["api_server"]["timeout"] == 600
        assert "apple_silicon" in bento_config

    def test_to_ray_serve_config(self):
        """Test conversion to Ray Serve configuration dict"""
        config = BentoMLConfig(
            service_name="test_service",
            project_name="test_project",
            workers=3,
            enable_apple_silicon_optimization=False,  # Disable to preserve worker count
        )

        ray_config = config.to_ray_serve_config()

        assert ray_config["deployment_name"] == "test_project_test_service"
        assert ray_config["num_replicas"] == 3
        assert "ray_actor_options" in ray_config

    def test_from_dict(self):
        """Test creation from dictionary"""
        config_dict = {
            "service_name": "dict_service",
            "project_name": "dict_project",
            "model_framework": "pytorch",
            "workers": 5,
            "enable_apple_silicon_optimization": False,  # Disable to preserve worker count
            "apple_silicon": {
                "enable_mps": False,
                "max_batch_size": 16,
            },
        }

        config = BentoMLConfig.from_dict(config_dict)

        assert config.service_name == "dict_service"
        assert config.project_name == "dict_project"
        assert config.model_framework == ModelFramework.PYTORCH
        assert config.workers == 5
        assert config.apple_silicon.enable_mps is False
        assert config.apple_silicon.max_batch_size == 16

    def test_detect_auto_configuration(self):
        """Test auto-detection of optimal configuration"""
        config = BentoMLConfig.detect(project_name="auto_project")

        assert config.project_name == "auto_project"

        # Check that framework is set based on hardware
        system = platform.system()
        machine = platform.machine()
        is_apple_silicon = system == "Darwin" and machine == "arm64"

        if is_apple_silicon:
            assert config.model_framework == ModelFramework.MLX
            assert config.serving_backend == ServingBackend.HYBRID
            assert config.enable_apple_silicon_optimization is True
        else:
            assert config.model_framework == ModelFramework.PYTORCH
            assert config.serving_backend == ServingBackend.BENTOML


class TestGetBentoMLConfig:
    """Tests for get_bentoml_config helper function"""

    def test_get_default_config(self):
        """Test getting default configuration"""
        config = get_bentoml_config()

        assert config.project_name == "default"
        assert isinstance(config, BentoMLConfig)

    def test_get_custom_project_config(self):
        """Test getting configuration for specific project"""
        config = get_bentoml_config(project_name="custom_project")

        assert config.project_name == "custom_project"

    def test_override_model_framework(self):
        """Test overriding model framework"""
        config = get_bentoml_config(
            project_name="test_project",
            model_framework=ModelFramework.PYTORCH,
        )

        assert config.project_name == "test_project"
        assert config.model_framework == ModelFramework.PYTORCH


class TestModelFrameworkEnum:
    """Tests for ModelFramework enum"""

    def test_enum_values(self):
        """Test ModelFramework enum values"""
        assert ModelFramework.MLX.value == "mlx"
        assert ModelFramework.PYTORCH.value == "pytorch"
        assert ModelFramework.TRANSFORMERS.value == "transformers"
        assert ModelFramework.ONNX.value == "onnx"

    def test_enum_from_string(self):
        """Test creating enum from string"""
        framework = ModelFramework("mlx")
        assert framework == ModelFramework.MLX


class TestServingBackendEnum:
    """Tests for ServingBackend enum"""

    def test_enum_values(self):
        """Test ServingBackend enum values"""
        assert ServingBackend.BENTOML.value == "bentoml"
        assert ServingBackend.RAY_SERVE.value == "ray_serve"
        assert ServingBackend.HYBRID.value == "hybrid"

    def test_enum_from_string(self):
        """Test creating enum from string"""
        backend = ServingBackend("hybrid")
        assert backend == ServingBackend.HYBRID
