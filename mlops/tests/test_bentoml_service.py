"""Tests for BentoML Service Implementation"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from mlops.serving.bentoml.service import (
    BentoMLError,
    MLXBentoService,
    create_bentoml_service,
    create_lora_service,
)
from mlops.serving.bentoml.runner import MLXModelRunner


class TestBentoMLError:
    """Tests for BentoMLError exception"""

    def test_basic_error(self):
        """Test basic error creation"""
        error = BentoMLError("Test error")
        assert str(error) == "Test error"
        assert error.operation is None
        assert error.details == {}

    def test_error_with_operation(self):
        """Test error with operation"""
        error = BentoMLError("Test error", operation="predict")
        assert error.operation == "predict"

    def test_error_with_details(self):
        """Test error with details"""
        details = {"service_name": "test_service"}
        error = BentoMLError("Test error", details=details)
        assert error.details == details


class TestMLXBentoService:
    """Tests for MLXBentoService"""

    @pytest.fixture
    def mock_runner(self):
        """Create mock runner"""
        runner = Mock(spec=MLXModelRunner)
        runner.is_loaded = False
        runner.load_model = Mock()
        runner.predict = Mock(return_value={"prediction": "test"})
        runner.unload_model = Mock()
        runner.get_memory_usage = Mock(
            return_value={
                "total_mb": 100.0,
                "mlx_available": True,
            }
        )
        return runner

    def test_initialization(self, mock_runner):
        """Test service initialization"""
        service = MLXBentoService(
            runner=mock_runner,
            service_name="test_service",
            project_name="test_project",
        )

        assert service.runner == mock_runner
        assert service.service_name == "test_service"
        assert service.project_name == "test_project"
        assert service._is_ready is False

    def test_load_success(self, mock_runner):
        """Test successful model loading"""
        service = MLXBentoService(mock_runner)
        service.load()

        assert service._is_ready is True
        mock_runner.load_model.assert_called_once()

    def test_load_failure(self, mock_runner):
        """Test model loading failure"""
        mock_runner.load_model.side_effect = RuntimeError("Load failed")
        service = MLXBentoService(mock_runner)

        with pytest.raises(BentoMLError) as exc_info:
            service.load()

        assert "Service initialization failed" in str(exc_info.value)
        assert exc_info.value.operation == "load"

    def test_predict_success(self, mock_runner):
        """Test successful prediction"""
        service = MLXBentoService(
            mock_runner,
            service_name="test_service",
            project_name="test_project",
        )
        service._is_ready = True
        mock_runner.is_loaded = True

        result = service.predict({"input": "test"})

        assert "prediction" in result
        assert result["service_name"] == "test_service"
        assert result["project_name"] == "test_project"
        mock_runner.predict.assert_called_once_with({"input": "test"})

    def test_predict_not_ready(self, mock_runner):
        """Test prediction when service not ready"""
        service = MLXBentoService(mock_runner)

        with pytest.raises(BentoMLError) as exc_info:
            service.predict({"input": "test"})

        assert "Service not ready" in str(exc_info.value)
        assert exc_info.value.operation == "predict"

    def test_predict_failure(self, mock_runner):
        """Test prediction failure"""
        mock_runner.predict.side_effect = RuntimeError("Prediction failed")
        service = MLXBentoService(mock_runner)
        service._is_ready = True
        mock_runner.is_loaded = True

        with pytest.raises(BentoMLError) as exc_info:
            service.predict({"input": "test"})

        assert "Prediction failed" in str(exc_info.value)
        assert exc_info.value.operation == "predict"

    def test_health_check_healthy(self, mock_runner):
        """Test health check when service is healthy"""
        service = MLXBentoService(
            mock_runner,
            service_name="test_service",
            project_name="test_project",
        )
        service._is_ready = True
        mock_runner.is_loaded = True

        health = service.health_check()

        assert health["status"] == "healthy"
        assert health["service_name"] == "test_service"
        assert health["project_name"] == "test_project"
        assert health["model_loaded"] is True
        assert health["memory_usage_mb"] == 100.0
        assert health["mlx_available"] is True

    def test_health_check_not_ready(self, mock_runner):
        """Test health check when service is not ready"""
        service = MLXBentoService(mock_runner)

        health = service.health_check()

        assert health["status"] == "not_ready"
        assert health["model_loaded"] is False

    def test_health_check_failure(self, mock_runner):
        """Test health check when it fails"""
        mock_runner.get_memory_usage.side_effect = RuntimeError("Memory check failed")
        service = MLXBentoService(mock_runner, service_name="test_service")

        health = service.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health

    def test_unload_success(self, mock_runner):
        """Test successful unload"""
        service = MLXBentoService(mock_runner)
        service._is_ready = True
        service.unload()

        assert service._is_ready is False
        mock_runner.unload_model.assert_called_once()

    def test_unload_failure(self, mock_runner):
        """Test unload failure"""
        mock_runner.unload_model.side_effect = RuntimeError("Unload failed")
        service = MLXBentoService(mock_runner)
        service._is_ready = True

        with pytest.raises(BentoMLError) as exc_info:
            service.unload()

        assert "Service unload failed" in str(exc_info.value)
        assert exc_info.value.operation == "unload"

    def test_is_ready_property(self, mock_runner):
        """Test is_ready property"""
        service = MLXBentoService(mock_runner)

        assert service.is_ready is False

        service._is_ready = True
        assert service.is_ready is True


class TestCreateBentoMLService:
    """Tests for create_bentoml_service factory function"""

    @patch("mlops.serving.bentoml.service.create_runner")
    @patch("mlops.serving.bentoml.service.bentoml")
    def test_create_service_basic(self, mock_bentoml, mock_create_runner):
        """Test basic service creation"""
        mock_runner = Mock()
        mock_create_runner.return_value = mock_runner

        mock_bento_runner = Mock()
        mock_bentoml.Runner.return_value = mock_bento_runner

        mock_service = Mock()
        mock_bentoml.Service.return_value = mock_service

        service = create_bentoml_service(
            model_path="test/path",
            service_name="test_service",
            project_name="test_project",
        )

        mock_create_runner.assert_called_once()
        mock_bentoml.Runner.assert_called_once()
        mock_bentoml.Service.assert_called_once_with(
            "test_service",
            runners=[mock_bento_runner],
        )

    @patch("mlops.serving.bentoml.service.create_runner")
    @patch("mlops.serving.bentoml.service.bentoml")
    def test_create_service_with_model_type(self, mock_bentoml, mock_create_runner):
        """Test service creation with specific model type"""
        mock_runner = Mock()
        mock_create_runner.return_value = mock_runner

        mock_service = Mock()
        mock_bentoml.Service.return_value = mock_service
        mock_bentoml.Runner.return_value = Mock()

        service = create_bentoml_service(
            model_path="test/path",
            model_type="lora",
        )

        mock_create_runner.assert_called_once_with(
            "test/path",
            model_type="lora",
        )


class TestCreateLoRAService:
    """Tests for create_lora_service helper function"""

    @patch("mlops.serving.bentoml.service.create_bentoml_service")
    def test_create_lora_service(self, mock_create_service):
        """Test LoRA service creation"""
        mock_service = Mock()
        mock_create_service.return_value = mock_service

        service = create_lora_service(
            model_path="test/lora/path",
            service_name="lora_service",
            project_name="lora_project",
        )

        mock_create_service.assert_called_once_with(
            model_path="test/lora/path",
            service_name="lora_service",
            project_name="lora_project",
            model_type="lora",
        )

    @patch("mlops.serving.bentoml.service.create_bentoml_service")
    def test_create_lora_service_defaults(self, mock_create_service):
        """Test LoRA service creation with defaults"""
        mock_service = Mock()
        mock_create_service.return_value = mock_service

        service = create_lora_service(model_path="test/path")

        mock_create_service.assert_called_once_with(
            model_path="test/path",
            service_name="lora_service",
            project_name="lora-finetuning-mlx",
            model_type="lora",
        )
