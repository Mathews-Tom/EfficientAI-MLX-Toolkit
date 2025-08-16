"""
Unit tests for DSPy deployment components.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dspy_toolkit.deployment import (
    DSPyFastAPIApp,
    DSPyMonitor,
    DSPyStreamingEndpoint,
    PerformanceMetrics,
    create_dspy_app,
    create_streaming_endpoint,
)
from dspy_toolkit.deployment.monitoring import AlertRule, SystemMetrics
from dspy_toolkit.types import DSPyConfig


class TestDSPyFastAPIApp:
    """Test cases for DSPy FastAPI application."""

    @pytest.fixture
    def mock_framework(self):
        """Mock DSPy framework for testing."""
        framework = Mock()
        framework.health_check.return_value = {
            "overall_status": "healthy",
            "components": {},
            "issues": [],
        }
        framework.module_manager.list_modules.return_value = ["test_module"]
        framework.signature_registry.list_project_signatures.return_value = [
            "test_signature"
        ]
        framework.get_framework_stats.return_value = {"test": "stats"}
        return framework

    @pytest.fixture
    def test_config(self, tmp_path):
        """Test configuration."""
        return DSPyConfig(
            model_provider="test", model_name="test-model", cache_dir=tmp_path / "cache"
        )

    @patch("dspy_toolkit.deployment.fastapi_integration.FASTAPI_AVAILABLE", True)
    @patch("dspy_toolkit.deployment.fastapi_integration.FastAPI")
    def test_fastapi_app_creation(self, mock_fastapi, mock_framework):
        """Test FastAPI app creation."""
        mock_app = Mock()
        mock_fastapi.return_value = mock_app

        dspy_app = DSPyFastAPIApp(mock_framework)

        assert dspy_app.framework == mock_framework
        assert dspy_app.monitor is not None
        mock_fastapi.assert_called_once()

    @patch("dspy_toolkit.deployment.fastapi_integration.FASTAPI_AVAILABLE", False)
    def test_fastapi_unavailable_error(self, mock_framework):
        """Test error when FastAPI is unavailable."""
        from dspy_toolkit.exceptions import DSPyIntegrationError

        with pytest.raises(DSPyIntegrationError, match="FastAPI is not available"):
            DSPyFastAPIApp(mock_framework)

    def test_create_dspy_app(self, test_config):
        """Test DSPy app creation function."""
        with (
            patch(
                "dspy_toolkit.deployment.fastapi_integration.DSPyFramework"
            ) as mock_framework_class,
            patch(
                "dspy_toolkit.deployment.fastapi_integration.DSPyFastAPIApp"
            ) as mock_app_class,
        ):

            mock_framework = Mock()
            mock_framework_class.return_value = mock_framework
            mock_app = Mock()
            mock_app_class.return_value = mock_app

            result = create_dspy_app(test_config)

            mock_framework_class.assert_called_once_with(test_config)
            mock_app_class.assert_called_once_with(mock_framework)
            assert result == mock_app


class TestDSPyMonitor:
    """Test cases for DSPy monitoring."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create test monitor."""
        return DSPyMonitor(
            export_path=tmp_path / "monitoring",
            enable_system_monitoring=False,  # Disable for testing
        )

    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.metrics_collector is not None
        assert monitor.alert_manager is not None
        assert monitor.export_path.exists()

    def test_record_request(self, monitor):
        """Test request recording."""
        metrics = PerformanceMetrics(
            execution_time=1.5,
            input_tokens=100,
            output_tokens=200,
            memory_usage=1024,
            timestamp=1234567890,
            success=True,
        )

        monitor.record_request("test_project", "test_module", metrics)

        assert len(monitor.metrics_collector.performance_metrics) == 1
        assert monitor.metrics_collector.request_counts["success"] == 1

    @pytest.mark.asyncio
    async def test_get_metrics(self, monitor):
        """Test metrics retrieval."""
        # Add some test data
        metrics = PerformanceMetrics(
            execution_time=1.0,
            input_tokens=50,
            output_tokens=100,
            memory_usage=512,
            timestamp=1234567890,
            success=True,
        )
        monitor.record_request("test", "test", metrics)

        result = await monitor.get_metrics()

        assert "performance" in result
        assert "system" in result
        assert "errors" in result
        assert "alerts" in result

    def test_alert_rule_management(self, monitor):
        """Test alert rule management."""
        rule = AlertRule(
            name="test_rule",
            metric_name="test_metric",
            threshold=10.0,
            comparison="gt",
            window_minutes=5,
        )

        monitor.add_alert_rule(rule)

        assert len(monitor.alert_manager.alert_rules) > 0
        assert any(r.name == "test_rule" for r in monitor.alert_manager.alert_rules)

    @pytest.mark.asyncio
    async def test_export_metrics(self, monitor, tmp_path):
        """Test metrics export."""
        # Add some test data
        metrics = PerformanceMetrics(
            execution_time=1.0,
            input_tokens=50,
            output_tokens=100,
            memory_usage=512,
            timestamp=1234567890,
            success=True,
        )
        monitor.record_request("test", "test", metrics)

        await monitor.export_metrics("test_export.json")

        export_file = monitor.export_path / "test_export.json"
        assert export_file.exists()


class TestDSPyStreamingEndpoint:
    """Test cases for DSPy streaming endpoint."""

    @pytest.fixture
    def mock_framework(self):
        """Mock framework for streaming tests."""
        framework = Mock()
        mock_module = Mock()
        framework.get_project_module.return_value = mock_module
        return framework

    @patch("dspy_toolkit.deployment.streaming.STREAMING_AVAILABLE", True)
    def test_streaming_endpoint_creation(self, mock_framework):
        """Test streaming endpoint creation."""
        endpoint = DSPyStreamingEndpoint(mock_framework)

        assert endpoint.framework == mock_framework
        assert endpoint.config is not None

    @patch("dspy_toolkit.deployment.streaming.STREAMING_AVAILABLE", False)
    def test_streaming_unavailable_error(self, mock_framework):
        """Test error when streaming dependencies are unavailable."""
        from dspy_toolkit.exceptions import DSPyIntegrationError

        with pytest.raises(
            DSPyIntegrationError, match="Streaming dependencies not available"
        ):
            DSPyStreamingEndpoint(mock_framework)

    def test_create_streaming_endpoint(self):
        """Test streaming endpoint creation function."""
        mock_framework = Mock()

        with patch(
            "dspy_toolkit.deployment.streaming.DSPyStreamingEndpoint"
        ) as mock_endpoint_class:
            mock_endpoint = Mock()
            mock_endpoint_class.return_value = mock_endpoint

            result = create_streaming_endpoint(mock_framework)

            mock_endpoint_class.assert_called_once_with(mock_framework, None)
            assert result == mock_endpoint


class TestPerformanceMetrics:
    """Test cases for performance metrics."""

    def test_performance_metrics_creation(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(
            execution_time=1.5,
            input_tokens=100,
            output_tokens=200,
            memory_usage=1024,
            timestamp=1234567890,
            success=True,
        )

        assert metrics.execution_time == 1.5
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 200
        assert metrics.memory_usage == 1024
        assert metrics.timestamp == 1234567890
        assert metrics.success == True
        assert metrics.error_message is None

    def test_performance_metrics_with_error(self):
        """Test performance metrics with error."""
        metrics = PerformanceMetrics(
            execution_time=0.5,
            input_tokens=50,
            output_tokens=0,
            memory_usage=512,
            timestamp=1234567890,
            success=False,
            error_message="Test error",
        )

        assert metrics.success == False
        assert metrics.error_message == "Test error"


class TestSystemMetrics:
    """Test cases for system metrics."""

    def test_system_metrics_creation(self):
        """Test system metrics creation."""
        metrics = SystemMetrics(
            cpu_usage=0.75,
            memory_usage=0.60,
            disk_usage=0.45,
            network_io=1024.0,
            timestamp=1234567890,
        )

        assert metrics.cpu_usage == 0.75
        assert metrics.memory_usage == 0.60
        assert metrics.disk_usage == 0.45
        assert metrics.network_io == 1024.0
        assert metrics.timestamp == 1234567890


class TestAlertRule:
    """Test cases for alert rules."""

    def test_alert_rule_creation(self):
        """Test alert rule creation."""
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu_usage",
            threshold=0.8,
            comparison="gt",
            window_minutes=5,
            enabled=True,
        )

        assert rule.name == "high_cpu"
        assert rule.metric_name == "cpu_usage"
        assert rule.threshold == 0.8
        assert rule.comparison == "gt"
        assert rule.window_minutes == 5
        assert rule.enabled == True


@pytest.mark.integration
class TestDeploymentIntegration:
    """Integration tests for deployment components."""

    @pytest.mark.asyncio
    async def test_end_to_end_monitoring(self, tmp_path):
        """Test end-to-end monitoring workflow."""
        monitor = DSPyMonitor(
            export_path=tmp_path / "monitoring", enable_system_monitoring=False
        )

        # Record some metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                execution_time=1.0 + i * 0.1,
                input_tokens=100 + i * 10,
                output_tokens=200 + i * 20,
                memory_usage=1024 + i * 100,
                timestamp=1234567890 + i,
                success=i < 4,  # Last one fails
            )
            monitor.record_request("test_project", "test_module", metrics)

        # Get metrics summary
        summary = await monitor.get_metrics(60)

        assert summary["performance"]["total_requests"] == 5
        assert summary["performance"]["success_rate"] == 0.8  # 4/5 successful
        assert summary["errors"]["total_errors"] == 1

        # Export metrics
        await monitor.export_metrics("integration_test.json")

        export_file = monitor.export_path / "integration_test.json"
        assert export_file.exists()

        # Cleanup
        monitor.cleanup()

    def test_fastapi_integration_with_real_components(self):
        """Test FastAPI integration with real components."""
        # This would require actual FastAPI and DSPy integration
        # Skip for now as it requires complex setup
        pytest.skip("Requires full FastAPI integration - run manually for full testing")
