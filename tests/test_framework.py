"""
Unit tests for DSPy Framework core functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from dspy_toolkit.exceptions import DSPyIntegrationError
from dspy_toolkit.framework import DSPyFramework
from dspy_toolkit.types import DSPyConfig


class TestDSPyFramework:
    """Test cases for DSPy Framework."""

    @pytest.fixture
    def test_config(self, tmp_path):
        """Test configuration for DSPy framework."""
        return DSPyConfig(
            model_provider="mlx", model_name="test-model", cache_dir=tmp_path / "cache"
        )

    @pytest.fixture
    def mock_components(self):
        """Mock framework components."""
        with (
            patch("dspy_toolkit.framework.SignatureRegistry") as mock_registry,
            patch("dspy_toolkit.framework.ModuleManager") as mock_manager,
            patch("dspy_toolkit.framework.OptimizerEngine") as mock_optimizer,
        ):

            yield {
                "registry": mock_registry,
                "manager": mock_manager,
                "optimizer": mock_optimizer,
            }

    def test_framework_initialization(self, test_config, mock_components):
        """Test framework initialization."""
        with patch("dspy_toolkit.framework.DSPyFramework.setup_llm_provider"):
            framework = DSPyFramework(test_config)

            assert framework.config == test_config
            assert framework.signature_registry is not None
            assert framework.module_manager is not None
            assert framework.optimizer_engine is not None

    def test_signature_registration(self, test_config, mock_components):
        """Test project signature registration."""
        with patch("dspy_toolkit.framework.DSPyFramework.setup_llm_provider"):
            framework = DSPyFramework(test_config)

            # Mock signature registry
            mock_registry = framework.signature_registry

            signatures = {"test_sig": Mock}
            framework.register_project_signatures("test_project", signatures)

            mock_registry.register_project.assert_called_once_with(
                "test_project", signatures
            )

    def test_framework_stats(self, test_config, mock_components):
        """Test framework statistics retrieval."""
        with patch("dspy_toolkit.framework.DSPyFramework.setup_llm_provider"):
            framework = DSPyFramework(test_config)

            # Mock component stats
            framework.signature_registry.get_registry_stats.return_value = {
                "signatures": 5
            }
            framework.module_manager.get_manager_stats.return_value = {"modules": 3}
            framework.optimizer_engine.get_optimizer_stats.return_value = {
                "optimizations": 2
            }

            stats = framework.get_framework_stats()

            assert "framework" in stats
            assert "signatures" in stats
            assert "modules" in stats
            assert "optimizer" in stats

    def test_health_check(self, test_config, mock_components):
        """Test framework health check."""
        with patch("dspy_toolkit.framework.DSPyFramework.setup_llm_provider"):
            framework = DSPyFramework(test_config)

            # Mock healthy components
            framework.signature_registry.get_registry_stats.return_value = {}
            framework.module_manager.get_manager_stats.return_value = {}
            framework.optimizer_engine.get_optimizer_stats.return_value = {}

            health = framework.health_check()

            assert "overall_status" in health
            assert "components" in health
            assert "issues" in health

    def test_cache_clearing(self, test_config, mock_components):
        """Test cache clearing functionality."""
        with patch("dspy_toolkit.framework.DSPyFramework.setup_llm_provider"):
            framework = DSPyFramework(test_config)

            framework.clear_all_caches()

            framework.signature_registry.clear_cache.assert_called_once()
            framework.module_manager.clear_cache.assert_called_once()

    @patch("dspy_toolkit.framework.MLXLLMProvider")
    def test_mlx_provider_setup(
        self, mock_provider_class, test_config, mock_components
    ):
        """Test MLX provider setup."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.hardware_info = Mock()
        mock_provider_class.return_value = mock_provider

        with (
            patch("dspy_toolkit.framework.setup_mlx_provider_for_dspy"),
            patch("dspy_toolkit.framework.dspy"),
        ):

            framework = DSPyFramework(test_config)

            assert framework.llm_provider == mock_provider
            assert framework.hardware_info == mock_provider.hardware_info

    def test_provider_setup_failure(self, test_config, mock_components):
        """Test provider setup failure handling."""
        with patch(
            "dspy_toolkit.framework.MLXLLMProvider",
            side_effect=Exception("Provider failed"),
        ):

            with pytest.raises(DSPyIntegrationError):
                DSPyFramework(test_config)


class TestFrameworkIntegration:
    """Integration tests for DSPy Framework."""

    def test_framework_with_real_components(self, tmp_path):
        """Test framework with real component instances."""
        config = DSPyConfig(
            model_provider="test",  # Use test provider to avoid MLX dependency
            model_name="test-model",
            cache_dir=tmp_path / "cache",
        )

        # This test would require mocking the provider setup
        # In a real scenario, we'd test with actual components
        pytest.skip(
            "Requires real component integration - run manually for full testing"
        )
