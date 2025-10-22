"""Tests for BentoML Model Packager"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mlops.serving.bentoml.packager import (
    ModelPackager,
    PackageConfig,
    package_model,
)
from mlops.serving.bentoml.config import BentoMLConfig, ModelFramework


class TestPackageConfig:
    """Tests for PackageConfig"""

    def test_default_initialization(self, tmp_path):
        """Test default PackageConfig creation"""
        config = PackageConfig(
            model_path=tmp_path,
            model_name="test_model",
        )

        assert config.model_path == tmp_path
        assert config.model_name == "test_model"
        assert config.model_version is None
        assert config.model_framework == ModelFramework.MLX
        assert config.service_name == "mlx_model_service"
        assert config.project_name == "default"
        assert "project" in config.labels
        assert config.metadata["framework"] == "mlx"
        assert config.metadata["apple_silicon_optimized"] is True

    def test_custom_initialization(self, tmp_path):
        """Test custom PackageConfig creation"""
        config = PackageConfig(
            model_path=tmp_path,
            model_name="custom_model",
            model_version="v1.0",
            service_name="custom_service",
            project_name="custom_project",
            labels={"env": "prod"},
        )

        assert config.model_name == "custom_model"
        assert config.model_version == "v1.0"
        assert config.service_name == "custom_service"
        assert config.project_name == "custom_project"
        assert config.labels["env"] == "prod"
        assert config.labels["project"] == "custom_project"


class TestModelPackager:
    """Tests for ModelPackager"""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory with files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir)

            # Create model files
            (model_path / "model.safetensors").write_text("model weights")
            (model_path / "config.json").write_text('{"model": "config"}')
            (model_path / "tokenizer.json").write_text('{"tokenizer": "config"}')

            # Create file to be excluded
            (model_path / "test.pyc").write_text("compiled")

            yield model_path

    @pytest.fixture
    def packager_config(self, temp_model_dir):
        """Create PackageConfig for testing"""
        return PackageConfig(
            model_path=temp_model_dir,
            model_name="test_model",
            service_name="test_service",
            project_name="test_project",
        )

    def test_initialization(self, packager_config):
        """Test ModelPackager initialization"""
        packager = ModelPackager(packager_config)

        assert packager.config == packager_config
        assert packager.bento_tag is None

    def test_validate_model_path_success(self, temp_model_dir):
        """Test successful model path validation"""
        config = PackageConfig(
            model_path=temp_model_dir,
            model_name="test_model",
        )
        packager = ModelPackager(config)

        assert packager.validate_model_path() is True

    def test_validate_model_path_not_exists(self, tmp_path):
        """Test validation with non-existent path"""
        nonexistent_path = tmp_path / "nonexistent"
        config = PackageConfig(
            model_path=nonexistent_path,
            model_name="test_model",
        )
        packager = ModelPackager(config)

        with pytest.raises(ValueError) as exc_info:
            packager.validate_model_path()

        assert "does not exist" in str(exc_info.value)

    def test_validate_model_path_not_directory(self, tmp_path):
        """Test validation with file instead of directory"""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        config = PackageConfig(
            model_path=file_path,
            model_name="test_model",
        )
        packager = ModelPackager(config)

        with pytest.raises(ValueError) as exc_info:
            packager.validate_model_path()

        assert "not a directory" in str(exc_info.value)

    def test_collect_model_files(self, packager_config):
        """Test collecting model files"""
        packager = ModelPackager(packager_config)
        files = packager.collect_model_files()

        # Should collect .safetensors and .json files
        assert len(files) > 0

        # Check that .pyc files are excluded
        pyc_files = [f for f in files if f.suffix == ".pyc"]
        assert len(pyc_files) == 0

    @patch("mlops.serving.bentoml.packager.bentoml")
    def test_create_bento_model(self, mock_bentoml, packager_config):
        """Test BentoML model creation"""
        mock_model = Mock()
        mock_model.tag.name = "test_model"
        mock_model.tag.version = "v1.0"
        mock_model.path = str(packager_config.model_path / "bento_store")

        # Create the path
        Path(mock_model.path).mkdir(parents=True, exist_ok=True)

        mock_bentoml.models.create.return_value = mock_model

        packager = ModelPackager(packager_config)
        model = packager.create_bento_model()

        assert model == mock_model
        mock_bentoml.models.create.assert_called_once()

    def test_create_bentofile(self, packager_config, tmp_path):
        """Test bentofile.yaml creation"""
        packager = ModelPackager(packager_config)
        output_path = tmp_path / "bentofile.yaml"

        bentofile_path = packager.create_bentofile(output_path)

        assert bentofile_path.exists()
        content = bentofile_path.read_text()

        assert "service:" in content
        assert packager_config.service_name in content
        assert "python:" in content
        assert "packages:" in content
        assert "mlx" in content

    def test_build_bento(self, packager_config, tmp_path):
        """Test Bento package building"""
        packager = ModelPackager(packager_config)

        # Mock bentofile creation
        with patch.object(packager, "create_bentofile"):
            bento_tag = packager.build_bento()

        assert bento_tag is not None
        assert packager_config.service_name in bento_tag
        assert packager.bento_tag == bento_tag

    @patch("mlops.serving.bentoml.packager.bentoml")
    def test_package_model_complete(self, mock_bentoml, packager_config):
        """Test complete packaging workflow"""
        mock_model = Mock()
        mock_model.tag.name = "test_model"
        mock_model.tag.version = "v1.0"
        mock_model.path = str(packager_config.model_path / "bento_store")
        Path(mock_model.path).mkdir(parents=True, exist_ok=True)

        mock_bentoml.models.create.return_value = mock_model

        packager = ModelPackager(packager_config)

        with patch.object(packager, "create_bentofile"):
            result = packager.package_model(
                build_bento=True,
                create_docker=False,
            )

        assert result["success"] is True
        assert result["model_name"] == "test_model"
        assert "model_tag" in result
        assert "bento_tag" in result

    @patch("mlops.serving.bentoml.packager.bentoml")
    def test_package_model_with_docker(self, mock_bentoml, packager_config):
        """Test packaging with Docker image creation"""
        mock_model = Mock()
        mock_model.tag.name = "test_model"
        mock_model.tag.version = "v1.0"
        mock_model.path = str(packager_config.model_path / "bento_store")
        Path(mock_model.path).mkdir(parents=True, exist_ok=True)

        mock_bentoml.models.create.return_value = mock_model

        packager = ModelPackager(packager_config)

        with patch.object(packager, "create_bentofile"):
            result = packager.package_model(
                build_bento=True,
                create_docker=True,
            )

        assert result["success"] is True
        assert "docker_tag" in result

    @patch("mlops.serving.bentoml.packager.bentoml")
    def test_package_model_failure(self, mock_bentoml, packager_config):
        """Test packaging failure handling"""
        mock_bentoml.models.create.side_effect = RuntimeError("Package failed")

        packager = ModelPackager(packager_config)
        result = packager.package_model()

        assert result["success"] is False
        assert "error" in result

    def test_from_config(self, temp_model_dir):
        """Test creating packager from BentoMLConfig"""
        bentoml_config = BentoMLConfig(
            model_name="test_model",
            service_name="test_service",
            project_name="test_project",
        )

        packager = ModelPackager.from_config(
            bentoml_config=bentoml_config,
            model_path=temp_model_dir,
        )

        assert packager.config.model_name == "test_model"
        assert packager.config.service_name == "test_service"
        assert packager.config.project_name == "test_project"


class TestPackageModelHelper:
    """Tests for package_model helper function"""

    @patch("mlops.serving.bentoml.packager.ModelPackager")
    def test_package_model_basic(self, mock_packager_class, tmp_path):
        """Test basic model packaging"""
        mock_packager = Mock()
        mock_packager.package_model.return_value = {
            "success": True,
            "model_name": "test_model",
        }
        mock_packager_class.return_value = mock_packager

        result = package_model(
            model_path=tmp_path,
            model_name="test_model",
            project_name="test_project",
        )

        assert result["success"] is True
        assert result["model_name"] == "test_model"
        mock_packager.package_model.assert_called_once_with(build_bento=True)

    @patch("mlops.serving.bentoml.packager.ModelPackager")
    def test_package_model_no_bento_build(self, mock_packager_class, tmp_path):
        """Test packaging without Bento build"""
        mock_packager = Mock()
        mock_packager.package_model.return_value = {"success": True}
        mock_packager_class.return_value = mock_packager

        result = package_model(
            model_path=tmp_path,
            model_name="test_model",
            build_bento=False,
        )

        mock_packager.package_model.assert_called_once_with(build_bento=False)
