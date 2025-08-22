"""Test CLI interface."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from typer.testing import CliRunner
import yaml

from src.cli import app


class TestCLICommands:
    """Test CLI command structure and basic functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_info_command(self):
        """Test info command structure."""
        result = self.runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0
        assert "Show information about the Style Transfer framework" in result.stdout
    
    def test_validate_command(self):
        """Test validate command structure."""
        result = self.runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate configuration file" in result.stdout
    
    def test_transfer_command(self):
        """Test transfer command structure."""
        result = self.runner.invoke(app, ["transfer", "--help"])
        assert result.exit_code == 0
        assert "Perform style transfer on an image" in result.stdout
        assert "--content-image" in result.stdout
        assert "--style-image" in result.stdout
    
    @pytest.mark.skip(reason="Training command is under development")
    def test_train_command(self):
        """Test train command structure."""
        result = self.runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 2  # Command not found
    
    def test_convert_command(self):
        """Test convert command structure."""
        result = self.runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "Convert trained model to Core ML format" in result.stdout
        assert "--model-path" in result.stdout
    
    @pytest.mark.skip(reason="Serving command is under development")
    def test_serve_command(self):
        """Test serve command structure."""
        result = self.runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 2  # Command not found
    
    def test_benchmark_command(self):
        """Test benchmark command structure."""
        result = self.runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "Benchmark style transfer performance" in result.stdout
        assert "--model-path" in result.stdout


class TestCLIInfo:
    """Test CLI info command functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @patch('src.cli.Path.exists')
    @patch('builtins.open')
    def test_info_command_with_config(self, mock_open, mock_exists):
        """Test info command with existing config."""
        mock_exists.return_value = True
        
        # Mock config data
        config_data = {
            'diffusion': {'model_name': 'test-model', 'num_inference_steps': 25},
            'style_transfer': {'style_strength': 0.9},
            'coreml': {'optimize_for_apple_silicon': True},
            'hardware': {'prefer_mlx': True}
        }
        
        mock_open.return_value.__enter__.return_value.read.return_value = yaml.dump(config_data)
        
        with patch('yaml.safe_load', return_value=config_data):
            result = self.runner.invoke(app, ["info"])
            
        assert result.exit_code == 0
        assert "Core ML Stable Diffusion Style Transfer Information" in result.stdout
    
    @patch('src.cli.Path.exists')
    def test_info_command_missing_config(self, mock_exists):
        """Test info command with missing config."""
        mock_exists.return_value = False
        
        result = self.runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        assert "Configuration file not found" in result.stdout


class TestCLIValidation:
    """Test CLI validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_validate_missing_config(self):
        """Test validate with missing config."""
        result = self.runner.invoke(app, ["validate", "--config", "nonexistent.yaml"])
        
        assert result.exit_code == 1
        assert "Configuration file not found" in result.output
    
    @patch('src.cli.Path.exists')
    @patch('builtins.open')
    def test_validate_valid_config(self, mock_open, mock_exists):
        """Test validate with valid config."""
        mock_exists.return_value = True
        
        config_data = {
            'diffusion': {'model_name': 'test', 'num_inference_steps': 50},
            'style_transfer': {'style_strength': 0.8, 'content_strength': 0.6},
            'coreml': {'optimize_for_apple_silicon': True, 'precision': 'float16'}
        }
        
        with patch('yaml.safe_load', return_value=config_data):
            with patch('src.diffusion.DiffusionConfig.from_dict') as mock_diff:
                with patch('src.style_transfer.StyleTransferConfig.from_dict') as mock_style:
                    with patch('src.coreml.CoreMLConfig.from_dict') as mock_coreml:
                        
                        mock_diff.return_value = Mock()
                        mock_style.return_value = Mock()
                        mock_coreml.return_value = Mock()
                        
                        result = self.runner.invoke(app, ["validate"])
        
        assert result.exit_code == 0
        assert "All configurations are valid" in result.stdout


class TestCLITransfer:
    """Test CLI transfer command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_transfer_missing_content_image(self):
        """Test transfer with missing content image."""
        result = self.runner.invoke(app, [
            "transfer",
            "--content-image", "missing.jpg",
            "--style-image", "style.jpg",
            "--output", "output.png"
        ])
        
        assert result.exit_code == 1
        assert "Content image not found" in result.output
    
    def test_transfer_missing_style_image(self):
        """Test transfer with missing style image."""
        # Create a temporary content image for the test
        with self.runner.isolated_filesystem():
            with open("content.jpg", "w") as f:
                f.write("fake image")
            
            result = self.runner.invoke(app, [
                "transfer", 
                "--content-image", "content.jpg",
                "--style-image", "missing.jpg",
                "--output", "output.png"
            ])
        
        assert result.exit_code == 1
        assert "Style image not found" in result.output


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_help_command(self):
        """Test main help command."""
        result = self.runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Core ML Stable Diffusion Style Transfer Framework" in result.stdout
        assert "info" in result.stdout
        assert "validate" in result.stdout
        assert "transfer" in result.stdout
        assert "train" in result.stdout
    
    def test_command_parameter_validation(self):
        """Test command parameter validation."""
        # Test with invalid parameters
        result = self.runner.invoke(app, [
            "transfer",
            "--content-image", "",  # Empty path
            "--output", "test.png"
        ])
        
        # Should fail due to validation (exact error may vary)
        assert result.exit_code != 0