"""
Tests for the CLI interface.
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

# Import the CLI app
from src.cli import app

runner = CliRunner()


def test_info_command():
    """Test the info command."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Model Compression Framework Information" in result.stdout


def test_validate_command():
    """Test the validate command."""
    result = runner.invoke(app, ["validate"])
    # May fail if config is invalid, but should not crash
    assert result.exit_code in [0, 1]
    assert "Validating configuration" in result.stdout


def test_info_with_custom_config():
    """Test info command with custom config path."""
    result = runner.invoke(app, ["info", "--config", "configs/default.yaml"])
    assert result.exit_code == 0


@pytest.mark.slow
def test_quantize_command_help():
    """Test quantize command help."""
    result = runner.invoke(app, ["quantize", "--help"])
    assert result.exit_code == 0
    assert "Quantize a model" in result.stdout


@pytest.mark.slow  
def test_prune_command_help():
    """Test prune command help."""
    result = runner.invoke(app, ["prune", "--help"])
    assert result.exit_code == 0
    assert "Prune a model" in result.stdout


def test_cli_app_exists():
    """Test that CLI app is properly defined."""
    assert app is not None
    
    # Basic test that app can be invoked
    assert callable(app)
    
    # Test that app has expected attributes of a Typer app
    assert hasattr(app, '__call__')
    
    # Test that the app name is set correctly
    if hasattr(app, 'info') and hasattr(app.info, 'name'):
        assert app.info.name == "model-compression-framework"