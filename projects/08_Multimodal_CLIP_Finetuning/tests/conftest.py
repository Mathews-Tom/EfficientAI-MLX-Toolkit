"""Pytest configuration and fixtures for CLIP fine-tuning tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    from config import CLIPFinetuningConfig

    return CLIPFinetuningConfig(
        model_name="openai/clip-vit-base-patch32",
        domain="general",
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=2,
        use_mps=False,  # Default to CPU for tests
        mixed_precision=False,
        output_dir=project_root / "outputs" / "test",
    )


@pytest.fixture
def mps_config():
    """Provide a configuration with MPS enabled for testing."""
    from config import CLIPFinetuningConfig

    return CLIPFinetuningConfig(
        model_name="openai/clip-vit-base-patch32",
        domain="medical",
        use_mps=True,
        mixed_precision=True,
        output_dir=project_root / "outputs" / "test_mps",
    )
