"""Configuration management for meta-learning PEFT."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        # Default to configs/default.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        output_path: Path to save configuration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
