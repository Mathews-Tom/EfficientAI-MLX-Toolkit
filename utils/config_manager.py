"""
Configuration management system using pathlib for file handling.

This module provides centralized configuration management for the EfficientAI-MLX-Toolkit,
supporting YAML, JSON, and TOML formats with validation and error handling.
"""

import json
import logging
import os
import tomllib
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, TypeVar

import tomli_w
import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigurationError(Exception):
    """Raised when configuration operations fail."""

    def __init__(
        self,
        message: str,
        config_path: Path | None = None,
        details: Mapping[str, str | int | float] | None = None,
    ) -> None:
        super().__init__(message)
        self.config_path = config_path
        self.details = dict(details or {})


class ConfigManager:
    """
    Centralized configuration management with pathlib support.

    This class provides configuration loading, validation, and management
    capabilities for the EfficientAI-MLX-Toolkit, supporting multiple
    configuration formats.

    Attributes:
        config_path: Path to the configuration file
        config_data: Loaded configuration data
    """

    def __init__(
        self,
        config_path: Path,
        environment_prefix: str = "EFFICIENTAI",
        profile: str = "default"
    ) -> None:
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to configuration file using pathlib
            environment_prefix: Prefix for environment variable overrides
            profile: Configuration profile (default, development, production, etc.)

        Raises:
            ConfigurationError: If configuration file format is unsupported
        """
        self.config_path = config_path
        self.environment_prefix = environment_prefix
        self.profile = profile
        self.config_data: dict[str, str | int | float | bool | list | dict] = {}

        # Validate file format
        self._validate_format()

        # Load configuration if file exists
        if self.config_path.exists():
            self.load()

        # Apply environment overrides
        self._apply_environment_overrides()

    def _validate_format(self) -> None:
        """Validate that the configuration file format is supported."""
        suffix = self.config_path.suffix.lower()
        supported_formats = [".json", ".yaml", ".yml", ".toml"]

        if suffix not in supported_formats:
            raise ConfigurationError(
                f"Unsupported configuration format: {suffix}",
                config_path=self.config_path,
                details={"supported_formats": supported_formats},
            )

    def load(self) -> None:
        """
        Load configuration from file.

        Raises:
            ConfigurationError: If loading fails
        """
        try:
            logger.info("Loading configuration from %s", self.config_path)

            suffix = self.config_path.suffix.lower()
            content = self.config_path.read_text(encoding="utf-8")

            if suffix == ".json":
                self.config_data = json.loads(content)
            elif suffix in [".yaml", ".yml"]:
                self.config_data = yaml.safe_load(content)
            elif suffix == ".toml":
                self.config_data = tomllib.loads(content)

            logger.info("Configuration loaded successfully")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {self.config_path}",
                config_path=self.config_path,
            ) from e

    def save(self) -> None:
        """
        Save configuration to file.

        Raises:
            ConfigurationError: If saving fails
        """
        try:
            logger.info("Saving configuration to %s", self.config_path)

            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            suffix = self.config_path.suffix.lower()

            if suffix == ".json":
                content = json.dumps(self.config_data, indent=2, ensure_ascii=False)
            elif suffix in [".yaml", ".yml"]:
                content = yaml.dump(self.config_data, default_flow_style=False, allow_unicode=True)
            elif suffix == ".toml":
                content = tomli_w.dumps(self.config_data)
            else:
                raise ConfigurationError(f"Unsupported format for saving: {suffix}")

            self.config_path.write_text(content, encoding="utf-8")
            logger.info("Configuration saved successfully")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration to {self.config_path}", config_path=self.config_path
            ) from e

    def get(
        self, key: str, default: str | int | float | bool | list | dict | None = None
    ) -> str | int | float | bool | list | dict | None:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = ConfigManager(Path("config.yaml"))
            >>> value = config.get("database.host", "localhost")
        """
        try:
            keys = key.split(".")
            value = self.config_data

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value

        except Exception:
            logger.warning("Failed to get configuration key %s, using default", key)
            return default

    def set(self, key: str, value: str | int | float | bool | list | dict) -> None:
        """
        Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set

        Example:
            >>> config = ConfigManager(Path("config.yaml"))
            >>> config.set("database.host", "localhost")
        """
        keys = key.split(".")
        current = self.config_data

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            elif not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]

        # Set the final value
        current[keys[-1]] = value
        logger.debug("Set configuration key %s to %s", key, value)

    def update(self, updates: Mapping[str, str | int | float | bool | list | dict]) -> None:
        """
        Update configuration with multiple key-value pairs.

        Args:
            updates: Dictionary of configuration updates

        Example:
            >>> config = ConfigManager(Path("config.yaml"))
            >>> config.update({"database.host": "localhost", "database.port": 5432})
        """
        for key, value in updates.items():
            self.set(key, value)

        logger.info("Updated configuration with %d keys", len(updates))

    def validate_schema(self, schema: Mapping[str, type]) -> bool:
        """
        Validate configuration against a schema.

        Args:
            schema: Dictionary mapping keys to expected types

        Returns:
            True if validation passes

        Raises:
            ConfigurationError: If validation fails

        Example:
            >>> schema = {"database.host": str, "database.port": int}
            >>> config.validate_schema(schema)
        """
        errors: list[str] = []

        for key, expected_type in schema.items():
            value = self.get(key)

            if value is None:
                errors.append(f"Missing required key: {key}")
            elif not isinstance(value, expected_type):
                errors.append(
                    f"Invalid type for {key}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        if errors:
            raise ConfigurationError(
                "Configuration validation failed",
                config_path=self.config_path,
                details={"validation_errors": errors},
            )

        logger.info("Configuration validation passed")
        return True

    def to_dict(self) -> dict[str, str | int | float | bool | list | dict]:
        """
        Get configuration as dictionary.

        Returns:
            Copy of configuration data
        """
        return dict(self.config_data)

    def merge_from_file(self, other_config_path: Path) -> None:
        """
        Merge configuration from another file.

        Args:
            other_config_path: Path to configuration file to merge

        Raises:
            ConfigurationError: If merge fails
        """
        try:
            other_config = ConfigManager(other_config_path)
            self._deep_merge(self.config_data, other_config.config_data)
            logger.info("Merged configuration from %s", other_config_path)

        except Exception as e:
            raise ConfigurationError(
                f"Failed to merge configuration from {other_config_path}",
                config_path=self.config_path,
            ) from e

    def _deep_merge(self, target: MutableMapping, source: Mapping) -> None:
        """Deep merge source dictionary into target dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        prefix = f"{self.environment_prefix}_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace("_", ".")

                # Try to parse the value as JSON first, then fall back to string
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value

                self.set(config_key, parsed_value)
                logger.debug("Applied environment override: %s = %s", config_key, parsed_value)

    def get_with_type(self, key: str, expected_type: type[T], default: T | None = None) -> T:
        """
        Get configuration value with type checking.

        Args:
            key: Configuration key
            expected_type: Expected type of the value
            default: Default value if key not found

        Returns:
            Configuration value cast to expected type

        Raises:
            ConfigurationError: If value cannot be cast to expected type
        """
        value = self.get(key, default)

        if value is None:
            if default is not None:
                return default
            raise ConfigurationError(
                f"Configuration key '{key}' not found and no default provided"
            )

        try:
            if expected_type == bool and isinstance(value, str):
                # Handle string boolean values
                return expected_type(value.lower() in ("true", "1", "yes", "on"))
            return expected_type(value)
        except (ValueError, TypeError) as e:
            raise ConfigurationError(
                f"Cannot convert '{key}' value '{value}' to {expected_type.__name__}",
                config_path=self.config_path,
            ) from e

    def validate_required_keys(self, required_keys: list[str]) -> bool:
        """
        Validate that all required keys are present.

        Args:
            required_keys: List of required configuration keys

        Returns:
            True if all keys are present

        Raises:
            ConfigurationError: If any required keys are missing
        """
        missing_keys = []

        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)

        if missing_keys:
            raise ConfigurationError(
                f"Missing required configuration keys: {', '.join(missing_keys)}",
                config_path=self.config_path,
                details={"missing_keys": missing_keys}
            )

        return True

    def get_profile_config(self, profile: str | None = None) -> dict[str, Any]:
        """
        Get configuration for a specific profile.

        Args:
            profile: Profile name (uses instance profile if None)

        Returns:
            Profile-specific configuration
        """
        profile = profile or self.profile
        profile_key = f"profiles.{profile}"

        profile_config = self.get(profile_key, {})
        if not isinstance(profile_config, dict):
            profile_config = {}

        # Merge with base configuration
        base_config = {k: v for k, v in self.config_data.items() if not k.startswith("profiles.")}
        merged_config = dict(base_config)
        self._deep_merge(merged_config, profile_config)

        return merged_config

    def set_profile_config(self, profile_config: dict[str, Any], profile: str | None = None) -> None:
        """
        Set configuration for a specific profile.

        Args:
            profile_config: Profile configuration dictionary
            profile: Profile name (uses instance profile if None)
        """
        profile = profile or self.profile
        profile_key = f"profiles.{profile}"

        self.set(profile_key, profile_config)
        logger.info("Updated profile configuration: %s", profile)

    def list_profiles(self) -> list[str]:
        """
        List available configuration profiles.

        Returns:
            List of profile names
        """
        profiles_config = self.get("profiles", {})
        if isinstance(profiles_config, dict):
            return list(profiles_config.keys())
        return []
