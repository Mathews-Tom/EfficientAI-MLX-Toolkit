"""
Centralized configuration management for DSPy Integration Framework.
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml

from ..exceptions import DSPyIntegrationError
from ..types import DSPyConfig

logger = logging.getLogger(__name__)


@dataclass
class ConfigSchema:
    """Schema definition for configuration validation."""

    name: str
    version: str
    required_fields: list[str]
    optional_fields: list[str]
    field_types: dict[str, type]
    validation_rules: dict[str, str | int | float | bool]


class ConfigManager:
    """Centralized configuration management system."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize configuration manager."""
        self.config_dir = config_dir or Path(".dspy_config")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Configuration files
        self.global_config_file = self.config_dir / "global.yaml"
        self.project_configs_dir = self.config_dir / "projects"
        self.project_configs_dir.mkdir(exist_ok=True)

        # Schema registry
        self.schemas: dict[str, ConfigSchema] = {}

        # Configuration cache
        self._config_cache: dict[str, dict[str, str | int | float | bool]] = {}
        self._cache_timestamps: dict[str, float] = {}

        # Initialize default schemas
        self._register_default_schemas()

        logger.info("Configuration manager initialized with config dir: %s", self.config_dir)

    def _register_default_schemas(self):
        """Register default configuration schemas."""
        # DSPy Framework schema
        framework_schema = ConfigSchema(
            name="dspy_framework",
            version="1.0.0",
            required_fields=["model_provider", "model_name"],
            optional_fields=[
                "optimization_level",
                "cache_dir",
                "enable_tracing",
                "max_retries",
            ],
            field_types={
                "model_provider": str,
                "model_name": str,
                "optimization_level": int,
                "cache_dir": str,
                "enable_tracing": bool,
                "max_retries": int,
            },
            validation_rules={
                "optimization_level": {"min": 0, "max": 3},
                "max_retries": {"min": 1, "max": 10},
            },
        )
        self.register_schema(framework_schema)

        # Project configuration schema
        project_schema = ConfigSchema(
            name="project_config",
            version="1.0.0",
            required_fields=["name", "type"],
            optional_fields=[
                "description",
                "signatures",
                "modules",
                "optimization_settings",
            ],
            field_types={
                "name": str,
                "type": str,
                "description": str,
                "signatures": dict,
                "modules": dict,
                "optimization_settings": dict,
            },
            validation_rules={
                "type": {
                    "allowed_values": [
                        "lora",
                        "diffusion",
                        "clip",
                        "federated",
                        "general",
                    ]
                },
            },
        )
        self.register_schema(project_schema)

    def register_schema(self, schema: ConfigSchema):
        """Register a configuration schema."""
        self.schemas[schema.name] = schema
        logger.info("Registered configuration schema: %s v%s", schema.name, schema.version)

    def validate_config(
        self, config: dict[str, str | int | float | bool], schema_name: str
    ) -> bool:
        """Validate configuration against schema."""
        if schema_name not in self.schemas:
            raise DSPyIntegrationError(f"Unknown schema: {schema_name}")

        schema = self.schemas[schema_name]

        # Check required fields
        for field in schema.required_fields:
            if field not in config:
                raise DSPyIntegrationError(f"Missing required field: {field}")

        # Check field types
        for field, value in config.items():
            if field in schema.field_types:
                expected_type = schema.field_types[field]
                if not isinstance(value, expected_type):
                    raise DSPyIntegrationError(
                        f"Field {field} should be {expected_type.__name__}, got {type(value).__name__}"
                    )

        # Check validation rules
        for field, rules in schema.validation_rules.items():
            if field in config:
                value = config[field]

                if "min" in rules and value < rules["min"]:
                    raise DSPyIntegrationError(
                        f"Field {field} value {value} below minimum {rules['min']}"
                    )

                if "max" in rules and value > rules["max"]:
                    raise DSPyIntegrationError(
                        f"Field {field} value {value} above maximum {rules['max']}"
                    )

                if "allowed_values" in rules and value not in rules["allowed_values"]:
                    raise DSPyIntegrationError(
                        f"Field {field} value {value} not in allowed values: {rules['allowed_values']}"
                    )

        return True

    def load_global_config(self) -> dict[str, str | int | float | bool]:
        """Load global configuration."""
        if not self.global_config_file.exists():
            # Create default global config
            default_config = {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "framework": {
                    "model_provider": "mlx",
                    "model_name": "mlx/mlx-7b",
                    "optimization_level": 2,
                    "enable_tracing": False,
                    "max_retries": 3,
                },
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "monitoring": {
                    "enable_metrics": True,
                    "export_interval": 300,
                },
            }
            self.save_global_config(default_config)
            return default_config

        try:
            with open(self.global_config_file, "r") as f:
                config = yaml.safe_load(f)

            # Validate framework config if present
            if "framework" in config:
                self.validate_config(config["framework"], "dspy_framework")

            # Cache config
            self._config_cache["global"] = config
            self._cache_timestamps["global"] = self.global_config_file.stat().st_mtime

            return config

        except Exception as e:
            logger.error("Failed to load global config: %s", e)
            raise DSPyIntegrationError(f"Global config loading failed: {e}") from e

    def save_global_config(self, config: dict[str, str | int | float | bool]):
        """Save global configuration."""
        try:
            # Validate framework config if present
            if "framework" in config:
                self.validate_config(config["framework"], "dspy_framework")

            # Add metadata
            config["updated_at"] = datetime.now().isoformat()

            with open(self.global_config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            # Update cache
            self._config_cache["global"] = config
            self._cache_timestamps["global"] = self.global_config_file.stat().st_mtime

            logger.info("Global configuration saved")

        except Exception as e:
            logger.error("Failed to save global config: %s", e)
            raise DSPyIntegrationError(f"Global config saving failed: {e}") from e

    def load_project_config(self, project_name: str) -> dict[str, str | int | float | bool]:
        """Load project-specific configuration."""
        project_config_file = self.project_configs_dir / f"{project_name}.yaml"

        if not project_config_file.exists():
            # Create default project config
            default_config = {
                "name": project_name,
                "type": "general",
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "description": f"Configuration for {project_name} project",
                "signatures": {},
                "modules": {},
                "optimization_settings": {
                    "default_optimizer": "auto",
                    "max_attempts": 3,
                    "metrics": ["accuracy"],
                },
            }
            self.save_project_config(project_name, default_config)
            return default_config

        try:
            with open(project_config_file, "r") as f:
                config = yaml.safe_load(f)

            # Validate project config
            self.validate_config(config, "project_config")

            # Cache config
            cache_key = f"project_{project_name}"
            self._config_cache[cache_key] = config
            self._cache_timestamps[cache_key] = project_config_file.stat().st_mtime

            return config

        except Exception as e:
            logger.error("Failed to load project config for %s: %s", project_name, e)
            raise DSPyIntegrationError(f"Project config loading failed: {e}") from e

    def save_project_config(self, project_name: str, config: dict[str, str | int | float | bool]):
        """Save project-specific configuration."""
        try:
            # Validate project config
            self.validate_config(config, "project_config")

            # Add metadata
            config["updated_at"] = datetime.now().isoformat()

            project_config_file = self.project_configs_dir / f"{project_name}.yaml"
            with open(project_config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            # Update cache
            cache_key = f"project_{project_name}"
            self._config_cache[cache_key] = config
            self._cache_timestamps[cache_key] = project_config_file.stat().st_mtime

            logger.info("Project configuration saved for %s", project_name)

        except Exception as e:
            logger.error("Failed to save project config for %s: %s", project_name, e)
            raise DSPyIntegrationError(f"Project config saving failed: {e}") from e

    def get_merged_config(
        self, project_name: str | None = None
    ) -> dict[str, str | int | float | bool]:
        """Get merged configuration (global + project)."""
        global_config = self.load_global_config()

        if project_name:
            project_config = self.load_project_config(project_name)

            # Merge configurations (project overrides global)
            merged_config = global_config.copy()
            merged_config.update(project_config)

            # Special handling for nested dictionaries
            if "framework" in global_config and "framework" in project_config:
                merged_framework = global_config["framework"].copy()
                merged_framework.update(project_config.get("framework", {}))
                merged_config["framework"] = merged_framework

            return merged_config

        return global_config

    def create_dspy_config(self, project_name: str | None = None) -> DSPyConfig:
        """Create DSPyConfig from managed configuration."""
        merged_config = self.get_merged_config(project_name)
        framework_config = merged_config.get("framework", {})

        # Map configuration to DSPyConfig
        cache_dir = Path(framework_config.get("cache_dir", ".dspy_cache"))
        if project_name:
            cache_dir = cache_dir / project_name

        return DSPyConfig(
            model_provider=framework_config.get("model_provider", "mlx"),
            model_name=framework_config.get("model_name", "mlx/mlx-7b"),
            optimization_level=framework_config.get("optimization_level", 2),
            cache_dir=cache_dir,
            enable_tracing=framework_config.get("enable_tracing", False),
            max_retries=framework_config.get("max_retries", 3),
            api_key=framework_config.get("api_key"),
            base_url=framework_config.get("base_url"),
        )

    def list_projects(self) -> list[str]:
        """List all configured projects."""
        project_files = list(self.project_configs_dir.glob("*.yaml"))
        return [f.stem for f in project_files]

    def delete_project_config(self, project_name: str):
        """Delete project configuration."""
        project_config_file = self.project_configs_dir / f"{project_name}.yaml"

        if project_config_file.exists():
            project_config_file.unlink()

            # Remove from cache
            cache_key = f"project_{project_name}"
            self._config_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)

            logger.info("Deleted project configuration for %s", project_name)
        else:
            logger.warning("Project configuration not found: %s", project_name)

    def export_config(self, export_path: Path, include_projects: bool = True):
        """Export all configurations to a directory."""
        try:
            export_path.mkdir(parents=True, exist_ok=True)

            # Export global config
            global_config = self.load_global_config()
            with open(export_path / "global.yaml", "w") as f:
                yaml.dump(global_config, f, default_flow_style=False, indent=2)

            if include_projects:
                # Export project configs
                projects_dir = export_path / "projects"
                projects_dir.mkdir(exist_ok=True)

                for project_name in self.list_projects():
                    project_config = self.load_project_config(project_name)
                    with open(projects_dir / f"{project_name}.yaml", "w") as f:
                        yaml.dump(project_config, f, default_flow_style=False, indent=2)

            # Export schemas
            schemas_data = {name: asdict(schema) for name, schema in self.schemas.items()}
            with open(export_path / "schemas.yaml", "w") as f:
                yaml.dump(schemas_data, f, default_flow_style=False, indent=2)

            logger.info("Configuration exported to %s", export_path)

        except Exception as e:
            logger.error("Failed to export configuration: %s", e)
            raise DSPyIntegrationError(f"Configuration export failed: {e}") from e

    def import_config(self, import_path: Path):
        """Import configurations from a directory."""
        try:
            # Import global config
            global_config_file = import_path / "global.yaml"
            if global_config_file.exists():
                with open(global_config_file, "r") as f:
                    global_config = yaml.safe_load(f)
                self.save_global_config(global_config)

            # Import project configs
            projects_dir = import_path / "projects"
            if projects_dir.exists():
                for project_file in projects_dir.glob("*.yaml"):
                    project_name = project_file.stem
                    with open(project_file, "r") as f:
                        project_config = yaml.safe_load(f)
                    self.save_project_config(project_name, project_config)

            # Import schemas
            schemas_file = import_path / "schemas.yaml"
            if schemas_file.exists():
                with open(schemas_file, "r") as f:
                    schemas_data = yaml.safe_load(f)

                for name, schema_dict in schemas_data.items():
                    schema = ConfigSchema(**schema_dict)
                    self.register_schema(schema)

            logger.info("Configuration imported from %s", import_path)

        except Exception as e:
            logger.error("Failed to import configuration: %s", e)
            raise DSPyIntegrationError(f"Configuration import failed: {e}") from e

    def get_config_status(self) -> dict[str, str | int | float | bool]:
        """Get configuration system status."""
        try:
            global_config = self.load_global_config()
            projects = self.list_projects()

            return {
                "config_dir": str(self.config_dir),
                "global_config_exists": self.global_config_file.exists(),
                "global_config_version": global_config.get("version", "unknown"),
                "num_projects": len(projects),
                "projects": projects,
                "num_schemas": len(self.schemas),
                "schemas": list(self.schemas.keys()),
                "cache_size": len(self._config_cache),
                "last_updated": global_config.get("updated_at", "unknown"),
            }

        except Exception as e:
            logger.error("Failed to get config status: %s", e)
            return {"error": str(e)}

    def clear_cache(self):
        """Clear configuration cache."""
        self._config_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Configuration cache cleared")

    def reload_config(self, project_name: str | None = None):
        """Reload configuration from disk."""
        if project_name:
            cache_key = f"project_{project_name}"
            self._config_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
            self.load_project_config(project_name)
        else:
            self._config_cache.pop("global", None)
            self._cache_timestamps.pop("global", None)
            self.load_global_config()

        logger.info("Configuration reloaded for %s", project_name or "global")

    def watch_config_changes(self, callback: callable):
        """Watch for configuration file changes (simplified implementation)."""
        # This is a simplified implementation
        # In production, you might use watchdog or similar library
        import threading
        import time

        def watch_thread():
            while True:
                try:
                    # Check global config
                    if self.global_config_file.exists():
                        current_mtime = self.global_config_file.stat().st_mtime
                        cached_mtime = self._cache_timestamps.get("global", 0)

                        if current_mtime > cached_mtime:
                            logger.info("Global config file changed, reloading")
                            self.reload_config()
                            callback("global", self.load_global_config())

                    # Check project configs
                    for project_file in self.project_configs_dir.glob("*.yaml"):
                        project_name = project_file.stem
                        cache_key = f"project_{project_name}"

                        current_mtime = project_file.stat().st_mtime
                        cached_mtime = self._cache_timestamps.get(cache_key, 0)

                        if current_mtime > cached_mtime:
                            logger.info(
                                "Project config file changed for %s, reloading", project_name
                            )
                            self.reload_config(project_name)
                            callback(project_name, self.load_project_config(project_name))

                    time.sleep(5)  # Check every 5 seconds

                except Exception as e:
                    logger.error("Config watching error: %s", e)
                    time.sleep(10)  # Wait longer on error

        watch_thread = threading.Thread(target=watch_thread, daemon=True)
        watch_thread.start()
        logger.info("Started configuration file watching")
