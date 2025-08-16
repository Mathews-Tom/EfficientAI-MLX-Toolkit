"""
Unit tests for DSPy management components.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dspy_toolkit.exceptions import DSPyIntegrationError
from dspy_toolkit.management import (
    ConfigManager,
    ConfigSchema,
    DocumentationGenerator,
    Migration,
    MigrationManager,
    ProgramManager,
    ProgramVersion,
)
from dspy_toolkit.types import DSPyConfig


class TestConfigManager:
    """Test cases for configuration manager."""

    def test_config_manager_initialization(self, temp_dir):
        """Test configuration manager initialization."""
        config_manager = ConfigManager(temp_dir / "config")

        assert config_manager.config_dir.exists()
        assert config_manager.global_config_file.parent.exists()
        assert config_manager.project_configs_dir.exists()
        assert len(config_manager.schemas) > 0

    def test_schema_registration(self, temp_dir):
        """Test schema registration."""
        config_manager = ConfigManager(temp_dir / "config")

        schema = ConfigSchema(
            name="test_schema",
            version="1.0.0",
            required_fields=["name"],
            optional_fields=["description"],
            field_types={"name": str, "description": str},
            validation_rules={},
        )

        config_manager.register_schema(schema)

        assert "test_schema" in config_manager.schemas
        assert config_manager.schemas["test_schema"] == schema

    def test_config_validation(self, temp_dir):
        """Test configuration validation."""
        config_manager = ConfigManager(temp_dir / "config")

        # Valid config
        valid_config = {
            "model_provider": "mlx",
            "model_name": "test-model",
            "optimization_level": 2,
        }

        assert config_manager.validate_config(valid_config, "dspy_framework") == True

        # Invalid config - missing required field
        invalid_config = {
            "model_provider": "mlx",
            # missing model_name
        }

        with pytest.raises(DSPyIntegrationError, match="Missing required field"):
            config_manager.validate_config(invalid_config, "dspy_framework")

    def test_global_config_operations(self, temp_dir):
        """Test global configuration operations."""
        config_manager = ConfigManager(temp_dir / "config")

        # Load default config
        config = config_manager.load_global_config()
        assert "framework" in config
        assert config["framework"]["model_provider"] == "mlx"

        # Modify and save config
        config["framework"]["optimization_level"] = 3
        config_manager.save_global_config(config)

        # Reload and verify
        reloaded_config = config_manager.load_global_config()
        assert reloaded_config["framework"]["optimization_level"] == 3

    def test_project_config_operations(self, temp_dir):
        """Test project configuration operations."""
        config_manager = ConfigManager(temp_dir / "config")

        project_name = "test_project"

        # Load default project config
        config = config_manager.load_project_config(project_name)
        assert config["name"] == project_name
        assert config["type"] == "general"

        # Modify and save
        config["description"] = "Updated description"
        config_manager.save_project_config(project_name, config)

        # Reload and verify
        reloaded_config = config_manager.load_project_config(project_name)
        assert reloaded_config["description"] == "Updated description"

    def test_dspy_config_creation(self, temp_dir):
        """Test DSPyConfig creation from managed configuration."""
        config_manager = ConfigManager(temp_dir / "config")

        # Create DSPyConfig
        dspy_config = config_manager.create_dspy_config()

        assert isinstance(dspy_config, DSPyConfig)
        assert dspy_config.model_provider == "mlx"
        assert dspy_config.optimization_level == 2

    def test_config_export_import(self, temp_dir):
        """Test configuration export and import."""
        config_manager = ConfigManager(temp_dir / "config")
        export_dir = temp_dir / "export"

        # Create test project
        test_config = {
            "name": "export_test",
            "type": "test",
            "description": "Test project for export",
        }
        config_manager.save_project_config("export_test", test_config)

        # Export configuration
        config_manager.export_config(export_dir)

        assert (export_dir / "global.yaml").exists()
        assert (export_dir / "projects" / "export_test.yaml").exists()
        assert (export_dir / "schemas.yaml").exists()

        # Create new config manager and import
        new_config_manager = ConfigManager(temp_dir / "new_config")
        new_config_manager.import_config(export_dir)

        # Verify import
        imported_config = new_config_manager.load_project_config("export_test")
        assert imported_config["description"] == "Test project for export"


class TestProgramManager:
    """Test cases for program manager."""

    def test_program_manager_initialization(self, temp_dir):
        """Test program manager initialization."""
        program_manager = ProgramManager(temp_dir / "programs")

        assert program_manager.programs_dir.exists()
        assert program_manager.metadata_dir.exists()
        assert program_manager.programs_store.exists()
        assert program_manager.versions_dir.exists()

    def test_program_publishing(self, temp_dir, mock_dspy_module):
        """Test program publishing."""
        program_manager = ProgramManager(temp_dir / "programs")

        # Publish program
        program_key = program_manager.publish_program(
            name="test_program",
            program=mock_dspy_module,
            project="test_project",
            signature_name="test_signature",
            version="1.0.0",
            description="Test program",
            performance_metrics={"accuracy": 0.95},
            created_by="test_user",
        )

        assert program_key == "test_project:test_program"
        assert program_key in program_manager.registry

        # Verify program file exists
        program_file = program_manager.programs_store / f"{program_key}_1.0.0.pkl"
        assert program_file.exists()

    def test_program_downloading(self, temp_dir, mock_dspy_module):
        """Test program downloading."""
        program_manager = ProgramManager(temp_dir / "programs")

        # Publish program first
        program_manager.publish_program(
            name="download_test",
            program=mock_dspy_module,
            project="test_project",
            signature_name="test_signature",
            version="1.0.0",
        )

        # Download program
        downloaded_program = program_manager.download_program(
            "download_test", "test_project"
        )

        assert downloaded_program is not None
        # Verify download count increased
        program_info = program_manager.get_program_info("download_test", "test_project")
        assert program_info["total_downloads"] == 1

    def test_program_versioning(self, temp_dir, mock_dspy_module):
        """Test program versioning."""
        program_manager = ProgramManager(temp_dir / "programs")

        # Publish version 1.0.0
        program_manager.publish_program(
            name="version_test",
            program=mock_dspy_module,
            project="test_project",
            signature_name="test_signature",
            version="1.0.0",
        )

        # Publish version 2.0.0
        program_manager.publish_program(
            name="version_test",
            program=mock_dspy_module,
            project="test_project",
            signature_name="test_signature",
            version="2.0.0",
        )

        # Verify versions
        versions = program_manager.get_program_versions("version_test", "test_project")
        assert len(versions) == 2

        version_numbers = [v["version"] for v in versions]
        assert "1.0.0" in version_numbers
        assert "2.0.0" in version_numbers

    def test_program_search(self, temp_dir, mock_dspy_module):
        """Test program search functionality."""
        program_manager = ProgramManager(temp_dir / "programs")

        # Publish test programs
        program_manager.publish_program(
            name="search_test_1",
            program=mock_dspy_module,
            project="project_a",
            signature_name="test_signature",
            description="First test program",
        )

        program_manager.publish_program(
            name="search_test_2",
            program=mock_dspy_module,
            project="project_b",
            signature_name="test_signature",
            description="Second test program",
        )

        # Search by name
        results = program_manager.search_programs("search_test")
        assert len(results) == 2

        # Search by project
        results = program_manager.search_programs("test", project="project_a")
        assert len(results) == 1
        assert results[0]["project"] == "project_a"

    def test_program_deletion(self, temp_dir, mock_dspy_module):
        """Test program deletion."""
        program_manager = ProgramManager(temp_dir / "programs")

        # Publish program
        program_manager.publish_program(
            name="delete_test",
            program=mock_dspy_module,
            project="test_project",
            signature_name="test_signature",
            version="1.0.0",
        )

        # Verify program exists
        assert (
            program_manager.get_program_info("delete_test", "test_project") is not None
        )

        # Delete program
        program_manager.delete_program("delete_test", "test_project")

        # Verify program is deleted
        assert program_manager.get_program_info("delete_test", "test_project") is None


class TestMigrationManager:
    """Test cases for migration manager."""

    def test_migration_manager_initialization(self, temp_dir):
        """Test migration manager initialization."""
        migration_manager = MigrationManager(temp_dir / "migrations")

        assert migration_manager.migrations_dir.exists()
        assert migration_manager.migration_scripts_dir.exists()
        assert len(migration_manager.migrations) > 0  # Built-in migrations

    def test_migration_registration(self, temp_dir):
        """Test migration registration."""
        migration_manager = MigrationManager(temp_dir / "migrations")

        def test_migrate(context):
            return True

        def test_rollback(context):
            return True

        migration = Migration(
            id="test_migration",
            version_from="1.0.0",
            version_to="2.0.0",
            description="Test migration",
            migration_func=test_migrate,
            rollback_func=test_rollback,
        )

        migration_manager.register_migration(migration)

        assert "test_migration" in migration_manager.migrations
        assert migration_manager.migrations["test_migration"] == migration

    def test_migration_script_creation(self, temp_dir):
        """Test migration script creation."""
        migration_manager = MigrationManager(temp_dir / "migrations")

        script_path = migration_manager.create_migration_script(
            migration_id="test_script_migration",
            version_from="1.0.0",
            version_to="2.0.0",
            description="Test script migration",
        )

        assert script_path.exists()
        assert "test_script_migration" in script_path.name

        # Verify script content
        content = script_path.read_text()
        assert 'MIGRATION_ID = "test_script_migration"' in content
        assert "def migrate(context):" in content
        assert "def rollback(context):" in content

    def test_migration_application(self, temp_dir):
        """Test migration application."""
        migration_manager = MigrationManager(temp_dir / "migrations")

        # Create test migration
        def test_migrate(context):
            context["test_applied"] = True
            return True

        migration = Migration(
            id="apply_test",
            version_from="1.0.0",
            version_to="2.0.0",
            description="Test application",
            migration_func=test_migrate,
        )

        migration_manager.register_migration(migration)

        # Apply migration
        context = {}
        success = migration_manager.apply_migration("apply_test", context)

        assert success == True
        assert context["test_applied"] == True

        # Verify migration record
        applied_records = [r for r in migration_manager.applied_migrations if r.success]
        assert len(applied_records) == 1
        assert applied_records[0].migration_id == "apply_test"

    def test_migration_rollback(self, temp_dir):
        """Test migration rollback."""
        migration_manager = MigrationManager(temp_dir / "migrations")

        # Create test migration with rollback
        def test_migrate(context):
            context["test_value"] = "migrated"
            return True

        def test_rollback(context):
            context["test_value"] = "rolled_back"
            return True

        migration = Migration(
            id="rollback_test",
            version_from="1.0.0",
            version_to="2.0.0",
            description="Test rollback",
            migration_func=test_migrate,
            rollback_func=test_rollback,
        )

        migration_manager.register_migration(migration)

        # Apply migration
        context = {}
        migration_manager.apply_migration("rollback_test", context)
        assert context["test_value"] == "migrated"

        # Rollback migration
        success = migration_manager.rollback_migration("rollback_test", context)

        assert success == True
        assert context["test_value"] == "rolled_back"

    def test_pending_migrations(self, temp_dir):
        """Test pending migrations detection."""
        migration_manager = MigrationManager(temp_dir / "migrations")

        # Create test migrations
        migration1 = Migration(
            id="pending_test_1",
            version_from="1.0.0",
            version_to="1.1.0",
            description="First pending",
            migration_func=lambda ctx: True,
        )

        migration2 = Migration(
            id="pending_test_2",
            version_from="1.1.0",
            version_to="2.0.0",
            description="Second pending",
            migration_func=lambda ctx: True,
        )

        migration_manager.register_migration(migration1)
        migration_manager.register_migration(migration2)

        # Get pending migrations for version 1.0.0
        pending = migration_manager.get_pending_migrations("1.0.0")

        # Should include both migrations
        pending_ids = [m.id for m in pending]
        assert "pending_test_1" in pending_ids
        assert "pending_test_2" in pending_ids


class TestDocumentationGenerator:
    """Test cases for documentation generator."""

    def test_documentation_generator_initialization(self, temp_dir):
        """Test documentation generator initialization."""
        doc_generator = DocumentationGenerator(temp_dir / "docs")

        assert doc_generator.output_dir.exists()
        assert len(doc_generator.templates) > 0

    def test_signature_documentation(self, temp_dir, mock_signature):
        """Test signature documentation generation."""
        doc_generator = DocumentationGenerator(temp_dir / "docs")

        sig_docs = doc_generator.generate_signature_docs(mock_signature)

        assert "name" in sig_docs
        assert "fields" in sig_docs
        assert "examples" in sig_docs
        assert sig_docs["name"] == mock_signature.__name__

    def test_module_documentation(self, temp_dir, mock_dspy_module):
        """Test module documentation generation."""
        doc_generator = DocumentationGenerator(temp_dir / "docs")

        module_docs = doc_generator.generate_module_docs(
            mock_dspy_module, "test_module"
        )

        assert "name" in module_docs
        assert "type" in module_docs
        assert "methods" in module_docs
        assert "examples" in module_docs
        assert module_docs["name"] == "test_module"

    def test_api_documentation(self, temp_dir, mock_framework):
        """Test API documentation generation."""
        doc_generator = DocumentationGenerator(temp_dir / "docs")

        api_docs = doc_generator.generate_api_docs(mock_framework)

        assert "title" in api_docs
        assert "endpoints" in api_docs
        assert "schemas" in api_docs
        assert "examples" in api_docs

    @patch(
        "dspy_toolkit.management.documentation_generator.LoRAOptimizationSignature"
    )
    def test_full_documentation_generation(
        self, mock_lora_sig, temp_dir, mock_framework
    ):
        """Test full documentation generation."""
        doc_generator = DocumentationGenerator(temp_dir / "docs")

        # Mock signature registry
        mock_framework.signature_registry.get_all_signatures.return_value = {
            "test_project": {"test_sig": mock_lora_sig}
        }

        # Generate all docs
        generated_files = doc_generator.generate_all_docs(mock_framework)

        assert "api" in generated_files
        assert "index" in generated_files

        # Verify files exist
        for file_path in generated_files.values():
            assert Path(file_path).exists()


@pytest.mark.integration
class TestManagementIntegration:
    """Integration tests for management components."""

    def test_config_and_program_integration(self, temp_dir, mock_dspy_module):
        """Test integration between config and program managers."""
        # Initialize managers
        config_manager = ConfigManager(temp_dir / "config")
        program_manager = ProgramManager(temp_dir / "programs")

        # Create project config
        project_config = {
            "name": "integration_test",
            "type": "lora",
            "description": "Integration test project",
        }
        config_manager.save_project_config("integration_test", project_config)

        # Publish program for the project
        program_manager.publish_program(
            name="integration_program",
            program=mock_dspy_module,
            project="integration_test",
            signature_name="test_signature",
            description="Integration test program",
        )

        # Verify integration
        projects = config_manager.list_projects()
        assert "integration_test" in projects

        programs = program_manager.list_programs(project="integration_test")
        assert len(programs) == 1
        assert programs[0]["name"] == "integration_program"

    def test_migration_with_config_changes(self, temp_dir):
        """Test migrations that modify configuration."""
        config_manager = ConfigManager(temp_dir / "config")
        migration_manager = MigrationManager(temp_dir / "migrations")

        # Create migration that modifies config
        def config_migration(context):
            # Simulate config modification
            config = config_manager.load_global_config()
            config["framework"]["optimization_level"] = 3
            config_manager.save_global_config(config)
            return True

        migration = Migration(
            id="config_update_test",
            version_from="1.0.0",
            version_to="2.0.0",
            description="Update config optimization level",
            migration_func=config_migration,
        )

        migration_manager.register_migration(migration)

        # Apply migration
        context = {"config_manager": config_manager}
        success = migration_manager.apply_migration("config_update_test", context)

        assert success == True

        # Verify config was updated
        updated_config = config_manager.load_global_config()
        assert updated_config["framework"]["optimization_level"] == 3
