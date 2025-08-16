"""
Migration manager for DSPy Integration Framework component versioning.
"""

import importlib.util
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from ..exceptions import DSPyIntegrationError

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Migration definition."""

    id: str
    version_from: str
    version_to: str
    description: str
    migration_func: Callable
    rollback_func: Callable | None = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class MigrationRecord:
    """Record of applied migration."""

    migration_id: str
    applied_at: str
    applied_by: str
    success: bool
    error_message: str | None = None
    rollback_available: bool = False


class MigrationManager:
    """Manager for component migrations and versioning."""

    def __init__(self, migrations_dir: Path | None = None):
        """Initialize migration manager."""
        self.migrations_dir = migrations_dir or Path(".dspy_migrations")
        self.migrations_dir.mkdir(parents=True, exist_ok=True)

        # Migration files
        self.migrations_file = self.migrations_dir / "migrations.json"
        self.applied_migrations_file = self.migrations_dir / "applied.json"
        self.migration_scripts_dir = self.migrations_dir / "scripts"
        self.migration_scripts_dir.mkdir(exist_ok=True)

        # Migration registry
        self.migrations: dict[str, Migration] = {}
        self.applied_migrations: list[MigrationRecord] = []

        # Load existing data
        self._load_migrations()
        self._load_applied_migrations()

        # Register built-in migrations
        self._register_builtin_migrations()

        logger.info("Migration manager initialized with directory: %s", self.migrations_dir)

    def _load_migrations(self):
        """Load migration definitions."""
        if self.migrations_file.exists():
            try:
                with open(self.migrations_file, "r") as f:
                    migrations_data = json.load(f)

                # Note: We can't serialize functions, so we only load metadata
                # Actual migration functions need to be registered separately
                for migration_id, migration_dict in migrations_data.items():
                    # Create placeholder migration (function will be registered later)
                    migration = Migration(
                        id=migration_dict["id"],
                        version_from=migration_dict["version_from"],
                        version_to=migration_dict["version_to"],
                        description=migration_dict["description"],
                        migration_func=lambda context: None,  # Placeholder
                        created_at=migration_dict["created_at"],
                    )
                    self.migrations[migration_id] = migration

            except Exception as e:
                logger.error("Failed to load migrations: %s", e)
                self.migrations = {}

    def _save_migrations(self):
        """Save migration definitions."""
        try:
            migrations_data = {}
            for migration_id, migration in self.migrations.items():
                migrations_data[migration_id] = {
                    "id": migration.id,
                    "version_from": migration.version_from,
                    "version_to": migration.version_to,
                    "description": migration.description,
                    "created_at": migration.created_at,
                }

            with open(self.migrations_file, "w") as f:
                json.dump(migrations_data, f, indent=2)

        except Exception as e:
            logger.error("Failed to save migrations: %s", e)
            raise DSPyIntegrationError(f"Migration save failed: {e}") from e

    def _load_applied_migrations(self):
        """Load applied migration records."""
        if self.applied_migrations_file.exists():
            try:
                with open(self.applied_migrations_file, "r") as f:
                    applied_data = json.load(f)

                self.applied_migrations = [MigrationRecord(**record) for record in applied_data]

            except Exception as e:
                logger.error("Failed to load applied migrations: %s", e)
                self.applied_migrations = []

    def _save_applied_migrations(self):
        """Save applied migration records."""
        try:
            applied_data = [asdict(record) for record in self.applied_migrations]

            with open(self.applied_migrations_file, "w") as f:
                json.dump(applied_data, f, indent=2)

        except Exception as e:
            logger.error("Failed to save applied migrations: %s", e)
            raise DSPyIntegrationError(f"Applied migrations save failed: {e}") from e

    def register_migration(self, migration: Migration):
        """Register a migration."""
        try:
            if migration.id in self.migrations:
                logger.warning("Migration %s already exists, overwriting", migration.id)

            self.migrations[migration.id] = migration
            self._save_migrations()

            logger.info(
                "Registered migration: %s (%s -> %s)",
                migration.id,
                migration.version_from,
                migration.version_to,
            )

        except Exception as e:
            logger.error("Failed to register migration %s: %s", migration.id, e)
            raise DSPyIntegrationError(f"Migration registration failed: {e}") from e

    def load_migration_script(self, script_path: Path) -> Migration:
        """Load migration from Python script."""
        try:
            # Load the migration script as a module
            spec = importlib.util.spec_from_file_location("migration_module", script_path)
            if spec is None or spec.loader is None:
                raise DSPyIntegrationError(f"Could not load migration script: {script_path}")

            migration_module = importlib.util.module_from_spec(spec)
            sys.modules["migration_module"] = migration_module
            spec.loader.exec_module(migration_module)

            # Extract migration information
            if not hasattr(migration_module, "MIGRATION_ID"):
                raise DSPyIntegrationError("Migration script must define MIGRATION_ID")

            if not hasattr(migration_module, "migrate"):
                raise DSPyIntegrationError("Migration script must define migrate() function")

            migration = Migration(
                id=migration_module.MIGRATION_ID,
                version_from=getattr(migration_module, "VERSION_FROM", "unknown"),
                version_to=getattr(migration_module, "VERSION_TO", "unknown"),
                description=getattr(migration_module, "DESCRIPTION", "No description"),
                migration_func=migration_module.migrate,
                rollback_func=getattr(migration_module, "rollback", None),
            )

            return migration

        except Exception as e:
            logger.error("Failed to load migration script %s: %s", script_path, e)
            raise DSPyIntegrationError(f"Migration script loading failed: {e}") from e

    def create_migration_script(
        self, migration_id: str, version_from: str, version_to: str, description: str
    ) -> Path:
        """Create a migration script template."""
        try:
            script_name = f"{migration_id}.py"
            script_path = self.migration_scripts_dir / script_name

            if script_path.exists():
                raise DSPyIntegrationError(f"Migration script already exists: {script_path}")

            script_template = f'''"""
Migration: {migration_id}
From: {version_from}
To: {version_to}
Description: {description}
"""

MIGRATION_ID = "{migration_id}"
VERSION_FROM = "{version_from}"
VERSION_TO = "{version_to}"
DESCRIPTION = "{description}"


def migrate(context):
    """
    Perform the migration.
    
    Args:
        context: Migration context containing framework components
    
    Returns:
        bool: True if migration successful, False otherwise
    """
    # TODO: Implement migration logic
    print(f"Running migration {{MIGRATION_ID}}")
    
    # Example migration operations:
    # - Update configuration files
    # - Migrate data structures
    # - Update component interfaces
    # - Transform stored programs
    
    return True


def rollback(context):
    """
    Rollback the migration (optional).
    
    Args:
        context: Migration context containing framework components
    
    Returns:
        bool: True if rollback successful, False otherwise
    """
    # TODO: Implement rollback logic
    print(f"Rolling back migration {{MIGRATION_ID}}")
    
    return True
'''

            with open(script_path, "w") as f:
                f.write(script_template)

            logger.info("Created migration script: %s", script_path)
            return script_path

        except Exception as e:
            logger.error("Failed to create migration script: %s", e)
            raise DSPyIntegrationError(f"Migration script creation failed: {e}") from e

    def get_pending_migrations(self, current_version: str) -> list[Migration]:
        """Get migrations that need to be applied."""
        applied_ids = {record.migration_id for record in self.applied_migrations if record.success}

        pending = []
        for migration in self.migrations.values():
            if migration.id not in applied_ids and self._should_apply_migration(
                migration, current_version
            ):
                pending.append(migration)

        # Sort by version order
        pending.sort(key=lambda m: m.version_to)
        return pending

    def _should_apply_migration(self, migration: Migration, current_version: str) -> bool:
        """Check if migration should be applied for current version."""
        # Simple version comparison (in production, use proper version parsing)
        try:
            current_parts = [int(x) for x in current_version.split(".")]
            from_parts = [int(x) for x in migration.version_from.split(".")]
            to_parts = [int(x) for x in migration.version_to.split(".")]

            return current_parts >= from_parts and current_parts < to_parts
        except ValueError:
            # Fallback to string comparison
            return (
                current_version >= migration.version_from and current_version < migration.version_to
            )

    def apply_migration(
        self,
        migration_id: str,
        context: dict[str, str | int | float | bool],
        applied_by: str = "system",
    ) -> bool:
        """Apply a specific migration."""
        try:
            if migration_id not in self.migrations:
                raise DSPyIntegrationError(f"Migration {migration_id} not found")

            migration = self.migrations[migration_id]

            # Check if already applied
            applied_ids = {
                record.migration_id for record in self.applied_migrations if record.success
            }
            if migration_id in applied_ids:
                logger.info("Migration %s already applied", migration_id)
                return True

            logger.info("Applying migration %s: %s", migration_id, migration.description)

            # Apply migration
            success = migration.migration_func(context)

            # Record migration
            record = MigrationRecord(
                migration_id=migration_id,
                applied_at=datetime.now().isoformat(),
                applied_by=applied_by,
                success=success,
                rollback_available=migration.rollback_func is not None,
            )

            if not success:
                record.error_message = "Migration function returned False"

            self.applied_migrations.append(record)
            self._save_applied_migrations()

            if success:
                logger.info("Migration %s applied successfully", migration_id)
            else:
                logger.error("Migration %s failed", migration_id)

            return success

        except Exception as e:
            logger.error("Failed to apply migration %s: %s", migration_id, e)

            # Record failed migration
            record = MigrationRecord(
                migration_id=migration_id,
                applied_at=datetime.now().isoformat(),
                applied_by=applied_by,
                success=False,
                error_message=str(e),
                rollback_available=False,
            )

            self.applied_migrations.append(record)
            self._save_applied_migrations()

            return False

    def rollback_migration(
        self,
        migration_id: str,
        context: dict[str, str | int | float | bool],
        applied_by: str = "system",
    ) -> bool:
        """Rollback a specific migration."""
        try:
            if migration_id not in self.migrations:
                raise DSPyIntegrationError(f"Migration {migration_id} not found")

            migration = self.migrations[migration_id]

            if migration.rollback_func is None:
                raise DSPyIntegrationError(f"Migration {migration_id} has no rollback function")

            # Check if migration was applied
            applied_record = None
            for record in self.applied_migrations:
                if record.migration_id == migration_id and record.success:
                    applied_record = record
                    break

            if applied_record is None:
                logger.warning("Migration %s was not applied, nothing to rollback", migration_id)
                return True

            logger.info("Rolling back migration %s", migration_id)

            # Perform rollback
            success = migration.rollback_func(context)

            if success:
                # Mark as rolled back (remove from applied migrations)
                self.applied_migrations = [
                    record
                    for record in self.applied_migrations
                    if not (record.migration_id == migration_id and record.success)
                ]
                self._save_applied_migrations()

                logger.info("Migration %s rolled back successfully", migration_id)
            else:
                logger.error("Migration %s rollback failed", migration_id)

            return success

        except Exception as e:
            logger.error("Failed to rollback migration %s: %s", migration_id, e)
            return False

    def apply_pending_migrations(
        self,
        current_version: str,
        context: dict[str, str | int | float | bool],
        applied_by: str = "system",
    ) -> int:
        """Apply all pending migrations."""
        pending_migrations = self.get_pending_migrations(current_version)

        if not pending_migrations:
            logger.info("No pending migrations")
            return 0

        applied_count = 0

        for migration in pending_migrations:
            try:
                success = self.apply_migration(migration.id, context, applied_by)
                if success:
                    applied_count += 1
                else:
                    logger.error("Migration %s failed, stopping migration process", migration.id)
                    break

            except Exception as e:
                logger.error("Error applying migration %s: %s", migration.id, e)
                break

        logger.info(
            "Applied %d out of %d pending migrations", applied_count, len(pending_migrations)
        )
        return applied_count

    def get_migration_status(self) -> dict[str, str | int | float | bool]:
        """Get migration system status."""
        try:
            total_migrations = len(self.migrations)
            applied_migrations = len([r for r in self.applied_migrations if r.success])
            failed_migrations = len([r for r in self.applied_migrations if not r.success])

            # Get latest applied migration
            latest_applied = None
            if self.applied_migrations:
                latest_record = max(
                    [r for r in self.applied_migrations if r.success],
                    key=lambda r: r.applied_at,
                    default=None,
                )
                if latest_record:
                    latest_applied = {
                        "migration_id": latest_record.migration_id,
                        "applied_at": latest_record.applied_at,
                    }

            return {
                "total_migrations": total_migrations,
                "applied_migrations": applied_migrations,
                "failed_migrations": failed_migrations,
                "pending_migrations": total_migrations - applied_migrations,
                "latest_applied": latest_applied,
                "migrations_dir": str(self.migrations_dir),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("Failed to get migration status: %s", e)
            return {"error": str(e)}

    def list_migrations(
        self, include_applied: bool = True
    ) -> list[dict[str, str | int | float | bool]]:
        """List all migrations with their status."""
        applied_ids = {record.migration_id for record in self.applied_migrations if record.success}
        failed_ids = {
            record.migration_id for record in self.applied_migrations if not record.success
        }

        migrations_list = []

        for migration in self.migrations.values():
            status = "pending"
            if migration.id in applied_ids:
                status = "applied"
            elif migration.id in failed_ids:
                status = "failed"

            if include_applied or status != "applied":
                migration_info = {
                    "id": migration.id,
                    "version_from": migration.version_from,
                    "version_to": migration.version_to,
                    "description": migration.description,
                    "status": status,
                    "created_at": migration.created_at,
                    "rollback_available": migration.rollback_func is not None,
                }

                # Add application details if applied
                if status in ["applied", "failed"]:
                    for record in self.applied_migrations:
                        if record.migration_id == migration.id:
                            migration_info.update(
                                {
                                    "applied_at": record.applied_at,
                                    "applied_by": record.applied_by,
                                    "error_message": record.error_message,
                                }
                            )
                            break

                migrations_list.append(migration_info)

        # Sort by creation date
        migrations_list.sort(key=lambda m: m["created_at"])
        return migrations_list

    def _register_builtin_migrations(self):
        """Register built-in migrations."""

        # Example: Configuration format migration
        def migrate_config_v1_to_v2(context):
            """Migrate configuration from v1 to v2 format."""
            try:
                # This would contain actual migration logic
                logger.info("Migrating configuration format from v1 to v2")
                return True
            except Exception as e:
                logger.error("Config migration failed: %s", e)
                return False

        def rollback_config_v2_to_v1(context):
            """Rollback configuration from v2 to v1 format."""
            try:
                logger.info("Rolling back configuration format from v2 to v1")
                return True
            except Exception as e:
                logger.error("Config rollback failed: %s", e)
                return False

        config_migration = Migration(
            id="config_v1_to_v2",
            version_from="1.0.0",
            version_to="2.0.0",
            description="Migrate configuration format from v1 to v2",
            migration_func=migrate_config_v1_to_v2,
            rollback_func=rollback_config_v2_to_v1,
        )

        self.register_migration(config_migration)

        # Example: Program storage migration
        def migrate_programs_v1_to_v2(context):
            """Migrate program storage format."""
            try:
                logger.info("Migrating program storage format")
                return True
            except Exception as e:
                logger.error("Program storage migration failed: %s", e)
                return False

        programs_migration = Migration(
            id="programs_v1_to_v2",
            version_from="1.0.0",
            version_to="2.0.0",
            description="Migrate program storage format",
            migration_func=migrate_programs_v1_to_v2,
        )

        self.register_migration(programs_migration)
