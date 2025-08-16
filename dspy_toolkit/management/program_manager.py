"""
Program manager for sharing optimized DSPy programs across projects.
"""

import hashlib
import json
import logging
import pickle
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy

from ..exceptions import DSPyIntegrationError

logger = logging.getLogger(__name__)


@dataclass
class ProgramVersion:
    """Version information for a DSPy program."""

    version: str
    created_at: str
    created_by: str
    description: str
    performance_metrics: dict[str, float]
    optimization_config: dict[str, str | int | float | bool]
    program_hash: str
    tags: list[str]


@dataclass
class ProgramMetadata:
    """Metadata for a DSPy program."""

    name: str
    project: str
    signature_name: str
    module_type: str
    current_version: str
    versions: list[ProgramVersion]
    created_at: str
    updated_at: str
    total_downloads: int
    rating: float
    description: str


class ProgramManager:
    """Manager for sharing and versioning optimized DSPy programs."""

    def __init__(self, programs_dir: Path | None = None):
        """Initialize program manager."""
        self.programs_dir = programs_dir or Path(".dspy_programs")
        self.programs_dir.mkdir(parents=True, exist_ok=True)

        # Directory structure
        self.metadata_dir = self.programs_dir / "metadata"
        self.programs_store = self.programs_dir / "programs"
        self.versions_dir = self.programs_dir / "versions"

        for directory in [self.metadata_dir, self.programs_store, self.versions_dir]:
            directory.mkdir(exist_ok=True)

        # Program registry
        self.registry_file = self.programs_dir / "registry.json"
        self._load_registry()

        logger.info("Program manager initialized with directory: %s", self.programs_dir)

    def _load_registry(self):
        """Load program registry."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    self.registry = json.load(f)
            except Exception as e:
                logger.error("Failed to load program registry: %s", e)
                self.registry = {}
        else:
            self.registry = {}

    def _save_registry(self):
        """Save program registry."""
        try:
            with open(self.registry_file, "w") as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error("Failed to save program registry: %s", e)
            raise DSPyIntegrationError(f"Registry save failed: {e}") from e

    def _calculate_program_hash(self, program: dspy.Module) -> str:
        """Calculate hash for a DSPy program."""
        try:
            # Serialize program to bytes for hashing
            program_bytes = pickle.dumps(program)
            return hashlib.sha256(program_bytes).hexdigest()
        except Exception as e:
            logger.warning("Failed to calculate program hash: %s", e)
            return "unknown"

    def publish_program(
        self,
        name: str,
        program: dspy.Module,
        project: str,
        signature_name: str,
        version: str = "1.0.0",
        description: str = "",
        performance_metrics: dict[str, float | None] = None,
        optimization_config: dict[str, Any | None] = None,
        tags: list[str | None] = None,
        created_by: str = "unknown",
    ) -> str:
        """Publish a DSPy program to the shared repository."""

        try:
            program_hash = self._calculate_program_hash(program)

            # Create version info
            version_info = ProgramVersion(
                version=version,
                created_at=datetime.now().isoformat(),
                created_by=created_by,
                description=description,
                performance_metrics=performance_metrics or {},
                optimization_config=optimization_config or {},
                program_hash=program_hash,
                tags=tags or [],
            )

            # Check if program already exists
            program_key = f"{project}:{name}"

            if program_key in self.registry:
                # Update existing program
                metadata = ProgramMetadata(**self.registry[program_key])

                # Check if version already exists
                existing_versions = [v.version for v in metadata.versions]
                if version in existing_versions:
                    raise DSPyIntegrationError(
                        f"Version {version} already exists for program {name}"
                    )

                metadata.versions.append(version_info)
                metadata.current_version = version
                metadata.updated_at = datetime.now().isoformat()

            else:
                # Create new program metadata
                metadata = ProgramMetadata(
                    name=name,
                    project=project,
                    signature_name=signature_name,
                    module_type=type(program).__name__,
                    current_version=version,
                    versions=[version_info],
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    total_downloads=0,
                    rating=0.0,
                    description=description,
                )

            # Save program file
            program_file = self.programs_store / f"{program_key}_{version}.pkl"
            with open(program_file, "wb") as f:
                pickle.dump(program, f)

            # Save metadata
            metadata_file = self.metadata_dir / f"{program_key}.json"
            with open(metadata_file, "w") as f:
                json.dump(asdict(metadata), f, indent=2)

            # Update registry
            self.registry[program_key] = asdict(metadata)
            self._save_registry()

            logger.info("Published program %s v%s for project %s", name, version, project)
            return program_key

        except Exception as e:
            logger.error("Failed to publish program %s: %s", name, e)
            raise DSPyIntegrationError(f"Program publishing failed: {e}") from e

    def download_program(self, name: str, project: str, version: str | None = None) -> dspy.Module:
        """Download a DSPy program from the shared repository."""

        try:
            program_key = f"{project}:{name}"

            if program_key not in self.registry:
                raise DSPyIntegrationError(f"Program {name} not found for project {project}")

            metadata = ProgramMetadata(**self.registry[program_key])

            # Determine version to download
            if version is None:
                version = metadata.current_version
            else:
                # Check if version exists
                available_versions = [v.version for v in metadata.versions]
                if version not in available_versions:
                    raise DSPyIntegrationError(f"Version {version} not found for program {name}")

            # Load program file
            program_file = self.programs_store / f"{program_key}_{version}.pkl"

            if not program_file.exists():
                raise DSPyIntegrationError(f"Program file not found: {program_file}")

            with open(program_file, "rb") as f:
                program = pickle.load(f)

            # Update download count
            metadata.total_downloads += 1
            metadata_file = self.metadata_dir / f"{program_key}.json"
            with open(metadata_file, "w") as f:
                json.dump(asdict(metadata), f, indent=2)

            # Update registry
            self.registry[program_key] = asdict(metadata)
            self._save_registry()

            logger.info("Downloaded program %s v%s for project %s", name, version, project)
            return program

        except Exception as e:
            logger.error("Failed to download program %s: %s", name, e)
            raise DSPyIntegrationError(f"Program download failed: {e}") from e

    def list_programs(
        self, project: str | None = None
    ) -> list[dict[str, str | int | float | bool]]:
        """List available programs."""
        programs = []

        for _, metadata_dict in self.registry.items():
            metadata = ProgramMetadata(**metadata_dict)

            if project is None or metadata.project == project:
                program_info = {
                    "name": metadata.name,
                    "project": metadata.project,
                    "current_version": metadata.current_version,
                    "description": metadata.description,
                    "total_downloads": metadata.total_downloads,
                    "rating": metadata.rating,
                    "created_at": metadata.created_at,
                    "updated_at": metadata.updated_at,
                    "num_versions": len(metadata.versions),
                }
                programs.append(program_info)

        return programs

    def get_program_info(self, name: str, project: str) -> dict[str, Any | None]:
        """Get detailed information about a program."""
        program_key = f"{project}:{name}"

        if program_key not in self.registry:
            return None

        metadata = ProgramMetadata(**self.registry[program_key])
        return asdict(metadata)

    def get_program_versions(
        self, name: str, project: str
    ) -> list[dict[str, str | int | float | bool]]:
        """Get all versions of a program."""
        program_key = f"{project}:{name}"

        if program_key not in self.registry:
            return []

        metadata = ProgramMetadata(**self.registry[program_key])
        return [asdict(version) for version in metadata.versions]

    def delete_program(self, name: str, project: str, version: str | None = None):
        """Delete a program or specific version."""
        try:
            program_key = f"{project}:{name}"

            if program_key not in self.registry:
                raise DSPyIntegrationError(f"Program {name} not found for project {project}")

            metadata = ProgramMetadata(**self.registry[program_key])

            if version is None:
                # Delete entire program
                # Remove all program files
                for v in metadata.versions:
                    program_file = self.programs_store / f"{program_key}_{v.version}.pkl"
                    if program_file.exists():
                        program_file.unlink()

                # Remove metadata file
                metadata_file = self.metadata_dir / f"{program_key}.json"
                if metadata_file.exists():
                    metadata_file.unlink()

                # Remove from registry
                del self.registry[program_key]

                logger.info("Deleted program %s for project %s", name, project)

            else:
                # Delete specific version
                # Check if version exists
                version_to_delete = None
                for v in metadata.versions:
                    if v.version == version:
                        version_to_delete = v
                        break

                if version_to_delete is None:
                    raise DSPyIntegrationError(f"Version {version} not found for program {name}")

                # Remove program file
                program_file = self.programs_store / f"{program_key}_{version}.pkl"
                if program_file.exists():
                    program_file.unlink()

                # Remove version from metadata
                metadata.versions = [v for v in metadata.versions if v.version != version]

                if not metadata.versions:
                    # No versions left, delete entire program
                    self.delete_program(name, project)
                    return

                # Update current version if necessary
                if metadata.current_version == version:
                    # Set to latest version
                    latest_version = max(metadata.versions, key=lambda v: v.created_at)
                    metadata.current_version = latest_version.version

                metadata.updated_at = datetime.now().isoformat()

                # Save updated metadata
                metadata_file = self.metadata_dir / f"{program_key}.json"
                with open(metadata_file, "w") as f:
                    json.dump(asdict(metadata), f, indent=2)

                # Update registry
                self.registry[program_key] = asdict(metadata)

                logger.info(
                    "Deleted version %s of program %s for project %s", version, name, project
                )

            self._save_registry()

        except Exception as e:
            logger.error("Failed to delete program %s: %s", name, e)
            raise DSPyIntegrationError(f"Program deletion failed: {e}") from e

    def search_programs(
        self,
        query: str,
        project: str | None = None,
        tags: list[str | None] = None,
    ) -> list[dict[str, str | int | float | bool]]:
        """Search for programs by name, description, or tags."""
        results = []
        query_lower = query.lower()

        for program_key, metadata_dict in self.registry.items():
            metadata = ProgramMetadata(**metadata_dict)

            # Filter by project if specified
            if project and metadata.project != project:
                continue

            # Search in name and description
            match_found = False

            if query_lower in metadata.name.lower() or query_lower in metadata.description.lower():
                match_found = True

            # Search in tags
            if not match_found and tags:
                for version in metadata.versions:
                    if any(tag in version.tags for tag in tags):
                        match_found = True
                        break

            if match_found:
                program_info = {
                    "name": metadata.name,
                    "project": metadata.project,
                    "current_version": metadata.current_version,
                    "description": metadata.description,
                    "total_downloads": metadata.total_downloads,
                    "rating": metadata.rating,
                    "relevance_score": self._calculate_relevance(metadata, query, tags),
                }
                results.append(program_info)

        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    def _calculate_relevance(
        self, metadata: ProgramMetadata, query: str, tags: list[str | None]
    ) -> float:
        """Calculate relevance score for search results."""
        score = 0.0
        query_lower = query.lower()

        # Name match (highest weight)
        if query_lower in metadata.name.lower():
            score += 10.0

        # Description match
        if query_lower in metadata.description.lower():
            score += 5.0

        # Tag matches
        if tags:
            for version in metadata.versions:
                matching_tags = set(tags) & set(version.tags)
                score += len(matching_tags) * 3.0

        # Popularity boost
        score += min(metadata.total_downloads * 0.1, 5.0)

        # Rating boost
        score += metadata.rating

        return score

    def rate_program(self, name: str, project: str, rating: float):
        """Rate a program (simplified rating system)."""
        try:
            if not 0.0 <= rating <= 5.0:
                raise DSPyIntegrationError("Rating must be between 0.0 and 5.0")

            program_key = f"{project}:{name}"

            if program_key not in self.registry:
                raise DSPyIntegrationError(f"Program {name} not found for project {project}")

            metadata = ProgramMetadata(**self.registry[program_key])

            # Simple average rating (in production, you'd want a more sophisticated system)
            current_rating = metadata.rating
            total_ratings = metadata.total_downloads  # Simplified assumption

            if total_ratings == 0:
                new_rating = rating
            else:
                new_rating = ((current_rating * total_ratings) + rating) / (total_ratings + 1)

            metadata.rating = new_rating
            metadata.updated_at = datetime.now().isoformat()

            # Save updated metadata
            metadata_file = self.metadata_dir / f"{program_key}.json"
            with open(metadata_file, "w") as f:
                json.dump(asdict(metadata), f, indent=2)

            # Update registry
            self.registry[program_key] = asdict(metadata)
            self._save_registry()

            logger.info("Rated program %s with %s stars", name, rating)

        except Exception as e:
            logger.error("Failed to rate program %s: %s", name, e)
            raise DSPyIntegrationError(f"Program rating failed: {e}") from e

    def export_programs(self, export_dir: Path, project: str | None = None):
        """Export programs to a directory."""
        try:
            export_dir.mkdir(parents=True, exist_ok=True)

            programs_to_export = []

            for program_key, metadata_dict in self.registry.items():
                metadata = ProgramMetadata(**metadata_dict)

                if project is None or metadata.project == project:
                    programs_to_export.append((program_key, metadata))

            # Export metadata
            export_registry = {}
            for program_key, metadata in programs_to_export:
                export_registry[program_key] = asdict(metadata)

            with open(export_dir / "registry.json", "w") as f:
                json.dump(export_registry, f, indent=2)

            # Export program files
            programs_export_dir = export_dir / "programs"
            programs_export_dir.mkdir(exist_ok=True)

            for program_key, metadata in programs_to_export:
                for version in metadata.versions:
                    src_file = self.programs_store / f"{program_key}_{version.version}.pkl"
                    dst_file = programs_export_dir / f"{program_key}_{version.version}.pkl"

                    if src_file.exists():
                        shutil.copy2(src_file, dst_file)

            logger.info("Exported %d programs to %s", len(programs_to_export), export_dir)

        except Exception as e:
            logger.error("Failed to export programs: %s", e)
            raise DSPyIntegrationError(f"Program export failed: {e}") from e

    def import_programs(self, import_dir: Path):
        """Import programs from a directory."""
        try:
            registry_file = import_dir / "registry.json"
            programs_import_dir = import_dir / "programs"

            if not registry_file.exists():
                raise DSPyIntegrationError("Registry file not found in import directory")

            # Load import registry
            with open(registry_file, "r") as f:
                import_registry = json.load(f)

            imported_count = 0

            for program_key, metadata_dict in import_registry.items():
                metadata = ProgramMetadata(**metadata_dict)

                # Import program files
                for version in metadata.versions:
                    src_file = programs_import_dir / f"{program_key}_{version.version}.pkl"
                    dst_file = self.programs_store / f"{program_key}_{version.version}.pkl"

                    if src_file.exists():
                        shutil.copy2(src_file, dst_file)

                # Save metadata
                metadata_file = self.metadata_dir / f"{program_key}.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata_dict, f, indent=2)

                # Update registry
                self.registry[program_key] = metadata_dict
                imported_count += 1

            self._save_registry()

            logger.info("Imported %d programs from %s", imported_count, import_dir)

        except Exception as e:
            logger.error("Failed to import programs: %s", e)
            raise DSPyIntegrationError(f"Program import failed: {e}") from e

    def get_manager_stats(self) -> dict[str, str | int | float | bool]:
        """Get program manager statistics."""
        try:
            total_programs = len(self.registry)
            total_versions = sum(len(metadata["versions"]) for metadata in self.registry.values())
            total_downloads = sum(
                metadata["total_downloads"] for metadata in self.registry.values()
            )

            # Project breakdown
            projects = {}
            for metadata_dict in self.registry.values():
                project = metadata_dict["project"]
                if project not in projects:
                    projects[project] = 0
                projects[project] += 1

            # Calculate storage usage
            storage_usage = 0
            try:
                for file_path in self.programs_store.rglob("*.pkl"):
                    storage_usage += file_path.stat().st_size
            except Exception:
                storage_usage = 0

            return {
                "total_programs": total_programs,
                "total_versions": total_versions,
                "total_downloads": total_downloads,
                "projects": projects,
                "storage_usage_mb": storage_usage / (1024 * 1024),
                "programs_dir": str(self.programs_dir),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("Failed to get manager stats: %s", e)
            return {"error": str(e)}

    def cleanup_old_versions(self, keep_versions: int = 5):
        """Clean up old versions of programs."""
        try:
            cleaned_count = 0

            for program_key, metadata_dict in self.registry.items():
                metadata = ProgramMetadata(**metadata_dict)

                if len(metadata.versions) > keep_versions:
                    # Sort versions by creation date
                    sorted_versions = sorted(
                        metadata.versions, key=lambda v: v.created_at, reverse=True
                    )

                    # Keep only the latest versions
                    versions_to_keep = sorted_versions[:keep_versions]
                    versions_to_delete = sorted_versions[keep_versions:]

                    # Delete old version files
                    for version in versions_to_delete:
                        program_file = self.programs_store / f"{program_key}_{version.version}.pkl"
                        if program_file.exists():
                            program_file.unlink()
                            cleaned_count += 1

                    # Update metadata
                    metadata.versions = versions_to_keep
                    metadata.updated_at = datetime.now().isoformat()

                    # Save updated metadata
                    metadata_file = self.metadata_dir / f"{program_key}.json"
                    with open(metadata_file, "w") as f:
                        json.dump(asdict(metadata), f, indent=2)

                    # Update registry
                    self.registry[program_key] = asdict(metadata)

            self._save_registry()

            logger.info("Cleaned up %d old program versions", cleaned_count)
            return cleaned_count

        except Exception as e:
            logger.error("Failed to cleanup old versions: %s", e)
            raise DSPyIntegrationError(f"Cleanup failed: {e}") from e
