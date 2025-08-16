"""
Signature registry for DSPy Integration Framework.
"""

# Standard library imports
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Type

# Third-party imports
import dspy

# Local imports
from .exceptions import ModuleRegistrationError, SignatureValidationError
from .interfaces import SignatureRegistryInterface

logger = logging.getLogger(__name__)


class SignatureRegistry(SignatureRegistryInterface):
    """Registry for managing project-specific DSPy signatures."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the signature registry."""
        self.cache_dir = cache_dir or Path(".dspy_cache/signatures")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._signatures: dict[str, dict[str, Type]] = {}
        self._metadata: dict[str, dict[str, str | int | float | bool]] = {}

        # Load existing signatures from cache
        self._load_from_cache()

    def register_project(self, project_name: str, signatures: dict[str, Type]) -> None:
        """Register signatures for a project."""
        try:
            # Validate all signatures before registering
            for sig_name, sig_class in signatures.items():
                if not self.validate_signature(sig_class):
                    raise SignatureValidationError(
                        f"Invalid signature {sig_name} for project {project_name}"
                    )

            # Register signatures
            if project_name not in self._signatures:
                self._signatures[project_name] = {}
                self._metadata[project_name] = {}

            self._signatures[project_name].update(signatures)

            # Update metadata
            for sig_name in signatures.keys():
                self._metadata[project_name][sig_name] = {
                    "registered_at": datetime.now().isoformat(),
                    "signature_type": signatures[sig_name].__name__,
                    "module": signatures[sig_name].__module__,
                }

            # Save to cache
            self._save_to_cache()

            logger.info("Registered %d signatures for project %s", len(signatures), project_name)

        except Exception as e:
            logger.error("Failed to register signatures for project %s: %s", project_name, e)
            raise ModuleRegistrationError("Signature registration failed") from e

    def get_project_signatures(self, project_name: str) -> dict[str, Type]:
        """Get signatures for a specific project."""
        return self._signatures.get(project_name, {})

    def get_all_signatures(self) -> dict[str, dict[str, Type]]:
        """Get all registered signatures."""
        return self._signatures.copy()

    def validate_signature(self, signature: Type) -> bool:
        """Validate a DSPy signature."""
        try:
            # Check if it's a DSPy signature class
            if not issubclass(signature, dspy.Signature):
                logger.warning("Signature %s is not a DSPy Signature subclass", signature)
                return False

            # Check if it has required attributes
            if not hasattr(signature, "__annotations__"):
                logger.warning("Signature %s has no annotations", signature)
                return False

            # Validate field annotations
            annotations = getattr(signature, "__annotations__", {})
            has_input = False
            has_output = False

            for field_name, field_type in annotations.items():
                # Check for InputField and OutputField
                if hasattr(signature, field_name):
                    field_obj = getattr(signature, field_name)
                    if isinstance(field_obj, dspy.InputField):
                        has_input = True
                    elif isinstance(field_obj, dspy.OutputField):
                        has_output = True

            if not has_input:
                logger.warning("Signature %s has no input fields", signature)
                return False

            if not has_output:
                logger.warning("Signature %s has no output fields", signature)
                return False

            return True

        except Exception as e:
            logger.error("Signature validation failed for %s: %s", signature, e)
            return False

    def get_signature_metadata(
        self, project_name: str, signature_name: str
    ) -> dict[str, Any | None]:
        """Get metadata for a specific signature."""
        project_metadata = self._metadata.get(project_name, {})
        return project_metadata.get(signature_name)

    def list_projects(self) -> list[str]:
        """List all registered projects."""
        return list(self._signatures.keys())

    def list_project_signatures(self, project_name: str) -> list[str]:
        """List signature names for a specific project."""
        return list(self._signatures.get(project_name, {}).keys())

    def remove_project(self, project_name: str) -> None:
        """Remove all signatures for a project."""
        if project_name in self._signatures:
            del self._signatures[project_name]
            del self._metadata[project_name]
            self._save_to_cache()
            logger.info("Removed project %s from registry", project_name)

    def remove_signature(self, project_name: str, signature_name: str) -> None:
        """Remove a specific signature from a project."""
        if project_name in self._signatures and signature_name in self._signatures[project_name]:
            del self._signatures[project_name][signature_name]
            del self._metadata[project_name][signature_name]
            self._save_to_cache()
            logger.info("Removed signature %s from project %s", signature_name, project_name)

    def search_signatures(self, query: str) -> dict[str, list[str]]:
        """Search for signatures by name or metadata."""
        results = {}
        query_lower = query.lower()

        for project_name, signatures in self._signatures.items():
            matching_signatures = []

            for sig_name, sig_class in signatures.items():
                # Search in signature name
                if query_lower in sig_name.lower():
                    matching_signatures.append(sig_name)
                    continue

                # Search in signature class name
                if query_lower in sig_class.__name__.lower():
                    matching_signatures.append(sig_name)
                    continue

                # Search in metadata
                metadata = self.get_signature_metadata(project_name, sig_name)
                if metadata:
                    metadata_str = json.dumps(metadata).lower()
                    if query_lower in metadata_str:
                        matching_signatures.append(sig_name)

            if matching_signatures:
                results[project_name] = matching_signatures

        return results

    def get_registry_stats(self) -> dict[str, str | int | float | bool]:
        """Get statistics about the registry."""
        total_signatures = sum(len(sigs) for sigs in self._signatures.values())

        return {
            "total_projects": len(self._signatures),
            "total_signatures": total_signatures,
            "projects": {
                project: len(signatures) for project, signatures in self._signatures.items()
            },
            "cache_dir": str(self.cache_dir),
            "last_updated": datetime.now().isoformat(),
        }

    def _save_to_cache(self) -> None:
        """Save registry to cache."""
        try:
            # Save metadata as JSON
            metadata_file = self.cache_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)

            # Save signature class references (simplified)
            registry_file = self.cache_dir / "registry.json"
            registry_data = {}

            for project_name, signatures in self._signatures.items():
                registry_data[project_name] = {}
                for sig_name, sig_class in signatures.items():
                    registry_data[project_name][sig_name] = {
                        "class_name": sig_class.__name__,
                        "module": sig_class.__module__,
                    }

            with open(registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)

            logger.debug("Registry saved to cache")

        except Exception as e:
            logger.error("Failed to save registry to cache: %s", e)

    def _load_from_cache(self) -> None:
        """Load registry from cache."""
        try:
            metadata_file = self.cache_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    self._metadata = json.load(f)

                logger.debug("Registry metadata loaded from cache")

            # Note: Signature classes need to be re-registered as they can't be easily serialized
            # This is by design - signatures should be registered at runtime

        except Exception as e:
            logger.warning("Failed to load registry from cache: %s", e)
            self._metadata = {}

    def clear_cache(self) -> None:
        """Clear the registry cache."""
        try:
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file():
                    cache_file.unlink()

            logger.info("Registry cache cleared")

        except Exception as e:
            logger.error("Failed to clear registry cache: %s", e)

    def export_registry(self, export_path: Path) -> None:
        """Export registry to a file."""
        try:
            export_data = {
                "signatures": {},
                "metadata": self._metadata,
                "stats": self.get_registry_stats(),
                "exported_at": datetime.now().isoformat(),
            }

            # Export signature information (not the actual classes)
            for project_name, signatures in self._signatures.items():
                export_data["signatures"][project_name] = {}
                for sig_name, sig_class in signatures.items():
                    export_data["signatures"][project_name][sig_name] = {
                        "class_name": sig_class.__name__,
                        "module": sig_class.__module__,
                        "docstring": sig_class.__doc__,
                        "annotations": str(getattr(sig_class, "__annotations__", {})),
                    }

            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info("Registry exported to %s", export_path)

        except Exception as e:
            logger.error("Failed to export registry: %s", e)
            raise ModuleRegistrationError("Registry export failed") from e
