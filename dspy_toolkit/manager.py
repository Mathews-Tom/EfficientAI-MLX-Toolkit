"""
Module manager for DSPy Integration Framework.
"""

# Standard library imports
import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Third-party imports
import dspy

# Local imports
from .exceptions import DSPyIntegrationError, ModuleRegistrationError
from .interfaces import ModuleManagerInterface

logger = logging.getLogger(__name__)


class ModuleManager(ModuleManagerInterface):
    """Manager for DSPy modules with persistence and versioning."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the module manager."""
        self.cache_dir = cache_dir or Path(".dspy_cache/modules")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._modules: dict[str, dspy.Module] = {}
        self._metadata: dict[str, dict[str, str | int | float | bool]] = {}

        # Load existing modules from cache
        self._load_modules_metadata()

    def register_module(
        self, name: str, module: dspy.Module, metadata: dict[str, str | int | float | bool]
    ) -> None:
        """Register a DSPy module."""
        try:
            # Validate module
            if not isinstance(module, dspy.Module):
                raise ModuleRegistrationError(f"Object {name} is not a DSPy Module")

            # Store module
            self._modules[name] = module

            # Store metadata with additional information
            self._metadata[name] = {
                **metadata,
                "registered_at": datetime.now().isoformat(),
                "module_type": type(module).__name__,
                "module_hash": self._calculate_module_hash(module),
                "version": metadata.get("version", "1.0.0"),
            }

            # Save to disk
            self._save_module_to_disk(name, module)
            self._save_metadata()

            logger.info("Registered module %s of type %s", name, type(module).__name__)

        except Exception as e:
            logger.error("Failed to register module %s: %s", name, e)
            raise ModuleRegistrationError("Module registration failed") from e

    def get_module(self, name: str) -> dspy.Module | None:
        """Get a registered module by name."""
        if name in self._modules:
            return self._modules[name]

        # Try to load from disk if not in memory
        try:
            module = self._load_module_from_disk(name)
            if module:
                self._modules[name] = module
                return module
        except Exception as e:
            logger.warning("Failed to load module %s from disk: %s", name, e)

        return None

    def list_modules(self) -> list[str]:
        """List all registered module names."""
        # Combine in-memory and disk-stored modules
        disk_modules = set()
        try:
            for module_file in self.cache_dir.glob("*.pkl"):
                disk_modules.add(module_file.stem)
        except Exception as e:
            logger.warning("Failed to list disk modules: %s", e)

        return list(set(self._modules.keys()) | disk_modules)

    def save_module(self, name: str, path: str) -> None:
        """Save a module to a specific path."""
        module = self.get_module(name)
        if not module:
            raise ModuleRegistrationError(f"Module {name} not found")

        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Use DSPy's built-in save method if available
            if hasattr(module, "save"):
                module.save(str(save_path))
            else:
                # Fallback to pickle
                with open(save_path, "wb") as f:
                    pickle.dump(module, f)

            logger.info("Saved module %s to %s", name, path)

        except Exception as e:
            logger.error("Failed to save module %s to %s: %s", name, path, e)
            raise ModuleRegistrationError("Module save failed") from e

    def load_module(self, name: str, path: str) -> dspy.Module:
        """Load a module from a specific path."""
        try:
            load_path = Path(path)
            if not load_path.exists():
                raise ModuleRegistrationError(f"Module file {path} does not exist")

            # Try DSPy's built-in load method first
            try:
                # This would work if DSPy has a standard load method
                module = dspy.Module.load(str(load_path))
            except (AttributeError, NotImplementedError):
                # Fallback to pickle
                with open(load_path, "rb") as f:
                    module = pickle.load(f)

            # Register the loaded module
            metadata = {
                "loaded_from": str(path),
                "loaded_at": datetime.now().isoformat(),
            }
            self.register_module(name, module, metadata)

            logger.info("Loaded module %s from %s", name, path)
            return module

        except Exception as e:
            logger.error("Failed to load module %s from %s: %s", name, path, e)
            raise ModuleRegistrationError("Module load failed") from e

    def get_module_metadata(self, name: str) -> dict[str, Any | None]:
        """Get metadata for a specific module."""
        return self._metadata.get(name)

    def update_module_metadata(
        self, name: str, metadata: dict[str, str | int | float | bool]
    ) -> None:
        """Update metadata for a module."""
        if name not in self._metadata:
            raise ModuleRegistrationError(f"Module {name} not found")

        self._metadata[name].update(metadata)
        self._metadata[name]["updated_at"] = datetime.now().isoformat()
        self._save_metadata()

        logger.info("Updated metadata for module %s", name)

    def remove_module(self, name: str) -> None:
        """Remove a module from the manager."""
        try:
            # Remove from memory
            if name in self._modules:
                del self._modules[name]

            # Remove metadata
            if name in self._metadata:
                del self._metadata[name]

            # Remove from disk
            module_file = self.cache_dir / f"{name}.pkl"
            if module_file.exists():
                module_file.unlink()

            self._save_metadata()
            logger.info("Removed module %s", name)

        except Exception as e:
            logger.error("Failed to remove module %s: %s", name, e)
            raise ModuleRegistrationError("Module removal failed") from e

    def search_modules(self, query: str) -> list[str]:
        """Search for modules by name or metadata."""
        results = []
        query_lower = query.lower()

        for module_name in self.list_modules():
            # Search in module name
            if query_lower in module_name.lower():
                results.append(module_name)
                continue

            # Search in metadata
            metadata = self.get_module_metadata(module_name)
            if metadata:
                metadata_str = json.dumps(metadata).lower()
                if query_lower in metadata_str:
                    results.append(module_name)

        return results

    def get_module_versions(self, name: str) -> list[str]:
        """Get all versions of a module."""
        # This is a simplified implementation
        # In a full implementation, you might store multiple versions
        metadata = self.get_module_metadata(name)
        if metadata and "version" in metadata:
            return [metadata["version"]]
        return []

    def create_module_snapshot(self, name: str, snapshot_name: str) -> None:
        """Create a snapshot of a module."""
        module = self.get_module(name)
        if not module:
            raise ModuleRegistrationError(f"Module {name} not found")

        try:
            snapshot_path = self.cache_dir / "snapshots" / f"{name}_{snapshot_name}.pkl"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)

            with open(snapshot_path, "wb") as f:
                pickle.dump(module, f)

            # Update metadata
            metadata = self.get_module_metadata(name) or {}
            if "snapshots" not in metadata:
                metadata["snapshots"] = []

            metadata["snapshots"].append(
                {
                    "name": snapshot_name,
                    "created_at": datetime.now().isoformat(),
                    "path": str(snapshot_path),
                }
            )

            self.update_module_metadata(name, metadata)
            logger.info("Created snapshot %s for module %s", snapshot_name, name)

        except Exception as e:
            logger.error("Failed to create snapshot for module %s: %s", name, e)
            raise ModuleRegistrationError("Snapshot creation failed") from e

    def restore_module_snapshot(self, name: str, snapshot_name: str) -> None:
        """Restore a module from a snapshot."""
        metadata = self.get_module_metadata(name)
        if not metadata or "snapshots" not in metadata:
            raise ModuleRegistrationError(f"No snapshots found for module {name}")

        # Find the snapshot
        snapshot_info = None
        for snapshot in metadata["snapshots"]:
            if snapshot["name"] == snapshot_name:
                snapshot_info = snapshot
                break

        if not snapshot_info:
            raise ModuleRegistrationError(f"Snapshot {snapshot_name} not found for module {name}")

        try:
            snapshot_path = Path(snapshot_info["path"])
            if not snapshot_path.exists():
                raise ModuleRegistrationError(f"Snapshot file {snapshot_path} does not exist")

            with open(snapshot_path, "rb") as f:
                module = pickle.load(f)

            # Register the restored module
            restore_metadata = {
                **metadata,
                "restored_from": snapshot_name,
                "restored_at": datetime.now().isoformat(),
            }
            self.register_module(name, module, restore_metadata)

            logger.info("Restored module %s from snapshot %s", name, snapshot_name)

        except Exception as e:
            logger.error("Failed to restore module %s from snapshot %s: %s", name, snapshot_name, e)
            raise ModuleRegistrationError("Snapshot restoration failed") from e

    def get_manager_stats(self) -> dict[str, str | int | float | bool]:
        """Get statistics about the module manager."""
        total_modules = len(self.list_modules())
        memory_modules = len(self._modules)

        # Calculate total disk usage
        total_size = 0
        try:
            for module_file in self.cache_dir.rglob("*.pkl"):
                total_size += module_file.stat().st_size
        except Exception as e:
            logger.warning("Failed to calculate disk usage: %s", e)

        return {
            "total_modules": total_modules,
            "memory_modules": memory_modules,
            "disk_modules": total_modules - memory_modules,
            "cache_dir": str(self.cache_dir),
            "total_disk_usage_mb": total_size / (1024 * 1024),
            "last_updated": datetime.now().isoformat(),
        }

    def _calculate_module_hash(self, module: dspy.Module) -> str:
        """Calculate a hash for the module (simplified)."""
        try:
            # This is a simplified hash based on module type and string representation
            module_str = f"{type(module).__name__}_{str(module)}"
            return hashlib.md5(module_str.encode()).hexdigest()
        except Exception:
            return "unknown"

    def _save_module_to_disk(self, name: str, module: dspy.Module) -> None:
        """Save a module to disk."""
        try:
            module_file = self.cache_dir / f"{name}.pkl"
            with open(module_file, "wb") as f:
                pickle.dump(module, f)
        except Exception as e:
            logger.warning("Failed to save module %s to disk: %s", name, e)

    def _load_module_from_disk(self, name: str) -> dspy.Module | None:
        """Load a module from disk."""
        try:
            module_file = self.cache_dir / f"{name}.pkl"
            if module_file.exists():
                with open(module_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning("Failed to load module %s from disk: %s", name, e)
        return None

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            metadata_file = self.cache_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save metadata: %s", e)

    def _load_modules_metadata(self) -> None:
        """Load modules metadata from disk."""
        try:
            metadata_file = self.cache_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    self._metadata = json.load(f)
        except Exception as e:
            logger.warning("Failed to load metadata: %s", e)
            self._metadata = {}

    def clear_cache(self) -> None:
        """Clear all cached modules and metadata."""
        try:
            # Clear memory
            self._modules.clear()
            self._metadata.clear()

            # Clear disk cache
            for cache_file in self.cache_dir.rglob("*"):
                if cache_file.is_file():
                    cache_file.unlink()

            logger.info("Module manager cache cleared")

        except Exception as e:
            logger.error("Failed to clear module manager cache: %s", e)

    def export_modules(self, export_dir: Path) -> None:
        """Export all modules to a directory."""
        try:
            export_dir.mkdir(parents=True, exist_ok=True)

            for module_name in self.list_modules():
                module_path = export_dir / f"{module_name}.pkl"
                self.save_module(module_name, str(module_path))

            # Export metadata
            metadata_path = export_dir / "modules_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self._metadata, f, indent=2)

            logger.info("Exported all modules to %s", export_dir)

        except Exception as e:
            logger.error("Failed to export modules: %s", e)
            raise ModuleRegistrationError("Module export failed") from e
