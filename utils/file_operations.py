"""
Pathlib-based file operation utilities with cross-platform compatibility.

This module provides standardized file operations for the EfficientAI-MLX-Toolkit,
with comprehensive error handling, validation, and safety checks.
"""

import hashlib
import json
import logging
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileOperationError(Exception):
    """Raised when file operations fail."""

    def __init__(
        self,
        message: str,
        file_path: Path | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.file_path = file_path
        self.operation = operation
        self.details = dict(details or {})


class SafeFileHandler:
    """
    Safe file operation handler with pathlib support.

    Provides atomic operations, backup capabilities, and comprehensive validation.
    """

    def __init__(
        self, enable_backups: bool = True, backup_dir: Path | None = None
    ) -> None:
        """
        Initialize safe file handler.

        Args:
            enable_backups: Whether to create backups before modifications
            backup_dir: Directory for backups (default: same directory as original file)
        """
        self.enable_backups = enable_backups
        self.backup_dir = backup_dir
        self.logger = logging.getLogger(__name__)

    def safe_write(
        self,
        file_path: Path,
        content: str | bytes,
        encoding: str = "utf-8",
        create_backup: bool | None = None,
    ) -> Path:
        """
        Safely write content to file with atomic operation.

        Args:
            file_path: Target file path
            content: Content to write
            encoding: Text encoding (ignored for bytes content)
            create_backup: Override instance backup setting

        Returns:
            Path to written file

        Raises:
            FileOperationError: If write operation fails
        """
        create_backup = (
            create_backup if create_backup is not None else self.enable_backups
        )

        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if requested and file exists
            backup_path = None
            if create_backup and file_path.exists():
                backup_path = self._create_backup(file_path)

            # Write to temporary file first (atomic operation)
            temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

            if isinstance(content, bytes):
                temp_path.write_bytes(content)
            else:
                temp_path.write_text(content, encoding=encoding)

            # Atomic move to final location
            temp_path.replace(file_path)

            self.logger.info("Successfully wrote file: %s", file_path)
            if backup_path:
                self.logger.debug("Created backup: %s", backup_path)

            return file_path

        except Exception as e:
            raise FileOperationError(
                f"Failed to write file: {file_path}",
                file_path=file_path,
                operation="write",
                details={"encoding": encoding, "content_type": type(content).__name__},
            ) from e

    def safe_read(
        self,
        file_path: Path,
        encoding: str = "utf-8",
        binary_mode: bool = False,
    ) -> str | bytes:
        """
        Safely read file content with validation.

        Args:
            file_path: File path to read
            encoding: Text encoding (ignored in binary mode)
            binary_mode: Whether to read in binary mode

        Returns:
            File content as string or bytes

        Raises:
            FileOperationError: If read operation fails
        """
        try:
            if not file_path.exists():
                raise FileOperationError(
                    f"File not found: {file_path}",
                    file_path=file_path,
                    operation="read",
                )

            if not file_path.is_file():
                raise FileOperationError(
                    f"Path is not a regular file: {file_path}",
                    file_path=file_path,
                    operation="read",
                )

            if binary_mode:
                content = file_path.read_bytes()
            else:
                content = file_path.read_text(encoding=encoding)

            self.logger.debug("Successfully read file: %s", file_path)
            return content

        except FileOperationError:
            raise
        except Exception as e:
            raise FileOperationError(
                f"Failed to read file: {file_path}",
                file_path=file_path,
                operation="read",
                details={"encoding": encoding, "binary_mode": binary_mode},
            ) from e

    def safe_copy(
        self,
        source_path: Path,
        destination_path: Path,
        preserve_metadata: bool = True,
    ) -> Path:
        """
        Safely copy file with metadata preservation.

        Args:
            source_path: Source file path
            destination_path: Destination file path
            preserve_metadata: Whether to preserve file metadata

        Returns:
            Path to destination file

        Raises:
            FileOperationError: If copy operation fails
        """
        try:
            if not source_path.exists():
                raise FileOperationError(
                    f"Source file not found: {source_path}",
                    file_path=source_path,
                    operation="copy",
                )

            # Ensure destination directory exists
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            if preserve_metadata:
                shutil.copy2(source_path, destination_path)
            else:
                shutil.copy(source_path, destination_path)

            self.logger.info(
                "Successfully copied: %s -> %s", source_path, destination_path
            )
            return destination_path

        except Exception as e:
            raise FileOperationError(
                f"Failed to copy file: {source_path} -> {destination_path}",
                file_path=source_path,
                operation="copy",
            ) from e

    def safe_move(self, source_path: Path, destination_path: Path) -> Path:
        """
        Safely move/rename file.

        Args:
            source_path: Source file path
            destination_path: Destination file path

        Returns:
            Path to destination file

        Raises:
            FileOperationError: If move operation fails
        """
        try:
            if not source_path.exists():
                raise FileOperationError(
                    f"Source file not found: {source_path}",
                    file_path=source_path,
                    operation="move",
                )

            # Ensure destination directory exists
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            source_path.rename(destination_path)

            self.logger.info(
                "Successfully moved: %s -> %s", source_path, destination_path
            )
            return destination_path

        except Exception as e:
            raise FileOperationError(
                f"Failed to move file: {source_path} -> {destination_path}",
                file_path=source_path,
                operation="move",
            ) from e

    def safe_delete(self, file_path: Path, create_backup: bool | None = None) -> bool:
        """
        Safely delete file with optional backup.

        Args:
            file_path: File path to delete
            create_backup: Override instance backup setting

        Returns:
            True if file was deleted

        Raises:
            FileOperationError: If delete operation fails
        """
        create_backup = (
            create_backup if create_backup is not None else self.enable_backups
        )

        try:
            if not file_path.exists():
                self.logger.warning("File does not exist: %s", file_path)
                return False

            # Create backup if requested
            if create_backup:
                self._create_backup(file_path)

            file_path.unlink()

            self.logger.info("Successfully deleted file: %s", file_path)
            return True

        except Exception as e:
            raise FileOperationError(
                f"Failed to delete file: {file_path}",
                file_path=file_path,
                operation="delete",
            ) from e

    def _create_backup(self, file_path: Path) -> Path:
        """Create backup of file."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}_backup{file_path.suffix}"

        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = self.backup_dir / backup_name
        else:
            backup_path = file_path.parent / backup_name

        shutil.copy2(file_path, backup_path)
        return backup_path


class FileValidator:
    """
    File validation and integrity checking utilities.
    """

    @staticmethod
    def validate_path(file_path: Path, must_exist: bool = True) -> bool:
        """
        Validate file path properties.

        Args:
            file_path: Path to validate
            must_exist: Whether file must exist

        Returns:
            True if validation passes

        Raises:
            FileOperationError: If validation fails
        """
        try:
            # Check if path is absolute (recommended for consistency)
            if not file_path.is_absolute():
                logger.warning("Using relative path: %s", file_path)

            # Check existence
            if must_exist and not file_path.exists():
                raise FileOperationError(
                    f"Path does not exist: {file_path}",
                    file_path=file_path,
                    operation="validate",
                )

            # Check if it's a file (not directory)
            if file_path.exists() and not file_path.is_file():
                raise FileOperationError(
                    f"Path is not a regular file: {file_path}",
                    file_path=file_path,
                    operation="validate",
                )

            return True

        except FileOperationError:
            raise
        except Exception as e:
            raise FileOperationError(
                f"Path validation failed: {file_path}",
                file_path=file_path,
                operation="validate",
            ) from e

    @staticmethod
    def check_file_integrity(
        file_path: Path, expected_hash: str | None = None
    ) -> dict[str, str]:
        """
        Check file integrity using checksums.

        Args:
            file_path: File path to check
            expected_hash: Expected SHA256 hash (optional)

        Returns:
            Dictionary with hash information

        Raises:
            FileOperationError: If integrity check fails
        """
        try:
            if not file_path.exists():
                raise FileOperationError(
                    f"File not found: {file_path}",
                    file_path=file_path,
                    operation="integrity_check",
                )

            # Calculate SHA256 hash
            sha256_hash = hashlib.sha256()
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

            calculated_hash = sha256_hash.hexdigest()

            result = {
                "file_path": str(file_path),
                "sha256": calculated_hash,
                "file_size": str(file_path.stat().st_size),
            }

            if expected_hash:
                if calculated_hash == expected_hash:
                    result["integrity_status"] = "valid"
                else:
                    result["integrity_status"] = "invalid"
                    raise FileOperationError(
                        f"File integrity check failed: {file_path}",
                        file_path=file_path,
                        operation="integrity_check",
                        details={
                            "expected": expected_hash,
                            "calculated": calculated_hash,
                        },
                    )
            else:
                result["integrity_status"] = "calculated"

            return result

        except FileOperationError:
            raise
        except Exception as e:
            raise FileOperationError(
                f"Integrity check failed: {file_path}",
                file_path=file_path,
                operation="integrity_check",
            ) from e

    @staticmethod
    def validate_file_format(file_path: Path, expected_formats: list[str]) -> bool:
        """
        Validate file format based on extension.

        Args:
            file_path: File path to validate
            expected_formats: List of expected extensions (e.g., ['.json', '.yaml'])

        Returns:
            True if format is valid

        Raises:
            FileOperationError: If format validation fails
        """
        suffix = file_path.suffix.lower()

        if suffix not in expected_formats:
            raise FileOperationError(
                f"Invalid file format: {suffix}",
                file_path=file_path,
                operation="format_validation",
                details={"expected_formats": expected_formats, "actual_format": suffix},
            )

        return True


class CrossPlatformUtils:
    """
    Cross-platform compatibility utilities.
    """

    @staticmethod
    def normalize_path(path: Path) -> Path:
        """
        Normalize path for cross-platform compatibility.

        Args:
            path: Path to normalize

        Returns:
            Normalized path
        """
        return Path(path).resolve()

    @staticmethod
    def get_temp_dir() -> Path:
        """Get cross-platform temporary directory."""
        return Path(tempfile.gettempdir())

    @staticmethod
    def get_home_dir() -> Path:
        """Get user home directory."""
        return Path.home()

    @staticmethod
    def is_case_sensitive_filesystem() -> bool:
        """Check if filesystem is case-sensitive."""
        return platform.system() != "Windows"

    @staticmethod
    def safe_filename(filename: str, replacement: str = "_") -> str:
        """
        Create safe filename for cross-platform use.

        Args:
            filename: Original filename
            replacement: Character to replace invalid characters

        Returns:
            Safe filename
        """
        # Invalid characters for most filesystems
        invalid_chars = '<>:"/\\|?*'

        safe_name = filename
        for char in invalid_chars:
            safe_name = safe_name.replace(char, replacement)

        # Remove leading/trailing dots and spaces
        safe_name = safe_name.strip(". ")

        # Ensure not empty
        if not safe_name:
            safe_name = "unnamed"

        return safe_name

    @staticmethod
    def get_available_space(path: Path) -> int:
        """
        Get available disk space in bytes.

        Args:
            path: Path to check (file or directory)

        Returns:
            Available space in bytes
        """
        if path.is_file():
            path = path.parent

        stat = shutil.disk_usage(path)
        return stat.free


# Convenience functions for common operations
def read_json_file(file_path: Path) -> dict[str, Any]:
    """
    Read and parse JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileOperationError: If read or parse fails
    """
    handler = SafeFileHandler()

    try:
        content = handler.safe_read(file_path)
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise FileOperationError(
            f"Invalid JSON format: {file_path}",
            file_path=file_path,
            operation="json_read",
        ) from e


def write_json_file(file_path: Path, data: dict[str, Any], indent: int = 2) -> Path:
    """
    Write data to JSON file.

    Args:
        file_path: Path to JSON file
        data: Data to write
        indent: JSON indentation

    Returns:
        Path to written file

    Raises:
        FileOperationError: If write fails
    """
    handler = SafeFileHandler()
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    return handler.safe_write(file_path, content)


def find_files(
    directory: Path,
    pattern: str = "*",
    recursive: bool = True,
    file_types: list[str] | None = None,
) -> list[Path]:
    """
    Find files matching pattern and criteria.

    Args:
        directory: Directory to search
        pattern: Glob pattern
        recursive: Whether to search recursively
        file_types: List of file extensions to filter (e.g., ['.py', '.json'])

    Returns:
        List of matching file paths
    """
    if not directory.exists():
        return []

    if recursive:
        files = directory.rglob(pattern)
    else:
        files = directory.glob(pattern)

    # Filter by file types if specified
    if file_types:
        files = (f for f in files if f.suffix.lower() in file_types)

    # Return only regular files
    return [f for f in files if f.is_file()]
