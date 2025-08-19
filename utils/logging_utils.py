"""
Centralized logging utilities with pathlib support for Apple Silicon optimization tracking.

This module provides structured logging capabilities optimized for the EfficientAI-MLX-Toolkit,
with special support for tracking Apple Silicon optimizations and performance metrics.
"""

import json
import logging
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
    enable_apple_silicon_tracking: bool = True,
    structured_format: bool = True,
    enable_rotation: bool = True,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up centralized logging with pathlib-based file handling.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file using pathlib
        enable_apple_silicon_tracking: Enable Apple Silicon optimization tracking
        structured_format: Use structured JSON logging format
        enable_rotation: Enable log file rotation
        max_file_size_mb: Maximum file size in MB before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance

    Example:
        >>> from pathlib import Path
        >>> logger = setup_logging(
        ...     log_level="DEBUG",
        ...     log_file=Path("logs/toolkit.log"),
        ...     enable_apple_silicon_tracking=True
        ... )
    """
    # Create logs directory if log_file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if structured_format:
        formatter = StructuredFormatter(enable_apple_silicon_tracking)
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        if enable_rotation:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")

        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting Apple Silicon optimization")
    """
    return logging.getLogger(name)


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter with Apple Silicon optimization tracking.
    """

    def __init__(self, enable_apple_silicon_tracking: bool = True) -> None:
        super().__init__()
        self.enable_apple_silicon_tracking = enable_apple_silicon_tracking

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add Apple Silicon specific tracking
        if self.enable_apple_silicon_tracking:
            log_entry.update(self._get_apple_silicon_context(record))

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, default=str)

    def _get_apple_silicon_context(
        self, record: logging.LogRecord
    ) -> dict[str, str | int | float | bool]:
        """Extract Apple Silicon optimization context from log record."""
        context: dict[str, str | int | float | bool] = {}

        # Check for Apple Silicon specific attributes
        apple_silicon_attrs = [
            "mlx_enabled",
            "mps_enabled",
            "optimization_level",
            "memory_usage",
            "performance_metrics",
            "hardware_type",
        ]

        for attr in apple_silicon_attrs:
            if hasattr(record, attr):
                context[attr] = getattr(record, attr)

        return context


class AppleSiliconLogger:
    """
    Specialized logger for Apple Silicon optimization tracking.
    """

    def __init__(self, logger_name: str) -> None:
        self.logger = get_logger(logger_name)

    def log_optimization_start(self, optimization_type: str, model_name: str) -> None:
        """Log the start of an Apple Silicon optimization."""
        self.logger.info(
            "Starting %s optimization for %s",
            optimization_type,
            model_name,
            extra={
                "extra_fields": {
                    "event_type": "optimization_start",
                    "optimization_type": optimization_type,
                    "model_name": model_name,
                }
            },
        )

    def log_optimization_complete(
        self, optimization_type: str, model_name: str, performance_metrics: Mapping[str, float]
    ) -> None:
        """Log the completion of an Apple Silicon optimization."""
        self.logger.info(
            "Completed %s optimization for %s",
            optimization_type,
            model_name,
            extra={
                "extra_fields": {
                    "event_type": "optimization_complete",
                    "optimization_type": optimization_type,
                    "model_name": model_name,
                    "performance_metrics": dict(performance_metrics),
                }
            },
        )

    def log_hardware_detection(self, hardware_info: Mapping[str, str | int | float | bool]) -> None:
        """Log Apple Silicon hardware detection results."""
        self.logger.info(
            "Apple Silicon hardware detected",
            extra={
                "extra_fields": {
                    "event_type": "hardware_detection",
                    "hardware_info": dict(hardware_info),
                }
            },
        )

    def log_memory_usage(self, memory_stats: Mapping[str, float]) -> None:
        """Log memory usage statistics for unified memory architecture."""
        self.logger.debug(
            "Memory usage statistics",
            extra={
                "extra_fields": {
                    "event_type": "memory_usage",
                    "memory_stats": dict(memory_stats),
                }
            },
        )


class LogManager:
    """
    Log file management and analysis utilities.
    """

    def __init__(self, log_dir: Path) -> None:
        """Initialize log manager with pathlib-based directory handling."""
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

    def cleanup_old_logs(self, max_age_days: int = 30) -> list[Path]:
        """
        Clean up old log files based on age.

        Args:
            max_age_days: Maximum age in days for log files

        Returns:
            List of removed log file paths
        """
        removed_files: list[Path] = []
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        try:
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    removed_files.append(log_file)

            self.logger.info("Cleaned up %d old log files", len(removed_files))

        except Exception as e:
            self.logger.error("Failed to cleanup old logs: %s", e)

        return removed_files

    def get_log_statistics(self) -> dict[str, int | float]:
        """
        Get statistics about log files in the directory.

        Returns:
            Dictionary with log statistics
        """
        stats: dict[str, int | float] = {
            "total_files": 0,
            "total_size_mb": 0.0,
            "oldest_file_days": 0.0,
            "newest_file_days": 0.0,
        }

        try:
            log_files = list(self.log_dir.glob("*.log*"))
            stats["total_files"] = len(log_files)

            if log_files:
                total_size = sum(f.stat().st_size for f in log_files)
                stats["total_size_mb"] = total_size / (1024 * 1024)

                current_time = time.time()
                mtimes = [f.stat().st_mtime for f in log_files]

                stats["oldest_file_days"] = (current_time - min(mtimes)) / (24 * 60 * 60)
                stats["newest_file_days"] = (current_time - max(mtimes)) / (24 * 60 * 60)

        except Exception as e:
            self.logger.error("Failed to get log statistics: %s", e)

        return stats

    def search_logs(
        self, pattern: str, max_results: int = 100, case_sensitive: bool = False
    ) -> list[dict[str, str]]:
        """
        Search for patterns in log files.

        Args:
            pattern: Search pattern
            max_results: Maximum number of results
            case_sensitive: Whether search is case sensitive

        Returns:
            List of matching log entries
        """
        results: list[dict[str, str]] = []

        try:
            import re

            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)

            for log_file in self.log_dir.glob("*.log*"):
                if len(results) >= max_results:
                    break

                try:
                    content = log_file.read_text(encoding="utf-8", errors="ignore")
                    for line_num, line in enumerate(content.splitlines(), 1):
                        if regex.search(line):
                            results.append({
                                "file": str(log_file),
                                "line_number": str(line_num),
                                "content": line.strip(),
                            })

                            if len(results) >= max_results:
                                break

                except Exception as e:
                    self.logger.warning("Error reading log file %s: %s", log_file, e)

        except Exception as e:
            self.logger.error("Failed to search logs: %s", e)

        return results

    def archive_logs(self, archive_dir: Path | None = None) -> Path:
        """
        Archive log files to a compressed archive.

        Args:
            archive_dir: Directory to store archive (default: log_dir/archives)

        Returns:
            Path to created archive
        """
        import tarfile
        from datetime import datetime

        if archive_dir is None:
            archive_dir = self.log_dir / "archives"

        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"logs_{timestamp}.tar.gz"

        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                for log_file in self.log_dir.glob("*.log*"):
                    if log_file != archive_path:
                        tar.add(log_file, arcname=log_file.name)

            self.logger.info("Archived logs to %s", archive_path)

        except Exception as e:
            self.logger.error("Failed to archive logs: %s", e)
            raise

        return archive_path
