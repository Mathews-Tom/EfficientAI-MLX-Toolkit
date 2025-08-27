"""Tests for utility modules."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from utils import (
    AppleSiliconLogger,
    BenchmarkRunner,
    ConfigManager,
    CrossPlatformUtils,
    FileValidator,
    LogManager,
    SafeFileHandler,
    create_performance_plot,
    find_files,
    read_json_file,
    setup_logging,
    write_json_file,
)
from utils.benchmark_runner import BenchmarkError
from utils.config_manager import ConfigurationError
from utils.file_operations import FileOperationError


class TestConfigManager:
    """Test configuration management functionality."""

    def test_json_config_creation(self) -> None:
        """Test creating and loading JSON configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"

            # Create config manager
            config = ConfigManager(config_path)

            # Set some values
            config.set("database.host", "localhost")
            config.set("database.port", 5432)
            config.set("debug", True)

            # Save configuration
            config.save()

            # Verify file exists and has correct content
            assert config_path.exists()

            # Load in new instance
            config2 = ConfigManager(config_path)

            assert config2.get("database.host") == "localhost"
            assert config2.get("database.port") == 5432
            assert config2.get("debug") is True

    def test_nested_key_access(self) -> None:
        """Test nested key access with dot notation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nested_config.json"
            config = ConfigManager(config_path)

            # Set nested values
            config.set("app.database.host", "localhost")
            config.set("app.database.port", 5432)
            config.set("app.logging.level", "INFO")

            # Test retrieval
            assert config.get("app.database.host") == "localhost"
            assert config.get("app.database.port") == 5432
            assert config.get("app.logging.level") == "INFO"

            # Test default values
            assert config.get("nonexistent.key", "default") == "default"

    def test_unsupported_format_error(self) -> None:
        """Test error handling for unsupported file formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test.xml"

            with pytest.raises(ConfigurationError) as exc_info:
                ConfigManager(config_path)

            assert "Unsupported configuration format" in str(exc_info.value)


class TestLoggingUtils:
    """Test logging utilities."""

    def test_logger_creation(self) -> None:
        """Test logger creation and configuration."""
        from utils.logging_utils import get_logger
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"

    def test_apple_silicon_logger(self) -> None:
        """Test Apple Silicon specific logger."""
        logger = AppleSiliconLogger("test_apple_logger")

        # Test that methods don't raise exceptions
        logger.log_optimization_start("MLX", "test_model")
        logger.log_optimization_complete("MLX", "test_model", {"accuracy": 0.95})
        logger.log_hardware_detection({"is_apple_silicon": True})
        logger.log_memory_usage({"rss_mb": 100.0})

    def test_structured_logging_setup(self) -> None:
        """Test structured logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Setup logging with file output
            setup_logging(log_level="DEBUG", log_file=log_file, structured_format=True)

            # Log a message
            from utils.logging_utils import get_logger
            logger = get_logger("test")
            logger.info("Test message")

            # Verify log file was created
            assert log_file.exists()

            # Verify log content is JSON
            log_content = log_file.read_text()
            assert log_content.strip()  # Should have content

            # Try to parse as JSON (structured format)
            try:
                json.loads(log_content.strip().split("\n")[0])
            except json.JSONDecodeError:
                pytest.fail("Log output is not valid JSON")


class TestBenchmarkRunner:
    """Test benchmark runner functionality."""

    def test_benchmark_execution(self) -> None:
        """Test basic benchmark execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            runner = BenchmarkRunner(results_dir)

            def sample_benchmark() -> dict[str, float]:
                return {"accuracy": 0.95, "speed": 100.0}

            result = runner.run_benchmark("test_benchmark", sample_benchmark, iterations=2)

            assert result.name == "test_benchmark"
            assert result.success is True
            assert result.execution_time > 0
            assert "accuracy" in result.performance_metrics
            assert "speed" in result.performance_metrics

    def test_benchmark_failure_handling(self) -> None:
        """Test benchmark failure handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            runner = BenchmarkRunner(results_dir)

            def failing_benchmark() -> dict[str, float]:
                raise ValueError("Test error")

            with pytest.raises(BenchmarkError):
                runner.run_benchmark("failing_test", failing_benchmark)

    def test_results_saving(self) -> None:
        """Test benchmark results saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            runner = BenchmarkRunner(results_dir)

            def sample_benchmark() -> dict[str, float]:
                return {"metric": 1.0}

            runner.run_benchmark("test", sample_benchmark)

            # Save results
            output_file = runner.save_results()

            assert output_file.exists()
            assert output_file.suffix == ".json"

            # Verify content
            results_data = json.loads(output_file.read_text())
            assert "results" in results_data
            assert "hardware_info" in results_data
            assert len(results_data["results"]) == 1

    def test_benchmark_comparison(self) -> None:
        """Test benchmark comparison functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            runner = BenchmarkRunner(results_dir)

            # Run baseline benchmark
            def baseline_benchmark() -> dict[str, float]:
                return {"accuracy": 0.8, "speed": 50.0}

            runner.run_benchmark("baseline", baseline_benchmark)

            # Run improved benchmark
            def improved_benchmark() -> dict[str, float]:
                return {"accuracy": 0.9, "speed": 75.0}

            runner.run_benchmark("improved", improved_benchmark)

            # Compare benchmarks
            comparisons = runner.compare_benchmarks("baseline", ["improved"])

            assert "improved" in comparisons
            assert "accuracy_improvement" in comparisons["improved"]
            assert "speed_improvement" in comparisons["improved"]


class TestLogManager:
    """Test log management utilities."""

    def test_log_manager_creation(self) -> None:
        """Test log manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            manager = LogManager(log_dir)

            assert log_dir.exists()
            assert manager.log_dir == log_dir

    def test_log_statistics(self) -> None:
        """Test log statistics collection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            manager = LogManager(log_dir)

            # Create test log files
            (log_dir / "test1.log").write_text("Log content 1")
            (log_dir / "test2.log").write_text("Log content 2")

            stats = manager.get_log_statistics()

            assert stats["total_files"] == 2
            assert stats["total_size_mb"] > 0

    def test_log_search(self) -> None:
        """Test log search functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            manager = LogManager(log_dir)

            # Create test log file with searchable content
            (log_dir / "search_test.log").write_text(
                "ERROR: Test error message\n"
                "INFO: Normal operation\n"
                "ERROR: Another error\n"
            )

            results = manager.search_logs("ERROR")
            assert len(results) == 2

            # Test case insensitive search
            results = manager.search_logs("error", case_sensitive=False)
            assert len(results) == 2


class TestConfigManagerEnhanced:
    """Test enhanced configuration management features."""

    def test_environment_overrides(self) -> None:
        """Test environment variable overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_data = {"key1": "original", "nested": {"key2": "original"}}
            config_path.write_text(json.dumps(config_data))

            with patch.dict(os.environ, {
                "TESTAPP_KEY1": "overridden",
                "TESTAPP_NESTED__KEY2": "nested_override"
            }):
                config = ConfigManager(config_path, environment_prefix="TESTAPP")

                assert config.get("key1") == "overridden"
                # Note: nested overrides require specific implementation

    def test_profile_configuration(self) -> None:
        """Test profile-based configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_data = {
                "base_setting": "base_value",
                "profiles": {
                    "development": {
                        "debug": True,
                        "base_setting": "dev_override"
                    },
                    "production": {
                        "debug": False,
                        "base_setting": "prod_override"
                    }
                }
            }
            config_path.write_text(json.dumps(config_data))

            config = ConfigManager(config_path, profile="development")
            profile_config = config.get_profile_config()

            assert profile_config["debug"] is True
            assert profile_config["base_setting"] == "dev_override"

    def test_required_keys_validation(self) -> None:
        """Test validation of required configuration keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_data = {"existing_key": "value"}
            config_path.write_text(json.dumps(config_data))

            config = ConfigManager(config_path)

            # Should pass for existing key
            assert config.validate_required_keys(["existing_key"]) is True

            # Should fail for missing key
            with pytest.raises(ConfigurationError):
                config.validate_required_keys(["missing_key"])

    def test_typed_configuration_access(self) -> None:
        """Test type-safe configuration access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_data = {
                "string_val": "test",
                "int_val": 42,
                "bool_val": True,
                "float_val": 3.14
            }
            config_path.write_text(json.dumps(config_data))

            config = ConfigManager(config_path)

            assert config.get_with_type("string_val", str) == "test"
            assert config.get_with_type("int_val", int) == 42
            assert config.get_with_type("bool_val", bool) is True
            assert config.get_with_type("float_val", float) == 3.14

            # Test type conversion error
            with pytest.raises(ConfigurationError):
                config.get_with_type("string_val", int)


class TestFileOperations:
    """Test file operation utilities."""

    def test_safe_file_handler_write_read(self) -> None:
        """Test safe file write and read operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SafeFileHandler()
            test_file = Path(temp_dir) / "test.txt"
            content = "Test content"

            # Test write
            result_path = handler.safe_write(test_file, content)
            assert result_path == test_file
            assert test_file.exists()

            # Test read
            read_content = handler.safe_read(test_file)
            assert read_content == content

    def test_safe_file_handler_backup(self) -> None:
        """Test backup creation during file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SafeFileHandler(enable_backups=True)
            test_file = Path(temp_dir) / "backup_test.txt"

            # Create initial file
            test_file.write_text("Original content")

            # Overwrite with backup enabled
            handler.safe_write(test_file, "New content")

            # Check backup was created
            backup_files = list(Path(temp_dir).glob("backup_test_*_backup.txt"))
            assert len(backup_files) == 1
            assert backup_files[0].read_text() == "Original content"

    def test_file_copy_move_delete(self) -> None:
        """Test file copy, move, and delete operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SafeFileHandler()
            source_file = Path(temp_dir) / "source.txt"
            copy_file = Path(temp_dir) / "copy.txt"
            move_file = Path(temp_dir) / "moved.txt"

            # Create source file
            source_file.write_text("Test content")

            # Test copy
            handler.safe_copy(source_file, copy_file)
            assert copy_file.exists()
            assert copy_file.read_text() == "Test content"

            # Test move
            handler.safe_move(copy_file, move_file)
            assert not copy_file.exists()
            assert move_file.exists()

            # Test delete
            handler.safe_delete(move_file, create_backup=False)
            assert not move_file.exists()

    def test_file_validator(self) -> None:
        """Test file validation utilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.json"
            test_file.write_text('{"test": "data"}')

            # Test path validation
            assert FileValidator.validate_path(test_file) is True

            # Test format validation
            assert FileValidator.validate_file_format(test_file, [".json"]) is True

            with pytest.raises(FileOperationError):
                FileValidator.validate_file_format(test_file, [".xml"])

    def test_file_integrity_check(self) -> None:
        """Test file integrity checking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "integrity_test.txt"
            test_content = "Test content"
            test_file.write_text(test_content)

            result = FileValidator.check_file_integrity(test_file)

            assert "sha256" in result
            assert "file_size" in result
            assert "integrity_status" in result

    def test_cross_platform_utils(self) -> None:
        """Test cross-platform utilities."""
        # Test safe filename creation
        unsafe_name = "file<>:name|with?invalid*chars"
        safe_name = CrossPlatformUtils.safe_filename(unsafe_name)

        # Should not contain any invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            assert char not in safe_name

        # Test path normalization
        test_path = Path("./test/../normalized")
        normalized = CrossPlatformUtils.normalize_path(test_path)
        assert normalized.is_absolute()

    def test_json_convenience_functions(self) -> None:
        """Test JSON convenience functions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.json"
            test_data = {"key": "value", "number": 42}

            # Test write
            write_json_file(test_file, test_data)
            assert test_file.exists()

            # Test read
            read_data = read_json_file(test_file)
            assert read_data == test_data

    def test_find_files_functionality(self) -> None:
        """Test file finding functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Create test files
            (base_dir / "test1.py").write_text("# Python file")
            (base_dir / "test2.py").write_text("# Another Python file")
            (base_dir / "readme.txt").write_text("Text file")

            # Create subdirectory with files
            sub_dir = base_dir / "subdir"
            sub_dir.mkdir()
            (sub_dir / "test3.py").write_text("# Nested Python file")

            # Test finding all files
            all_files = find_files(base_dir)
            assert len(all_files) == 4

            # Test filtering by file type
            py_files = find_files(base_dir, file_types=[".py"])
            assert len(py_files) == 3

            # Test non-recursive search
            root_files = find_files(base_dir, recursive=False)
            assert len(root_files) == 3  # Only files in root


class TestPlottingUtilities:
    """Test plotting utilities."""

    @pytest.mark.slow
    def test_create_performance_plot(self) -> None:
        """Test performance plot creation."""
        pytest.importorskip("matplotlib")

        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_results = [
                {"name": "baseline", "execution_time": 1.5},
                {"name": "optimized", "execution_time": 0.8}
            ]

            output_path = Path(temp_dir) / "performance.png"
            result_path = create_performance_plot(benchmark_results, output_path)

            assert result_path == output_path
            assert output_path.exists()


class TestIntegration:
    """Integration tests for utilities."""

    def test_config_logging_integration(self) -> None:
        """Test integration between configuration and logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.json"
            config_data = {
                "logging": {
                    "level": "DEBUG",
                    "file": "test.log"
                }
            }
            config_file.write_text(json.dumps(config_data))

            config = ConfigManager(config_file)
            log_file = Path(temp_dir) / config.get("logging.file")

            setup_logging(
                log_level=config.get("logging.level"),
                log_file=log_file
            )

            # Test that logging works
            from utils.logging_utils import get_logger
            logger = get_logger("integration_test")
            logger.info("Integration test message")

            assert log_file.exists()

    def test_file_operations_with_config(self) -> None:
        """Test file operations with configuration management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config for file operations
            config_file = Path(temp_dir) / "file_config.json"
            config_data = {
                "file_operations": {
                    "enable_backups": True,
                    "backup_count": 5
                }
            }
            config_file.write_text(json.dumps(config_data))

            config = ConfigManager(config_file)

            # Use config in file operations
            handler = SafeFileHandler(
                enable_backups=config.get("file_operations.enable_backups")
            )

            test_file = Path(temp_dir) / "configured_test.txt"
            handler.safe_write(test_file, "Test with config")

            assert test_file.exists()
            assert test_file.read_text() == "Test with config"
