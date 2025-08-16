"""
Pytest configuration and fixtures for DSPy Integration Framework tests.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from dspy_toolkit.framework import DSPyFramework
from dspy_toolkit.types import DSPyConfig, HardwareInfo


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return DSPyConfig(
        model_provider="test",
        model_name="test-model",
        cache_dir=temp_dir / "cache",
        optimization_level=1,
        enable_tracing=False,
        max_retries=2,
    )


@pytest.fixture
def mock_hardware_info():
    """Mock hardware information."""
    return HardwareInfo(
        device_type="test",
        total_memory=16,
        available_memory=12,
        metal_available=False,
        mps_available=False,
        optimization_level=1,
    )


@pytest.fixture
def mock_mlx_provider(mock_hardware_info):
    """Mock MLX provider for testing."""
    provider = Mock()
    provider.is_available.return_value = True
    provider.hardware_info = mock_hardware_info
    provider.get_provider_info.return_value = {
        "provider_type": "MockMLXProvider",
        "model_name": "test-model",
        "hardware_info": mock_hardware_info.__dict__,
        "initialized": True,
    }
    provider.benchmark_performance.return_value = {
        "average_time": 1.0,
        "tokens_per_second": 100.0,
        "hardware_type": "test",
    }
    return provider


@pytest.fixture
def mock_dspy_module():
    """Mock DSPy module for testing."""
    module = Mock()
    module.__call__ = Mock(return_value={"answer": "test response"})
    module.__dict__ = {"answer": "test response"}
    return module


@pytest.fixture
def mock_signature():
    """Mock DSPy signature for testing."""
    import dspy

    class MockSignature(dspy.Signature):
        """Mock signature for testing."""

        input_field = dspy.InputField(desc="Test input")
        output_field = dspy.OutputField(desc="Test output")

    return MockSignature


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return [
        {"input": "test input 1", "expected_output": "test output 1"},
        {"input": "test input 2", "expected_output": "test output 2"},
        {"input": "test input 3", "expected_output": "test output 3"},
        {"input": "test input 4", "expected_output": "test output 4"},
        {"input": "test input 5", "expected_output": "test output 5"},
    ]


@pytest.fixture
def mock_framework(test_config, mock_mlx_provider):
    """Mock DSPy framework for testing."""
    with pytest.MonkeyPatch().context() as m:
        # Mock the framework initialization to avoid actual setup
        m.setattr("dspy_toolkit.framework.DSPyFramework._initialize", Mock())
        m.setattr("dspy_toolkit.framework.DSPyFramework.setup_llm_provider", Mock())

        framework = DSPyFramework.__new__(DSPyFramework)
        framework.config = test_config
        framework.llm_provider = mock_mlx_provider
        framework.hardware_info = mock_mlx_provider.hardware_info

        # Mock components
        framework.signature_registry = Mock()
        framework.module_manager = Mock()
        framework.optimizer_engine = Mock()

        # Mock methods
        framework.health_check = Mock(
            return_value={
                "overall_status": "healthy",
                "components": {"test": "healthy"},
                "issues": [],
            }
        )
        framework.get_framework_stats = Mock(
            return_value={
                "framework": {"test": "stats"},
                "signatures": {"total_signatures": 5},
                "modules": {"total_modules": 3},
                "optimizer": {"total_optimizations": 2},
            }
        )

        yield framework


@pytest.fixture
def performance_metrics():
    """Sample performance metrics for testing."""
    from dspy_toolkit.deployment.monitoring import PerformanceMetrics

    return PerformanceMetrics(
        execution_time=1.5,
        input_tokens=100,
        output_tokens=200,
        memory_usage=1024,
        timestamp=1234567890,
        success=True,
    )


@pytest.fixture
def system_metrics():
    """Sample system metrics for testing."""
    from dspy_toolkit.deployment.monitoring import SystemMetrics

    return SystemMetrics(
        cpu_usage=0.75,
        memory_usage=0.60,
        disk_usage=0.45,
        network_io=1024.0,
        timestamp=1234567890,
    )


@pytest.fixture
def alert_rule():
    """Sample alert rule for testing."""
    from dspy_toolkit.deployment.monitoring import AlertRule

    return AlertRule(
        name="test_alert",
        metric_name="test_metric",
        threshold=10.0,
        comparison="gt",
        window_minutes=5,
        enabled=True,
    )


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")
    config.addinivalue_line("markers", "memory: marks tests as memory profiling tests")
    config.addinivalue_line(
        "markers", "apple_silicon: marks tests that require Apple Silicon hardware"
    )
    config.addinivalue_line(
        "markers", "requires_mlx: marks tests that require MLX installation"
    )
    config.addinivalue_line(
        "markers", "requires_dspy: marks tests that require DSPy installation"
    )


# Skip conditions for optional dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle optional dependencies."""
    skip_mlx = pytest.mark.skip(reason="MLX not available")
    skip_dspy = pytest.mark.skip(reason="DSPy not available")
    skip_apple_silicon = pytest.mark.skip(reason="Not running on Apple Silicon")

    try:
        import mlx

        mlx_available = True
    except ImportError:
        mlx_available = False

    try:
        import dspy

        dspy_available = True
    except ImportError:
        dspy_available = False

    import platform

    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

    for item in items:
        if "requires_mlx" in item.keywords and not mlx_available:
            item.add_marker(skip_mlx)
        if "requires_dspy" in item.keywords and not dspy_available:
            item.add_marker(skip_dspy)
        if "apple_silicon" in item.keywords and not is_apple_silicon:
            item.add_marker(skip_apple_silicon)


# Utility functions for tests
class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def create_mock_dspy_example(input_data: dict, output_data: dict):
        """Create a mock DSPy example."""
        try:
            import dspy

            return dspy.Example(**input_data, **output_data)
        except ImportError:
            # Return a simple dict if DSPy is not available
            return {**input_data, **output_data}

    @staticmethod
    def assert_performance_metrics(metrics, expected_ranges: dict):
        """Assert performance metrics are within expected ranges."""
        for metric_name, (min_val, max_val) in expected_ranges.items():
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                assert (
                    min_val <= value <= max_val
                ), f"{metric_name} {value} not in range [{min_val}, {max_val}]"

    @staticmethod
    def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true."""
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)

        return False

    @staticmethod
    async def async_wait_for_condition(condition_func, timeout=5.0, interval=0.1):
        """Async version of wait_for_condition."""
        import asyncio

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if (
                await condition_func()
                if asyncio.iscoroutinefunction(condition_func)
                else condition_func()
            ):
                return True
            await asyncio.sleep(interval)

        return False


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils


# Memory profiling utilities
class MemoryProfiler:
    """Simple memory profiler for tests."""

    def __init__(self):
        self.start_memory = 0
        self.peak_memory = 0
        self.end_memory = 0

    def __enter__(self):
        import os

        import psutil

        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss
        self.peak_memory = self.start_memory
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import os

        import psutil

        process = psutil.Process(os.getpid())
        self.end_memory = process.memory_info().rss
        self.peak_memory = max(self.peak_memory, self.end_memory)

    def get_profile(self):
        """Get memory profile results."""
        return {
            "start_memory": self.start_memory,
            "peak_memory": self.peak_memory,
            "end_memory": self.end_memory,
            "memory_increase": self.end_memory - self.start_memory,
        }


@pytest.fixture
def memory_profiler():
    """Provide memory profiler."""
    try:
        import psutil

        return MemoryProfiler
    except ImportError:
        pytest.skip("psutil not available for memory profiling")


# Performance benchmarking utilities
class BenchmarkResult:
    """Benchmark result container."""

    def __init__(self, name: str, duration: float, iterations: int):
        self.name = name
        self.duration = duration
        self.iterations = iterations
        self.avg_time = duration / iterations if iterations > 0 else 0
        self.ops_per_second = iterations / duration if duration > 0 else 0


class SimpleBenchmark:
    """Simple benchmark utility for tests."""

    def __init__(self, name: str = "benchmark"):
        self.name = name
        self.results = []

    def run(self, func, iterations: int = 100, *args, **kwargs):
        """Run benchmark."""
        import time

        start_time = time.time()

        for _ in range(iterations):
            func(*args, **kwargs)

        end_time = time.time()
        duration = end_time - start_time

        result = BenchmarkResult(self.name, duration, iterations)
        self.results.append(result)

        return result

    async def async_run(self, func, iterations: int = 100, *args, **kwargs):
        """Run async benchmark."""
        import time

        start_time = time.time()

        for _ in range(iterations):
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)

        end_time = time.time()
        duration = end_time - start_time

        result = BenchmarkResult(self.name, duration, iterations)
        self.results.append(result)

        return result


@pytest.fixture
def benchmark():
    """Provide simple benchmark utility."""
    return SimpleBenchmark
