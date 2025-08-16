"""
Performance and benchmark tests for DSPy Integration Framework.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from dspy_toolkit.deployment.monitoring import DSPyMonitor, PerformanceMetrics
from dspy_toolkit.framework import DSPyFramework
from dspy_toolkit.types import DSPyConfig


@pytest.mark.benchmark
class TestFrameworkPerformance:
    """Performance tests for framework components."""

    def test_signature_registry_performance(self, benchmark, mock_signature):
        """Benchmark signature registry operations."""
        from dspy_toolkit.registry import SignatureRegistry

        registry = SignatureRegistry()

        def register_signatures():
            signatures = {f"sig_{i}": mock_signature for i in range(10)}
            registry.register_project("perf_test", signatures)

        result = benchmark(register_signatures)

        # Should complete quickly
        assert result.avg_time < 0.1

    def test_module_manager_performance(self, benchmark, mock_dspy_module):
        """Benchmark module manager operations."""
        from dspy_toolkit.manager import ModuleManager

        manager = ModuleManager()

        def register_modules():
            for i in range(10):
                manager.register_module(
                    f"module_{i}", mock_dspy_module, {"test": "metadata"}
                )

        result = benchmark(register_modules)

        # Should handle multiple registrations efficiently
        assert result.avg_time < 0.5

    def test_optimizer_engine_performance(
        self, benchmark, mock_dspy_module, sample_dataset
    ):
        """Benchmark optimizer engine operations."""
        from dspy_toolkit.optimizer import OptimizerEngine

        engine = OptimizerEngine()

        # Mock the actual optimization to focus on engine overhead
        with patch.object(engine, "_run_optimizer", return_value=mock_dspy_module):

            def optimize_module():
                return engine.optimize(
                    mock_dspy_module, sample_dataset[:3], ["accuracy"]
                )

            result = benchmark(optimize_module)

            # Should complete optimization quickly
            assert result.avg_time < 1.0

    def test_circuit_breaker_performance(self, benchmark):
        """Benchmark circuit breaker overhead."""
        from dspy_toolkit.recovery.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker("perf_test")

        def fast_operation():
            return "success"

        def protected_operation():
            return cb.call(fast_operation)

        result = benchmark(protected_operation)

        # Circuit breaker should add minimal overhead
        assert result.avg_time < 0.001  # Less than 1ms

    def test_retry_handler_performance(self, benchmark):
        """Benchmark retry handler overhead for successful operations."""
        from dspy_toolkit.recovery.retry_handler import RetryHandler

        retry_handler = RetryHandler()

        def fast_operation():
            return "success"

        def retryable_operation():
            return retry_handler.execute(fast_operation)

        result = benchmark(retryable_operation)

        # Retry handler should add minimal overhead for successful operations
        assert result.avg_time < 0.001  # Less than 1ms


@pytest.mark.benchmark
class TestDeploymentPerformance:
    """Performance tests for deployment components."""

    @pytest.mark.asyncio
    async def test_fastapi_handler_performance(self, mock_framework, benchmark):
        """Benchmark FastAPI request handling."""
        with patch(
            "dspy_toolkit.deployment.fastapi_integration.FASTAPI_AVAILABLE", True
        ):
            from dspy_toolkit.deployment.fastapi_integration import (
                DSPyFastAPIApp,
                DSPyRequest,
            )

            app = DSPyFastAPIApp(mock_framework)

            # Setup mock module
            mock_module = Mock()
            mock_framework.get_project_module.return_value = mock_module

            request = DSPyRequest(
                inputs={"test": "input"},
                project_name="test_project",
                module_name="test_module",
            )

            background_tasks = Mock()

            with patch.object(
                app, "_execute_module_async", return_value={"answer": "test"}
            ):

                async def handle_request():
                    return await app._handle_prediction(request, background_tasks)

                # Use async benchmark
                result = await benchmark.async_run(handle_request, iterations=50)

                # Should handle requests efficiently
                assert result.avg_time < 0.1  # Less than 100ms per request

    def test_monitoring_performance(self, benchmark, performance_metrics):
        """Benchmark monitoring overhead."""
        monitor = DSPyMonitor(enable_system_monitoring=False)

        def record_metrics():
            monitor.record_request("test_project", "test_module", performance_metrics)

        result = benchmark(record_metrics)

        # Monitoring should add minimal overhead
        assert result.avg_time < 0.001  # Less than 1ms

        monitor.cleanup()

    @pytest.mark.asyncio
    async def test_streaming_performance(self, mock_framework, benchmark):
        """Benchmark streaming endpoint performance."""
        with patch("dspy_toolkit.deployment.streaming.STREAMING_AVAILABLE", True):
            from dspy_toolkit.deployment.streaming import DSPyStreamingEndpoint

            endpoint = DSPyStreamingEndpoint(mock_framework)

            # Mock module
            mock_module = Mock()
            mock_framework.get_project_module.return_value = mock_module

            with patch.object(
                endpoint, "_execute_module_async", return_value="stream_result"
            ):

                async def stream_prediction():
                    chunks = []
                    async for chunk in endpoint.stream_prediction(
                        "test_project", "test_module", {"test": "input"}
                    ):
                        chunks.append(chunk)
                    return chunks

                result = await benchmark.async_run(stream_prediction, iterations=10)

                # Streaming should be efficient
                assert result.avg_time < 1.0  # Less than 1 second per stream


@pytest.mark.benchmark
class TestRecoveryPerformance:
    """Performance tests for recovery systems."""

    def test_health_checker_performance(self, benchmark):
        """Benchmark health checker performance."""
        from dspy_toolkit.recovery.health_checker import HealthChecker

        checker = HealthChecker()

        def fast_health_check():
            return {"status": "healthy"}

        checker.add_health_check("fast_component", fast_health_check)

        async def check_all_health():
            return await checker.check_all()

        # Run sync benchmark for async function
        def sync_health_check():
            return asyncio.run(check_all_health())

        result = benchmark(sync_health_check)

        # Health checks should be fast
        assert result.avg_time < 0.1  # Less than 100ms

    def test_fallback_manager_performance(self, benchmark):
        """Benchmark fallback manager performance."""
        from dspy_toolkit.recovery.fallback_manager import FallbackManager

        manager = FallbackManager()

        def fast_primary():
            return "primary_success"

        def fast_fallback():
            return "fallback_success"

        manager.set_primary(fast_primary)
        manager.add_fallback(fast_fallback, "backup")

        def execute_with_fallback():
            return manager.execute()

        result = benchmark(execute_with_fallback)

        # Fallback manager should add minimal overhead for successful primary
        assert result.avg_time < 0.001  # Less than 1ms


@pytest.mark.memory
class TestMemoryUsage:
    """Memory usage tests."""

    def test_framework_memory_usage(self, test_config, memory_profiler):
        """Test framework memory usage."""
        with memory_profiler() as profiler:
            with (
                patch("dspy_toolkit.framework.MLXLLMProvider"),
                patch("dspy_toolkit.framework.setup_mlx_provider_for_dspy"),
                patch("dspy_toolkit.framework.dspy"),
            ):

                framework = DSPyFramework(test_config)

                # Perform various operations
                for i in range(10):
                    framework.health_check()
                    framework.get_framework_stats()

        profile = profiler.get_profile()
        memory_increase_mb = profile["memory_increase"] / (1024 * 1024)

        # Framework should not use excessive memory
        assert memory_increase_mb < 100  # Less than 100MB

    def test_signature_registry_memory_usage(self, mock_signature, memory_profiler):
        """Test signature registry memory usage."""
        from dspy_toolkit.registry import SignatureRegistry

        with memory_profiler() as profiler:
            registry = SignatureRegistry()

            # Register many signatures
            for project_id in range(10):
                signatures = {f"sig_{i}": mock_signature for i in range(100)}
                registry.register_project(f"project_{project_id}", signatures)

        profile = profiler.get_profile()
        memory_increase_mb = profile["memory_increase"] / (1024 * 1024)

        # Should handle many signatures efficiently
        assert memory_increase_mb < 50  # Less than 50MB for 1000 signatures

    def test_module_manager_memory_usage(self, mock_dspy_module, memory_profiler):
        """Test module manager memory usage."""
        from dspy_toolkit.manager import ModuleManager

        with memory_profiler() as profiler:
            manager = ModuleManager()

            # Register many modules
            for i in range(100):
                manager.register_module(f"module_{i}", mock_dspy_module, {"id": i})

        profile = profiler.get_profile()
        memory_increase_mb = profile["memory_increase"] / (1024 * 1024)

        # Should handle many modules efficiently
        assert memory_increase_mb < 100  # Less than 100MB for 100 modules

    def test_monitoring_memory_usage(self, performance_metrics, memory_profiler):
        """Test monitoring system memory usage."""
        with memory_profiler() as profiler:
            monitor = DSPyMonitor(enable_system_monitoring=False)

            # Record many metrics
            for i in range(1000):
                monitor.record_request(
                    f"project_{i % 10}", f"module_{i % 5}", performance_metrics
                )

        profile = profiler.get_profile()
        memory_increase_mb = profile["memory_increase"] / (1024 * 1024)

        # Monitoring should be memory efficient
        assert memory_increase_mb < 50  # Less than 50MB for 1000 metrics

        monitor.cleanup()


@pytest.mark.benchmark
class TestConcurrencyPerformance:
    """Concurrency and threading performance tests."""

    @pytest.mark.asyncio
    async def test_concurrent_framework_operations(self, mock_framework):
        """Test concurrent framework operations."""

        async def framework_operation(operation_id):
            # Simulate various framework operations
            mock_framework.health_check()
            mock_framework.get_framework_stats()
            await asyncio.sleep(0.01)  # Simulate some work
            return f"result_{operation_id}"

        # Run many concurrent operations
        start_time = time.time()
        tasks = [framework_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Should complete efficiently
        total_time = end_time - start_time
        assert total_time < 2.0  # Should complete in less than 2 seconds
        assert len(results) == 50
        assert all(result.startswith("result_") for result in results)

    def test_thread_safety(self, mock_framework):
        """Test thread safety of framework components."""
        results = []
        errors = []

        def worker_thread(thread_id):
            try:
                for i in range(10):
                    # Perform thread-safe operations
                    health = mock_framework.health_check()
                    stats = mock_framework.get_framework_stats()
                    results.append(f"thread_{thread_id}_op_{i}")
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(5)]

            # Wait for completion
            for future in futures:
                future.result(timeout=5.0)

        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 operations each

    @pytest.mark.asyncio
    async def test_concurrent_monitoring(self, performance_metrics):
        """Test concurrent monitoring operations."""
        monitor = DSPyMonitor(enable_system_monitoring=False)

        async def record_metrics_batch(batch_id):
            for i in range(10):
                monitor.record_request(
                    f"project_{batch_id}", f"module_{i}", performance_metrics
                )
            return batch_id

        # Run concurrent monitoring
        start_time = time.time()
        tasks = [record_metrics_batch(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Should handle concurrent recording efficiently
        total_time = end_time - start_time
        assert total_time < 1.0  # Should complete quickly
        assert len(results) == 10

        # Verify all metrics were recorded
        summary = await monitor.get_metrics()
        assert (
            summary["performance"]["total_requests"] == 100
        )  # 10 batches * 10 metrics

        monitor.cleanup()


@pytest.mark.benchmark
@pytest.mark.slow
class TestScalabilityPerformance:
    """Scalability performance tests."""

    def test_large_dataset_optimization(self, mock_dspy_module, benchmark):
        """Test optimization performance with large datasets."""
        from dspy_toolkit.optimizer import OptimizerEngine

        # Create large dataset
        large_dataset = [
            {"input": f"input_{i}", "expected_output": f"output_{i}"}
            for i in range(1000)
        ]

        engine = OptimizerEngine()

        with patch.object(engine, "_run_optimizer", return_value=mock_dspy_module):

            def optimize_large_dataset():
                return engine.optimize(
                    mock_dspy_module, large_dataset[:100], ["accuracy"]
                )

            result = benchmark.run(optimize_large_dataset, iterations=1)

            # Should handle large datasets reasonably
            assert result.duration < 10.0  # Less than 10 seconds

    def test_many_signatures_performance(self, mock_signature, benchmark):
        """Test performance with many registered signatures."""
        from dspy_toolkit.registry import SignatureRegistry

        registry = SignatureRegistry()

        def register_many_signatures():
            # Register signatures for many projects
            for project_id in range(50):
                signatures = {f"sig_{i}": mock_signature for i in range(20)}
                registry.register_project(f"project_{project_id}", signatures)

        result = benchmark.run(register_many_signatures, iterations=1)

        # Should handle many signatures efficiently
        assert result.duration < 5.0  # Less than 5 seconds for 1000 signatures

        # Verify retrieval performance
        def get_all_signatures():
            return registry.get_all_signatures()

        retrieval_result = benchmark.run(get_all_signatures, iterations=10)
        assert retrieval_result.avg_time < 0.1  # Fast retrieval

    def test_high_throughput_monitoring(self, performance_metrics, benchmark):
        """Test monitoring system under high throughput."""
        monitor = DSPyMonitor(enable_system_monitoring=False)

        def high_throughput_recording():
            # Record many metrics quickly
            for i in range(100):
                monitor.record_request(
                    f"project_{i % 10}", f"module_{i % 5}", performance_metrics
                )

        result = benchmark.run(high_throughput_recording, iterations=10)

        # Should handle high throughput efficiently
        assert result.avg_time < 1.0  # Less than 1 second for 100 metrics

        # Verify system remains responsive
        summary = asyncio.run(monitor.get_metrics())
        assert (
            summary["performance"]["total_requests"] == 1000
        )  # 10 iterations * 100 metrics

        monitor.cleanup()
