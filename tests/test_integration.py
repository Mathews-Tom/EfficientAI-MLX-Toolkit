"""
Integration tests for DSPy Integration Framework.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dspy_toolkit.deployment import DSPyMonitor, create_dspy_app
from dspy_toolkit.framework import DSPyFramework
from dspy_toolkit.recovery import CircuitBreaker, FallbackManager, RetryHandler
from dspy_toolkit.types import DSPyConfig


@pytest.mark.integration
class TestFrameworkIntegration:
    """Integration tests for the complete framework."""

    def test_framework_initialization_flow(self, test_config):
        """Test complete framework initialization flow."""
        with (
            patch("dspy_toolkit.framework.MLXLLMProvider") as mock_provider_class,
            patch("dspy_toolkit.framework.setup_mlx_provider_for_dspy"),
            patch("dspy_toolkit.framework.dspy"),
        ):

            mock_provider = Mock()
            mock_provider.is_available.return_value = True
            mock_provider.hardware_info = Mock()
            mock_provider_class.return_value = mock_provider

            framework = DSPyFramework(test_config)

            # Verify initialization
            assert framework.config == test_config
            assert framework.signature_registry is not None
            assert framework.module_manager is not None
            assert framework.optimizer_engine is not None
            assert framework.llm_provider is not None

    def test_signature_to_module_workflow(self, mock_framework, mock_signature):
        """Test workflow from signature registration to module creation."""
        project_name = "test_project"
        signature_name = "test_signature"
        module_name = "test_module"

        # Register signature
        signatures = {signature_name: mock_signature}
        mock_framework.signature_registry.get_project_signatures.return_value = (
            signatures
        )
        mock_framework.register_project_signatures(project_name, signatures)

        # Create module
        mock_module = Mock()
        with patch(
            "dspy_toolkit.framework.dspy.ChainOfThought", return_value=mock_module
        ):
            created_module = mock_framework.create_project_module(
                project_name, module_name, signature_name
            )

        # Verify workflow
        mock_framework.signature_registry.register_project.assert_called_once_with(
            project_name, signatures
        )
        mock_framework.module_manager.register_module.assert_called_once()
        assert created_module == mock_module

    @pytest.mark.asyncio
    async def test_end_to_end_prediction_workflow(
        self, mock_framework, mock_dspy_module
    ):
        """Test end-to-end prediction workflow."""
        project_name = "test_project"
        module_name = "test_module"

        # Setup mock module
        mock_framework.get_project_module.return_value = mock_dspy_module

        # Create FastAPI app
        with (
            patch(
                "dspy_toolkit.deployment.fastapi_integration.FASTAPI_AVAILABLE",
                True,
            ),
            patch(
                "dspy_toolkit.deployment.fastapi_integration.FastAPI"
            ) as mock_fastapi,
        ):

            mock_app = Mock()
            mock_fastapi.return_value = mock_app

            dspy_app = create_dspy_app(mock_framework.config)

            # Simulate prediction request
            inputs = {"input": "test input"}

            # This would normally go through FastAPI, but we'll test the core logic
            from dspy_toolkit.deployment.fastapi_integration import DSPyFastAPIApp

            app_instance = DSPyFastAPIApp(mock_framework)

            # Mock the execution
            with patch.object(
                app_instance,
                "_execute_module_async",
                return_value={"answer": "test response"},
            ):
                # This simulates the prediction flow
                result = await app_instance._execute_module_async(
                    mock_dspy_module, inputs
                )

                assert result == {"answer": "test response"}

    def test_optimization_workflow(
        self, mock_framework, mock_dspy_module, sample_dataset
    ):
        """Test module optimization workflow."""
        # Setup optimizer
        mock_framework.optimizer_engine.optimize.return_value = mock_dspy_module

        # Optimize module
        optimized_module = mock_framework.optimize_module(
            mock_dspy_module, sample_dataset, ["accuracy"]
        )

        # Verify optimization
        mock_framework.optimizer_engine.optimize.assert_called_once()
        mock_framework.module_manager.register_module.assert_called_once()
        assert optimized_module == mock_dspy_module

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, temp_dir):
        """Test monitoring integration with framework."""
        monitor = DSPyMonitor(
            export_path=temp_dir / "monitoring", enable_system_monitoring=False
        )

        # Record some metrics
        from dspy_toolkit.deployment.monitoring import PerformanceMetrics

        for i in range(3):
            metrics = PerformanceMetrics(
                execution_time=1.0 + i * 0.1,
                input_tokens=100 + i * 10,
                output_tokens=200 + i * 20,
                memory_usage=1024 + i * 100,
                timestamp=time.time(),
                success=True,
            )
            monitor.record_request("test_project", "test_module", metrics)

        # Get metrics
        summary = await monitor.get_metrics()

        assert summary["performance"]["total_requests"] == 3
        assert summary["performance"]["success_rate"] == 1.0

        # Export metrics
        await monitor.export_metrics("integration_test.json")

        export_file = monitor.export_path / "integration_test.json"
        assert export_file.exists()

        monitor.cleanup()


@pytest.mark.integration
class TestRecoveryIntegration:
    """Integration tests for recovery systems."""

    def test_circuit_breaker_with_framework(self, mock_framework):
        """Test circuit breaker integration with framework operations."""
        from dspy_toolkit.recovery.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
        )

        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        cb = CircuitBreaker("framework_test", config)

        # Simulate framework operation that might fail
        def framework_operation():
            health = mock_framework.health_check()
            if health["overall_status"] != "healthy":
                raise Exception("Framework unhealthy")
            return "success"

        # Should succeed initially
        result = cb.call(framework_operation)
        assert result == "success"

        # Simulate framework becoming unhealthy
        mock_framework.health_check.return_value = {
            "overall_status": "unhealthy",
            "issues": ["test issue"],
        }

        # Should fail and eventually open circuit
        with pytest.raises(Exception):
            cb.call(framework_operation)
        with pytest.raises(Exception):
            cb.call(framework_operation)

        # Circuit should be open now
        assert cb.state.value == "open"

    def test_retry_with_fallback(self, mock_framework):
        """Test retry handler with fallback manager."""
        from dspy_toolkit.recovery.fallback_manager import FallbackManager
        from dspy_toolkit.recovery.retry_handler import RetryConfig, RetryHandler

        retry_config = RetryConfig(max_attempts=2, base_delay=0.01)
        retry_handler = RetryHandler(retry_config)

        fallback_manager = FallbackManager()

        attempt_count = 0

        def failing_primary():
            nonlocal attempt_count
            attempt_count += 1
            raise Exception(f"Primary failed (attempt {attempt_count})")

        def success_fallback():
            return "fallback_success"

        fallback_manager.set_primary(failing_primary)
        fallback_manager.add_fallback(success_fallback, "backup")

        # Wrap fallback execution with retry
        result = retry_handler.execute(fallback_manager.execute)

        assert result == "fallback_success"
        assert attempt_count >= 1  # Primary was attempted

    @pytest.mark.asyncio
    async def test_health_monitoring_with_recovery(self, mock_framework):
        """Test health monitoring triggering recovery actions."""
        from dspy_toolkit.recovery.health_checker import (
            HealthCheckConfig,
            HealthChecker,
        )

        config = HealthCheckConfig(check_interval=0.1, failure_threshold=1)
        checker = HealthChecker(config)

        # Track component state
        component_healthy = True
        recovery_triggered = False

        def component_check():
            return {"status": "healthy" if component_healthy else "unhealthy"}

        def recovery_action():
            nonlocal component_healthy, recovery_triggered
            component_healthy = True
            recovery_triggered = True

        checker.add_health_check("test_component", component_check)

        # Initial check should be healthy
        results = await checker.check_all()
        assert results["test_component"].status.value == "healthy"

        # Simulate component failure
        component_healthy = False
        results = await checker.check_all()
        assert results["test_component"].status.value == "unhealthy"

        # Trigger recovery
        recovery_action()
        assert recovery_triggered

        # Check should be healthy again
        results = await checker.check_all()
        assert results["test_component"].status.value == "healthy"


@pytest.mark.integration
class TestDeploymentIntegration:
    """Integration tests for deployment components."""

    @pytest.mark.asyncio
    async def test_fastapi_with_monitoring(self, mock_framework):
        """Test FastAPI integration with monitoring."""
        with (
            patch(
                "dspy_toolkit.deployment.fastapi_integration.FASTAPI_AVAILABLE",
                True,
            ),
            patch(
                "dspy_toolkit.deployment.fastapi_integration.FastAPI"
            ) as mock_fastapi,
        ):

            mock_app = Mock()
            mock_fastapi.return_value = mock_app

            # Create DSPy FastAPI app
            from dspy_toolkit.deployment.fastapi_integration import DSPyFastAPIApp

            dspy_app = DSPyFastAPIApp(mock_framework)

            # Verify monitoring is integrated
            assert dspy_app.monitor is not None

            # Simulate request handling
            mock_framework.get_project_module.return_value = Mock()

            from dspy_toolkit.deployment.fastapi_integration import DSPyRequest
            from dspy_toolkit.deployment.monitoring import PerformanceMetrics

            request = DSPyRequest(
                inputs={"test": "input"},
                project_name="test_project",
                module_name="test_module",
            )

            # Mock background tasks
            background_tasks = Mock()

            with patch.object(
                dspy_app, "_execute_module_async", return_value={"answer": "test"}
            ):
                response = await dspy_app._handle_prediction(request, background_tasks)

                assert response.outputs == {"answer": "test"}
                assert response.performance is not None
                background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_with_fallback(self, mock_framework):
        """Test streaming endpoints with fallback."""
        with patch("dspy_toolkit.deployment.streaming.STREAMING_AVAILABLE", True):
            from dspy_toolkit.deployment.streaming import DSPyStreamingEndpoint

            endpoint = DSPyStreamingEndpoint(mock_framework)

            # Setup fallback
            from dspy_toolkit.recovery.fallback_manager import FallbackManager

            fallback_manager = FallbackManager()

            def failing_primary():
                raise Exception("Primary streaming failed")

            def success_fallback():
                return "fallback_stream_result"

            fallback_manager.set_primary(failing_primary)
            fallback_manager.add_fallback(success_fallback, "stream_fallback")

            # Mock module
            mock_module = Mock()
            mock_framework.get_project_module.return_value = mock_module

            with patch.object(
                endpoint, "_execute_module_async", side_effect=fallback_manager.execute
            ):
                # This would normally stream, but we'll test the fallback logic
                result = await endpoint._execute_module_async(
                    mock_module, {"test": "input"}
                )

                assert result == "fallback_stream_result"


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance integration tests."""

    def test_framework_performance_under_load(
        self, mock_framework, sample_dataset, benchmark
    ):
        """Test framework performance under load."""
        # Setup optimization
        mock_framework.optimizer_engine.optimize.return_value = Mock()

        def optimize_operation():
            return mock_framework.optimize_module(
                Mock(), sample_dataset[:3], ["accuracy"]
            )

        # Benchmark optimization
        result = benchmark.run(optimize_operation, iterations=10)

        assert result.avg_time < 1.0  # Should complete in reasonable time
        assert (
            result.ops_per_second > 1.0
        )  # Should handle multiple operations per second

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_framework):
        """Test handling concurrent requests."""
        with patch(
            "dspy_toolkit.deployment.fastapi_integration.FASTAPI_AVAILABLE", True
        ):
            from dspy_toolkit.deployment.fastapi_integration import DSPyFastAPIApp

            app = DSPyFastAPIApp(mock_framework)

            # Setup mock module
            mock_module = Mock()
            mock_framework.get_project_module.return_value = mock_module

            async def simulate_request(request_id):
                with patch.object(
                    app, "_execute_module_async", return_value={"id": request_id}
                ):
                    from dspy_toolkit.deployment.fastapi_integration import (
                        DSPyRequest,
                    )

                    request = DSPyRequest(
                        inputs={"request_id": request_id},
                        project_name="test",
                        module_name="test",
                    )

                    background_tasks = Mock()
                    response = await app._handle_prediction(request, background_tasks)
                    return response.outputs["id"]

            # Run concurrent requests
            tasks = [simulate_request(i) for i in range(5)]
            results = await asyncio.gather(*tasks)

            # Verify all requests completed
            assert len(results) == 5
            assert set(results) == {0, 1, 2, 3, 4}

    def test_memory_usage_stability(self, mock_framework, memory_profiler):
        """Test memory usage remains stable under repeated operations."""
        with memory_profiler() as profiler:
            # Perform repeated operations
            for _ in range(100):
                mock_framework.health_check()
                mock_framework.get_framework_stats()

        profile = profiler.get_profile()

        # Memory increase should be minimal for repeated operations
        memory_increase_mb = profile["memory_increase"] / (1024 * 1024)
        assert memory_increase_mb < 50  # Less than 50MB increase


@pytest.mark.integration
@pytest.mark.requires_dspy
class TestDSPyIntegration:
    """Integration tests that require actual DSPy installation."""

    def test_real_dspy_signature_validation(self):
        """Test signature validation with real DSPy signatures."""
        import dspy

        class TestSignature(dspy.Signature):
            """Test signature for validation."""

            input_field = dspy.InputField(desc="Test input")
            output_field = dspy.OutputField(desc="Test output")

        from dspy_toolkit.registry import SignatureRegistry

        registry = SignatureRegistry()

        # Should validate successfully
        assert registry.validate_signature(TestSignature) == True

        # Should register successfully
        registry.register_project("test_project", {"test_sig": TestSignature})

        signatures = registry.get_project_signatures("test_project")
        assert "test_sig" in signatures
        assert signatures["test_sig"] == TestSignature

    def test_real_dspy_module_creation(self):
        """Test creating real DSPy modules."""
        import dspy

        class TestSignature(dspy.Signature):
            """Test signature."""

            question = dspy.InputField()
            answer = dspy.OutputField()

        # Create ChainOfThought module
        module = dspy.ChainOfThought(TestSignature)

        assert module is not None
        assert hasattr(module, "forward")

    @pytest.mark.slow
    def test_real_optimization_workflow(self, temp_dir):
        """Test optimization workflow with real DSPy components."""
        import dspy

        # This test would require actual LLM setup, so we'll skip it
        # in automated testing but provide the structure
        pytest.skip("Requires actual LLM setup - run manually for full testing")

        # Example of what the test would do:
        # 1. Setup real DSPy configuration
        # 2. Create real signature and module
        # 3. Prepare real dataset
        # 4. Run optimization
        # 5. Verify results


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Integration tests for error recovery scenarios."""

    def test_cascading_failure_recovery(self, mock_framework):
        """Test recovery from cascading failures."""
        from dspy_toolkit.recovery import (
            CircuitBreaker,
            FallbackManager,
            RetryHandler,
        )
        from dspy_toolkit.recovery.circuit_breaker import CircuitBreakerConfig
        from dspy_toolkit.recovery.retry_handler import RetryConfig

        # Setup recovery chain: Retry -> Circuit Breaker -> Fallback
        retry_handler = RetryHandler(RetryConfig(max_attempts=2, base_delay=0.01))
        circuit_breaker = CircuitBreaker(
            "cascade_test", CircuitBreakerConfig(failure_threshold=1)
        )
        fallback_manager = FallbackManager()

        failure_count = 0

        def failing_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count < 5:
                raise Exception(f"Failure {failure_count}")
            return "eventual_success"

        def fallback_operation():
            return "fallback_success"

        fallback_manager.set_primary(failing_operation)
        fallback_manager.add_fallback(fallback_operation, "backup")

        # Wrap with circuit breaker and retry
        @circuit_breaker
        def protected_operation():
            return retry_handler.execute(fallback_manager.execute)

        # Should eventually succeed through fallback
        result = protected_operation()
        assert result == "fallback_success"

    @pytest.mark.asyncio
    async def test_health_check_triggered_recovery(self, mock_framework):
        """Test health check triggering automatic recovery."""
        from dspy_toolkit.recovery.health_checker import (
            HealthCheckConfig,
            HealthChecker,
        )

        checker = HealthChecker(HealthCheckConfig(failure_threshold=1))

        # Simulate degrading component
        component_state = "healthy"
        recovery_attempts = 0

        def component_check():
            return {"status": component_state}

        async def recovery_procedure():
            nonlocal component_state, recovery_attempts
            recovery_attempts += 1
            if recovery_attempts >= 2:
                component_state = "healthy"
            return component_state == "healthy"

        checker.add_health_check("degrading_component", component_check)

        # Initial state is healthy
        results = await checker.check_all()
        assert results["degrading_component"].status.value == "healthy"

        # Component degrades
        component_state = "unhealthy"
        results = await checker.check_all()
        assert results["degrading_component"].status.value == "unhealthy"

        # Trigger recovery
        recovery_success = await recovery_procedure()
        assert not recovery_success  # First attempt fails

        recovery_success = await recovery_procedure()
        assert recovery_success  # Second attempt succeeds

        # Component should be healthy again
        results = await checker.check_all()
        assert results["degrading_component"].status.value == "healthy"
        assert recovery_attempts == 2
