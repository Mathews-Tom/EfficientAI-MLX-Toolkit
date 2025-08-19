#!/usr/bin/env python3
"""
Quick test script to verify all imports work correctly after compliance fixes.
"""


def test_core_imports():
    """Test that all core utilities import without defensive checks."""
    print("Testing core imports...")

    # Test logging utilities
    from utils.logging_utils import AppleSiliconLogger, get_logger, setup_logging

    print("✅ Logging utilities imported successfully")

    # Test config manager
    from utils.config_manager import ConfigManager, ConfigurationError

    print("✅ Config manager imported successfully")

    # Test benchmark runner
    from utils.benchmark_runner import BenchmarkError, BenchmarkRunner

    print("✅ Benchmark runner imported successfully")

    # Test main package
    from efficientai_mlx_toolkit import BenchmarkRunner, ConfigManager, setup_logging

    print("✅ Main package imports successful")


def test_optional_imports():
    """Test optional imports that should fail gracefully."""
    print("\nTesting optional imports...")

    # Test visualization imports (should work if --extra visualization installed)
    try:
        from utils.plotting_utils import create_performance_plot

        print("✅ Visualization utilities available")
    except ImportError as e:
        print(f"ℹ️  Visualization not available: {e}")
        print("   Install with: uv sync --extra visualization")

    # Test MLX (truly optional - Apple Silicon specific)
    try:
        import mlx.core as mx

        print("✅ MLX available (Apple Silicon)")
    except ImportError:
        print("ℹ️  MLX not available (not on Apple Silicon or not installed)")


if __name__ == "__main__":
    print("🧪 Testing import compliance...")
    test_core_imports()
    test_optional_imports()
    print("\n✅ All import tests completed!")
