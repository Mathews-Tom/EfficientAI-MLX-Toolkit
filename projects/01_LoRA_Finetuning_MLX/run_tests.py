#!/usr/bin/env uv run python3
"""
Comprehensive test runner for LoRA Fine-tuning Framework.

This script runs all tests with proper configuration and reporting.
"""

import sys
import subprocess
from pathlib import Path
import argparse
import time


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    duration = end_time - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {description} - PASSED ({duration:.2f}s)")
        return True
    else:
        print(f"\n❌ {description} - FAILED ({duration:.2f}s)")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="LoRA Framework Test Runner")
    parser.add_argument("--fast", action="store_true", help="Run only fast tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, default=1, help="Run tests in parallel")
    parser.add_argument("--pattern", "-k", type=str, help="Run tests matching pattern")
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    original_dir = Path.cwd()
    
    try:
        import os
        os.chdir(project_dir)
        
        print("🚀 LoRA Fine-tuning Framework - Test Suite")
        print(f"📁 Project Directory: {project_dir}")
        print(f"🐍 Python: {sys.executable}")
        
        # Check if pytest is available
        try:
            import pytest
            print(f"✅ pytest version: {pytest.__version__}")
        except ImportError:
            print("❌ pytest not installed. Run: uv add pytest")
            return 1
        
        # Check if MLX is available
        try:
            import mlx.core as mx
            print(f"✅ MLX available: {mx.metal.is_available() if hasattr(mx, 'metal') else 'Unknown'}")
        except ImportError:
            print("⚠️  MLX not available - some tests will be skipped")
        
        success_count = 0
        total_tests = 0
        
        # Build base pytest command with uv run prefix
        base_cmd = ["uv", "run", "pytest"]
        
        if args.verbose:
            base_cmd.append("-v")
        
        if args.parallel > 1:
            base_cmd.extend(["-n", str(args.parallel)])
        
        if args.pattern:
            base_cmd.extend(["-k", args.pattern])
        
        if args.coverage:
            base_cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:coverage_html",
                "--cov-report=xml:coverage.xml",
            ])
        
        # Test categories to run
        test_categories = []
        
        if args.fast or not (args.integration or args.benchmark):
            test_categories.append(("Unit Tests", [
                "tests/test_lora.py::TestLoRAConfig",
                "tests/test_lora.py::TestLoRALayers", 
                "tests/test_training.py::TestTrainingState",
                "tests/test_training.py::TestOptimizers",
                "tests/test_training.py::TestCallbacks",
                "tests/test_inference.py::TestInferenceResult",
                "tests/test_inference.py::TestGenerationRequests",
                "tests/test_cli.py::TestCLICommands",
            ]))
        
        if args.integration or not args.fast:
            test_categories.append(("Integration Tests", [
                "tests/ -m integration"
            ]))
        
        if args.benchmark:
            test_categories.append(("Benchmark Tests", [
                "tests/ -m benchmark"
            ]))
        
        # If no specific category requested, run all basic tests
        if not test_categories:
            test_categories = [
                ("All Tests", ["tests/"])
            ]
        
        # Run each test category
        for category_name, test_paths in test_categories:
            total_tests += 1
            
            cmd = base_cmd + test_paths
            
            if run_command(cmd, f"{category_name}"):
                success_count += 1
        
        # Additional checks
        additional_checks = []
        
        # Check imports work correctly
        additional_checks.append(("Import Tests", [
            "uv", "run", "python", "-c", 
            "import sys; sys.path.insert(0, 'src'); "
            "from lora import LoRAConfig; "
            "from training import LoRATrainer; "
            "from inference import LoRAInferenceEngine; "
            "print('✅ All imports successful')"
        ]))
        
        # Validate configuration
        if Path("configs/default.yaml").exists():
            additional_checks.append(("Config Validation", [
                "uv", "run", "python", "src/cli.py", "validate", "--config", "configs/default.yaml"
            ]))
        
        # Run additional checks
        for check_name, check_cmd in additional_checks:
            total_tests += 1
            if run_command(check_cmd, check_name):
                success_count += 1
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Passed: {success_count}/{total_tests}")
        print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
        
        if args.coverage and success_count > 0:
            print(f"📋 Coverage report generated: coverage_html/index.html")
        
        if success_count == total_tests:
            print(f"\n🎉 All tests passed!")
            return 0
        else:
            print(f"\n💥 Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        return 1
    
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    sys.exit(main())