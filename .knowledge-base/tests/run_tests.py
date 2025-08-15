#!/usr/bin/env python3
"""
Test runner for Knowledge Base system.

This script provides a convenient way to run tests with different configurations.
"""

import argparse
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    print("Error: pytest is not installed. Please install it with:")
    print("  uv add pytest")
    sys.exit(1)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Knowledge Base tests")

    # Test selection options
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument(
        "--suite",
        choices=["unit", "integration", "e2e", "all"],
        default="all",
        help="Run specific test suite",
    )
    parser.add_argument(
        "--module",
        choices=[
            "models",
            "indexer",
            "search",
            "validation",
            "isolated",
            "integration",
            "all",
        ],
        help="Run tests for specific module (within unit tests)",
    )
    parser.add_argument(
        "--isolated-only",
        action="store_true",
        help="Run only isolated tests (recommended for CI)",
    )
    parser.add_argument(
        "--integration-only", action="store_true", help="Run only integration tests"
    )

    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )

    # Pytest options
    parser.add_argument(
        "--failfast", "-x", action="store_true", help="Stop on first failure"
    )
    parser.add_argument(
        "--pdb", action="store_true", help="Drop into debugger on failures"
    )

    args = parser.parse_args()

    # Build pytest arguments
    pytest_args = []

    # Test selection
    if args.unit:
        pytest_args.extend(["-m", "unit"])
    elif args.integration:
        pytest_args.extend(["-m", "integration"])

    if args.fast:
        pytest_args.extend(["-m", "not slow"])

    # Test suite selection
    if args.isolated_only:
        pytest_args.append("test_isolated.py")
    elif args.integration_only:
        pytest_args.append("integration/")
    elif args.module:
        if args.module == "isolated":
            pytest_args.append("test_isolated.py")
        elif args.module == "integration":
            pytest_args.append("integration/")
        elif args.module != "all":
            pytest_args.append(f"test_{args.module}.py")

    # Output options
    if args.verbose:
        pytest_args.append("-v")
    elif args.quiet:
        pytest_args.append("-q")

    if args.coverage:
        try:
            import pytest_cov

            pytest_args.extend(
                ["--cov=../meta", "--cov-report=html", "--cov-report=term-missing"]
            )
        except ImportError:
            print("Warning: pytest-cov not installed. Coverage reporting disabled.")

    # Pytest options
    if args.failfast:
        pytest_args.append("-x")

    if args.pdb:
        pytest_args.append("--pdb")

    # Change to test directory
    test_dir = Path(__file__).parent
    original_cwd = Path.cwd()

    try:
        import os

        os.chdir(test_dir)

        print(f"Running tests in {test_dir}")
        if pytest_args:
            print(f"Pytest arguments: {' '.join(pytest_args)}")

        # Run pytest
        exit_code = pytest.main(pytest_args)

        if exit_code == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code {exit_code}")

        return exit_code

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    sys.exit(main())
