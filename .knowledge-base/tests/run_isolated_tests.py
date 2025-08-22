#!/usr/bin/env python3
"""
Simple test runner for isolated tests only.

This script runs only the isolated tests that don't depend on
existing knowledge base content, making it perfect for CI/CD
and development workflows.
"""

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    print("Error: pytest is not installed. Please install it with:")
    print("  uv add pytest")
    sys.exit(1)


def main():
    """Run isolated tests only."""
    # Change to test directory
    test_dir = Path(__file__).parent
    original_cwd = Path.cwd()

    try:
        import os

        os.chdir(test_dir)

        print("🧪 Running isolated Knowledge Base tests...")
        print("These tests are independent of existing knowledge base content.\n")

        # Run only isolated tests with verbose output
        pytest_args = ["unit/test_isolated.py", "-v", "--tb=short", "--color=yes"]

        exit_code = pytest.main(pytest_args)

        if exit_code == 0:
            print("\n✅ All isolated tests passed!")
            print("💡 These tests will remain stable as the knowledge base grows.")
        else:
            print(f"\n❌ Some isolated tests failed (exit code: {exit_code})")
            print("🔧 Check the output above for details.")

        return exit_code

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    sys.exit(main())
