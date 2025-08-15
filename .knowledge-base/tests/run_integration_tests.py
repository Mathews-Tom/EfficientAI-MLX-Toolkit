#!/usr/bin/env python3
"""
Integration test runner for Knowledge Base system.

This script runs comprehensive integration tests that verify
end-to-end workflows and component interactions.
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
    """Run integration tests."""
    # Change to test directory
    test_dir = Path(__file__).parent
    original_cwd = Path.cwd()

    try:
        import os

        os.chdir(test_dir)

        print("🔧 Running Knowledge Base integration tests...")
        print("These tests verify end-to-end workflows and component integration.\n")

        # Run integration tests with appropriate settings
        pytest_args = [
            "integration/",
            "-v",
            "--tb=short",
            "--color=yes",
            "-m",
            "integration",
        ]

        exit_code = pytest.main(pytest_args)

        if exit_code == 0:
            print("\n✅ All integration tests passed!")
            print("🎯 End-to-end workflows are working correctly.")
        else:
            print(f"\n❌ Some integration tests failed (exit code: {exit_code})")
            print("🔧 Check the output above for details.")

        return exit_code

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    sys.exit(main())
