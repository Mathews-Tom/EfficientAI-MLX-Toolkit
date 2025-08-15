"""
Pytest configuration for Knowledge Base tests.

This file contains shared fixtures and configuration for all tests.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add the .meta directory to Python path for all tests
kb_meta_path = Path(__file__).parent.parent / ".meta"
sys.path.insert(0, str(kb_meta_path))


@pytest.fixture
def temp_kb_dir():
    """Create a temporary directory for knowledge base testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_entry_content():
    """Provide sample entry content for testing."""
    return """---
title: "Sample Entry"
category: "test-category"
tags: ["test", "sample", "example"]
difficulty: "intermediate"
last_updated: "2024-01-15"
contributors: ["test-user", "sample-contributor"]
---

# Sample Entry

## Problem/Context
This is a sample entry used for testing purposes.

## Solution/Pattern
The solution involves creating comprehensive test cases.

## Code Example
```python
def sample_function():
    \"\"\"A sample function for testing.\"\"\"
    return "Hello, World!"

# Usage example
result = sample_function()
print(result)
```

## Gotchas/Pitfalls
- Remember to include proper error handling
- Always validate input parameters

## Performance Impact
This is a lightweight example with minimal performance impact.

## Related Knowledge
- [Testing Best Practices](../testing/best-practices.md)
- [Code Quality Guidelines](../quality/guidelines.md)
"""


@pytest.fixture
def sample_frontmatter():
    """Provide sample frontmatter for testing."""
    return {
        "title": "Sample Entry",
        "category": "test-category",
        "tags": ["test", "sample", "example"],
        "difficulty": "intermediate",
        "last_updated": "2024-01-15",
        "contributors": ["test-user", "sample-contributor"],
    }


@pytest.fixture
def invalid_frontmatter():
    """Provide invalid frontmatter for testing."""
    return {
        "title": "",  # Empty title
        "category": "Invalid_Category",  # Invalid format
        "tags": "not-a-list",  # Should be list
        "difficulty": "invalid",  # Invalid difficulty
        "last_updated": "invalid-date",  # Invalid date format
        "contributors": [],  # Empty contributors
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names and locations."""
    for item in items:
        # Mark integration tests
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "slow" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
