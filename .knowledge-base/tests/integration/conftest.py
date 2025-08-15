"""
Pytest configuration for Knowledge Base integration tests.

This file contains shared fixtures and configuration for integration tests.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add the .meta directory to Python path for all tests
kb_meta_path = Path(__file__).parent.parent.parent / ".meta"
sys.path.insert(0, str(kb_meta_path))


@pytest.fixture
def temp_kb_dir():
    """Create a temporary directory for knowledge base testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def realistic_kb_structure(temp_kb_dir):
    """Create a realistic knowledge base structure for integration testing."""
    import shutil

    kb_path = temp_kb_dir / ".knowledge-base"

    # Create comprehensive directory structure
    categories = [
        "apple-silicon",
        "mlx-framework",
        "performance",
        "troubleshooting",
        "deployment",
        "testing",
    ]

    for category in categories:
        (kb_path / "categories" / category).mkdir(parents=True)

    (kb_path / "patterns" / "training-patterns").mkdir(parents=True)
    (kb_path / "patterns" / "inference-patterns").mkdir(parents=True)
    (kb_path / "templates").mkdir(parents=True)
    (kb_path / ".meta").mkdir(parents=True)

    # Copy templates from the real knowledge base
    real_kb_path = Path(__file__).parent.parent.parent
    real_templates_dir = real_kb_path / "templates"

    if real_templates_dir.exists():
        for template_file in real_templates_dir.glob("*.md"):
            shutil.copy2(template_file, kb_path / "templates")
    else:
        # Create a basic template if the real one doesn't exist
        basic_template = """---
title: "{title}"
category: "{category}"
tags: {tags}
difficulty: "{difficulty}"
last_updated: "{last_updated}"
contributors: {contributors}
---

# {title}

## Problem/Context
Describe the problem or context that this entry addresses.

## Solution/Pattern
Provide the solution or pattern.

## Code Example
```python
# Add code example here
```

## Related Knowledge
- Link to related entries
"""
        (kb_path / "templates" / "entry-template.md").write_text(basic_template)

    # Create required files
    (kb_path / "README.md").write_text("# Test Knowledge Base")
    (kb_path / ".meta" / "contribution-guide.md").write_text(
        "# Test Contribution Guide"
    )

    return kb_path


@pytest.fixture
def sample_integration_entries():
    """Provide sample entries for integration testing."""
    return [
        {
            "title": "Integration Test Entry 1",
            "category": "apple-silicon",
            "tags": ["integration", "test", "apple-silicon"],
            "difficulty": "intermediate",
            "content": """# Integration Test Entry 1

## Problem/Context
This is a test entry for integration testing.

## Solution/Pattern
Integration testing approach for knowledge base.

## Code Example
```python
def integration_test():
    return "success"
```
""",
        },
        {
            "title": "Integration Test Entry 2",
            "category": "mlx-framework",
            "tags": ["integration", "test", "mlx"],
            "difficulty": "advanced",
            "content": """# Integration Test Entry 2

## Problem/Context
Advanced integration testing scenarios.

## Solution/Pattern
Complex integration patterns.

## Code Example
```python
import mlx.core as mx

def advanced_integration():
    return mx.array([1, 2, 3])
```
""",
        },
    ]


def pytest_configure(config):
    """Configure pytest with custom markers for integration tests."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "cli: marks tests that require CLI functionality"
    )
    config.addinivalue_line(
        "markers", "filesystem: marks tests that perform file system operations"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark integration tests."""
    for item in items:
        # Mark all tests in integration folder as integration tests
        item.add_marker(pytest.mark.integration)

        # Mark CLI tests
        if "cli" in item.name.lower():
            item.add_marker(pytest.mark.cli)

        # Mark filesystem tests
        if "workflow" in item.name.lower() or "file" in item.name.lower():
            item.add_marker(pytest.mark.filesystem)

        # Mark slow tests
        if (
            "slow" in item.name.lower()
            or "complete" in item.name.lower()
            or "lifecycle" in item.name.lower()
        ):
            item.add_marker(pytest.mark.slow)
