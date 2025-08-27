# Knowledge Base Test Suite

This directory contains comprehensive unit and integration tests for the Knowledge Base system.

## Overview

The test suite covers all major components of the knowledge base:

- **Models** (`test_models.py`): Tests for `KnowledgeBaseEntry` and `KnowledgeBaseIndex` classes
- **Indexer** (`test_indexer.py`): Tests for the indexing system including file scanning and index building
- **Search** (`test_search.py`): Tests for search functionality, filtering, and ranking
- **Validation** (`test_validation.py`): Tests for content validation and consistency checking

## Setup

### Install Dependencies

```bash
# Install test dependencies
uv add --group test -r requirements.txt

# Or install individual packages
uv add pytest pytest-cov pytest-mock
```

### Verify Installation

```bash
python -c "import pytest; print('pytest installed successfully')"
```

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage

# Run only unit tests
python run_tests.py --unit

# Run only integration tests
python run_tests.py --integration
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific test file
pytest test_models.py

# Run specific test class
pytest test_models.py::TestKnowledgeBaseEntry

# Run specific test method
pytest test_models.py::TestKnowledgeBaseEntry::test_entry_creation_valid

# Run with coverage
pytest --cov=../meta --cov-report=html

# Run in parallel
pytest -n auto
```

### Test Categories

Tests are organized by suite and markers:

```bash
# Run by test suite
pytest unit/              # Unit tests only
pytest integration/       # Integration tests only
pytest e2e/              # End-to-end tests only

# Run by markers
pytest -m unit           # Unit tests (fast)
pytest -m integration    # Integration tests (moderate)
pytest -m e2e           # End-to-end tests (slow)
pytest -m "not slow"    # Skip slow tests

# Run specific modules
pytest unit/test_models.py        # Models only
pytest unit/test_isolated.py      # Isolated tests (recommended for CI)
pytest integration/               # All integration tests
```

### Integration Tests

The `integration/` directory contains comprehensive integration tests that verify:

- **End-to-end contribution workflow** (`test_contribution_workflow.py`)
  - Complete entry lifecycle from creation to maintenance
  - Multi-contributor workflows
  - Category and tag organization
  - Incremental updates and error handling

- **Cross-reference functionality** (`test_cross_references.py`)
  - Cross-reference creation and detection
  - Link validation and broken link detection
  - Bidirectional relationships
  - Consistency checking

- **CLI and maintenance tools** (`test_cli_integration.py`)
  - CLI command workflows
  - Maintenance tool integration
  - Error handling and edge cases
  - Reporting and analytics

**Running Integration Tests:**
```bash
# All integration tests
python run_integration_tests.py

# Specific integration test files
pytest integration/test_contribution_workflow.py
pytest integration/test_cross_references.py
pytest integration/test_cli_integration.py

# Integration tests with coverage
python run_tests.py --integration-only --coverage
```

### Isolated Tests (Recommended)

The `test_isolated.py` file contains tests that are completely independent of any existing knowledge base content. These tests:

- ✅ Create their own temporary knowledge base structures
- ✅ Use predictable, controlled test data
- ✅ Are resilient to changes in the actual knowledge base
- ✅ Run consistently in any environment
- ✅ Are ideal for continuous integration

**Use isolated tests for**:
- New feature development
- Regression testing
- Continuous integration
- Testing core functionality

**Example isolated test**:
```python
def test_isolated_search_functionality(self):
    """Test search with completely controlled data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create known test entries
        kb_path, test_entries = self.create_minimal_test_kb(Path(temp_dir))

        # Test with predictable results
        indexer = KnowledgeBaseIndexer(kb_path)
        index = indexer.build_index()

        # Assertions based on known test data
        assert len(index.entries) == len(test_entries)
```

## Test Structure

The tests are organized into three main categories in dedicated subfolders:

### Unit Tests (`unit/`)

Fast, isolated tests that focus on individual functions and classes:

- **`test_models.py`** - Data model validation and behavior
- **`test_indexer.py`** - Individual indexing functions
- **`test_search.py`** - Search and filtering logic
- **`test_validation.py`** - Validation functions
- **`test_isolated.py`** - Completely isolated tests (recommended for CI)

### Integration Tests (`integration/`)

Tests that verify component interactions and workflows:

- **`test_contribution_workflow.py`** - End-to-end contribution processes
- **`test_cli_integration.py`** - CLI command integration with system components
- Cross-component functionality and data flow

### End-to-End Tests (`e2e/`)

Complete workflow tests that simulate real user scenarios:

- **`test_complete_workflows.py`** - Full developer and team workflows
- Multi-step processes from contribution to maintenance
- Realistic usage patterns and complex interactions

### Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `temp_kb_dir`: Temporary directory for testing
- `sample_entry_content`: Valid entry content
- `sample_frontmatter`: Valid frontmatter data
- `invalid_frontmatter`: Invalid frontmatter for error testing

## Writing Tests

### Test Naming Convention

- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<functionality>_<scenario>`

### Test Independence and Isolation

**CRITICAL**: Tests should be completely independent of the actual knowledge base content:

✅ **DO**: Create isolated test data
```python
def test_search_functionality(self):
    # Create temporary knowledge base with known content
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_path = self.create_test_kb_with_known_entries(temp_dir)
        # Test with predictable data
```

❌ **DON'T**: Depend on real knowledge base entries
```python
def test_search_functionality(self):
    # This will break when knowledge base content changes!
    results = searcher.search("mlx")
    assert len(results) == 2  # Fragile assumption
```

### Example Test Structure

```python
class TestKnowledgeBaseEntry:
    """Test cases for KnowledgeBaseEntry class."""

    def test_entry_creation_valid(self):
        """Test creating a valid knowledge base entry."""
        # Arrange - Create test data
        entry_data = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "intermediate",
            "content_path": Path("test.md"),
            "last_updated": datetime.now(),
            "contributors": ["test-user"],
        }

        # Act
        entry = KnowledgeBaseEntry(**entry_data)

        # Assert
        assert entry.title == "Test Entry"
        assert entry.category == "test-category"

    def test_entry_validation_invalid_difficulty(self):
        """Test validation fails for invalid difficulty."""
        with pytest.raises(ValueError, match="Invalid difficulty"):
            KnowledgeBaseEntry(
                title="Test",
                category="test",
                tags=["test"],
                difficulty="invalid",  # This should fail
                content_path=Path("test.md"),
                last_updated=datetime.now(),
                contributors=["test-user"],
            )
```

### Best Practices

1. **Create isolated test data** - Never depend on real knowledge base content
2. **Use temporary directories** for file system tests
3. **Use descriptive test names** that explain what is being tested
4. **Follow AAA pattern** (Arrange, Act, Assert)
5. **Test both success and failure cases**
6. **Use fixtures** for common test data setup
7. **Mock external dependencies** in unit tests
8. **Clean up resources** in tests (use context managers)
9. **Test edge cases** and boundary conditions
10. **Make assertions specific** - avoid brittle exact counts when possible

### Robust vs Fragile Tests

**Robust Test** (Recommended):
```python
def test_search_returns_relevant_results(self):
    # Create known test data
    test_entries = self.create_test_entries_with_mlx_tag()
    index = KnowledgeBaseIndex(entries=test_entries)
    searcher = KnowledgeBaseSearcher(index)

    results = searcher.search("mlx")

    # Test behavior, not exact counts
    assert len(results.results) > 0
    assert all("mlx" in result.entry.tags for result in results.results)
```

**Fragile Test** (Avoid):
```python
def test_search_returns_two_mlx_results(self):
    # Depends on current knowledge base content
    results = searcher.search("mlx")
    assert len(results.results) == 2  # Will break when content changes!
```

## Coverage

Generate coverage reports to ensure comprehensive testing:

```bash
# Generate HTML coverage report
pytest --cov=../meta --cov-report=html

# View coverage in terminal
pytest --cov=../meta --cov-report=term-missing

# Set coverage threshold
pytest --cov=../meta --cov-fail-under=80
```

Coverage reports are generated in `htmlcov/` directory.

## Continuous Integration

The test suite is designed to run in CI environments:

```bash
# CI-friendly test run
pytest --tb=short --disable-warnings --color=yes

# With coverage for CI
pytest --cov=../meta --cov-report=xml --cov-report=term
```

## Debugging Tests

### Running with Debugger

```bash
# Drop into pdb on failure
pytest --pdb

# Drop into pdb on first failure
pytest --pdb -x
```

### Verbose Output

```bash
# Verbose output
pytest -v

# Very verbose output
pytest -vv

# Show local variables in tracebacks
pytest -l
```

### Filtering Tests

```bash
# Run tests matching pattern
pytest -k "test_search"

# Run tests NOT matching pattern
pytest -k "not slow"

# Combine patterns
pytest -k "test_search and not slow"
```

## Performance Testing

Some tests are marked as slow and can be skipped:

```bash
# Skip slow tests
pytest -m "not slow"

# Run only slow tests
pytest -m slow

# Set test timeout
pytest --timeout=60
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `.meta` directory is in Python path
2. **File Not Found**: Tests create temporary files; ensure proper cleanup
3. **Permission Errors**: Tests may need write access to temp directories

### Debug Import Issues

```python
import sys
from pathlib import Path

# Check if meta directory is in path
kb_meta_path = Path(__file__).parent.parent / ".meta"
print(f"Meta path: {kb_meta_path}")
print(f"Meta path exists: {kb_meta_path.exists()}")
print(f"In sys.path: {str(kb_meta_path) in sys.path}")
```

### Clean Test Environment

```bash
# Remove pytest cache
rm -rf .pytest_cache __pycache__ */__pycache__

# Remove coverage files
rm -rf htmlcov .coverage
```

## Recommended Testing Workflow

### For Development
```bash
# Quick feedback loop - run isolated tests
python run_isolated_tests.py

# Run unit tests only (fast)
python run_tests.py --suite unit

# Run integration tests
python run_tests.py --suite integration

# Test specific module
python run_tests.py --module isolated

# Full test suite
python run_tests.py
```

### For Continuous Integration
```bash
# Recommended: Run only isolated tests in CI
python run_isolated_tests.py

# Alternative: Run unit tests with coverage
python run_tests.py --suite unit --coverage

# Full CI pipeline (unit + integration)
python run_tests.py --suite unit
python run_tests.py --suite integration
```

### For Comprehensive Testing
```bash
# Run everything including slow E2E tests
python run_tests.py --suite all

# Run E2E tests only (slow but comprehensive)
python run_tests.py --suite e2e
```

### Why Isolated Tests?

As the knowledge base grows with real content, tests that depend on specific entries will break. The isolated tests:

- ✅ **Always pass** regardless of knowledge base content
- ✅ **Test core functionality** with controlled data
- ✅ **Run fast** and are reliable for CI/CD
- ✅ **Provide clear feedback** when functionality breaks
- ✅ **Scale with the project** without maintenance overhead

## Contributing

When adding new functionality:

1. **Write isolated tests first** (TDD approach with controlled data)
2. **Use `test_isolated.py` as a template** for new tests
3. **Avoid depending on real knowledge base content** in tests
4. **Ensure good coverage** of new code with isolated tests
5. **Add integration tests** only when testing real file system interactions
6. **Update this README** if adding new test categories
7. **Run isolated tests** before submitting changes

### Test Checklist

- [ ] Isolated unit tests for new functions/classes
- [ ] Tests use temporary/controlled data (not real KB content)
- [ ] Integration tests for new workflows (if needed)
- [ ] Error handling tests with predictable inputs
- [ ] Edge case tests with known scenarios
- [ ] Performance tests (if applicable)
- [ ] Documentation updated

### Example: Adding a New Feature

```python
# ✅ Good: Isolated test
def test_new_feature_with_controlled_data(self):
    # Create predictable test data
    test_entries = [
        self.create_test_entry("Known Title", ["known", "tags"]),
        self.create_test_entry("Another Title", ["other", "tags"]),
    ]

    # Test with known inputs and expected outputs
    result = new_feature(test_entries)
    assert len(result) == 2
    assert result[0].title == "Known Title"

# ❌ Avoid: Test depending on real content
def test_new_feature_with_real_data(self):
    # This will break when KB content changes!
    all_entries = load_real_knowledge_base()
    result = new_feature(all_entries)
    assert len(result) == 42  # Fragile assumption
```