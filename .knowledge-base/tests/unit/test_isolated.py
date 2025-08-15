"""
Isolated tests that don't depend on real knowledge base content.

These tests create their own temporary knowledge base structures and are
completely independent of any existing knowledge base entries.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Add the .meta directory to Python path for imports
kb_meta_path = Path(__file__).parent.parent / ".meta"
sys.path.insert(0, str(kb_meta_path))

from indexer import KnowledgeBaseIndexer
from models import KnowledgeBaseEntry, KnowledgeBaseIndex
from search import KnowledgeBaseSearcher, SearchFilter, SearchMode, SortBy


class TestIsolatedKnowledgeBase:
    """Tests that create completely isolated knowledge base structures."""

    def create_minimal_test_kb(self, temp_dir: Path) -> tuple[Path, list[dict]]:
        """Create a minimal test knowledge base with known content."""
        kb_path = temp_dir / ".knowledge-base"

        # Create directory structure
        (kb_path / "categories" / "test-category").mkdir(parents=True)
        (kb_path / "categories" / "another-category").mkdir(parents=True)
        (kb_path / ".meta").mkdir(parents=True)

        # Define test entries with predictable content
        test_entries = [
            {
                "filename": "entry-one.md",
                "category": "test-category",
                "content": """---
title: "Test Entry One"
category: "test-category"
tags: ["test", "first", "example"]
difficulty: "beginner"
last_updated: "2024-01-01"
contributors: ["test-author"]
---

# Test Entry One

## Problem/Context
This is the first test entry.

## Solution/Pattern
Simple test solution.

## Code Example
```python
def test_function():
    return "test"
```
""",
            },
            {
                "filename": "entry-two.md",
                "category": "test-category",
                "content": """---
title: "Test Entry Two"
category: "test-category"
tags: ["test", "second", "advanced"]
difficulty: "intermediate"
last_updated: "2024-01-02"
contributors: ["test-author", "another-author"]
---

# Test Entry Two

## Problem/Context
This is the second test entry.

## Solution/Pattern
More complex test solution.

## Code Example
```python
def advanced_function():
    return {"result": "advanced"}
```
""",
            },
            {
                "filename": "entry-three.md",
                "category": "another-category",
                "content": """---
title: "Different Category Entry"
category: "another-category"
tags: ["different", "category", "test"]
difficulty: "advanced"
last_updated: "2024-01-03"
contributors: ["different-author"]
---

# Different Category Entry

## Problem/Context
This entry is in a different category.

## Solution/Pattern
Category-specific solution.
""",
            },
        ]

        # Create the test files
        for entry_data in test_entries:
            category_path = kb_path / "categories" / entry_data["category"]
            file_path = category_path / entry_data["filename"]
            file_path.write_text(entry_data["content"])

        return kb_path, test_entries

    def test_isolated_indexing_workflow(self):
        """Test complete indexing workflow with isolated data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path, expected_entries = self.create_minimal_test_kb(Path(temp_dir))

            # Create indexer and build index
            indexer = KnowledgeBaseIndexer(kb_path)
            index = indexer.build_index()

            # Verify we indexed the expected number of entries
            assert len(index.entries) == len(expected_entries)

            # Verify categories
            assert "test-category" in index.categories
            assert "another-category" in index.categories
            assert (
                len(index.categories["test-category"]) == 2
            )  # Two entries in test-category
            assert (
                len(index.categories["another-category"]) == 1
            )  # One entry in another-category

            # Verify tags
            assert "test" in index.tags
            assert "first" in index.tags
            assert "different" in index.tags

            # Verify specific entries exist
            entry_titles = [entry.title for entry in index.entries]
            assert "Test Entry One" in entry_titles
            assert "Test Entry Two" in entry_titles
            assert "Different Category Entry" in entry_titles

    def test_isolated_search_functionality(self):
        """Test search functionality with isolated data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path, expected_entries = self.create_minimal_test_kb(Path(temp_dir))

            # Build index and create searcher
            indexer = KnowledgeBaseIndexer(kb_path)
            index = indexer.build_index()
            searcher = KnowledgeBaseSearcher(index)

            # Test basic search
            results = searcher.search("test")
            assert len(results.results) >= 2  # At least two entries should match "test"

            # Test empty query returns all entries
            all_results = searcher.search("")
            assert len(all_results.results) == 3  # All three test entries

            # Test category filtering
            category_filter = SearchFilter(categories=["test-category"])
            category_results = searcher.search("test", filters=category_filter)
            assert len(category_results.results) == 2  # Only entries from test-category

            # Test tag filtering
            tag_filter = SearchFilter(tags=["first"])
            tag_results = searcher.search("", filters=tag_filter)
            assert len(tag_results.results) == 1  # Only one entry has "first" tag

            # Test difficulty filtering
            difficulty_filter = SearchFilter(difficulty=["advanced"])
            difficulty_results = searcher.search("", filters=difficulty_filter)
            assert len(difficulty_results.results) == 1  # Only one advanced entry

    def test_isolated_sorting_functionality(self):
        """Test result sorting with isolated data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path, expected_entries = self.create_minimal_test_kb(Path(temp_dir))

            # Build index and create searcher
            indexer = KnowledgeBaseIndexer(kb_path)
            index = indexer.build_index()
            searcher = KnowledgeBaseSearcher(index)

            # Get all entries and test title sorting
            results = searcher.search("", sort_by=SortBy.TITLE)
            titles = [result.entry.title for result in results.results]

            # Verify alphabetical sorting
            expected_sorted_titles = sorted(titles)
            assert titles == expected_sorted_titles

            # Test date sorting
            date_results = searcher.search("", sort_by=SortBy.DATE)
            dates = [result.entry.last_updated for result in date_results.results]

            # Should be sorted by date (most recent first)
            for i in range(len(dates) - 1):
                assert dates[i] >= dates[i + 1]

    def test_isolated_index_persistence(self):
        """Test index saving and loading with isolated data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path, expected_entries = self.create_minimal_test_kb(Path(temp_dir))

            # Build and save index
            indexer1 = KnowledgeBaseIndexer(kb_path)
            original_index = indexer1.build_index()

            # Create new indexer and load index
            indexer2 = KnowledgeBaseIndexer(kb_path)
            loaded_index = indexer2.load_index_from_file()

            # Verify loaded index matches original
            assert loaded_index is not None
            assert len(loaded_index.entries) == len(original_index.entries)
            assert loaded_index.categories.keys() == original_index.categories.keys()
            assert loaded_index.tags.keys() == original_index.tags.keys()

    def test_isolated_incremental_updates(self):
        """Test incremental index updates with isolated data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path, expected_entries = self.create_minimal_test_kb(Path(temp_dir))

            # Build initial index
            indexer = KnowledgeBaseIndexer(kb_path)
            initial_index = indexer.build_index()
            assert len(initial_index.entries) == 3

            # Add a new entry
            new_entry_content = """---
title: "New Test Entry"
category: "test-category"
tags: ["new", "test"]
difficulty: "beginner"
last_updated: "2024-01-04"
contributors: ["new-author"]
---

# New Test Entry

This is a newly added entry.
"""
            new_file = kb_path / "categories" / "test-category" / "new-entry.md"
            new_file.write_text(new_entry_content)

            # Perform incremental update
            updated_index = indexer.incremental_update()

            # Should detect the new file and rebuild
            assert updated_index is not None
            assert len(updated_index.entries) == 4  # Now has 4 entries

            # Verify new entry is indexed
            new_entry_titles = [entry.title for entry in updated_index.entries]
            assert "New Test Entry" in new_entry_titles


class TestRobustValidation:
    """Tests for validation that don't depend on external content."""

    def test_frontmatter_validation_comprehensive(self):
        """Test frontmatter validation with various scenarios."""
        from validation import validate_frontmatter

        # Valid frontmatter
        valid_fm = {
            "title": "Valid Entry",
            "category": "test-category",
            "tags": ["test", "valid"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-01",
            "contributors": ["author"],
        }
        assert len(validate_frontmatter(valid_fm)) == 0

        # Invalid frontmatter - missing fields
        invalid_fm = {"title": "Incomplete Entry"}
        errors = validate_frontmatter(invalid_fm)
        assert len(errors) > 0
        assert any("Missing required field" in error for error in errors)

        # Invalid frontmatter - wrong types
        wrong_types_fm = {
            "title": 123,  # Should be string
            "category": "test-category",
            "tags": "not-a-list",  # Should be list
            "difficulty": "invalid",  # Invalid value
            "last_updated": "invalid-date",  # Invalid format
            "contributors": [],  # Empty list
        }
        errors = validate_frontmatter(wrong_types_fm)
        assert len(errors) >= 5  # Multiple errors

    def test_content_validation_isolated(self):
        """Test content validation with controlled content."""
        from validation import validate_code_examples, validate_markdown_content

        # Complete content with all sections
        complete_content = """# Test Entry

## Problem/Context
This describes the problem in detail. We need to provide enough context
so that developers understand when and why to use this solution. The problem
typically involves specific scenarios where this approach is beneficial.

## Solution/Pattern
This describes the solution in comprehensive detail. The solution should
include step-by-step instructions and explain the reasoning behind each
decision. This helps developers understand not just what to do, but why.

## Code Example
```python
def example():
    return "Hello, World!"

# Additional example showing usage
result = example()
print(f"Result: {result}")
```

## Additional Notes
These additional notes provide extra context and considerations that
developers should keep in mind when implementing this solution.
"""
        warnings = validate_markdown_content(complete_content)
        assert len(warnings) == 0  # Should have no warnings

        # Incomplete content
        incomplete_content = "# Short Entry\n\nJust a title."
        warnings = validate_markdown_content(incomplete_content)
        assert len(warnings) > 0
        assert any("Missing recommended section" in warning for warning in warnings)

        # Valid code examples
        valid_code_content = """# Entry
```python
def valid_function():
    return True
```
"""
        errors = validate_code_examples(valid_code_content)
        assert len(errors) == 0

        # Invalid code examples
        invalid_code_content = """# Entry
```python
def invalid_function(
    # Missing closing parenthesis
```
"""
        errors = validate_code_examples(invalid_code_content)
        assert len(errors) > 0
        assert any("Python syntax error" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__])
