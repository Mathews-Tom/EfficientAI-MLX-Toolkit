"""
Unit tests for Knowledge Base indexing system.

Tests the KnowledgeBaseIndexer class including file scanning,
index building, and incremental updates.
"""

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

# Add the .meta directory to Python path for imports
kb_meta_path = Path(__file__).parent.parent / ".meta"
sys.path.insert(0, str(kb_meta_path))

from indexer import IndexingError, KnowledgeBaseIndexer, create_indexer
from models import KnowledgeBaseEntry


class TestKnowledgeBaseIndexer:
    """Test cases for KnowledgeBaseIndexer class."""

    def create_test_kb_structure(self, temp_dir: Path) -> Path:
        """Create a test knowledge base directory structure."""
        kb_path = temp_dir / ".knowledge-base"

        # Create directory structure
        (kb_path / "categories" / "test-category").mkdir(parents=True)
        (kb_path / "patterns" / "test-pattern").mkdir(parents=True)
        (kb_path / ".meta").mkdir(parents=True)

        # Create test entry files
        entry1_content = """---
title: "Test Entry 1"
category: "test-category"
tags: ["test", "example"]
difficulty: "intermediate"
last_updated: "2024-01-15"
contributors: ["test-user"]
---

# Test Entry 1

This is a test entry for indexing.
"""

        entry2_content = """---
title: "Test Entry 2"
category: "test-category"
tags: ["test", "advanced"]
difficulty: "advanced"
last_updated: "2024-01-16"
contributors: ["test-user", "another-user"]
---

# Test Entry 2

This is another test entry.
"""

        (kb_path / "categories" / "test-category" / "test-entry-1.md").write_text(
            entry1_content
        )
        (kb_path / "categories" / "test-category" / "test-entry-2.md").write_text(
            entry2_content
        )

        # Create README files (should be ignored)
        (kb_path / "categories" / "test-category" / "README.md").write_text(
            "# Test Category"
        )

        return kb_path

    def test_indexer_creation(self):
        """Test creating indexer instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()

            indexer = KnowledgeBaseIndexer(kb_path)

            assert indexer.kb_path == kb_path
            assert indexer.watch_files is True
            assert indexer.parallel_processing is True

    def test_indexer_creation_with_options(self):
        """Test creating indexer with custom options."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()
            index_file = kb_path / "custom_index.json"

            indexer = KnowledgeBaseIndexer(
                kb_path,
                index_file=index_file,
                watch_files=False,
                parallel_processing=False,
            )

            assert indexer.index_file == index_file
            assert indexer.watch_files is False
            assert indexer.parallel_processing is False

    def test_find_entry_files(self):
        """Test finding entry files in knowledge base."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path)

            entry_files = indexer._find_entry_files()

            # Should find 2 entry files (excluding README.md)
            assert len(entry_files) == 2

            # Check that README files are excluded
            readme_files = [f for f in entry_files if f.name == "README.md"]
            assert len(readme_files) == 0

    def test_process_single_file_valid(self):
        """Test processing a valid entry file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path)

            entry_file = kb_path / "categories" / "test-category" / "test-entry-1.md"
            entry = indexer._process_single_file(entry_file)

            assert entry is not None
            assert entry.title == "Test Entry 1"
            assert entry.category == "test-category"
            assert "test" in entry.tags

    def test_process_single_file_invalid(self):
        """Test processing an invalid entry file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()
            indexer = KnowledgeBaseIndexer(kb_path)

            # Create invalid file (no frontmatter)
            invalid_file = kb_path / "invalid.md"
            invalid_file.write_text("# Invalid Entry\n\nNo frontmatter here.")

            entry = indexer._process_single_file(invalid_file)

            assert entry is None
            assert indexer.stats["errors_encountered"] > 0

    def test_build_index_success(self):
        """Test successful index building."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path)

            index = indexer.build_index()

            assert len(index.entries) == 2
            assert "test-category" in index.categories
            assert "test" in index.tags
            assert "example" in index.tags
            assert "advanced" in index.tags

            # Check statistics
            assert indexer.stats["entries_indexed"] == 2
            assert indexer.stats["total_files_scanned"] == 2
            assert indexer.stats["last_index_time"] is not None

    def test_build_index_empty_directory(self):
        """Test building index from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()
            (kb_path / "categories").mkdir()
            (kb_path / "patterns").mkdir()

            indexer = KnowledgeBaseIndexer(kb_path)
            index = indexer.build_index()

            assert len(index.entries) == 0
            assert len(index.categories) == 0
            assert len(index.tags) == 0

    def test_build_full_text_index(self):
        """Test building full-text search index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path)

            index = indexer.build_index()
            entries = index.entries

            full_text_index = indexer._build_full_text_index(entries)

            # Should contain words from titles, tags, categories
            assert "test" in full_text_index
            assert "entry" in full_text_index
            assert "indexing" in full_text_index  # From content

    def test_tokenize_text(self):
        """Test text tokenization for search."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()
            indexer = KnowledgeBaseIndexer(kb_path)

            text = "This is a test with some WORDS and numbers123!"
            tokens = indexer._tokenize_text(text)

            assert "test" in tokens
            assert "words" in tokens
            assert "numbers123" in tokens
            # Stop words should be filtered out
            # Note: "this" is not in the actual stop words list, so we test actual stop words
            assert "is" not in tokens
            assert "a" not in tokens
            assert "and" not in tokens

    def test_extract_searchable_text(self):
        """Test extracting searchable text from entry."""
        entry = KnowledgeBaseEntry(
            title="Test Entry",
            category="test-category",
            tags=["test", "example"],
            difficulty="intermediate",
            content_path=Path("test.md"),
            last_updated=datetime.now(),
            contributors=["test-user"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()
            indexer = KnowledgeBaseIndexer(kb_path)

            searchable_text = indexer._extract_searchable_text(entry)

            assert "Test Entry" in searchable_text
            assert "test-category" in searchable_text
            assert "test example" in searchable_text
            assert "test-user" in searchable_text

    def test_save_and_load_index(self):
        """Test saving and loading index from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path)

            # Build and save index
            original_index = indexer.build_index()

            # Create new indexer and load index
            new_indexer = KnowledgeBaseIndexer(kb_path)
            loaded_index = new_indexer.load_index_from_file()

            assert loaded_index is not None
            assert len(loaded_index.entries) == len(original_index.entries)
            assert loaded_index.categories.keys() == original_index.categories.keys()
            assert loaded_index.tags.keys() == original_index.tags.keys()

    def test_load_index_no_file(self):
        """Test loading index when no file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()
            indexer = KnowledgeBaseIndexer(kb_path)

            loaded_index = indexer.load_index_from_file()

            assert loaded_index is None

    def test_load_index_corrupted_file(self):
        """Test loading index from corrupted file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()
            (kb_path / ".meta").mkdir()

            # Create corrupted index file
            index_file = kb_path / ".meta" / "index.json"
            index_file.write_text("invalid json content")

            indexer = KnowledgeBaseIndexer(kb_path)
            loaded_index = indexer.load_index_from_file()

            assert loaded_index is None

    def test_file_change_detection(self):
        """Test file change detection for incremental updates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path)

            entry_file = kb_path / "categories" / "test-category" / "test-entry-1.md"

            # First check - should be considered changed (new file)
            assert indexer._has_file_changed(entry_file) is True

            # Second check - should not be changed
            assert indexer._has_file_changed(entry_file) is False

            # Modify file
            content = entry_file.read_text()
            entry_file.write_text(content + "\n# Additional content")

            # Should be detected as changed
            assert indexer._has_file_changed(entry_file) is True

    def test_incremental_update_no_changes(self):
        """Test incremental update when no files changed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path)

            # Build initial index
            indexer.build_index()

            # Try incremental update
            updated_index = indexer.incremental_update()

            # Should return None (no changes)
            assert updated_index is None

    def test_incremental_update_with_changes(self):
        """Test incremental update when files changed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path)

            # Build initial index
            indexer.build_index()

            # Modify a file
            entry_file = kb_path / "categories" / "test-category" / "test-entry-1.md"
            content = entry_file.read_text()
            entry_file.write_text(content.replace("Test Entry 1", "Modified Entry 1"))

            # Try incremental update
            updated_index = indexer.incremental_update()

            # Should return updated index
            assert updated_index is not None
            assert len(updated_index.entries) == 2

    def test_get_index_stats(self):
        """Test getting indexing statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path)

            # Build index
            indexer.build_index()

            stats = indexer.get_index_stats()

            assert "total_files_scanned" in stats
            assert "entries_indexed" in stats
            assert "errors_encountered" in stats
            assert "last_index_time" in stats
            assert "index_build_duration" in stats
            assert "index_file_exists" in stats
            assert "kb_path" in stats

            assert stats["entries_indexed"] == 2
            assert stats["total_files_scanned"] == 2

    def test_force_rebuild(self):
        """Test forcing a complete rebuild."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path)

            # Build initial index
            indexer.build_index()
            initial_build_time = indexer.stats["last_index_time"]

            # Force rebuild
            time.sleep(0.1)  # Ensure different timestamp
            indexer.build_index(force_rebuild=True)
            rebuild_time = indexer.stats["last_index_time"]

            # Should have different build times
            assert rebuild_time != initial_build_time

    def test_parallel_processing_disabled(self):
        """Test indexing with parallel processing disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir))
            indexer = KnowledgeBaseIndexer(kb_path, parallel_processing=False)

            index = indexer.build_index()

            # Should still work correctly
            assert len(index.entries) == 2
            assert indexer.stats["entries_indexed"] == 2

    def test_indexing_error_handling(self):
        """Test error handling during indexing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create non-existent path
            kb_path = Path(temp_dir) / "nonexistent" / ".knowledge-base"
            indexer = KnowledgeBaseIndexer(kb_path)

            # Should handle gracefully
            index = indexer.build_index()
            assert len(index.entries) == 0

    def test_create_indexer_factory(self):
        """Test indexer factory function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()

            indexer = create_indexer(kb_path, watch_files=False)

            assert isinstance(indexer, KnowledgeBaseIndexer)
            assert indexer.kb_path == kb_path
            assert indexer.watch_files is False


class TestIndexingIntegration:
    """Integration tests for the indexing system."""

    def test_end_to_end_indexing(self):
        """Test complete end-to-end indexing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create comprehensive test structure
            kb_path = Path(temp_dir) / ".knowledge-base"

            # Multiple categories
            (kb_path / "categories" / "apple-silicon").mkdir(parents=True)
            (kb_path / "categories" / "mlx-framework").mkdir(parents=True)
            (kb_path / "patterns" / "training-patterns").mkdir(parents=True)
            (kb_path / ".meta").mkdir(parents=True)

            # Create various entry types
            entries_data = [
                {
                    "path": "categories/apple-silicon/memory-optimization.md",
                    "content": """---
title: "MLX Memory Optimization"
category: "apple-silicon"
tags: ["mlx", "memory", "optimization"]
difficulty: "advanced"
last_updated: "2024-01-15"
contributors: ["alice", "bob"]
---

# MLX Memory Optimization

Techniques for optimizing memory usage in MLX applications.

## Problem/Context
Memory management is crucial for Apple Silicon performance.

## Solution/Pattern
Use gradient checkpointing and mixed precision.

```python
import mlx.core as mx

def optimize_memory():
    mx.metal.set_memory_limit(8 * 1024**3)
```
""",
                },
                {
                    "path": "categories/mlx-framework/training-loops.md",
                    "content": """---
title: "MLX Training Loops"
category: "mlx-framework"
tags: ["mlx", "training", "loops"]
difficulty: "intermediate"
last_updated: "2024-01-16"
contributors: ["charlie"]
---

# MLX Training Loops

Best practices for implementing training loops in MLX.
""",
                },
                {
                    "path": "patterns/training-patterns/batch-processing.md",
                    "content": """---
title: "Batch Processing Patterns"
category: "training-patterns"
tags: ["batch", "processing", "performance"]
difficulty: "beginner"
last_updated: "2024-01-17"
contributors: ["alice"]
---

# Batch Processing Patterns

Efficient batch processing techniques.
""",
                },
            ]

            # Create entry files
            for entry_data in entries_data:
                file_path = kb_path / entry_data["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(entry_data["content"])

            # Build index
            indexer = KnowledgeBaseIndexer(kb_path)
            index = indexer.build_index()

            # Verify comprehensive indexing
            assert len(index.entries) == 3
            assert len(index.categories) == 3
            assert "apple-silicon" in index.categories
            assert "mlx-framework" in index.categories
            assert "training-patterns" in index.categories

            # Verify tag indexing
            assert "mlx" in index.tags
            assert "memory" in index.tags
            assert "optimization" in index.tags
            assert "training" in index.tags

            # Verify search functionality
            memory_entries = index.search("memory")
            assert len(memory_entries) >= 1  # At least one memory-related entry
            memory_titles = [e.title for e in memory_entries]
            assert "MLX Memory Optimization" in memory_titles

            mlx_entries = index.search("mlx")
            assert len(mlx_entries) >= 2  # At least two MLX-related entries

            # Verify category filtering
            apple_silicon_entries = index.get_by_category("apple-silicon")
            assert len(apple_silicon_entries) == 1

            # Verify tag filtering
            training_entries = index.get_by_tags(["training"])
            assert (
                len(training_entries) == 1
            )  # Only MLX Training Loops has "training" tag

            # Verify contributor filtering
            alice_entries = index.get_by_contributor("alice")
            assert len(alice_entries) == 2

            # Verify difficulty filtering
            advanced_entries = index.get_by_difficulty("advanced")
            assert len(advanced_entries) == 1

            # Verify statistics
            stats = index.get_statistics()
            assert stats["total_entries"] == 3
            assert stats["total_categories"] == 3
            assert stats["total_tags"] >= 6  # At least 6 unique tags

            # Verify index persistence
            new_indexer = KnowledgeBaseIndexer(kb_path)
            loaded_index = new_indexer.load_index_from_file()
            assert loaded_index is not None
            assert len(loaded_index.entries) == 3


if __name__ == "__main__":
    pytest.main([__file__])
