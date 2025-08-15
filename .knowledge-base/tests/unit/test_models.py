"""
Unit tests for Knowledge Base data models.

Tests the KnowledgeBaseEntry and KnowledgeBaseIndex classes including
validation, serialization, and core functionality.
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

# Add the .meta directory to Python path for imports
kb_meta_path = Path(__file__).parent.parent / ".meta"
sys.path.insert(0, str(kb_meta_path))

from models import KnowledgeBaseEntry, KnowledgeBaseIndex, validate_entry_frontmatter


class TestKnowledgeBaseEntry:
    """Test cases for KnowledgeBaseEntry class."""

    def test_entry_creation_valid(self):
        """Test creating a valid knowledge base entry."""
        entry = KnowledgeBaseEntry(
            title="Test Entry",
            category="test-category",
            tags=["test", "example"],
            difficulty="intermediate",
            content_path=Path("test.md"),
            last_updated=datetime(2024, 1, 15),
            contributors=["test-user"],
        )

        assert entry.title == "Test Entry"
        assert entry.category == "test-category"
        assert entry.tags == ["test", "example"]
        assert entry.difficulty == "intermediate"
        assert entry.usage_count == 0
        assert len(entry.contributors) == 1

    def test_entry_validation_invalid_difficulty(self):
        """Test validation fails for invalid difficulty."""
        with pytest.raises(ValueError, match="Invalid difficulty"):
            KnowledgeBaseEntry(
                title="Test Entry",
                category="test-category",
                tags=["test"],
                difficulty="invalid",  # Invalid difficulty
                content_path=Path("test.md"),
                last_updated=datetime.now(),
                contributors=["test-user"],
            )

    def test_entry_validation_invalid_category(self):
        """Test validation fails for invalid category format."""
        with pytest.raises(ValueError, match="Invalid category format"):
            KnowledgeBaseEntry(
                title="Test Entry",
                category="Test_Category",  # Invalid format (should be lowercase with hyphens)
                tags=["test"],
                difficulty="intermediate",
                content_path=Path("test.md"),
                last_updated=datetime.now(),
                contributors=["test-user"],
            )

    def test_entry_validation_empty_title(self):
        """Test validation fails for empty title."""
        with pytest.raises(ValueError, match="Title cannot be empty"):
            KnowledgeBaseEntry(
                title="   ",  # Empty title
                category="test-category",
                tags=["test"],
                difficulty="intermediate",
                content_path=Path("test.md"),
                last_updated=datetime.now(),
                contributors=["test-user"],
            )

    def test_entry_validation_empty_contributors(self):
        """Test validation fails for empty contributors list."""
        with pytest.raises(ValueError, match="At least one contributor"):
            KnowledgeBaseEntry(
                title="Test Entry",
                category="test-category",
                tags=["test"],
                difficulty="intermediate",
                content_path=Path("test.md"),
                last_updated=datetime.now(),
                contributors=[],  # Empty contributors
            )

    def test_entry_validation_invalid_tags(self):
        """Test validation fails for invalid tags."""
        with pytest.raises(ValueError, match="Invalid tag"):
            KnowledgeBaseEntry(
                title="Test Entry",
                category="test-category",
                tags=["test", ""],  # Empty tag
                difficulty="intermediate",
                content_path=Path("test.md"),
                last_updated=datetime.now(),
                contributors=["test-user"],
            )

    def test_update_usage(self):
        """Test usage counter increment."""
        entry = KnowledgeBaseEntry(
            title="Test Entry",
            category="test-category",
            tags=["test"],
            difficulty="intermediate",
            content_path=Path("test.md"),
            last_updated=datetime.now(),
            contributors=["test-user"],
        )

        assert entry.usage_count == 0
        entry.update_usage()
        assert entry.usage_count == 1
        entry.update_usage()
        assert entry.usage_count == 2

    def test_is_stale(self):
        """Test stale entry detection."""
        # Fresh entry (today)
        fresh_entry = KnowledgeBaseEntry(
            title="Fresh Entry",
            category="test-category",
            tags=["test"],
            difficulty="intermediate",
            content_path=Path("test.md"),
            last_updated=datetime.now(),
            contributors=["test-user"],
        )
        assert not fresh_entry.is_stale(30)

        # Stale entry (100 days ago)
        stale_date = datetime.now() - timedelta(days=100)
        stale_entry = KnowledgeBaseEntry(
            title="Stale Entry",
            category="test-category",
            tags=["test"],
            difficulty="intermediate",
            content_path=Path("test.md"),
            last_updated=stale_date,
            contributors=["test-user"],
        )
        assert stale_entry.is_stale(90)

    def test_matches_search(self):
        """Test search matching functionality."""
        entry = KnowledgeBaseEntry(
            title="MLX Memory Optimization",
            category="mlx-framework",
            tags=["mlx", "memory", "performance"],
            difficulty="intermediate",
            content_path=Path("test.md"),
            last_updated=datetime.now(),
            contributors=["test-user"],
        )

        # Should match title
        assert entry.matches_search("memory")
        assert entry.matches_search("MLX")

        # Should match tags
        assert entry.matches_search("performance")

        # Should match category
        assert entry.matches_search("mlx-framework")

        # Should not match unrelated terms
        assert not entry.matches_search("unrelated")

    def test_get_frontmatter(self):
        """Test frontmatter generation."""
        entry = KnowledgeBaseEntry(
            title="Test Entry",
            category="test-category",
            tags=["test", "example"],
            difficulty="intermediate",
            content_path=Path("test.md"),
            last_updated=datetime(2024, 1, 15),
            contributors=["test-user"],
        )

        frontmatter = entry.get_frontmatter()

        assert frontmatter["title"] == "Test Entry"
        assert frontmatter["category"] == "test-category"
        assert frontmatter["tags"] == ["test", "example"]
        assert frontmatter["difficulty"] == "intermediate"
        assert frontmatter["last_updated"] == "2024-01-15"
        assert frontmatter["contributors"] == ["test-user"]

    def test_from_file_valid(self):
        """Test creating entry from valid markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            content = """---
title: "Test Entry"
category: "test-category"
tags: ["test", "example"]
difficulty: "intermediate"
last_updated: "2024-01-15"
contributors: ["test-user"]
---

# Test Entry

This is test content.
"""
            f.write(content)
            f.flush()

            entry = KnowledgeBaseEntry.from_file(Path(f.name))

            assert entry.title == "Test Entry"
            assert entry.category == "test-category"
            assert entry.tags == ["test", "example"]
            assert entry.difficulty == "intermediate"
            assert entry.contributors == ["test-user"]

            # Cleanup
            Path(f.name).unlink()

    def test_from_file_missing_frontmatter(self):
        """Test error handling for file without frontmatter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            content = "# Test Entry\n\nThis has no frontmatter."
            f.write(content)
            f.flush()

            with pytest.raises(ValueError, match="missing frontmatter"):
                KnowledgeBaseEntry.from_file(Path(f.name))

            # Cleanup
            Path(f.name).unlink()

    def test_from_file_invalid_yaml(self):
        """Test error handling for invalid YAML frontmatter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            content = """---
title: "Test Entry"
invalid_yaml: [unclosed list
---

# Test Entry
"""
            f.write(content)
            f.flush()

            with pytest.raises(ValueError, match="Invalid YAML"):
                KnowledgeBaseEntry.from_file(Path(f.name))

            # Cleanup
            Path(f.name).unlink()

    def test_from_file_missing_required_fields(self):
        """Test error handling for missing required frontmatter fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            content = """---
title: "Test Entry"
# Missing required fields
---

# Test Entry
"""
            f.write(content)
            f.flush()

            with pytest.raises(ValueError, match="Missing required frontmatter field"):
                KnowledgeBaseEntry.from_file(Path(f.name))

            # Cleanup
            Path(f.name).unlink()

    def test_save_to_file(self):
        """Test saving entry to markdown file."""
        entry = KnowledgeBaseEntry(
            title="Test Entry",
            category="test-category",
            tags=["test", "example"],
            difficulty="intermediate",
            content_path=Path("test.md"),
            last_updated=datetime(2024, 1, 15),
            contributors=["test-user"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_entry.md"
            entry.save_to_file(file_path)

            assert file_path.exists()

            # Verify content
            content = file_path.read_text()
            assert "title: Test Entry" in content
            assert "category: test-category" in content
            assert "# Test Entry" in content

    def test_get_content_file_not_found(self):
        """Test error handling when content file doesn't exist."""
        entry = KnowledgeBaseEntry(
            title="Test Entry",
            category="test-category",
            tags=["test"],
            difficulty="intermediate",
            content_path=Path("nonexistent.md"),
            last_updated=datetime.now(),
            contributors=["test-user"],
        )

        with pytest.raises(FileNotFoundError):
            entry.get_content()


class TestKnowledgeBaseIndex:
    """Test cases for KnowledgeBaseIndex class."""

    def create_test_entry(
        self, title: str, category: str = "test", tags: list[str] = None
    ) -> KnowledgeBaseEntry:
        """Helper method to create test entries."""
        return KnowledgeBaseEntry(
            title=title,
            category=category,
            tags=tags or ["test"],
            difficulty="intermediate",
            content_path=Path(f"{title.lower().replace(' ', '_')}.md"),
            last_updated=datetime.now(),
            contributors=["test-user"],
        )

    def test_index_creation_empty(self):
        """Test creating empty index."""
        index = KnowledgeBaseIndex()

        assert len(index.entries) == 0
        assert len(index.categories) == 0
        assert len(index.tags) == 0

    def test_add_entry(self):
        """Test adding entries to index."""
        index = KnowledgeBaseIndex()
        entry = self.create_test_entry(
            "Test Entry", "test-category", ["test", "example"]
        )

        index.add_entry(entry)

        assert len(index.entries) == 1
        assert "test-category" in index.categories
        assert "test" in index.tags
        assert "example" in index.tags
        assert index.get_entry("Test Entry") == entry

    def test_add_duplicate_entry(self):
        """Test error when adding duplicate entry."""
        index = KnowledgeBaseIndex()
        entry1 = self.create_test_entry("Test Entry")
        entry2 = self.create_test_entry("Test Entry")  # Same title

        index.add_entry(entry1)

        with pytest.raises(ValueError, match="already exists"):
            index.add_entry(entry2)

    def test_remove_entry(self):
        """Test removing entries from index."""
        index = KnowledgeBaseIndex()
        entry = self.create_test_entry("Test Entry", "test-category", ["test"])

        index.add_entry(entry)
        assert len(index.entries) == 1

        # Remove existing entry
        result = index.remove_entry("Test Entry")
        assert result is True
        assert len(index.entries) == 0
        assert index.get_entry("Test Entry") is None

        # Try to remove non-existent entry
        result = index.remove_entry("Non-existent")
        assert result is False

    def test_get_by_category(self):
        """Test filtering entries by category."""
        index = KnowledgeBaseIndex()

        entry1 = self.create_test_entry("Entry 1", "category-a")
        entry2 = self.create_test_entry("Entry 2", "category-b")
        entry3 = self.create_test_entry("Entry 3", "category-a")

        index.add_entry(entry1)
        index.add_entry(entry2)
        index.add_entry(entry3)

        category_a_entries = index.get_by_category("category-a")
        assert len(category_a_entries) == 2
        assert entry1 in category_a_entries
        assert entry3 in category_a_entries

        category_b_entries = index.get_by_category("category-b")
        assert len(category_b_entries) == 1
        assert entry2 in category_b_entries

        # Non-existent category
        empty_entries = index.get_by_category("non-existent")
        assert len(empty_entries) == 0

    def test_get_by_tags(self):
        """Test filtering entries by tags."""
        index = KnowledgeBaseIndex()

        entry1 = self.create_test_entry("Entry 1", tags=["python", "mlx"])
        entry2 = self.create_test_entry("Entry 2", tags=["javascript", "web"])
        entry3 = self.create_test_entry("Entry 3", tags=["python", "web"])

        index.add_entry(entry1)
        index.add_entry(entry2)
        index.add_entry(entry3)

        python_entries = index.get_by_tags(["python"])
        assert len(python_entries) == 2
        assert entry1 in python_entries
        assert entry3 in python_entries

        web_entries = index.get_by_tags(["web"])
        assert len(web_entries) == 2
        assert entry2 in web_entries
        assert entry3 in web_entries

        # Multiple tags (OR operation)
        multi_tag_entries = index.get_by_tags(["mlx", "javascript"])
        assert len(multi_tag_entries) == 2
        assert entry1 in multi_tag_entries
        assert entry2 in multi_tag_entries

    def test_get_by_difficulty(self):
        """Test filtering entries by difficulty."""
        index = KnowledgeBaseIndex()

        entry1 = KnowledgeBaseEntry(
            title="Beginner Entry",
            category="test",
            tags=["test"],
            difficulty="beginner",
            content_path=Path("test1.md"),
            last_updated=datetime.now(),
            contributors=["test-user"],
        )

        entry2 = KnowledgeBaseEntry(
            title="Advanced Entry",
            category="test",
            tags=["test"],
            difficulty="advanced",
            content_path=Path("test2.md"),
            last_updated=datetime.now(),
            contributors=["test-user"],
        )

        index.add_entry(entry1)
        index.add_entry(entry2)

        beginner_entries = index.get_by_difficulty("beginner")
        assert len(beginner_entries) == 1
        assert entry1 in beginner_entries

        advanced_entries = index.get_by_difficulty("advanced")
        assert len(advanced_entries) == 1
        assert entry2 in advanced_entries

    def test_get_by_contributor(self):
        """Test filtering entries by contributor."""
        index = KnowledgeBaseIndex()

        entry1 = KnowledgeBaseEntry(
            title="Entry 1",
            category="test",
            tags=["test"],
            difficulty="intermediate",
            content_path=Path("test1.md"),
            last_updated=datetime.now(),
            contributors=["alice", "bob"],
        )

        entry2 = KnowledgeBaseEntry(
            title="Entry 2",
            category="test",
            tags=["test"],
            difficulty="intermediate",
            content_path=Path("test2.md"),
            last_updated=datetime.now(),
            contributors=["bob", "charlie"],
        )

        index.add_entry(entry1)
        index.add_entry(entry2)

        alice_entries = index.get_by_contributor("alice")
        assert len(alice_entries) == 1
        assert entry1 in alice_entries

        bob_entries = index.get_by_contributor("bob")
        assert len(bob_entries) == 2
        assert entry1 in bob_entries
        assert entry2 in bob_entries

    def test_search_basic(self):
        """Test basic search functionality."""
        index = KnowledgeBaseIndex()

        entry1 = self.create_test_entry(
            "MLX Memory Optimization", "mlx-framework", ["mlx", "memory"]
        )
        entry2 = self.create_test_entry(
            "Python Performance Tips", "performance", ["python", "optimization"]
        )
        entry3 = self.create_test_entry(
            "Memory Management Guide", "general", ["memory", "guide"]
        )

        index.add_entry(entry1)
        index.add_entry(entry2)
        index.add_entry(entry3)

        # Search for "memory"
        memory_results = index.search("memory")
        assert len(memory_results) == 2
        assert entry1 in memory_results
        assert entry3 in memory_results

        # Search for "MLX"
        mlx_results = index.search("MLX")
        assert len(mlx_results) == 1
        assert entry1 in mlx_results

        # Search with category filter
        mlx_category_results = index.search("memory", category="mlx-framework")
        assert len(mlx_category_results) == 1
        assert entry1 in mlx_category_results

    def test_get_stale_entries(self):
        """Test getting stale entries."""
        index = KnowledgeBaseIndex()

        # Fresh entry
        fresh_entry = KnowledgeBaseEntry(
            title="Fresh Entry",
            category="test",
            tags=["test"],
            difficulty="intermediate",
            content_path=Path("fresh.md"),
            last_updated=datetime.now(),
            contributors=["test-user"],
        )

        # Stale entry
        stale_entry = KnowledgeBaseEntry(
            title="Stale Entry",
            category="test",
            tags=["test"],
            difficulty="intermediate",
            content_path=Path("stale.md"),
            last_updated=datetime.now() - timedelta(days=100),
            contributors=["test-user"],
        )

        index.add_entry(fresh_entry)
        index.add_entry(stale_entry)

        stale_entries = index.get_stale_entries(90)
        assert len(stale_entries) == 1
        assert stale_entry in stale_entries

    def test_get_popular_entries(self):
        """Test getting popular entries by usage."""
        index = KnowledgeBaseIndex()

        entry1 = self.create_test_entry("Entry 1")
        entry1.usage_count = 10

        entry2 = self.create_test_entry("Entry 2")
        entry2.usage_count = 5

        entry3 = self.create_test_entry("Entry 3")
        entry3.usage_count = 15

        index.add_entry(entry1)
        index.add_entry(entry2)
        index.add_entry(entry3)

        popular_entries = index.get_popular_entries(2)
        assert len(popular_entries) == 2
        assert popular_entries[0] == entry3  # Highest usage
        assert popular_entries[1] == entry1  # Second highest

    def test_get_statistics(self):
        """Test getting index statistics."""
        index = KnowledgeBaseIndex()

        entry1 = self.create_test_entry("Entry 1", "category-a", ["tag1", "tag2"])
        entry1.usage_count = 5

        entry2 = self.create_test_entry("Entry 2", "category-b", ["tag2", "tag3"])
        entry2.usage_count = 3

        index.add_entry(entry1)
        index.add_entry(entry2)

        stats = index.get_statistics()

        assert stats["total_entries"] == 2
        assert stats["total_categories"] == 2
        assert stats["total_tags"] == 3
        assert stats["total_usage"] == 8
        assert "category-a" in stats["category_distribution"]
        assert "category-b" in stats["category_distribution"]
        assert stats["difficulty_distribution"]["intermediate"] == 2


class TestValidateFrontmatter:
    """Test cases for frontmatter validation function."""

    def test_valid_frontmatter(self):
        """Test validation of valid frontmatter."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test", "example"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_entry_frontmatter(frontmatter)
        assert len(errors) == 0

    def test_missing_required_fields(self):
        """Test validation fails for missing required fields."""
        frontmatter = {
            "title": "Test Entry",
            # Missing other required fields
        }

        errors = validate_entry_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("Missing required field" in error for error in errors)

    def test_invalid_difficulty(self):
        """Test validation fails for invalid difficulty."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "invalid",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_entry_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("Invalid difficulty" in error for error in errors)

    def test_invalid_category_format(self):
        """Test validation fails for invalid category format."""
        frontmatter = {
            "title": "Test Entry",
            "category": "Test_Category",  # Invalid format
            "tags": ["test"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_entry_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("lowercase with hyphens" in error for error in errors)

    def test_invalid_tags_format(self):
        """Test validation fails for invalid tags format."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": "not-a-list",  # Should be a list
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_entry_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("Tags must be a list" in error for error in errors)

    def test_empty_tags(self):
        """Test validation fails for empty tags list."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": [],  # Empty list
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_entry_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("At least one tag is required" in error for error in errors)

    def test_invalid_contributors_format(self):
        """Test validation fails for invalid contributors format."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": "not-a-list",  # Should be a list
        }

        errors = validate_entry_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("Contributors must be a list" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__])
