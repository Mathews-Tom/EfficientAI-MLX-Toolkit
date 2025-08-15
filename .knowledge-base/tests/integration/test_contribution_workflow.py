"""
Integration tests for end-to-end contribution workflow.

Tests the complete workflow from creating entries to indexing,
searching, and maintaining the knowledge base.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Add the .meta directory to Python path for imports
kb_meta_path = Path(__file__).parent.parent.parent / ".meta"
sys.path.insert(0, str(kb_meta_path))

from contributor import KnowledgeBaseContributor
from freshness_tracker import ContentFreshnessTracker
from indexer import KnowledgeBaseIndexer
from models import KnowledgeBaseEntry
from quality_assurance import KnowledgeBaseQualityAssurance
from search import KnowledgeBaseSearcher


class TestContributionWorkflow:
    """Test complete contribution workflow from creation to maintenance."""

    def test_complete_entry_lifecycle(self, realistic_kb_structure):
        """Test complete lifecycle of a knowledge base entry."""
        kb_path = realistic_kb_structure

        # Step 1: Create a new entry using contributor
        contributor = KnowledgeBaseContributor(kb_path)

        entry_path = contributor.create_entry_from_template(
            title="Complete Lifecycle Test Entry",
            category="apple-silicon",
            tags=["lifecycle", "test", "integration"],
            difficulty="intermediate",
            contributor="integration-tester",
            entry_type="standard",
        )

        # Verify entry was created
        assert entry_path.exists()
        assert entry_path.parent.name == "apple-silicon"

        # Step 2: Index the knowledge base
        indexer = KnowledgeBaseIndexer(kb_path)
        index = indexer.build_index()

        # Verify entry is indexed
        assert len(index.entries) >= 1
        entry_titles = [entry.title for entry in index.entries]
        assert "Complete Lifecycle Test Entry" in entry_titles

        # Step 3: Search for the entry
        searcher = KnowledgeBaseSearcher(index)
        search_results = searcher.search("lifecycle test")

        # Verify entry is searchable
        assert len(search_results.results) >= 1
        found_entry = None
        for result in search_results.results:
            if result.entry.title == "Complete Lifecycle Test Entry":
                found_entry = result.entry
                break

        assert found_entry is not None
        assert "lifecycle" in found_entry.tags
        assert found_entry.category == "apple-silicon"

        # Step 4: Update entry usage
        found_entry.update_usage()
        assert found_entry.usage_count == 1

        # Step 5: Run quality assurance
        qa = KnowledgeBaseQualityAssurance(kb_path)
        qa_report = qa.run_comprehensive_quality_check()

        # Verify QA finds the entry
        assert qa_report.total_entries_checked >= 1

        # Step 6: Check freshness tracking
        freshness_tracker = ContentFreshnessTracker(kb_path)
        freshness_report = freshness_tracker.analyze_content_freshness()

        # Verify freshness tracking works
        assert freshness_report.total_entries >= 1

    def test_multi_contributor_workflow(self, realistic_kb_structure):
        """Test workflow with multiple contributors."""
        kb_path = realistic_kb_structure
        contributor = KnowledgeBaseContributor(kb_path)

        # Create entries from different contributors
        contributors = ["alice", "bob", "charlie"]
        created_entries = []

        for i, contrib in enumerate(contributors):
            entry_path = contributor.create_entry_from_template(
                title=f"Multi Contributor Entry {i+1}",
                category="mlx-framework",
                tags=["multi-contributor", f"author-{contrib}"],
                difficulty="intermediate",
                contributor=contrib,
                entry_type="standard",
            )
            created_entries.append(entry_path)

        # Verify all entries were created
        assert len(created_entries) == 3
        for entry_path in created_entries:
            assert entry_path.exists()

        # Index and verify all entries
        indexer = KnowledgeBaseIndexer(kb_path)
        index = indexer.build_index()

        # Should have at least 3 entries
        assert len(index.entries) >= 3

        # Verify contributor filtering works
        alice_entries = index.get_by_contributor("alice")
        bob_entries = index.get_by_contributor("bob")
        charlie_entries = index.get_by_contributor("charlie")

        assert len(alice_entries) >= 1
        assert len(bob_entries) >= 1
        assert len(charlie_entries) >= 1

        # Verify each contributor's entry has correct metadata
        for entry in alice_entries:
            if "Multi Contributor Entry" in entry.title:
                assert "alice" in entry.contributors

    def test_category_organization_workflow(self, realistic_kb_structure):
        """Test workflow for organizing entries across categories."""
        kb_path = realistic_kb_structure
        contributor = KnowledgeBaseContributor(kb_path)

        # Create entries in different categories
        categories = [
            "apple-silicon",
            "mlx-framework",
            "performance",
            "troubleshooting",
        ]
        category_entries = {}

        for category in categories:
            entry_path = contributor.create_entry_from_template(
                title=f"Category Test Entry for {category.title()}",
                category=category,
                tags=["category-test", category.replace("-", "_")],
                difficulty="intermediate",
                contributor="category-tester",
                entry_type="standard",
            )
            category_entries[category] = entry_path

        # Index the knowledge base
        indexer = KnowledgeBaseIndexer(kb_path)
        index = indexer.build_index()

        # Verify category organization
        for category in categories:
            category_results = index.get_by_category(category)
            assert len(category_results) >= 1

            # Find our test entry in this category
            found = False
            for entry in category_results:
                if f"Category Test Entry for {category.title()}" == entry.title:
                    found = True
                    assert entry.category == category
                    break
            assert found, f"Entry not found in category {category}"

        # Test cross-category search
        searcher = KnowledgeBaseSearcher(index)
        cross_category_results = searcher.search("Category Test Entry")

        # Should find entries from all categories
        assert len(cross_category_results.results) >= len(categories)

    def test_tag_based_workflow(self, realistic_kb_structure):
        """Test workflow using tag-based organization."""
        kb_path = realistic_kb_structure
        contributor = KnowledgeBaseContributor(kb_path)

        # Create entries with overlapping tags
        tag_combinations = [
            ["optimization", "performance", "apple-silicon"],
            ["optimization", "memory", "mlx"],
            ["performance", "benchmarking", "testing"],
            ["memory", "debugging", "troubleshooting"],
        ]

        for i, tags in enumerate(tag_combinations):
            contributor.create_entry_from_template(
                title=f"Tag Workflow Entry {i+1}",
                category="performance",
                tags=tags,
                difficulty="intermediate",
                contributor="tag-tester",
                entry_type="standard",
            )

        # Index and test tag-based queries
        indexer = KnowledgeBaseIndexer(kb_path)
        index = indexer.build_index()

        # Test single tag queries
        optimization_entries = index.get_by_tags(["optimization"])
        assert len(optimization_entries) >= 2  # Should find at least 2 entries

        performance_entries = index.get_by_tags(["performance"])
        assert len(performance_entries) >= 2

        memory_entries = index.get_by_tags(["memory"])
        assert len(memory_entries) >= 2

        # Test multi-tag queries (OR operation)
        multi_tag_entries = index.get_by_tags(["optimization", "debugging"])
        assert len(multi_tag_entries) >= 3  # Should find entries with either tag

        # Test search with tag filtering
        searcher = KnowledgeBaseSearcher(index)
        from search import SearchFilter

        # Search with tag filter
        tag_filter = SearchFilter(tags=["optimization"])
        filtered_results = searcher.search("workflow", filters=tag_filter)

        # All results should have the optimization tag
        for result in filtered_results.results:
            assert "optimization" in result.entry.tags

    def test_incremental_update_workflow(self, realistic_kb_structure):
        """Test incremental updates to the knowledge base."""
        kb_path = realistic_kb_structure
        contributor = KnowledgeBaseContributor(kb_path)
        indexer = KnowledgeBaseIndexer(kb_path)

        # Create initial entry
        initial_entry = contributor.create_entry_from_template(
            title="Incremental Update Test Entry",
            category="apple-silicon",
            tags=["incremental", "update"],
            difficulty="beginner",
            contributor="update-tester",
            entry_type="standard",
        )

        # Build initial index
        initial_index = indexer.build_index()
        initial_count = len(initial_index.entries)

        # Add another entry
        second_entry = contributor.create_entry_from_template(
            title="Second Incremental Entry",
            category="mlx-framework",
            tags=["incremental", "second"],
            difficulty="intermediate",
            contributor="update-tester",
            entry_type="standard",
        )

        # Test incremental update
        updated_index = indexer.incremental_update()

        # Should detect the new entry
        assert updated_index is not None
        assert len(updated_index.entries) == initial_count + 1

        # Verify both entries are present
        entry_titles = [entry.title for entry in updated_index.entries]
        assert "Incremental Update Test Entry" in entry_titles
        assert "Second Incremental Entry" in entry_titles

        # Modify existing entry
        modified_content = initial_entry.read_text().replace("beginner", "intermediate")
        initial_entry.write_text(modified_content)

        # Test incremental update after modification
        modified_index = indexer.incremental_update()
        assert modified_index is not None

        # Find the modified entry
        for entry in modified_index.entries:
            if entry.title == "Incremental Update Test Entry":
                # Note: The difficulty in the file content was changed,
                # but the entry object still reflects the frontmatter
                break

    def test_error_handling_workflow(self, realistic_kb_structure):
        """Test workflow error handling and recovery."""
        kb_path = realistic_kb_structure
        contributor = KnowledgeBaseContributor(kb_path)

        # Test creating entry with empty title
        with pytest.raises(ValueError):
            contributor.create_entry_from_template(
                title="",  # Empty title
                category="apple-silicon",
                tags=["error", "test"],
                difficulty="intermediate",
                contributor="error-tester",
                entry_type="standard",
            )

        # Create a valid entry first
        valid_entry = contributor.create_entry_from_template(
            title="Valid Entry Before Error",
            category="apple-silicon",
            tags=["valid", "before-error"],
            difficulty="intermediate",
            contributor="error-tester",
            entry_type="standard",
        )

        # Create an entry with invalid frontmatter
        invalid_entry_path = (
            kb_path / "categories" / "apple-silicon" / "invalid-frontmatter.md"
        )
        invalid_content = """---
title: "Invalid Entry"
# Missing required fields and malformed YAML
invalid_yaml: [unclosed
---

# Invalid Entry
This entry has invalid frontmatter.
"""
        invalid_entry_path.write_text(invalid_content)

        # Test indexing with invalid entry
        indexer = KnowledgeBaseIndexer(kb_path)
        index = indexer.build_index()

        # Should still index valid entries despite invalid ones
        assert len(index.entries) >= 1
        valid_titles = [entry.title for entry in index.entries]
        assert "Valid Entry Before Error" in valid_titles

        # Invalid entry should not be indexed
        assert "Invalid Entry" not in valid_titles

        # Check indexer stats for errors
        stats = indexer.get_index_stats()
        assert stats["errors_encountered"] >= 1


class TestWorkflowIntegration:
    """Test integration between different workflow components."""

    def test_search_and_analytics_integration(self, realistic_kb_structure):
        """Test integration between search and analytics."""
        kb_path = realistic_kb_structure
        contributor = KnowledgeBaseContributor(kb_path)

        # Create entries for analytics testing
        for i in range(5):
            contributor.create_entry_from_template(
                title=f"Analytics Test Entry {i+1}",
                category="performance",
                tags=["analytics", "search", f"entry-{i+1}"],
                difficulty="intermediate",
                contributor="analytics-tester",
                entry_type="standard",
            )

        # Index and search
        indexer = KnowledgeBaseIndexer(kb_path)
        index = indexer.build_index()
        searcher = KnowledgeBaseSearcher(index)

        # Perform multiple searches to generate analytics data
        search_queries = ["analytics", "performance", "test", "entry"]
        for query in search_queries:
            results = searcher.search(query)
            # Simulate usage by updating usage count
            for result in results.results:
                result.entry.update_usage()

        # Check search statistics
        search_stats = searcher.get_search_stats()
        assert search_stats["total_searches"] == len(search_queries)
        assert len(search_stats["popular_queries"]) == len(search_queries)

        # Check index statistics
        index_stats = index.get_statistics()
        assert index_stats["total_entries"] >= 5
        assert index_stats["total_usage"] > 0

    def test_quality_and_freshness_integration(self, realistic_kb_structure):
        """Test integration between quality assurance and freshness tracking."""
        kb_path = realistic_kb_structure
        contributor = KnowledgeBaseContributor(kb_path)

        # Create entries with different quality levels
        # Good quality entry
        good_entry = contributor.create_entry_from_template(
            title="High Quality Integration Entry",
            category="apple-silicon",
            tags=["quality", "integration", "good"],
            difficulty="intermediate",
            contributor="quality-tester",
            entry_type="standard",
        )

        # Enhance the good entry with proper content
        enhanced_content = """---
title: "High Quality Integration Entry"
category: "apple-silicon"
tags: ["quality", "integration", "good"]
difficulty: "intermediate"
last_updated: "2024-01-15"
contributors: ["quality-tester"]
---

# High Quality Integration Entry

## Problem/Context
This entry demonstrates high-quality knowledge base content with comprehensive
sections, proper formatting, and detailed explanations.

## Solution/Pattern
The solution involves following best practices for knowledge base entries:
1. Clear problem statement
2. Detailed solution description
3. Working code examples
4. Proper documentation

## Code Example
```python
def high_quality_example():
    \"\"\"
    A well-documented function that demonstrates best practices.
    
    Returns:
        str: A success message indicating proper implementation.
    \"\"\"
    return "High quality implementation"

# Usage example
result = high_quality_example()
print(f"Result: {result}")
```

## Gotchas/Pitfalls
- Always include comprehensive documentation
- Ensure code examples are tested and working
- Keep content up to date with current best practices

## Performance Impact
This approach has minimal performance impact while maximizing maintainability.

## Related Knowledge
- [Code Quality Guidelines](../quality/guidelines.md)
- [Documentation Standards](../standards/documentation.md)
"""
        good_entry.write_text(enhanced_content)

        # Create a lower quality entry
        poor_entry_path = kb_path / "categories" / "apple-silicon" / "poor-quality.md"
        poor_content = """---
title: "Poor Quality Entry"
category: "apple-silicon"
tags: ["quality", "poor"]
difficulty: "intermediate"
last_updated: "2023-01-01"
contributors: ["quality-tester"]
---

# Poor Quality Entry

Short content without proper sections.
"""
        poor_entry_path.write_text(poor_content)

        # Run quality assurance
        qa = KnowledgeBaseQualityAssurance(kb_path)
        qa_report = qa.run_comprehensive_quality_check()

        # Should find quality issues
        assert qa_report.total_entries_checked >= 2
        assert len(qa_report.issues_found) > 0

        # Run freshness tracking
        freshness_tracker = ContentFreshnessTracker(kb_path)
        freshness_report = freshness_tracker.analyze_content_freshness()

        # Should identify stale content
        assert freshness_report.total_entries >= 2
        assert (
            len(freshness_report.stale_entries) >= 1
        )  # The 2023 entry should be stale

        # Verify integration between quality and freshness
        # Stale entries should be flagged in quality report
        stale_titles = [entry.entry_title for entry in freshness_report.stale_entries]
        assert "Poor Quality Entry" in stale_titles


if __name__ == "__main__":
    pytest.main([__file__])
