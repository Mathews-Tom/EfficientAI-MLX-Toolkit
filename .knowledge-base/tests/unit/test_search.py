"""
Unit tests for Knowledge Base search system.

Tests the KnowledgeBaseSearcher class including text search,
filtering, ranking, and result processing.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add the .meta directory to Python path for imports
kb_meta_path = Path(__file__).parent.parent / ".meta"
sys.path.insert(0, str(kb_meta_path))

from models import KnowledgeBaseEntry, KnowledgeBaseIndex
from search import (
    KnowledgeBaseSearcher,
    SearchFilter,
    SearchMode,
    SearchResult,
    SortBy,
    create_searcher,
)


class TestSearchFilter:
    """Test cases for SearchFilter class."""

    def test_filter_creation_empty(self):
        """Test creating empty search filter."""
        filter_obj = SearchFilter()

        assert filter_obj.categories is None
        assert filter_obj.tags is None
        assert filter_obj.difficulty is None
        assert filter_obj.contributors is None
        assert filter_obj.date_range is None
        assert filter_obj.min_usage is None
        assert filter_obj.exclude_categories is None
        assert filter_obj.exclude_tags is None

    def test_filter_creation_with_values(self):
        """Test creating search filter with values."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

        filter_obj = SearchFilter(
            categories=["apple-silicon", "mlx-framework"],
            tags=["optimization", "performance"],
            difficulty=["intermediate", "advanced"],
            contributors=["alice", "bob"],
            date_range=(start_date, end_date),
            min_usage=5,
            exclude_categories=["deprecated"],
            exclude_tags=["old"],
        )

        assert filter_obj.categories == ["apple-silicon", "mlx-framework"]
        assert filter_obj.tags == ["optimization", "performance"]
        assert filter_obj.difficulty == ["intermediate", "advanced"]
        assert filter_obj.contributors == ["alice", "bob"]
        assert filter_obj.date_range == (start_date, end_date)
        assert filter_obj.min_usage == 5
        assert filter_obj.exclude_categories == ["deprecated"]
        assert filter_obj.exclude_tags == ["old"]


class TestSearchResult:
    """Test cases for SearchResult class."""

    def test_search_result_creation(self):
        """Test creating search result."""
        entry = KnowledgeBaseEntry(
            title="Test Entry",
            category="test",
            tags=["test"],
            difficulty="intermediate",
            content_path=Path("test.md"),
            last_updated=datetime.now(),
            contributors=["test-user"],
        )

        result = SearchResult(
            entry=entry,
            relevance_score=0.85,
            match_reasons=["Title match", "Tag match"],
            highlighted_text="**Test** Entry",
        )

        assert result.entry == entry
        assert result.relevance_score == 0.85
        assert result.match_reasons == ["Title match", "Tag match"]
        assert result.highlighted_text == "**Test** Entry"


class TestKnowledgeBaseSearcher:
    """Test cases for KnowledgeBaseSearcher class."""

    def create_test_index(self) -> KnowledgeBaseIndex:
        """Create a test index with sample entries."""
        entries = [
            KnowledgeBaseEntry(
                title="MLX Memory Optimization",
                category="apple-silicon",
                tags=["mlx", "memory", "optimization"],
                difficulty="advanced",
                content_path=Path("mlx-memory.md"),
                last_updated=datetime(2024, 1, 15),
                contributors=["alice", "bob"],
                usage_count=10,
            ),
            KnowledgeBaseEntry(
                title="Python Performance Tips",
                category="performance",
                tags=["python", "optimization", "tips"],
                difficulty="intermediate",
                content_path=Path("python-perf.md"),
                last_updated=datetime(2024, 1, 10),
                contributors=["charlie"],
                usage_count=5,
            ),
            KnowledgeBaseEntry(
                title="Memory Management Guide",
                category="general",
                tags=["memory", "management", "guide"],
                difficulty="beginner",
                content_path=Path("memory-guide.md"),
                last_updated=datetime(2024, 1, 20),
                contributors=["alice"],
                usage_count=15,
            ),
            KnowledgeBaseEntry(
                title="MLX Training Loops",
                category="mlx-framework",
                tags=["mlx", "training", "loops"],
                difficulty="intermediate",
                content_path=Path("mlx-training.md"),
                last_updated=datetime(2024, 1, 5),
                contributors=["bob", "charlie"],
                usage_count=8,
            ),
        ]

        return KnowledgeBaseIndex(entries=entries)

    def test_searcher_creation(self):
        """Test creating searcher instance."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        assert searcher.index == index
        assert searcher.full_text_index == {}
        assert searcher.search_stats["total_searches"] == 0

    def test_searcher_creation_with_full_text_index(self):
        """Test creating searcher with full-text index."""
        index = self.create_test_index()
        full_text_index = {
            "memory": ["MLX Memory Optimization", "Memory Management Guide"]
        }

        searcher = KnowledgeBaseSearcher(index, full_text_index)

        assert searcher.full_text_index == full_text_index

    def test_get_filtered_candidates_no_filters(self):
        """Test getting candidates without filters."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        candidates = searcher._get_filtered_candidates(None)

        assert len(candidates) == 4
        assert candidates == index.entries

    def test_get_filtered_candidates_by_category(self):
        """Test filtering candidates by category."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        filter_obj = SearchFilter(categories=["apple-silicon", "performance"])
        candidates = searcher._get_filtered_candidates(filter_obj)

        assert len(candidates) == 2
        categories = {entry.category for entry in candidates}
        assert categories == {"apple-silicon", "performance"}

    def test_get_filtered_candidates_exclude_categories(self):
        """Test filtering candidates by excluding categories."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        filter_obj = SearchFilter(exclude_categories=["general", "performance"])
        candidates = searcher._get_filtered_candidates(filter_obj)

        assert len(candidates) == 2
        categories = {entry.category for entry in candidates}
        assert "general" not in categories
        assert "performance" not in categories

    def test_get_filtered_candidates_by_tags(self):
        """Test filtering candidates by tags."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        filter_obj = SearchFilter(tags=["mlx", "python"])
        candidates = searcher._get_filtered_candidates(filter_obj)

        assert len(candidates) == 3  # 2 MLX entries + 1 Python entry

        # Check that all candidates have at least one of the specified tags
        for candidate in candidates:
            assert any(tag in candidate.tags for tag in ["mlx", "python"])

    def test_get_filtered_candidates_exclude_tags(self):
        """Test filtering candidates by excluding tags."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        filter_obj = SearchFilter(exclude_tags=["mlx"])
        candidates = searcher._get_filtered_candidates(filter_obj)

        assert len(candidates) == 2  # Should exclude 2 MLX entries

        # Check that no candidates have excluded tags
        for candidate in candidates:
            assert "mlx" not in candidate.tags

    def test_get_filtered_candidates_by_difficulty(self):
        """Test filtering candidates by difficulty."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        filter_obj = SearchFilter(difficulty=["intermediate", "advanced"])
        candidates = searcher._get_filtered_candidates(filter_obj)

        assert len(candidates) == 3  # Should exclude beginner entry
        difficulties = {entry.difficulty for entry in candidates}
        assert "beginner" not in difficulties

    def test_get_filtered_candidates_by_contributors(self):
        """Test filtering candidates by contributors."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        filter_obj = SearchFilter(contributors=["alice"])
        candidates = searcher._get_filtered_candidates(filter_obj)

        assert len(candidates) == 2  # Alice contributed to 2 entries

        # Check that all candidates include Alice as contributor
        for candidate in candidates:
            assert "alice" in candidate.contributors

    def test_get_filtered_candidates_by_date_range(self):
        """Test filtering candidates by date range."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        start_date = datetime(2024, 1, 12)
        end_date = datetime(2024, 1, 25)
        filter_obj = SearchFilter(date_range=(start_date, end_date))
        candidates = searcher._get_filtered_candidates(filter_obj)

        assert len(candidates) == 2  # 2 entries in date range

        # Check that all candidates are in date range
        for candidate in candidates:
            assert start_date <= candidate.last_updated <= end_date

    def test_get_filtered_candidates_by_min_usage(self):
        """Test filtering candidates by minimum usage."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        filter_obj = SearchFilter(min_usage=8)
        candidates = searcher._get_filtered_candidates(filter_obj)

        assert len(candidates) == 3  # 3 entries with usage >= 8

        # Check that all candidates meet minimum usage
        for candidate in candidates:
            assert candidate.usage_count >= 8

    def test_exact_search(self):
        """Test exact string matching search."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        results = searcher._exact_search("memory", index.entries)

        assert len(results) == 2  # 2 entries contain "memory"

        # Check that results have proper scoring and reasons
        for result in results:
            assert result.relevance_score > 0
            assert len(result.match_reasons) > 0
            assert isinstance(result.highlighted_text, str)

    def test_exact_search_case_insensitive(self):
        """Test exact search is case insensitive."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        results_lower = searcher._exact_search("mlx", index.entries)
        results_upper = searcher._exact_search("MLX", index.entries)

        assert len(results_lower) == len(results_upper)
        assert len(results_lower) == 2  # 2 MLX entries

    def test_fuzzy_search(self):
        """Test fuzzy search with partial matching."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        results = searcher._fuzzy_search("optimization", index.entries)

        assert len(results) >= 2  # At least 2 entries with optimization

        # Results should be sorted by relevance
        for i in range(len(results) - 1):
            assert results[i].relevance_score >= results[i + 1].relevance_score

    def test_tokenize_query(self):
        """Test query tokenization."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        query = "MLX memory optimization and performance"
        tokens = searcher._tokenize_query(query)

        assert "mlx" in tokens
        assert "memory" in tokens
        assert "optimization" in tokens
        assert "performance" in tokens
        # Stop words should be filtered out
        assert "and" not in tokens

    def test_calculate_field_score(self):
        """Test field relevance scoring."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        field_text = "MLX memory optimization techniques"
        query_terms = ["mlx", "memory"]

        score = searcher._calculate_field_score(field_text, query_terms, weight=1.0)

        assert score > 0
        assert score <= 1.0  # Should not exceed weight

    def test_fuzzy_match_score(self):
        """Test fuzzy matching between terms."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        # Exact match
        assert searcher._fuzzy_match_score("memory", "memory") == 1.0

        # Substring match
        assert searcher._fuzzy_match_score("mem", "memory") > 0

        # No match
        assert searcher._fuzzy_match_score("xyz", "memory") == 0.0

        # Too short
        assert searcher._fuzzy_match_score("ab", "memory") == 0.0

    def test_apply_popularity_boost(self):
        """Test popularity and recency boost."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        # Entry with high usage
        high_usage_entry = index.entries[2]  # Memory Management Guide, usage=15
        boosted_score = searcher._apply_popularity_boost(0.5, high_usage_entry)
        assert boosted_score > 0.5

        # Entry with recent update
        recent_entry = KnowledgeBaseEntry(
            title="Recent Entry",
            category="test",
            tags=["test"],
            difficulty="intermediate",
            content_path=Path("recent.md"),
            last_updated=datetime.now() - timedelta(days=5),  # 5 days ago
            contributors=["test-user"],
            usage_count=0,
        )
        recent_boosted_score = searcher._apply_popularity_boost(0.5, recent_entry)
        assert recent_boosted_score > 0.5

    def test_sort_results_by_relevance(self):
        """Test sorting results by relevance."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        # Create test results with different scores
        results = [
            SearchResult(index.entries[0], 0.3, ["match"]),
            SearchResult(index.entries[1], 0.8, ["match"]),
            SearchResult(index.entries[2], 0.5, ["match"]),
        ]

        sorted_results = searcher._sort_results(results, SortBy.RELEVANCE)

        # Should be sorted by relevance (descending)
        assert sorted_results[0].relevance_score == 0.8
        assert sorted_results[1].relevance_score == 0.5
        assert sorted_results[2].relevance_score == 0.3

    def test_sort_results_by_date(self):
        """Test sorting results by date."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        results = [
            SearchResult(index.entries[0], 0.5, ["match"]),  # 2024-01-15
            SearchResult(index.entries[1], 0.5, ["match"]),  # 2024-01-10
            SearchResult(index.entries[2], 0.5, ["match"]),  # 2024-01-20
        ]

        sorted_results = searcher._sort_results(results, SortBy.DATE)

        # Should be sorted by date (most recent first)
        assert sorted_results[0].entry.last_updated == datetime(2024, 1, 20)
        assert sorted_results[1].entry.last_updated == datetime(2024, 1, 15)
        assert sorted_results[2].entry.last_updated == datetime(2024, 1, 10)

    def test_sort_results_by_title(self):
        """Test sorting results by title."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        results = [
            SearchResult(index.entries[0], 0.5, ["match"]),  # MLX Memory Optimization
            SearchResult(index.entries[2], 0.5, ["match"]),  # Memory Management Guide
            SearchResult(index.entries[3], 0.5, ["match"]),  # MLX Training Loops
        ]

        sorted_results = searcher._sort_results(results, SortBy.TITLE)

        # Should be sorted alphabetically (case-insensitive)
        titles = [result.entry.title for result in sorted_results]
        expected_sorted = sorted(titles, key=str.lower)
        assert titles == expected_sorted

    def test_sort_results_by_usage(self):
        """Test sorting results by usage count."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        results = [
            SearchResult(index.entries[0], 0.5, ["match"]),  # usage=10
            SearchResult(index.entries[1], 0.5, ["match"]),  # usage=5
            SearchResult(index.entries[2], 0.5, ["match"]),  # usage=15
        ]

        sorted_results = searcher._sort_results(results, SortBy.USAGE)

        # Should be sorted by usage (descending)
        assert sorted_results[0].entry.usage_count == 15
        assert sorted_results[1].entry.usage_count == 10
        assert sorted_results[2].entry.usage_count == 5

    def test_sort_results_by_difficulty(self):
        """Test sorting results by difficulty."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        results = [
            SearchResult(index.entries[0], 0.5, ["match"]),  # advanced
            SearchResult(index.entries[1], 0.5, ["match"]),  # intermediate
            SearchResult(index.entries[2], 0.5, ["match"]),  # beginner
        ]

        sorted_results = searcher._sort_results(results, SortBy.DIFFICULTY)

        # Should be sorted by difficulty (beginner -> intermediate -> advanced)
        difficulties = [result.entry.difficulty for result in sorted_results]
        assert difficulties == ["beginner", "intermediate", "advanced"]

    def test_highlight_text(self):
        """Test text highlighting."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        text = "MLX Memory Optimization Guide"
        query = "memory optimization"

        highlighted = searcher._highlight_text(text, query)

        assert "**memory**" in highlighted.lower()
        assert "**optimization**" in highlighted.lower()

    def test_generate_suggestions_no_results(self):
        """Test generating suggestions when no results found."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        suggestions = searcher._generate_suggestions("nonexistent query", 0)

        assert len(suggestions) > 0
        assert any("different keywords" in suggestion for suggestion in suggestions)

    def test_generate_suggestions_few_results(self):
        """Test generating suggestions when few results found."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        suggestions = searcher._generate_suggestions("specific query", 2)

        assert len(suggestions) > 0
        assert any("broader" in suggestion for suggestion in suggestions)

    def test_search_full_workflow(self):
        """Test complete search workflow."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        # Search for memory-related entries
        results = searcher.search(
            query="memory optimization",
            mode=SearchMode.FUZZY,
            sort_by=SortBy.RELEVANCE,
            limit=5,
        )

        assert isinstance(results.results, list)
        assert results.total_matches >= 0
        assert results.query == "memory optimization"
        assert results.search_time > 0
        assert isinstance(results.suggestions, list)

        # Should find relevant entries
        assert len(results.results) >= 1

        # Results should be properly formatted
        for result in results.results:
            assert isinstance(result, SearchResult)
            assert result.relevance_score >= 0
            assert result.relevance_score <= 1.0
            assert isinstance(result.match_reasons, list)

    def test_search_with_filters(self):
        """Test search with filters applied."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        # Search with category filter
        filter_obj = SearchFilter(categories=["apple-silicon"])
        results = searcher.search(
            query="memory", filters=filter_obj, mode=SearchMode.FUZZY
        )

        # Should only return results from apple-silicon category
        for result in results.results:
            assert result.entry.category == "apple-silicon"

    def test_search_empty_query(self):
        """Test search with empty query."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        results = searcher.search("")

        # Should return all entries when query is empty
        assert len(results.results) == len(index.entries)
        # Verify we get the expected number of test entries
        assert len(results.results) == 4  # Our test index has 4 entries

    def test_suggest_similar_entries(self):
        """Test suggesting similar entries."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        # Get similar entries to MLX Memory Optimization
        reference_entry = index.entries[0]  # MLX Memory Optimization
        similar_results = searcher.suggest_similar_entries(reference_entry, limit=2)

        assert len(similar_results) <= 2

        # Should not include the reference entry itself
        for result in similar_results:
            assert result.entry.title != reference_entry.title

    def test_get_search_stats(self):
        """Test getting search statistics."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        # Perform some searches to generate stats
        searcher.search("memory")
        searcher.search("mlx")

        stats = searcher.get_search_stats()

        assert "total_searches" in stats
        assert "average_search_time" in stats
        assert "popular_queries" in stats
        assert "index_size" in stats
        assert "categories_available" in stats
        assert "tags_available" in stats

        assert stats["total_searches"] == 2
        assert stats["index_size"] == 4

    def test_update_search_stats(self):
        """Test updating search statistics."""
        index = self.create_test_index()
        searcher = KnowledgeBaseSearcher(index)

        # Initial state
        assert searcher.search_stats["total_searches"] == 0

        # Update query stats
        searcher._update_query_stats("test query")
        assert "test query" in searcher.search_stats["popular_queries"]
        assert searcher.search_stats["popular_queries"]["test query"] == 1

        # Update again
        searcher._update_query_stats("test query")
        assert searcher.search_stats["popular_queries"]["test query"] == 2

        # Update search time stats
        searcher.search_stats["total_searches"] = 1
        searcher._update_search_time_stats(0.5)
        assert searcher.search_stats["average_search_time"] == 0.5

        searcher.search_stats["total_searches"] = 2
        searcher._update_search_time_stats(1.0)
        assert searcher.search_stats["average_search_time"] == 0.75

    def test_create_searcher_factory(self):
        """Test searcher factory function."""
        index = self.create_test_index()
        full_text_index = {"test": ["entry1", "entry2"]}

        searcher = create_searcher(index, full_text_index)

        assert isinstance(searcher, KnowledgeBaseSearcher)
        assert searcher.index == index
        assert searcher.full_text_index == full_text_index


class TestSearchIntegration:
    """Integration tests for the search system."""

    def test_search_integration_with_indexer(self):
        """Test search integration with indexer."""
        # This would typically use the indexer to build a real index
        # For now, we'll create a comprehensive test scenario

        entries = [
            KnowledgeBaseEntry(
                title="Advanced MLX Memory Optimization Techniques",
                category="apple-silicon",
                tags=["mlx", "memory", "optimization", "advanced"],
                difficulty="advanced",
                content_path=Path("advanced-mlx-memory.md"),
                last_updated=datetime(2024, 1, 15),
                contributors=["alice", "bob"],
                usage_count=25,
            ),
            KnowledgeBaseEntry(
                title="Basic Python Memory Management",
                category="python-basics",
                tags=["python", "memory", "basics"],
                difficulty="beginner",
                content_path=Path("python-memory-basics.md"),
                last_updated=datetime(2024, 1, 10),
                contributors=["charlie"],
                usage_count=50,
            ),
            KnowledgeBaseEntry(
                title="MLX Training Performance Optimization",
                category="mlx-framework",
                tags=["mlx", "training", "performance", "optimization"],
                difficulty="intermediate",
                content_path=Path("mlx-training-perf.md"),
                last_updated=datetime(2024, 1, 20),
                contributors=["alice", "david"],
                usage_count=15,
            ),
            KnowledgeBaseEntry(
                title="Memory Profiling Tools and Techniques",
                category="debugging",
                tags=["memory", "profiling", "debugging", "tools"],
                difficulty="intermediate",
                content_path=Path("memory-profiling.md"),
                last_updated=datetime(2024, 1, 12),
                contributors=["bob", "charlie"],
                usage_count=30,
            ),
        ]

        index = KnowledgeBaseIndex(entries=entries)
        searcher = KnowledgeBaseSearcher(index)

        # Test comprehensive search scenarios

        # 1. Basic keyword search
        memory_results = searcher.search("memory optimization")
        assert len(memory_results.results) >= 2

        # 2. Category-filtered search
        mlx_filter = SearchFilter(categories=["apple-silicon", "mlx-framework"])
        mlx_results = searcher.search("optimization", filters=mlx_filter)
        for result in mlx_results.results:
            assert result.entry.category in ["apple-silicon", "mlx-framework"]

        # 3. Tag-based search
        tag_filter = SearchFilter(tags=["advanced", "debugging"])
        advanced_results = searcher.search("memory", filters=tag_filter)
        for result in advanced_results.results:
            assert any(tag in result.entry.tags for tag in ["advanced", "debugging"])

        # 4. Difficulty and usage filtering
        popular_filter = SearchFilter(
            difficulty=["intermediate", "advanced"], min_usage=20
        )
        popular_results = searcher.search("memory", filters=popular_filter)
        for result in popular_results.results:
            assert result.entry.difficulty in ["intermediate", "advanced"]
            assert result.entry.usage_count >= 20

        # 5. Date range filtering
        recent_filter = SearchFilter(
            date_range=(datetime(2024, 1, 15), datetime(2024, 1, 25))
        )
        recent_results = searcher.search("optimization", filters=recent_filter)
        for result in recent_results.results:
            assert (
                datetime(2024, 1, 15)
                <= result.entry.last_updated
                <= datetime(2024, 1, 25)
            )

        # 6. Contributor filtering
        alice_filter = SearchFilter(contributors=["alice"])
        alice_results = searcher.search("mlx", filters=alice_filter)
        for result in alice_results.results:
            assert "alice" in result.entry.contributors

        # 7. Exclusion filtering
        exclude_filter = SearchFilter(
            exclude_categories=["python-basics"], exclude_tags=["basics"]
        )
        advanced_only_results = searcher.search("memory", filters=exclude_filter)
        for result in advanced_only_results.results:
            assert result.entry.category != "python-basics"
            assert "basics" not in result.entry.tags

        # 8. Different search modes
        exact_results = searcher.search("MLX", mode=SearchMode.EXACT)
        fuzzy_results = searcher.search("MLX", mode=SearchMode.FUZZY)
        # Fuzzy should generally return more or equal results
        assert len(fuzzy_results.results) >= len(exact_results.results)

        # 9. Different sorting options
        relevance_results = searcher.search("memory", sort_by=SortBy.RELEVANCE)
        usage_results = searcher.search("memory", sort_by=SortBy.USAGE)
        date_results = searcher.search("memory", sort_by=SortBy.DATE)

        # Check that sorting actually affects order
        if len(usage_results.results) > 1:
            usage_counts = [r.entry.usage_count for r in usage_results.results]
            assert usage_counts == sorted(usage_counts, reverse=True)

        # 10. Limit functionality
        limited_results = searcher.search("memory", limit=2)
        assert len(limited_results.results) <= 2

        # 11. Similar entry suggestions
        reference_entry = entries[0]  # Advanced MLX Memory Optimization
        similar_results = searcher.suggest_similar_entries(reference_entry, limit=3)
        assert len(similar_results) <= 3
        for result in similar_results:
            assert result.entry.title != reference_entry.title


if __name__ == "__main__":
    pytest.main([__file__])
