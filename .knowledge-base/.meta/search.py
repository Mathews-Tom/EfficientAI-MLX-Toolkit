"""
Knowledge Base Search System

This module provides comprehensive search functionality for the knowledge base,
including text search, filtering, ranking, and advanced query processing.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from indexer import KnowledgeBaseIndexer
from models import KnowledgeBaseEntry, KnowledgeBaseIndex

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search mode options."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"  # For future LLM integration


class SortBy(Enum):
    """Sort options for search results."""

    RELEVANCE = "relevance"
    DATE = "date"
    TITLE = "title"
    USAGE = "usage"
    DIFFICULTY = "difficulty"


@dataclass
class SearchFilter:
    """
    Search filter configuration.

    Attributes:
        categories: Filter by specific categories
        tags: Filter by specific tags
        difficulty: Filter by difficulty levels
        contributors: Filter by contributors
        date_range: Filter by date range (start, end)
        min_usage: Minimum usage count
        exclude_categories: Categories to exclude
        exclude_tags: Tags to exclude
    """

    categories: list[str] | None = None
    tags: list[str] | None = None
    difficulty: list[str] | None = None
    contributors: list[str] | None = None
    date_range: tuple[datetime, datetime] | None = None
    min_usage: int | None = None
    exclude_categories: list[str] | None = None
    exclude_tags: list[str] | None = None


@dataclass
class SearchResult:
    """
    Individual search result with relevance scoring.

    Attributes:
        entry: The knowledge base entry
        relevance_score: Relevance score (0.0 to 1.0)
        match_reasons: List of reasons why this entry matched
        highlighted_text: Text with search terms highlighted
    """

    entry: KnowledgeBaseEntry
    relevance_score: float
    match_reasons: list[str]
    highlighted_text: str | None = None


@dataclass
class SearchResults:
    """
    Complete search results with metadata.

    Attributes:
        results: List of individual search results
        total_matches: Total number of matches found
        query: Original search query
        filters: Applied filters
        search_time: Time taken to perform search (seconds)
        suggestions: Query suggestions for better results
    """

    results: list[SearchResult]
    total_matches: int
    query: str
    filters: SearchFilter | None
    search_time: float
    suggestions: list[str]


class KnowledgeBaseSearcher:
    """
    Comprehensive search system for knowledge base entries.

    This class provides various search capabilities including text search,
    filtering, ranking, and result highlighting.
    """

    def __init__(
        self,
        index: KnowledgeBaseIndex,
        full_text_index: dict[str, list[str]] | None = None,
    ):
        self.index = index
        self.full_text_index = full_text_index or {}

        # Search statistics
        self.search_stats = {
            "total_searches": 0,
            "average_search_time": 0.0,
            "popular_queries": {},
            "popular_filters": {},
        }

    def search(
        self,
        query: str,
        filters: SearchFilter | None = None,
        mode: SearchMode = SearchMode.FUZZY,
        sort_by: SortBy = SortBy.RELEVANCE,
        limit: int | None = None,
    ) -> SearchResults:
        """
        Perform a comprehensive search of the knowledge base.

        Args:
            query: Search query string
            filters: Optional search filters
            mode: Search mode (exact, fuzzy, semantic)
            sort_by: How to sort results
            limit: Maximum number of results to return

        Returns:
            SearchResults object with matches and metadata
        """
        import time

        start_time = time.time()

        try:
            # Update search statistics
            self.search_stats["total_searches"] += 1
            self._update_query_stats(query)

            # Get candidate entries
            candidates = self._get_filtered_candidates(filters)

            # Handle empty query - return all candidates
            if not query.strip():
                matches = [
                    SearchResult(
                        entry=entry,
                        relevance_score=1.0,
                        match_reasons=["All entries"],
                        highlighted_text=entry.title,
                    )
                    for entry in candidates
                ]
            else:
                # Perform search based on mode
                if mode == SearchMode.EXACT:
                    matches = self._exact_search(query, candidates)
                elif mode == SearchMode.FUZZY:
                    matches = self._fuzzy_search(query, candidates)
                elif mode == SearchMode.SEMANTIC:
                    matches = self._semantic_search(
                        query, candidates
                    )  # Future implementation
                else:
                    matches = self._fuzzy_search(query, candidates)  # Default

            # Sort results
            sorted_matches = self._sort_results(matches, sort_by)

            # Apply limit
            if limit:
                sorted_matches = sorted_matches[:limit]

            # Generate suggestions
            suggestions = self._generate_suggestions(query, len(matches))

            # Calculate search time
            search_time = time.time() - start_time
            self._update_search_time_stats(search_time)

            return SearchResults(
                results=sorted_matches,
                total_matches=len(matches),
                query=query,
                filters=filters,
                search_time=search_time,
                suggestions=suggestions,
            )

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return SearchResults(
                results=[],
                total_matches=0,
                query=query,
                filters=filters,
                search_time=time.time() - start_time,
                suggestions=[],
            )

    def _get_filtered_candidates(
        self, filters: SearchFilter | None
    ) -> list[KnowledgeBaseEntry]:
        """
        Get candidate entries based on filters.

        Args:
            filters: Search filters to apply

        Returns:
            List of entries that pass the filters
        """
        candidates = self.index.entries.copy()

        if not filters:
            return candidates

        # Filter by categories
        if filters.categories:
            candidates = [e for e in candidates if e.category in filters.categories]

        # Exclude categories
        if filters.exclude_categories:
            candidates = [
                e for e in candidates if e.category not in filters.exclude_categories
            ]

        # Filter by tags
        if filters.tags:
            candidates = [
                e for e in candidates if any(tag in e.tags for tag in filters.tags)
            ]

        # Exclude tags
        if filters.exclude_tags:
            candidates = [
                e
                for e in candidates
                if not any(tag in e.tags for tag in filters.exclude_tags)
            ]

        # Filter by difficulty
        if filters.difficulty:
            candidates = [e for e in candidates if e.difficulty in filters.difficulty]

        # Filter by contributors
        if filters.contributors:
            candidates = [
                e
                for e in candidates
                if any(contrib in e.contributors for contrib in filters.contributors)
            ]

        # Filter by date range
        if filters.date_range:
            start_date, end_date = filters.date_range
            candidates = [
                e for e in candidates if start_date <= e.last_updated <= end_date
            ]

        # Filter by minimum usage
        if filters.min_usage is not None:
            candidates = [e for e in candidates if e.usage_count >= filters.min_usage]

        return candidates

    def _exact_search(
        self, query: str, candidates: list[KnowledgeBaseEntry]
    ) -> list[SearchResult]:
        """
        Perform exact string matching search.

        Args:
            query: Search query
            candidates: Candidate entries to search

        Returns:
            List of search results
        """
        results = []
        query_lower = query.lower()

        for entry in candidates:
            match_reasons = []
            score = 0.0

            # Check title (highest weight)
            if query_lower in entry.title.lower():
                match_reasons.append("Title match")
                score += 1.0

            # Check tags
            for tag in entry.tags:
                if query_lower in tag.lower():
                    match_reasons.append(f"Tag match: {tag}")
                    score += 0.5

            # Check category
            if query_lower in entry.category.lower():
                match_reasons.append("Category match")
                score += 0.3

            # Check content
            try:
                content = entry.get_content().lower()
                if query_lower in content:
                    match_reasons.append("Content match")
                    score += 0.2
            except Exception:
                pass  # Skip content search if file can't be read

            if match_reasons:
                results.append(
                    SearchResult(
                        entry=entry,
                        relevance_score=min(score, 1.0),
                        match_reasons=match_reasons,
                        highlighted_text=self._highlight_text(entry.title, query),
                    )
                )

        return results

    def _fuzzy_search(
        self, query: str, candidates: list[KnowledgeBaseEntry]
    ) -> list[SearchResult]:
        """
        Perform fuzzy search with partial matching and scoring.

        Args:
            query: Search query
            candidates: Candidate entries to search

        Returns:
            List of search results
        """
        results = []
        query_terms = self._tokenize_query(query)

        for entry in candidates:
            match_reasons = []
            score = 0.0

            # Search in different fields with different weights
            title_score = self._calculate_field_score(
                entry.title, query_terms, weight=1.0
            )
            if title_score > 0:
                match_reasons.append("Title relevance")
                score += title_score

            tag_score = self._calculate_field_score(
                " ".join(entry.tags), query_terms, weight=0.8
            )
            if tag_score > 0:
                match_reasons.append("Tag relevance")
                score += tag_score

            category_score = self._calculate_field_score(
                entry.category, query_terms, weight=0.6
            )
            if category_score > 0:
                match_reasons.append("Category relevance")
                score += category_score

            # Content search (lower weight, more expensive)
            try:
                content = entry.get_content()
                content_score = self._calculate_field_score(
                    content, query_terms, weight=0.4
                )
                if content_score > 0:
                    match_reasons.append("Content relevance")
                    score += content_score
            except Exception:
                pass

            # Boost score based on usage and recency
            score = self._apply_popularity_boost(score, entry)

            # Only include results with meaningful scores
            if score > 0.1:  # Minimum relevance threshold
                results.append(
                    SearchResult(
                        entry=entry,
                        relevance_score=min(score, 1.0),
                        match_reasons=match_reasons,
                        highlighted_text=self._highlight_text(entry.title, query),
                    )
                )

        return results

    def _semantic_search(
        self, query: str, candidates: list[KnowledgeBaseEntry]
    ) -> list[SearchResult]:
        """
        Perform semantic search using embeddings (future LLM integration).

        Args:
            query: Search query
            candidates: Candidate entries to search

        Returns:
            List of search results
        """
        # Placeholder for future LLM integration
        # For now, fall back to fuzzy search
        logger.info("Semantic search not yet implemented, falling back to fuzzy search")
        return self._fuzzy_search(query, candidates)

    def _tokenize_query(self, query: str) -> list[str]:
        """
        Tokenize search query into terms.

        Args:
            query: Search query string

        Returns:
            List of normalized query terms
        """
        # Split on whitespace and punctuation, convert to lowercase
        terms = re.findall(r"\b\w+\b", query.lower())

        # Remove very short terms and common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        filtered_terms = [
            term for term in terms if len(term) >= 2 and term not in stop_words
        ]

        return filtered_terms

    def _calculate_field_score(
        self, field_text: str, query_terms: list[str], weight: float = 1.0
    ) -> float:
        """
        Calculate relevance score for a specific field.

        Args:
            field_text: Text content of the field
            query_terms: List of query terms to match
            weight: Weight multiplier for this field

        Returns:
            Relevance score for this field
        """
        if not field_text or not query_terms:
            return 0.0

        field_lower = field_text.lower()
        field_terms = re.findall(r"\b\w+\b", field_lower)

        score = 0.0

        for query_term in query_terms:
            # Exact word match
            if query_term in field_terms:
                score += 1.0
            # Partial match (substring)
            elif query_term in field_lower:
                score += 0.5
            # Fuzzy match (edit distance)
            else:
                best_fuzzy_score = max(
                    (
                        self._fuzzy_match_score(query_term, field_term)
                        for field_term in field_terms
                    ),
                    default=0.0,
                )
                score += best_fuzzy_score

        # Normalize by number of query terms and apply weight
        normalized_score = (score / len(query_terms)) * weight
        return min(normalized_score, weight)  # Cap at weight value

    def _fuzzy_match_score(self, term1: str, term2: str) -> float:
        """
        Calculate fuzzy match score between two terms.

        Args:
            term1: First term
            term2: Second term

        Returns:
            Fuzzy match score (0.0 to 1.0)
        """
        if term1 == term2:
            return 1.0

        if len(term1) < 3 or len(term2) < 3:
            return 0.0  # Too short for fuzzy matching

        # Simple fuzzy matching based on common substrings
        if term1 in term2 or term2 in term1:
            return 0.7

        # Check for common prefix/suffix
        if term1.startswith(term2[:3]) or term2.startswith(term1[:3]):
            return 0.4

        if term1.endswith(term2[-3:]) or term2.endswith(term1[-3:]):
            return 0.3

        return 0.0

    def _apply_popularity_boost(
        self, base_score: float, entry: KnowledgeBaseEntry
    ) -> float:
        """
        Apply popularity and recency boost to search score.

        Args:
            base_score: Base relevance score
            entry: Knowledge base entry

        Returns:
            Boosted score
        """
        boost = 1.0

        # Usage popularity boost (up to 20% boost)
        if entry.usage_count > 0:
            usage_boost = min(entry.usage_count / 100.0, 0.2)
            boost += usage_boost

        # Recency boost (up to 10% boost for entries updated in last 30 days)
        days_since_update = (datetime.now() - entry.last_updated).days
        if days_since_update <= 30:
            recency_boost = (30 - days_since_update) / 30.0 * 0.1
            boost += recency_boost

        return base_score * boost

    def _sort_results(
        self, results: list[SearchResult], sort_by: SortBy
    ) -> list[SearchResult]:
        """
        Sort search results by specified criteria.

        Args:
            results: List of search results
            sort_by: Sort criteria

        Returns:
            Sorted list of results
        """
        if sort_by == SortBy.RELEVANCE:
            return sorted(results, key=lambda r: r.relevance_score, reverse=True)
        elif sort_by == SortBy.DATE:
            return sorted(results, key=lambda r: r.entry.last_updated, reverse=True)
        elif sort_by == SortBy.TITLE:
            return sorted(results, key=lambda r: r.entry.title.lower())
        elif sort_by == SortBy.USAGE:
            return sorted(results, key=lambda r: r.entry.usage_count, reverse=True)
        elif sort_by == SortBy.DIFFICULTY:
            difficulty_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
            return sorted(
                results, key=lambda r: difficulty_order.get(r.entry.difficulty, 1)
            )
        else:
            return results

    def _highlight_text(self, text: str, query: str) -> str:
        """
        Highlight search terms in text.

        Args:
            text: Text to highlight
            query: Search query

        Returns:
            Text with highlighted terms
        """
        if not query:
            return text

        query_terms = self._tokenize_query(query)
        highlighted = text

        for term in query_terms:
            # Case-insensitive highlighting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"**{term}**", highlighted)

        return highlighted

    def _generate_suggestions(self, query: str, num_results: int) -> list[str]:
        """
        Generate search suggestions for better results.

        Args:
            query: Original search query
            num_results: Number of results found

        Returns:
            List of suggestions
        """
        suggestions = []

        if num_results == 0:
            suggestions.append("Try using different keywords")
            suggestions.append("Check spelling of search terms")
            suggestions.append("Use broader search terms")
            suggestions.append("Try searching by category or tags")
        elif num_results < 3:
            suggestions.append("Try using broader search terms")
            suggestions.append("Remove some filters to see more results")

        # Suggest popular tags and categories
        popular_tags = sorted(
            self.index.tags.keys(), key=lambda t: len(self.index.tags[t]), reverse=True
        )[:5]
        if popular_tags:
            suggestions.append(f"Popular tags: {', '.join(popular_tags)}")

        return suggestions

    def _update_query_stats(self, query: str) -> None:
        """Update query statistics for analytics."""
        query_lower = query.lower()
        if query_lower in self.search_stats["popular_queries"]:
            self.search_stats["popular_queries"][query_lower] += 1
        else:
            self.search_stats["popular_queries"][query_lower] = 1

    def _update_search_time_stats(self, search_time: float) -> None:
        """Update search time statistics."""
        current_avg = self.search_stats["average_search_time"]
        total_searches = self.search_stats["total_searches"]

        # Calculate new average
        new_avg = ((current_avg * (total_searches - 1)) + search_time) / total_searches
        self.search_stats["average_search_time"] = new_avg

    def get_search_stats(self) -> dict[str, Any]:
        """
        Get search statistics.

        Returns:
            Dictionary with search statistics
        """
        return {
            **self.search_stats,
            "index_size": len(self.index.entries),
            "categories_available": list(self.index.categories.keys()),
            "tags_available": list(self.index.tags.keys()),
        }

    def suggest_similar_entries(
        self, entry: KnowledgeBaseEntry, limit: int = 5
    ) -> list[SearchResult]:
        """
        Find entries similar to the given entry.

        Args:
            entry: Reference entry
            limit: Maximum number of similar entries to return

        Returns:
            List of similar entries
        """
        # Create a query from the entry's tags and category
        query_terms = entry.tags + [entry.category]
        query = " ".join(query_terms)

        # Search for similar entries
        results = self.search(
            query, limit=limit + 1
        )  # +1 to account for the entry itself

        # Remove the original entry from results
        similar_results = [r for r in results.results if r.entry.title != entry.title]

        return similar_results[:limit]


def create_searcher(
    index: KnowledgeBaseIndex, full_text_index: dict[str, list[str]] | None = None
) -> KnowledgeBaseSearcher:
    """
    Factory function to create a knowledge base searcher.

    Args:
        index: Knowledge base index
        full_text_index: Optional full-text search index

    Returns:
        Configured searcher instance
    """
    return KnowledgeBaseSearcher(index, full_text_index)


def main():
    """
    Command-line interface for the searcher.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Base Search")
    parser.add_argument("kb_path", help="Path to knowledge base directory")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--tag", help="Filter by tag")
    parser.add_argument("--difficulty", help="Filter by difficulty")
    parser.add_argument("--limit", type=int, default=10, help="Maximum results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Load index
        indexer = KnowledgeBaseIndexer(Path(args.kb_path))
        index = indexer.load_index_from_file()

        if not index:
            print("No index found. Building index...")
            index = indexer.build_index()

        # Create searcher
        searcher = create_searcher(index)

        # Create filters
        filters = SearchFilter()
        if args.category:
            filters.categories = [args.category]
        if args.tag:
            filters.tags = [args.tag]
        if args.difficulty:
            filters.difficulty = [args.difficulty]

        # Perform search
        results = searcher.search(args.query, filters=filters, limit=args.limit)

        # Display results
        print(f"\nSearch Results for '{args.query}':")
        print(f"Found {results.total_matches} matches in {results.search_time:.3f}s\n")

        for i, result in enumerate(results.results, 1):
            entry = result.entry
            print(f"{i}. {result.highlighted_text or entry.title}")
            print(f"   Category: {entry.category} | Difficulty: {entry.difficulty}")
            print(f"   Tags: {', '.join(entry.tags)}")
            print(
                f"   Score: {result.relevance_score:.2f} | Reasons: {', '.join(result.match_reasons)}"
            )
            print(f"   Updated: {entry.last_updated.strftime('%Y-%m-%d')}")
            print()

        if results.suggestions:
            print("Suggestions:")
            for suggestion in results.suggestions:
                print(f"  • {suggestion}")

    except Exception as e:
        print(f"Search failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
