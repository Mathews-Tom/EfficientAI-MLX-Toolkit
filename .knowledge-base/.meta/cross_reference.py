"""
Knowledge Base Cross-Reference System

This module provides intelligent cross-referencing capabilities for knowledge base
entries, including automatic relationship detection, link validation, and
contextual recommendations.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indexer import KnowledgeBaseIndexer
from models import KnowledgeBaseEntry, KnowledgeBaseIndex

logger = logging.getLogger(__name__)


@dataclass
class CrossReference:
    """
    Represents a cross-reference between knowledge base entries.

    Attributes:
        source_entry: The entry that references another
        target_entry: The entry being referenced
        reference_type: Type of reference (explicit, implicit, related)
        strength: Strength of the relationship (0.0 to 1.0)
        context: Context where the reference was found
        bidirectional: Whether the reference works both ways
    """

    source_entry: KnowledgeBaseEntry
    target_entry: KnowledgeBaseEntry
    reference_type: str  # "explicit", "implicit", "related", "similar"
    strength: float
    context: str
    bidirectional: bool = False


@dataclass
class RelationshipGraph:
    """
    Graph representation of relationships between entries.

    Attributes:
        entries: All entries in the graph
        references: All cross-references
        adjacency: Adjacency list representation
        clusters: Groups of related entries
    """

    entries: list[KnowledgeBaseEntry]
    references: list[CrossReference]
    adjacency: dict[str, list[str]]  # entry_title -> [related_entry_titles]
    clusters: list[list[str]]  # Groups of related entry titles


class CrossReferenceAnalyzer:
    """
    Analyzes and manages cross-references between knowledge base entries.

    This class identifies relationships between entries, validates links,
    and provides recommendations for related content.
    """

    def __init__(self, index: KnowledgeBaseIndex):
        self.index = index
        self.references: list[CrossReference] = []
        self.relationship_graph: RelationshipGraph | None = None

        # Configuration
        self.similarity_threshold = 0.3
        self.tag_weight = 0.4
        self.category_weight = 0.3
        self.content_weight = 0.3

        # Statistics
        self.stats = {
            "explicit_references": 0,
            "implicit_references": 0,
            "broken_references": 0,
            "clusters_found": 0,
            "recommendations_generated": 0,
        }

    def analyze_all_references(self) -> RelationshipGraph:
        """
        Analyze all cross-references in the knowledge base.

        Returns:
            RelationshipGraph with all relationships
        """
        logger.info("Analyzing cross-references across knowledge base")

        self.references = []

        # Find explicit references (markdown links)
        self._find_explicit_references()

        # Find implicit references (content similarity)
        self._find_implicit_references()

        # Find related entries (tag/category similarity)
        self._find_related_entries()

        # Build relationship graph
        self.relationship_graph = self._build_relationship_graph()

        # Find clusters of related entries
        self._find_entry_clusters()

        logger.info(f"Found {len(self.references)} cross-references")
        return self.relationship_graph

    def _find_explicit_references(self) -> None:
        """Find explicit markdown links between entries."""
        for entry in self.index.entries:
            try:
                content = entry.get_content()

                # Find markdown links
                links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)

                for link_text, link_url in links:
                    # Check if it's an internal knowledge base link
                    if self._is_internal_link(link_url):
                        target_entry = self._resolve_internal_link(link_url, entry)

                        if target_entry:
                            ref = CrossReference(
                                source_entry=entry,
                                target_entry=target_entry,
                                reference_type="explicit",
                                strength=1.0,
                                context=f"Link: {link_text}",
                                bidirectional=False,
                            )
                            self.references.append(ref)
                            self.stats["explicit_references"] += 1
                        else:
                            self.stats["broken_references"] += 1
                            logger.warning(f"Broken link in {entry.title}: {link_url}")

            except Exception as e:
                logger.error(
                    f"Error analyzing explicit references for {entry.title}: {e}"
                )

    def _find_implicit_references(self) -> None:
        """Find implicit references based on content similarity."""
        entries = self.index.entries

        for i, entry1 in enumerate(entries):
            for entry2 in entries[i + 1 :]:
                similarity = self._calculate_content_similarity(entry1, entry2)

                if similarity > self.similarity_threshold:
                    # Create bidirectional implicit reference
                    ref1 = CrossReference(
                        source_entry=entry1,
                        target_entry=entry2,
                        reference_type="implicit",
                        strength=similarity,
                        context="Content similarity",
                        bidirectional=True,
                    )

                    ref2 = CrossReference(
                        source_entry=entry2,
                        target_entry=entry1,
                        reference_type="implicit",
                        strength=similarity,
                        context="Content similarity",
                        bidirectional=True,
                    )

                    self.references.extend([ref1, ref2])
                    self.stats["implicit_references"] += 2

    def _find_related_entries(self) -> None:
        """Find related entries based on tags and categories."""
        entries = self.index.entries

        for i, entry1 in enumerate(entries):
            for entry2 in entries[i + 1 :]:
                if entry1.title == entry2.title:
                    continue

                relatedness = self._calculate_relatedness(entry1, entry2)

                if relatedness > 0.2:  # Lower threshold for related entries
                    ref1 = CrossReference(
                        source_entry=entry1,
                        target_entry=entry2,
                        reference_type="related",
                        strength=relatedness,
                        context=self._get_relatedness_context(entry1, entry2),
                        bidirectional=True,
                    )

                    ref2 = CrossReference(
                        source_entry=entry2,
                        target_entry=entry1,
                        reference_type="related",
                        strength=relatedness,
                        context=self._get_relatedness_context(entry2, entry1),
                        bidirectional=True,
                    )

                    self.references.extend([ref1, ref2])

    def _is_internal_link(self, link_url: str) -> bool:
        """Check if a link is internal to the knowledge base."""
        # Internal links are relative paths or start with ../
        return (
            not link_url.startswith("http")
            and not link_url.startswith("#")
            and (".md" in link_url or link_url.startswith("../"))
        )

    def _resolve_internal_link(
        self, link_url: str, source_entry: KnowledgeBaseEntry
    ) -> KnowledgeBaseEntry | None:
        """Resolve an internal link to a target entry."""
        try:
            # Handle relative paths
            if link_url.startswith("../"):
                # Resolve relative to source entry location
                source_dir = source_entry.content_path.parent
                target_path = (source_dir / link_url).resolve()
            else:
                # Assume it's relative to knowledge base root
                target_path = (
                    self.index._title_to_entry[
                        list(self.index._title_to_entry.keys())[0]
                    ].content_path.parent.parent
                    / link_url
                ).resolve()

            # Find entry with matching path
            for entry in self.index.entries:
                if entry.content_path.resolve() == target_path:
                    return entry

            return None

        except Exception as e:
            logger.debug(f"Could not resolve link {link_url}: {e}")
            return None

    def _calculate_content_similarity(
        self, entry1: KnowledgeBaseEntry, entry2: KnowledgeBaseEntry
    ) -> float:
        """Calculate content similarity between two entries."""
        try:
            content1 = self._extract_searchable_content(entry1)
            content2 = self._extract_searchable_content(entry2)

            # Simple word-based similarity
            words1 = set(self._tokenize_content(content1))
            words2 = set(self._tokenize_content(content2))

            if not words1 or not words2:
                return 0.0

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.debug(f"Error calculating content similarity: {e}")
            return 0.0

    def _calculate_relatedness(
        self, entry1: KnowledgeBaseEntry, entry2: KnowledgeBaseEntry
    ) -> float:
        """Calculate overall relatedness between two entries."""
        # Tag similarity
        tags1 = set(entry1.tags)
        tags2 = set(entry2.tags)
        tag_similarity = len(tags1 & tags2) / len(tags1 | tags2) if tags1 | tags2 else 0

        # Category similarity
        category_similarity = 1.0 if entry1.category == entry2.category else 0.0

        # Content similarity (lighter weight)
        content_similarity = self._calculate_content_similarity(entry1, entry2)

        # Weighted combination
        relatedness = (
            tag_similarity * self.tag_weight
            + category_similarity * self.category_weight
            + content_similarity * self.content_weight
        )

        return relatedness

    def _get_relatedness_context(
        self, entry1: KnowledgeBaseEntry, entry2: KnowledgeBaseEntry
    ) -> str:
        """Get context explaining why entries are related."""
        contexts = []

        # Check tag overlap
        common_tags = set(entry1.tags) & set(entry2.tags)
        if common_tags:
            contexts.append(f"Common tags: {', '.join(common_tags)}")

        # Check category
        if entry1.category == entry2.category:
            contexts.append(f"Same category: {entry1.category}")

        # Check difficulty
        if entry1.difficulty == entry2.difficulty:
            contexts.append(f"Same difficulty: {entry1.difficulty}")

        return "; ".join(contexts) if contexts else "General similarity"

    def _extract_searchable_content(self, entry: KnowledgeBaseEntry) -> str:
        """Extract searchable content from an entry."""
        try:
            content = entry.get_content()

            # Remove frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = parts[2]

            # Remove markdown formatting
            content = re.sub(
                r"```.*?```", "", content, flags=re.DOTALL
            )  # Remove code blocks
            content = re.sub(r"`[^`]+`", "", content)  # Remove inline code
            content = re.sub(r"[#*_\[\]()]", "", content)  # Remove markdown chars

            return content

        except Exception:
            return ""

    def _tokenize_content(self, content: str) -> list[str]:
        """Tokenize content into words."""
        words = re.findall(r"\b\w+\b", content.lower())

        # Filter out common stop words and short words
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
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "this",
            "that",
            "these",
            "those",
        }

        filtered_words = [
            word for word in words if len(word) >= 3 and word not in stop_words
        ]

        return filtered_words

    def _build_relationship_graph(self) -> RelationshipGraph:
        """Build a relationship graph from cross-references."""
        adjacency = defaultdict(list)

        for ref in self.references:
            source_title = ref.source_entry.title
            target_title = ref.target_entry.title

            if target_title not in adjacency[source_title]:
                adjacency[source_title].append(target_title)

        return RelationshipGraph(
            entries=self.index.entries,
            references=self.references,
            adjacency=dict(adjacency),
            clusters=[],  # Will be populated by _find_entry_clusters
        )

    def _find_entry_clusters(self) -> None:
        """Find clusters of highly related entries."""
        if not self.relationship_graph:
            return

        # Simple clustering based on strong relationships
        visited = set()
        clusters = []

        for entry in self.index.entries:
            if entry.title in visited:
                continue

            cluster = self._find_cluster_from_entry(entry.title, visited)
            if len(cluster) > 1:  # Only keep clusters with multiple entries
                clusters.append(cluster)

        self.relationship_graph.clusters = clusters
        self.stats["clusters_found"] = len(clusters)

    def _find_cluster_from_entry(
        self, entry_title: str, visited: set[str]
    ) -> list[str]:
        """Find a cluster starting from a specific entry."""
        cluster = []
        to_visit = [entry_title]

        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue

            visited.add(current)
            cluster.append(current)

            # Find strongly related entries
            for ref in self.references:
                if (
                    ref.source_entry.title == current
                    and ref.strength > 0.5  # Strong relationship threshold
                    and ref.target_entry.title not in visited
                ):
                    to_visit.append(ref.target_entry.title)

        return cluster

    def get_related_entries(
        self, entry: KnowledgeBaseEntry, limit: int = 5, min_strength: float = 0.2
    ) -> list[tuple[KnowledgeBaseEntry, float, str]]:
        """
        Get entries related to the given entry.

        Args:
            entry: Source entry
            limit: Maximum number of related entries to return
            min_strength: Minimum relationship strength

        Returns:
            List of (related_entry, strength, context) tuples
        """
        related = []

        for ref in self.references:
            if ref.source_entry.title == entry.title and ref.strength >= min_strength:
                related.append((ref.target_entry, ref.strength, ref.context))

        # Sort by strength and limit
        related.sort(key=lambda x: x[1], reverse=True)
        self.stats["recommendations_generated"] += len(related[:limit])

        return related[:limit]

    def get_entry_cluster(self, entry: KnowledgeBaseEntry) -> list[str]:
        """
        Get the cluster that contains the given entry.

        Args:
            entry: Entry to find cluster for

        Returns:
            List of entry titles in the same cluster
        """
        if not self.relationship_graph:
            return []

        for cluster in self.relationship_graph.clusters:
            if entry.title in cluster:
                return cluster

        return []

    def validate_references(self) -> list[tuple[KnowledgeBaseEntry, str]]:
        """
        Validate all references and find broken ones.

        Returns:
            List of (entry, broken_link) tuples
        """
        broken_references = []

        for entry in self.index.entries:
            try:
                content = entry.get_content()
                links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)

                for link_text, link_url in links:
                    if self._is_internal_link(link_url):
                        target_entry = self._resolve_internal_link(link_url, entry)
                        if not target_entry:
                            broken_references.append((entry, link_url))

            except Exception as e:
                logger.error(f"Error validating references for {entry.title}: {e}")

        return broken_references

    def suggest_cross_references(
        self, entry: KnowledgeBaseEntry, limit: int = 3
    ) -> list[tuple[KnowledgeBaseEntry, str]]:
        """
        Suggest cross-references that could be added to an entry.

        Args:
            entry: Entry to suggest references for
            limit: Maximum number of suggestions

        Returns:
            List of (suggested_entry, reason) tuples
        """
        suggestions = []

        # Find related entries that aren't already referenced
        related = self.get_related_entries(entry, limit=limit * 2)

        try:
            content = entry.get_content()
            existing_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
            existing_targets = {
                self._resolve_internal_link(url, entry)
                for _, url in existing_links
                if self._is_internal_link(url)
            }
            existing_targets.discard(None)

            for related_entry, strength, context in related:
                if related_entry not in existing_targets:
                    reason = f"Related ({context}, strength: {strength:.2f})"
                    suggestions.append((related_entry, reason))

                    if len(suggestions) >= limit:
                        break

        except Exception as e:
            logger.error(f"Error suggesting cross-references for {entry.title}: {e}")

        return suggestions

    def get_reference_stats(self) -> dict[str, Any]:
        """Get cross-reference statistics."""
        return {
            **self.stats,
            "total_references": len(self.references),
            "total_entries": len(self.index.entries),
            "average_references_per_entry": (
                len(self.references) / len(self.index.entries)
                if self.index.entries
                else 0
            ),
            "reference_types": {
                ref_type: len(
                    [r for r in self.references if r.reference_type == ref_type]
                )
                for ref_type in ["explicit", "implicit", "related"]
            },
        }


def create_cross_reference_analyzer(
    index: KnowledgeBaseIndex,
) -> CrossReferenceAnalyzer:
    """
    Factory function to create a cross-reference analyzer.

    Args:
        index: Knowledge base index

    Returns:
        Configured analyzer instance
    """
    return CrossReferenceAnalyzer(index)


def main():
    """
    Command-line interface for cross-reference analysis.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Knowledge Base Cross-Reference Analyzer"
    )
    parser.add_argument("kb_path", help="Path to knowledge base directory")
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze all cross-references"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate existing references"
    )
    parser.add_argument("--entry", help="Show references for specific entry")
    parser.add_argument("--suggest", help="Suggest references for specific entry")
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

        # Create analyzer
        analyzer = create_cross_reference_analyzer(index)

        if args.analyze:
            print("🔍 Analyzing cross-references...")
            graph = analyzer.analyze_all_references()

            stats = analyzer.get_reference_stats()
            print("\n📊 Cross-Reference Analysis:")
            print(f"   Total references: {stats['total_references']}")
            print(f"   Explicit references: {stats['reference_types']['explicit']}")
            print(f"   Implicit references: {stats['reference_types']['implicit']}")
            print(f"   Related entries: {stats['reference_types']['related']}")
            print(f"   Broken references: {stats['broken_references']}")
            print(f"   Clusters found: {stats['clusters_found']}")
            print(
                f"   Average references per entry: {stats['average_references_per_entry']:.1f}"
            )

        elif args.validate:
            print("🔍 Validating references...")
            broken = analyzer.validate_references()

            if broken:
                print(f"\n❌ Found {len(broken)} broken references:")
                for entry, broken_link in broken:
                    print(f"   {entry.title}: {broken_link}")
            else:
                print("✅ All references are valid")

        elif args.entry:
            entry = index.get_entry(args.entry)
            if entry:
                analyzer.analyze_all_references()
                related = analyzer.get_related_entries(entry)

                print(f"\n🔗 Related entries for '{args.entry}':")
                for related_entry, strength, context in related:
                    print(
                        f"   {related_entry.title} (strength: {strength:.2f}, {context})"
                    )
            else:
                print(f"❌ Entry not found: {args.entry}")

        elif args.suggest:
            entry = index.get_entry(args.suggest)
            if entry:
                analyzer.analyze_all_references()
                suggestions = analyzer.suggest_cross_references(entry)

                print(f"\n💡 Suggested cross-references for '{args.suggest}':")
                for suggested_entry, reason in suggestions:
                    print(f"   {suggested_entry.title} - {reason}")
            else:
                print(f"❌ Entry not found: {args.suggest}")

        else:
            print("Specify --analyze, --validate, --entry, or --suggest")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
