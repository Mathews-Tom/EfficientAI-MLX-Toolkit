"""
Knowledge Base Data Models

This module defines the core data structures for the EfficientAI-MLX-Toolkit
Knowledge Base system, including entry management and validation.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

import yaml


@dataclass
class KnowledgeBaseEntry:
    """
    Represents a single knowledge base entry with metadata and content.

    This class provides a tool-agnostic way to manage knowledge base entries,
    including loading content, tracking usage, and validating metadata.

    Attributes:
        title: Human-readable title of the entry
        category: Category this entry belongs to (e.g., 'apple-silicon', 'mlx-framework')
        tags: List of tags for categorization and search
        difficulty: Difficulty level ('beginner', 'intermediate', 'advanced')
        content_path: Path to the markdown file containing the entry
        last_updated: Date when the entry was last modified
        contributors: List of contributor names
        usage_count: Number of times this entry has been accessed

    Example:
        >>> entry = KnowledgeBaseEntry(
        ...     title="MLX Memory Optimization",
        ...     category="mlx-framework",
        ...     tags=["mlx", "memory", "optimization"],
        ...     difficulty="intermediate",
        ...     content_path=Path(".knowledge-base/categories/mlx-framework/memory-optimization.md"),
        ...     last_updated=datetime.now(),
        ...     contributors=["developer1"]
        ... )
        >>> content = entry.get_content()
        >>> entry.update_usage()
    """

    title: str
    category: str
    tags: list[str]
    difficulty: str  # "beginner", "intermediate", "advanced"
    content_path: Path
    last_updated: datetime
    contributors: list[str]
    usage_count: int = 0

    def __post_init__(self) -> None:
        """Validate entry data after initialization."""
        self._validate()

    def _validate(self) -> None:
        """
        Validate entry data for consistency and correctness.

        Raises:
            ValueError: If any validation checks fail
        """
        # Validate difficulty level
        valid_difficulties = {"beginner", "intermediate", "advanced"}
        if self.difficulty not in valid_difficulties:
            raise ValueError(
                f"Invalid difficulty '{self.difficulty}'. "
                f"Must be one of: {', '.join(valid_difficulties)}"
            )

        # Validate category format (lowercase, hyphenated)
        if not re.match(r"^[a-z]+(-[a-z]+)*$", self.category):
            raise ValueError(
                f"Invalid category format '{self.category}'. "
                "Must be lowercase with hyphens (e.g., 'apple-silicon')"
            )

        # Validate tags format
        for tag in self.tags:
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError(
                    f"Invalid tag: '{tag}'. Tags must be non-empty strings"
                )

        # Validate contributors
        if not self.contributors:
            raise ValueError("At least one contributor must be specified")

        # Validate title
        if not self.title.strip():
            raise ValueError("Title cannot be empty")

    def get_content(self) -> str:
        """
        Load and return the markdown content of this entry.

        Returns:
            The full markdown content as a string

        Raises:
            FileNotFoundError: If the content file doesn't exist
            IOError: If the file cannot be read
        """
        try:
            return self.content_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Content file not found: {self.content_path}")
        except Exception as e:
            raise IOError(f"Error reading content file {self.content_path}: {e}")

    def update_usage(self) -> None:
        """
        Increment the usage counter for analytics tracking.

        This method is called whenever the entry is accessed or viewed
        to help track popular content and usage patterns.
        """
        self.usage_count += 1

    def get_frontmatter(self) -> dict[str, Union[str, list[str], int]]:
        """
        Generate frontmatter dictionary for this entry.

        Returns:
            Dictionary containing all metadata fields for YAML frontmatter
        """
        return {
            "title": self.title,
            "category": self.category,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "last_updated": self.last_updated.strftime("%Y-%m-%d"),
            "contributors": self.contributors,
        }

    def is_stale(self, days_threshold: int = 90) -> bool:
        """
        Check if this entry might be outdated.

        Args:
            days_threshold: Number of days after which an entry is considered stale

        Returns:
            True if the entry hasn't been updated within the threshold
        """
        days_since_update = (datetime.now() - self.last_updated).days
        return days_since_update > days_threshold

    def matches_search(self, query: str) -> bool:
        """
        Check if this entry matches a search query.

        Args:
            query: Search query string

        Returns:
            True if the entry matches the query (case-insensitive)
        """
        query_lower = query.lower()

        # Search in title
        if query_lower in self.title.lower():
            return True

        # Search in tags
        for tag in self.tags:
            if query_lower in tag.lower():
                return True

        # Search in category
        if query_lower in self.category.lower():
            return True

        return False

    @classmethod
    def from_file(cls, file_path: Path) -> "KnowledgeBaseEntry":
        """
        Create a KnowledgeBaseEntry from a markdown file with frontmatter.

        Args:
            file_path: Path to the markdown file

        Returns:
            KnowledgeBaseEntry instance

        Raises:
            ValueError: If frontmatter is invalid or missing required fields
            FileNotFoundError: If the file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")

        # Extract frontmatter
        if not content.startswith("---"):
            raise ValueError(f"File {file_path} missing frontmatter")

        try:
            # Split content into frontmatter and body
            parts = content.split("---", 2)
            if len(parts) < 3:
                raise ValueError("Invalid frontmatter format")

            frontmatter_text = parts[1].strip()
            frontmatter = yaml.safe_load(frontmatter_text)

            # Validate required fields
            required_fields = [
                "title",
                "category",
                "tags",
                "difficulty",
                "last_updated",
                "contributors",
            ]
            for field in required_fields:
                if field not in frontmatter:
                    raise ValueError(f"Missing required frontmatter field: {field}")

            # Parse date
            if isinstance(frontmatter["last_updated"], str):
                last_updated = datetime.strptime(
                    frontmatter["last_updated"], "%Y-%m-%d"
                )
            else:
                last_updated = frontmatter["last_updated"]

            return cls(
                title=frontmatter["title"],
                category=frontmatter["category"],
                tags=frontmatter["tags"],
                difficulty=frontmatter["difficulty"],
                content_path=file_path,
                last_updated=last_updated,
                contributors=frontmatter["contributors"],
                usage_count=frontmatter.get("usage_count", 0),
            )

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in frontmatter: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing frontmatter: {e}")

    def save_to_file(self, file_path: Path | None = None) -> None:
        """
        Save this entry to a markdown file with frontmatter.

        Args:
            file_path: Optional path to save to. If None, uses self.content_path
        """
        target_path = file_path or self.content_path

        # Read existing content (excluding frontmatter)
        if target_path.exists():
            existing_content = target_path.read_text(encoding="utf-8")
            if existing_content.startswith("---"):
                parts = existing_content.split("---", 2)
                body_content = parts[2] if len(parts) >= 3 else ""
            else:
                body_content = existing_content
        else:
            body_content = "\n# " + self.title + "\n\n[Content to be added]\n"

        # Generate new frontmatter
        frontmatter_dict = self.get_frontmatter()
        frontmatter_yaml = yaml.dump(
            frontmatter_dict, default_flow_style=False, sort_keys=False
        )

        # Combine frontmatter and content
        full_content = f"---\n{frontmatter_yaml}---{body_content}"

        # Ensure directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        target_path.write_text(full_content, encoding="utf-8")

        # Update content_path if we saved to a different location
        if file_path:
            self.content_path = file_path


def validate_entry_frontmatter(frontmatter: dict) -> list[str]:
    """
    Validate frontmatter dictionary for knowledge base entry.

    Args:
        frontmatter: Dictionary containing entry metadata

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Required fields
    required_fields = [
        "title",
        "category",
        "tags",
        "difficulty",
        "last_updated",
        "contributors",
    ]
    for _field in required_fields:
        if _field not in frontmatter:
            errors.append(f"Missing required field: {_field}")
        elif not frontmatter[_field]:
            errors.append(f"Field cannot be empty: {_field}")

    # Validate specific field formats
    if "difficulty" in frontmatter:
        valid_difficulties = {"beginner", "intermediate", "advanced"}
        if frontmatter["difficulty"] not in valid_difficulties:
            errors.append(
                f"Invalid difficulty. Must be one of: {', '.join(valid_difficulties)}"
            )

    if "category" in frontmatter:
        if not re.match(r"^[a-z]+(-[a-z]+)*$", frontmatter["category"]):
            errors.append(
                "Category must be lowercase with hyphens (e.g., 'apple-silicon')"
            )

    if "tags" in frontmatter:
        if not isinstance(frontmatter["tags"], list):
            errors.append("Tags must be a list")
        elif not frontmatter["tags"]:
            errors.append("At least one tag is required")

    if "contributors" in frontmatter:
        if not isinstance(frontmatter["contributors"], list):
            errors.append("Contributors must be a list")
        elif not frontmatter["contributors"]:
            errors.append("At least one contributor is required")

    return errors


@dataclass
class KnowledgeBaseIndex:
    """
    Manages and indexes knowledge base entries for efficient search and retrieval.

    This class provides the core functionality for organizing, searching, and
    filtering knowledge base entries. It maintains indexes by category and tags
    for fast lookups and supports various search strategies.

    Attributes:
        entries: List of all knowledge base entries
        categories: Dictionary mapping category names to entry titles
        tags: Dictionary mapping tag names to entry titles
        _title_to_entry: Internal mapping from titles to entries for fast lookup

    Example:
        >>> index = KnowledgeBaseIndex()
        >>> index.add_entry(entry)
        >>> results = index.search("memory optimization")
        >>> mlx_entries = index.get_by_category("mlx-framework")
        >>> performance_entries = index.get_by_tags(["performance", "optimization"])
    """

    entries: list[KnowledgeBaseEntry] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)
    tags: dict[str, list[str]] = field(default_factory=dict)
    _title_to_entry: dict[str, KnowledgeBaseEntry] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        """Build indexes after initialization."""
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Rebuild all internal indexes from the current entries list."""
        self.categories.clear()
        self.tags.clear()
        self._title_to_entry.clear()

        for entry in self.entries:
            self._add_to_indexes(entry)

    def _add_to_indexes(self, entry: KnowledgeBaseEntry) -> None:
        """Add a single entry to all indexes."""
        # Add to title mapping
        self._title_to_entry[entry.title] = entry

        # Add to category index
        if entry.category not in self.categories:
            self.categories[entry.category] = []
        if entry.title not in self.categories[entry.category]:
            self.categories[entry.category].append(entry.title)

        # Add to tag indexes
        for tag in entry.tags:
            if tag not in self.tags:
                self.tags[tag] = []
            if entry.title not in self.tags[tag]:
                self.tags[tag].append(entry.title)

    def _remove_from_indexes(self, entry: KnowledgeBaseEntry) -> None:
        """Remove a single entry from all indexes."""
        # Remove from title mapping
        self._title_to_entry.pop(entry.title, None)

        # Remove from category index
        if entry.category in self.categories:
            if entry.title in self.categories[entry.category]:
                self.categories[entry.category].remove(entry.title)
            if not self.categories[entry.category]:
                del self.categories[entry.category]

        # Remove from tag indexes
        for tag in entry.tags:
            if tag in self.tags:
                if entry.title in self.tags[tag]:
                    self.tags[tag].remove(entry.title)
                if not self.tags[tag]:
                    del self.tags[tag]

    def add_entry(self, entry: KnowledgeBaseEntry) -> None:
        """
        Add a new entry to the knowledge base index.

        Args:
            entry: The knowledge base entry to add

        Raises:
            ValueError: If an entry with the same title already exists
        """
        if entry.title in self._title_to_entry:
            raise ValueError(f"Entry with title '{entry.title}' already exists")

        self.entries.append(entry)
        self._add_to_indexes(entry)

    def remove_entry(self, title: str) -> bool:
        """
        Remove an entry from the knowledge base index.

        Args:
            title: Title of the entry to remove

        Returns:
            True if the entry was found and removed, False otherwise
        """
        if title not in self._title_to_entry:
            return False

        entry = self._title_to_entry[title]
        self.entries.remove(entry)
        self._remove_from_indexes(entry)
        return True

    def get_entry(self, title: str) -> KnowledgeBaseEntry | None:
        """
        Get an entry by its title.

        Args:
            title: Title of the entry to retrieve

        Returns:
            The knowledge base entry, or None if not found
        """
        return self._title_to_entry.get(title)

    def search(
        self, query: str, category: str | None = None
    ) -> list[KnowledgeBaseEntry]:
        """
        Search entries by title, tags, or content.

        This method performs a basic text-based search across entry titles,
        tags, and categories. For semantic search capabilities, use the
        LLM-enhanced version when available.

        Args:
            query: Search query string
            category: Optional category to limit search to

        Returns:
            List of matching knowledge base entries, sorted by relevance
        """
        if not query.strip():
            return self.entries.copy()

        # Filter by category if specified
        candidates = self.entries
        if category:
            candidates = self.get_by_category(category)

        # Find matching entries
        matches = []
        query_lower = query.lower()

        for entry in candidates:
            if entry.matches_search(query):
                matches.append(entry)

        # Sort by relevance (simple scoring based on title matches)
        def relevance_score(entry: KnowledgeBaseEntry) -> int:
            score = 0
            title_lower = entry.title.lower()

            # Exact title match gets highest score
            if query_lower == title_lower:
                score += 100
            # Title contains query
            elif query_lower in title_lower:
                score += 50
            # Tag exact match
            for tag in entry.tags:
                if query_lower == tag.lower():
                    score += 30
                elif query_lower in tag.lower():
                    score += 15
            # Category match
            if query_lower in entry.category.lower():
                score += 10

            return score

        matches.sort(key=relevance_score, reverse=True)
        return matches

    def get_by_category(self, category: str) -> list[KnowledgeBaseEntry]:
        """
        Get all entries in a specific category.

        Args:
            category: Category name to filter by

        Returns:
            List of entries in the specified category
        """
        if category not in self.categories:
            return []

        return [self._title_to_entry[title] for title in self.categories[category]]

    def get_by_tags(self, tags: list[str]) -> list[KnowledgeBaseEntry]:
        """
        Get entries that match any of the specified tags.

        Args:
            tags: List of tags to match against

        Returns:
            List of entries that have at least one of the specified tags
        """
        if not tags:
            return []

        matching_titles = set()
        for tag in tags:
            if tag in self.tags:
                matching_titles.update(self.tags[tag])

        return [self._title_to_entry[title] for title in matching_titles]

    def get_by_difficulty(self, difficulty: str) -> list[KnowledgeBaseEntry]:
        """
        Get all entries with a specific difficulty level.

        Args:
            difficulty: Difficulty level ('beginner', 'intermediate', 'advanced')

        Returns:
            List of entries with the specified difficulty level
        """
        return [entry for entry in self.entries if entry.difficulty == difficulty]

    def get_by_contributor(self, contributor: str) -> list[KnowledgeBaseEntry]:
        """
        Get all entries by a specific contributor.

        Args:
            contributor: Name of the contributor

        Returns:
            List of entries that include the specified contributor
        """
        return [entry for entry in self.entries if contributor in entry.contributors]

    def get_stale_entries(self, days_threshold: int = 90) -> list[KnowledgeBaseEntry]:
        """
        Get entries that might be outdated.

        Args:
            days_threshold: Number of days after which an entry is considered stale

        Returns:
            List of entries that haven't been updated within the threshold
        """
        return [entry for entry in self.entries if entry.is_stale(days_threshold)]

    def get_popular_entries(self, limit: int = 10) -> list[KnowledgeBaseEntry]:
        """
        Get the most frequently accessed entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of entries sorted by usage count (descending)
        """
        sorted_entries = sorted(self.entries, key=lambda e: e.usage_count, reverse=True)
        return sorted_entries[:limit]

    def get_statistics(self) -> dict[str, Union[int, dict[str, int]]]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary containing various statistics about the knowledge base
        """
        category_counts = {cat: len(titles) for cat, titles in self.categories.items()}
        difficulty_counts = {}
        for entry in self.entries:
            difficulty_counts[entry.difficulty] = (
                difficulty_counts.get(entry.difficulty, 0) + 1
            )

        return {
            "total_entries": len(self.entries),
            "total_categories": len(self.categories),
            "total_tags": len(self.tags),
            "category_distribution": category_counts,
            "difficulty_distribution": difficulty_counts,
            "total_usage": sum(entry.usage_count for entry in self.entries),
        }

    @classmethod
    def from_directory(cls, knowledge_base_path: Path) -> "KnowledgeBaseIndex":
        """
        Create a KnowledgeBaseIndex by scanning a knowledge base directory.

        Args:
            knowledge_base_path: Path to the .knowledge-base directory

        Returns:
            KnowledgeBaseIndex populated with entries from the directory

        Raises:
            FileNotFoundError: If the knowledge base directory doesn't exist
        """
        if not knowledge_base_path.exists():
            raise FileNotFoundError(
                f"Knowledge base directory not found: {knowledge_base_path}"
            )

        index = cls()

        # Scan categories directory
        categories_path = knowledge_base_path / "categories"
        if categories_path.exists():
            for category_dir in categories_path.iterdir():
                if category_dir.is_dir():
                    for md_file in category_dir.glob("*.md"):
                        if md_file.name != "README.md":
                            try:
                                entry = KnowledgeBaseEntry.from_file(md_file)
                                index.add_entry(entry)
                            except Exception as e:
                                # Log error but continue processing other files
                                print(
                                    f"Warning: Could not load entry from {md_file}: {e}"
                                )

        # Scan patterns directory
        patterns_path = knowledge_base_path / "patterns"
        if patterns_path.exists():
            for pattern_dir in patterns_path.iterdir():
                if pattern_dir.is_dir():
                    for md_file in pattern_dir.glob("*.md"):
                        if md_file.name != "README.md":
                            try:
                                entry = KnowledgeBaseEntry.from_file(md_file)
                                index.add_entry(entry)
                            except Exception as e:
                                print(
                                    f"Warning: Could not load entry from {md_file}: {e}"
                                )

        return index

    def save_index(self, index_file_path: Path) -> None:
        """
        Save the index metadata to a file for faster loading.

        Args:
            index_file_path: Path where to save the index file
        """
        index_data = {
            "statistics": self.get_statistics(),
            "categories": list(self.categories.keys()),
            "tags": list(self.tags.keys()),
            "entries": [
                {
                    "title": entry.title,
                    "category": entry.category,
                    "tags": entry.tags,
                    "difficulty": entry.difficulty,
                    "last_updated": entry.last_updated.isoformat(),
                    "contributors": entry.contributors,
                    "usage_count": entry.usage_count,
                    "file_path": str(entry.content_path),
                }
                for entry in self.entries
            ],
        }

        index_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_file_path, "w", encoding="utf-8") as f:
            yaml.dump(index_data, f, default_flow_style=False, sort_keys=False)
