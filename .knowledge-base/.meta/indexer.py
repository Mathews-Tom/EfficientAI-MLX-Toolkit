"""
Knowledge Base Indexing System

This module provides comprehensive indexing functionality for the knowledge base,
including file scanning, index building, and automatic updates.
"""

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from models import KnowledgeBaseEntry, KnowledgeBaseIndex
from validation import ParseError, ValidationError

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Raised when indexing operations fail."""

    pass


class KnowledgeBaseIndexer:
    """
    Comprehensive indexing system for knowledge base entries.

    This class handles scanning the knowledge base directory, building indexes,
    tracking file changes, and maintaining search-optimized data structures.

    Attributes:
        kb_path: Path to the knowledge base directory
        index_file: Path to the index cache file
        watch_files: Whether to monitor files for changes
        parallel_processing: Whether to use parallel processing for indexing
    """

    def __init__(
        self,
        kb_path: Path,
        index_file: Path | None = None,
        watch_files: bool = True,
        parallel_processing: bool = True,
    ):
        self.kb_path = Path(kb_path)
        self.index_file = index_file or (self.kb_path / ".meta" / "index.json")
        self.watch_files = watch_files
        self.parallel_processing = parallel_processing

        # File tracking for incremental updates
        self._file_hashes: dict[str, str] = {}
        self._last_scan_time: datetime | None = None

        # Statistics
        self.stats = {
            "total_files_scanned": 0,
            "entries_indexed": 0,
            "errors_encountered": 0,
            "last_index_time": None,
            "index_build_duration": 0.0,
        }

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of a file for change detection.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash string
        """
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return ""

    def _has_file_changed(self, file_path: Path) -> bool:
        """
        Check if a file has changed since last indexing.

        Args:
            file_path: Path to check

        Returns:
            True if file has changed or is new
        """
        file_key = str(file_path.relative_to(self.kb_path))
        current_hash = self._calculate_file_hash(file_path)

        if file_key not in self._file_hashes:
            # New file
            self._file_hashes[file_key] = current_hash
            return True

        if self._file_hashes[file_key] != current_hash:
            # File changed
            self._file_hashes[file_key] = current_hash
            return True

        return False

    def _find_entry_files(self) -> list[Path]:
        """
        Find all knowledge base entry files.

        Returns:
            List of paths to markdown files containing entries
        """
        entry_files = []

        # Scan categories directory
        categories_path = self.kb_path / "categories"
        if categories_path.exists():
            for category_dir in categories_path.iterdir():
                if category_dir.is_dir():
                    for md_file in category_dir.glob("*.md"):
                        if md_file.name != "README.md":
                            entry_files.append(md_file)

        # Scan patterns directory
        patterns_path = self.kb_path / "patterns"
        if patterns_path.exists():
            for pattern_dir in patterns_path.iterdir():
                if pattern_dir.is_dir():
                    for md_file in pattern_dir.glob("*.md"):
                        if md_file.name != "README.md":
                            entry_files.append(md_file)

        return entry_files

    def _process_single_file(self, file_path: Path) -> KnowledgeBaseEntry | None:
        """
        Process a single markdown file into a knowledge base entry.

        Args:
            file_path: Path to the markdown file

        Returns:
            KnowledgeBaseEntry if successful, None if failed
        """
        try:
            entry = KnowledgeBaseEntry.from_file(file_path)
            logger.debug(f"Successfully processed: {file_path}")
            return entry

        except (ValidationError, ParseError) as e:
            logger.error(f"Validation/Parse error in {file_path}: {e}")
            self.stats["errors_encountered"] += 1
            return None

        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}")
            self.stats["errors_encountered"] += 1
            return None

    def _build_full_text_index(
        self, entries: list[KnowledgeBaseEntry]
    ) -> dict[str, set[str]]:
        """
        Build full-text search index from entries.

        Args:
            entries: List of knowledge base entries

        Returns:
            Dictionary mapping words to entry titles containing them
        """
        word_to_entries = {}

        for entry in entries:
            # Get searchable text
            searchable_text = self._extract_searchable_text(entry)
            words = self._tokenize_text(searchable_text)

            for word in words:
                if word not in word_to_entries:
                    word_to_entries[word] = set()
                word_to_entries[word].add(entry.title)

        # Convert sets to lists for JSON serialization
        return {word: list(entries) for word, entries in word_to_entries.items()}

    def _extract_searchable_text(self, entry: KnowledgeBaseEntry) -> str:
        """
        Extract searchable text from an entry.

        Args:
            entry: Knowledge base entry

        Returns:
            Combined searchable text
        """
        searchable_parts = [
            entry.title,
            entry.category,
            " ".join(entry.tags),
            " ".join(entry.contributors),
        ]

        # Add content if available
        try:
            content = entry.get_content()
            # Remove frontmatter and markdown formatting for search
            content_lines = content.split("\n")
            content_body = []
            in_frontmatter = False
            frontmatter_count = 0

            for line in content_lines:
                if line.strip() == "---":
                    frontmatter_count += 1
                    if frontmatter_count == 2:
                        in_frontmatter = False
                    else:
                        in_frontmatter = True
                    continue

                if not in_frontmatter:
                    # Remove markdown formatting
                    clean_line = line.strip()
                    clean_line = clean_line.lstrip("#").strip()  # Remove headers
                    clean_line = clean_line.replace("**", "").replace(
                        "*", ""
                    )  # Remove bold/italic
                    clean_line = clean_line.replace("`", "")  # Remove code formatting
                    if clean_line:
                        content_body.append(clean_line)

            searchable_parts.append(" ".join(content_body))

        except Exception as e:
            logger.warning(f"Could not extract content from {entry.title}: {e}")

        return " ".join(searchable_parts)

    def _tokenize_text(self, text: str) -> set[str]:
        """
        Tokenize text for search indexing.

        Args:
            text: Text to tokenize

        Returns:
            Set of normalized tokens
        """
        import re

        # Convert to lowercase and split on non-alphanumeric characters
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out very short words and common stop words
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
        }

        filtered_words = {
            word for word in words if len(word) >= 2 and word not in stop_words
        }

        return filtered_words

    def build_index(self, force_rebuild: bool = False) -> KnowledgeBaseIndex:
        """
        Build or update the knowledge base index.

        Args:
            force_rebuild: Whether to force a complete rebuild

        Returns:
            Updated knowledge base index

        Raises:
            IndexingError: If indexing fails
        """
        start_time = time.time()
        logger.info("Starting knowledge base indexing...")

        try:
            # Find all entry files
            entry_files = self._find_entry_files()
            self.stats["total_files_scanned"] = len(entry_files)

            # Determine which files need processing
            if force_rebuild or not self.watch_files:
                files_to_process = entry_files
            else:
                files_to_process = [f for f in entry_files if self._has_file_changed(f)]

            logger.info(
                f"Processing {len(files_to_process)} files (of {len(entry_files)} total)"
            )

            # Process files
            entries = []
            if self.parallel_processing and len(files_to_process) > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_file = {
                        executor.submit(self._process_single_file, file_path): file_path
                        for file_path in files_to_process
                    }

                    for future in as_completed(future_to_file):
                        entry = future.result()
                        if entry:
                            entries.append(entry)
            else:
                # Sequential processing
                for file_path in files_to_process:
                    entry = self._process_single_file(file_path)
                    if entry:
                        entries.append(entry)

            # Create index
            index = KnowledgeBaseIndex(entries=entries)
            self.stats["entries_indexed"] = len(entries)

            # Build full-text search index
            full_text_index = self._build_full_text_index(entries)

            # Save index to file
            self._save_index_to_file(index, full_text_index)

            # Update statistics
            end_time = time.time()
            self.stats["last_index_time"] = datetime.now()
            self.stats["index_build_duration"] = end_time - start_time
            self._last_scan_time = datetime.now()

            logger.info(
                f"Indexing completed in {self.stats['index_build_duration']:.2f}s"
            )
            logger.info(
                f"Indexed {self.stats['entries_indexed']} entries with {self.stats['errors_encountered']} errors"
            )

            return index

        except Exception as e:
            raise IndexingError(f"Failed to build index: {e}") from e

    def _save_index_to_file(
        self, index: KnowledgeBaseIndex, full_text_index: dict[str, list[str]]
    ) -> None:
        """
        Save index data to file for persistence.

        Args:
            index: Knowledge base index
            full_text_index: Full-text search index
        """
        try:
            # Prepare index data with JSON-serializable stats
            serializable_stats = {}
            for key, value in self.stats.items():
                if isinstance(value, datetime):
                    serializable_stats[key] = value.isoformat()
                else:
                    serializable_stats[key] = value

            index_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "kb_path": str(self.kb_path),
                    "total_entries": len(index.entries),
                    "categories": list(index.categories.keys()),
                    "tags": list(index.tags.keys()),
                    "stats": serializable_stats,
                },
                "entries": [
                    {
                        "title": entry.title,
                        "category": entry.category,
                        "tags": entry.tags,
                        "difficulty": entry.difficulty,
                        "last_updated": entry.last_updated.isoformat(),
                        "contributors": entry.contributors,
                        "usage_count": entry.usage_count,
                        "file_path": str(entry.content_path.relative_to(self.kb_path)),
                    }
                    for entry in index.entries
                ],
                "indexes": {
                    "categories": {
                        cat: titles for cat, titles in index.categories.items()
                    },
                    "tags": {tag: titles for tag, titles in index.tags.items()},
                    "full_text": full_text_index,
                },
                "file_hashes": self._file_hashes,
            }

            # Ensure directory exists
            self.index_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Index saved to {self.index_file}")

        except Exception as e:
            logger.error(f"Failed to save index to file: {e}")
            raise IndexingError(f"Failed to save index: {e}") from e

    def load_index_from_file(self) -> KnowledgeBaseIndex | None:
        """
        Load index from file if it exists.

        Returns:
            Loaded knowledge base index, or None if file doesn't exist or is invalid
        """
        if not self.index_file.exists():
            logger.info("No existing index file found")
            return None

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            # Restore file hashes
            self._file_hashes = index_data.get("file_hashes", {})

            # Restore statistics
            self.stats.update(index_data.get("metadata", {}).get("stats", {}))

            # Recreate entries
            entries = []
            for entry_data in index_data.get("entries", []):
                try:
                    file_path = self.kb_path / entry_data["file_path"]
                    entry = KnowledgeBaseEntry.from_file(file_path)
                    entries.append(entry)
                except Exception as e:
                    logger.warning(
                        f"Could not load entry from {entry_data.get('file_path', 'unknown')}: {e}"
                    )

            index = KnowledgeBaseIndex(entries=entries)

            logger.info(
                f"Loaded index with {len(entries)} entries from {self.index_file}"
            )
            return index

        except Exception as e:
            logger.error(f"Failed to load index from file: {e}")
            return None

    def get_index_stats(self) -> dict[str, Any]:
        """
        Get indexing statistics.

        Returns:
            Dictionary with indexing statistics
        """
        return {
            **self.stats,
            "index_file_exists": self.index_file.exists(),
            "index_file_size": (
                self.index_file.stat().st_size if self.index_file.exists() else 0
            ),
            "kb_path": str(self.kb_path),
            "last_scan_time": (
                self._last_scan_time.isoformat() if self._last_scan_time else None
            ),
        }

    def incremental_update(self) -> KnowledgeBaseIndex | None:
        """
        Perform incremental update of the index.

        Returns:
            Updated index if changes were found, None otherwise
        """
        if not self.watch_files:
            logger.info("File watching disabled, performing full rebuild")
            return self.build_index(force_rebuild=True)

        # Check if any files have changed
        entry_files = self._find_entry_files()
        changed_files = [f for f in entry_files if self._has_file_changed(f)]

        if not changed_files:
            logger.info("No file changes detected")
            return None

        logger.info(f"Detected changes in {len(changed_files)} files, updating index")
        return self.build_index(force_rebuild=True)


def create_indexer(kb_path: Path, **kwargs) -> KnowledgeBaseIndexer:
    """
    Factory function to create a knowledge base indexer.

    Args:
        kb_path: Path to knowledge base directory
        **kwargs: Additional arguments for KnowledgeBaseIndexer

    Returns:
        Configured indexer instance
    """
    return KnowledgeBaseIndexer(kb_path, **kwargs)


def main():
    """
    Command-line interface for the indexer.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Base Indexer")
    parser.add_argument("kb_path", help="Path to knowledge base directory")
    parser.add_argument(
        "--force-rebuild", action="store_true", help="Force complete rebuild"
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Create indexer
    indexer = create_indexer(
        Path(args.kb_path), parallel_processing=not args.no_parallel
    )

    try:
        # Build index
        index = indexer.build_index(force_rebuild=args.force_rebuild)

        # Print statistics
        stats = indexer.get_index_stats()
        print("\nIndexing completed successfully!")
        print(f"Entries indexed: {stats['entries_indexed']}")
        print(f"Categories: {len(index.categories)}")
        print(f"Tags: {len(index.tags)}")
        print(f"Build time: {stats['index_build_duration']:.2f}s")
        print(f"Errors: {stats['errors_encountered']}")

    except IndexingError as e:
        print(f"Indexing failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
