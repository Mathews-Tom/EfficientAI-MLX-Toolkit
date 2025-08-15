"""
Knowledge Base Maintenance Tools

This module provides utilities for updating, maintaining, and managing existing
knowledge base entries, including conflict resolution, content updates, and
automated maintenance tasks.
"""

import difflib
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from indexer import KnowledgeBaseIndexer
from models import KnowledgeBaseEntry, KnowledgeBaseIndex
from validation import (
    parse_markdown_file,
    validate_frontmatter,
    validate_markdown_content,
)

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """
    Result of an entry update operation.

    Attributes:
        success: Whether the update was successful
        file_path: Path to the updated file
        changes_made: List of changes that were made
        warnings: List of warnings generated
        backup_path: Path to backup file (if created)
    """

    success: bool
    file_path: Path | None = None
    changes_made: list[str] = None
    warnings: list[str] = None
    backup_path: Path | None = None

    def __post_init__(self):
        if self.changes_made is None:
            self.changes_made = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class MaintenanceReport:
    """
    Report from maintenance operations.

    Attributes:
        stale_entries: Entries that haven't been updated recently
        broken_links: Entries with broken internal links
        validation_errors: Entries with validation issues
        duplicate_candidates: Potential duplicate entries
        orphaned_files: Files that aren't properly indexed
        suggestions: Maintenance suggestions
    """

    stale_entries: list[KnowledgeBaseEntry] = None
    broken_links: list[tuple[KnowledgeBaseEntry, list[str]]] = None
    validation_errors: list[tuple[KnowledgeBaseEntry, list[str]]] = None
    duplicate_candidates: list[tuple[KnowledgeBaseEntry, KnowledgeBaseEntry, float]] = (
        None
    )
    orphaned_files: list[Path] = None
    suggestions: list[str] = None

    def __post_init__(self):
        if self.stale_entries is None:
            self.stale_entries = []
        if self.broken_links is None:
            self.broken_links = []
        if self.validation_errors is None:
            self.validation_errors = []
        if self.duplicate_candidates is None:
            self.duplicate_candidates = []
        if self.orphaned_files is None:
            self.orphaned_files = []
        if self.suggestions is None:
            self.suggestions = []


class KnowledgeBaseMaintainer:
    """
    Tools for maintaining and updating knowledge base entries.

    This class provides utilities for updating existing entries, detecting
    maintenance issues, and performing automated maintenance tasks.
    """

    def __init__(
        self, kb_path: Path, backup_dir: Path | None = None, auto_backup: bool = True
    ):
        self.kb_path = Path(kb_path)
        self.backup_dir = backup_dir or (self.kb_path / ".meta" / "backups")
        self.auto_backup = auto_backup

        # Load current index
        self.indexer = KnowledgeBaseIndexer(self.kb_path)
        self.index = self.indexer.load_index_from_file()
        if not self.index:
            logger.info("No existing index found, building new index")
            self.index = self.indexer.build_index()

        # Statistics
        self.stats = {
            "entries_updated": 0,
            "backups_created": 0,
            "conflicts_resolved": 0,
            "maintenance_issues_found": 0,
            "maintenance_issues_fixed": 0,
        }

    def update_entry_metadata(
        self,
        entry_title: str,
        updates: dict[str, Any],
        contributor: str | None = None,
    ) -> UpdateResult:
        """
        Update metadata for an existing entry.

        Args:
            entry_title: Title of the entry to update
            updates: Dictionary of metadata updates
            contributor: Name of contributor making the update

        Returns:
            UpdateResult with operation details
        """
        try:
            # Find the entry
            entry = self.index.get_entry(entry_title)
            if not entry:
                return UpdateResult(
                    success=False, warnings=[f"Entry not found: {entry_title}"]
                )

            # Create backup if enabled
            backup_path = None
            if self.auto_backup:
                backup_path = self._create_backup(entry.content_path)

            # Read current content
            frontmatter, body = parse_markdown_file(entry.content_path)

            # Apply updates
            changes_made = []
            for key, value in updates.items():
                if key in frontmatter and frontmatter[key] != value:
                    old_value = frontmatter[key]
                    frontmatter[key] = value
                    changes_made.append(f"Updated {key}: {old_value} → {value}")
                elif key not in frontmatter:
                    frontmatter[key] = value
                    changes_made.append(f"Added {key}: {value}")

            # Update last_updated and contributors
            frontmatter["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            if contributor and contributor not in frontmatter.get("contributors", []):
                contributors = frontmatter.get("contributors", [])
                contributors.append(contributor)
                frontmatter["contributors"] = contributors
                changes_made.append(f"Added contributor: {contributor}")

            # Validate updates
            validation_errors = validate_frontmatter(frontmatter)
            if validation_errors:
                return UpdateResult(
                    success=False, warnings=validation_errors, backup_path=backup_path
                )

            # Write updated content
            self._write_entry_with_frontmatter(entry.content_path, frontmatter, body)

            # Update index
            self.indexer.incremental_update()

            self.stats["entries_updated"] += 1
            if backup_path:
                self.stats["backups_created"] += 1

            return UpdateResult(
                success=True,
                file_path=entry.content_path,
                changes_made=changes_made,
                backup_path=backup_path,
            )

        except Exception as e:
            logger.error(f"Failed to update entry {entry_title}: {e}")
            return UpdateResult(success=False, warnings=[f"Update failed: {e}"])

    def update_entry_content(
        self,
        entry_title: str,
        new_content: str,
        contributor: str | None = None,
        merge_strategy: str = "replace",
    ) -> UpdateResult:
        """
        Update the content of an existing entry.

        Args:
            entry_title: Title of the entry to update
            new_content: New content for the entry
            contributor: Name of contributor making the update
            merge_strategy: How to handle content updates ("replace", "append", "merge")

        Returns:
            UpdateResult with operation details
        """
        try:
            # Find the entry
            entry = self.index.get_entry(entry_title)
            if not entry:
                return UpdateResult(
                    success=False, warnings=[f"Entry not found: {entry_title}"]
                )

            # Create backup if enabled
            backup_path = None
            if self.auto_backup:
                backup_path = self._create_backup(entry.content_path)

            # Read current content
            frontmatter, current_body = parse_markdown_file(entry.content_path)

            # Apply content update based on strategy
            changes_made = []
            if merge_strategy == "replace":
                final_body = new_content
                changes_made.append("Replaced entire content")
            elif merge_strategy == "append":
                final_body = current_body + "\n\n" + new_content
                changes_made.append("Appended new content")
            elif merge_strategy == "merge":
                final_body = self._merge_content(current_body, new_content)
                changes_made.append("Merged content")
            else:
                return UpdateResult(
                    success=False,
                    warnings=[f"Invalid merge strategy: {merge_strategy}"],
                )

            # Update metadata
            frontmatter["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            if contributor and contributor not in frontmatter.get("contributors", []):
                contributors = frontmatter.get("contributors", [])
                contributors.append(contributor)
                frontmatter["contributors"] = contributors
                changes_made.append(f"Added contributor: {contributor}")

            # Validate content
            content_warnings = validate_markdown_content(final_body)

            # Write updated content
            self._write_entry_with_frontmatter(
                entry.content_path, frontmatter, final_body
            )

            # Update index
            self.indexer.incremental_update()

            self.stats["entries_updated"] += 1
            if backup_path:
                self.stats["backups_created"] += 1

            return UpdateResult(
                success=True,
                file_path=entry.content_path,
                changes_made=changes_made,
                warnings=content_warnings,
                backup_path=backup_path,
            )

        except Exception as e:
            logger.error(f"Failed to update content for {entry_title}: {e}")
            return UpdateResult(success=False, warnings=[f"Content update failed: {e}"])

    def resolve_conflict(
        self,
        entry_title: str,
        local_content: str,
        remote_content: str,
        contributor: str,
    ) -> UpdateResult:
        """
        Resolve conflicts between different versions of an entry.

        Args:
            entry_title: Title of the entry with conflicts
            local_content: Local version of the content
            remote_content: Remote version of the content
            contributor: Name of contributor resolving the conflict

        Returns:
            UpdateResult with resolution details
        """
        try:
            # Find the entry
            entry = self.index.get_entry(entry_title)
            if not entry:
                return UpdateResult(
                    success=False, warnings=[f"Entry not found: {entry_title}"]
                )

            # Create backup
            backup_path = self._create_backup(entry.content_path)

            # Parse both versions
            local_fm, local_body = self._parse_content_string(local_content)
            remote_fm, remote_body = self._parse_content_string(remote_content)

            # Merge frontmatter
            merged_fm = self._merge_frontmatter(local_fm, remote_fm, contributor)

            # Merge content with conflict markers
            merged_body = self._merge_content_with_markers(local_body, remote_body)

            # Write merged content
            self._write_entry_with_frontmatter(
                entry.content_path, merged_fm, merged_body
            )

            self.stats["conflicts_resolved"] += 1
            self.stats["backups_created"] += 1

            return UpdateResult(
                success=True,
                file_path=entry.content_path,
                changes_made=["Merged conflicting versions with conflict markers"],
                warnings=["Manual review required for conflict markers"],
                backup_path=backup_path,
            )

        except Exception as e:
            logger.error(f"Failed to resolve conflict for {entry_title}: {e}")
            return UpdateResult(
                success=False, warnings=[f"Conflict resolution failed: {e}"]
            )

    def run_maintenance_check(
        self,
        stale_days: int = 90,
        check_links: bool = True,
        check_duplicates: bool = True,
    ) -> MaintenanceReport:
        """
        Run comprehensive maintenance check on the knowledge base.

        Args:
            stale_days: Number of days after which entries are considered stale
            check_links: Whether to check for broken links
            check_duplicates: Whether to check for duplicate entries

        Returns:
            MaintenanceReport with findings
        """
        report = MaintenanceReport()

        try:
            # Check for stale entries
            cutoff_date = datetime.now() - timedelta(days=stale_days)
            for entry in self.index.entries:
                if entry.last_updated < cutoff_date:
                    report.stale_entries.append(entry)

            # Check for validation errors
            for entry in self.index.entries:
                try:
                    frontmatter, body = parse_markdown_file(entry.content_path)

                    # Validate frontmatter
                    fm_errors = validate_frontmatter(frontmatter)
                    content_warnings = validate_markdown_content(body)

                    all_errors = fm_errors + content_warnings
                    if all_errors:
                        report.validation_errors.append((entry, all_errors))

                except Exception as e:
                    report.validation_errors.append((entry, [f"Parse error: {e}"]))

            # Check for broken links
            if check_links:
                for entry in self.index.entries:
                    broken_links = self._check_entry_links(entry)
                    if broken_links:
                        report.broken_links.append((entry, broken_links))

            # Check for duplicates
            if check_duplicates:
                duplicates = self._find_duplicate_entries()
                report.duplicate_candidates.extend(duplicates)

            # Check for orphaned files
            orphaned = self._find_orphaned_files()
            report.orphaned_files.extend(orphaned)

            # Generate suggestions
            report.suggestions = self._generate_maintenance_suggestions(report)

            # Update statistics
            total_issues = (
                len(report.stale_entries)
                + len(report.validation_errors)
                + len(report.broken_links)
                + len(report.duplicate_candidates)
                + len(report.orphaned_files)
            )
            self.stats["maintenance_issues_found"] = total_issues

            logger.info(f"Maintenance check completed. Found {total_issues} issues.")

        except Exception as e:
            logger.error(f"Maintenance check failed: {e}")
            report.suggestions.append(f"Maintenance check failed: {e}")

        return report

    def _create_backup(self, file_path: Path) -> Path:
        """Create a backup of a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_filename

        # Ensure backup directory exists
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(file_path, backup_path)

        logger.info(f"Created backup: {backup_path}")
        return backup_path

    def _write_entry_with_frontmatter(
        self, file_path: Path, frontmatter: dict[str, Any], body: str
    ) -> None:
        """Write entry with frontmatter to file."""
        import yaml

        # Generate frontmatter YAML
        frontmatter_yaml = yaml.dump(
            frontmatter, default_flow_style=False, sort_keys=False
        )

        # Combine frontmatter and body
        full_content = f"---\n{frontmatter_yaml}---{body}"

        # Write to file
        file_path.write_text(full_content, encoding="utf-8")

    def _parse_content_string(self, content: str) -> tuple[dict[str, Any], str]:
        """Parse content string into frontmatter and body."""
        if not content.startswith("---"):
            return {}, content

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content

        import yaml

        frontmatter = yaml.safe_load(parts[1])
        body = parts[2]

        return frontmatter or {}, body

    def _merge_frontmatter(
        self, local_fm: dict[str, Any], remote_fm: dict[str, Any], contributor: str
    ) -> dict[str, Any]:
        """Merge frontmatter from two versions."""
        merged = local_fm.copy()

        # Merge contributors
        local_contributors = set(local_fm.get("contributors", []))
        remote_contributors = set(remote_fm.get("contributors", []))
        all_contributors = local_contributors | remote_contributors
        all_contributors.add(contributor)
        merged["contributors"] = sorted(list(all_contributors))

        # Use most recent date
        local_date = local_fm.get("last_updated", "1900-01-01")
        remote_date = remote_fm.get("last_updated", "1900-01-01")
        merged["last_updated"] = max(local_date, remote_date)

        # Merge tags
        local_tags = set(local_fm.get("tags", []))
        remote_tags = set(remote_fm.get("tags", []))
        merged["tags"] = sorted(list(local_tags | remote_tags))

        # For other fields, prefer remote if different
        for key, value in remote_fm.items():
            if key not in ["contributors", "last_updated", "tags"]:
                if key not in merged or merged[key] != value:
                    merged[key] = value

        return merged

    def _merge_content(self, local_content: str, remote_content: str) -> str:
        """Merge content from two versions."""
        # Simple merge - in practice, this could be more sophisticated
        local_lines = local_content.split("\n")
        remote_lines = remote_content.split("\n")

        # Use difflib to find differences and merge
        differ = difflib.unified_diff(local_lines, remote_lines, lineterm="")
        merged_lines = []

        for line in differ:
            if line.startswith("@@"):
                continue
            elif line.startswith("-"):
                # Removed line - skip for now
                continue
            elif line.startswith("+"):
                # Added line
                merged_lines.append(line[1:])
            else:
                # Unchanged line
                merged_lines.append(line[1:] if line.startswith(" ") else line)

        return "\n".join(merged_lines)

    def _merge_content_with_markers(
        self, local_content: str, remote_content: str
    ) -> str:
        """Merge content with conflict markers for manual resolution."""
        return f"""<<<<<<< LOCAL
{local_content}
=======
{remote_content}
>>>>>>> REMOTE

<!-- CONFLICT RESOLUTION NEEDED -->
<!-- Please review and resolve the conflicts above -->
<!-- Remove the conflict markers when done -->
"""

    def _check_entry_links(self, entry: KnowledgeBaseEntry) -> list[str]:
        """Check for broken internal links in an entry."""
        broken_links = []

        try:
            content = entry.get_content()

            # Find markdown links
            import re

            links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)

            for link_text, link_url in links:
                # Check internal links (relative paths)
                if not link_url.startswith("http") and not link_url.startswith("#"):
                    # Resolve relative to entry location
                    link_path = entry.content_path.parent / link_url
                    if not link_path.exists():
                        broken_links.append(f"Broken link: {link_text} -> {link_url}")

        except Exception as e:
            broken_links.append(f"Error checking links: {e}")

        return broken_links

    def _find_duplicate_entries(
        self,
    ) -> list[tuple[KnowledgeBaseEntry, KnowledgeBaseEntry, float]]:
        """Find potential duplicate entries."""
        duplicates = []
        entries = self.index.entries

        for i, entry1 in enumerate(entries):
            for entry2 in entries[i + 1 :]:
                similarity = self._calculate_similarity(entry1, entry2)
                if similarity > 0.7:  # 70% similarity threshold
                    duplicates.append((entry1, entry2, similarity))

        return duplicates

    def _calculate_similarity(
        self, entry1: KnowledgeBaseEntry, entry2: KnowledgeBaseEntry
    ) -> float:
        """Calculate similarity between two entries."""
        # Simple similarity based on title and tags
        title_similarity = difflib.SequenceMatcher(
            None, entry1.title.lower(), entry2.title.lower()
        ).ratio()

        tags1 = set(entry1.tags)
        tags2 = set(entry2.tags)
        tag_similarity = len(tags1 & tags2) / len(tags1 | tags2) if tags1 | tags2 else 0

        # Weighted average
        return (title_similarity * 0.7) + (tag_similarity * 0.3)

    def _find_orphaned_files(self) -> list[Path]:
        """Find markdown files that aren't properly indexed."""
        orphaned = []

        # Get all markdown files
        all_md_files = set()
        for category_dir in (self.kb_path / "categories").glob("*"):
            if category_dir.is_dir():
                for md_file in category_dir.glob("*.md"):
                    if md_file.name != "README.md":
                        all_md_files.add(md_file)

        # Get indexed files
        indexed_files = {entry.content_path for entry in self.index.entries}

        # Find orphaned files
        orphaned = list(all_md_files - indexed_files)

        return orphaned

    def _generate_maintenance_suggestions(self, report: MaintenanceReport) -> list[str]:
        """Generate maintenance suggestions based on report."""
        suggestions = []

        if report.stale_entries:
            suggestions.append(
                f"Consider updating {len(report.stale_entries)} stale entries"
            )

        if report.validation_errors:
            suggestions.append(f"Fix {len(report.validation_errors)} validation errors")

        if report.broken_links:
            suggestions.append(
                f"Repair {len(report.broken_links)} entries with broken links"
            )

        if report.duplicate_candidates:
            suggestions.append(
                f"Review {len(report.duplicate_candidates)} potential duplicate entries"
            )

        if report.orphaned_files:
            suggestions.append(
                f"Index or remove {len(report.orphaned_files)} orphaned files"
            )

        return suggestions

    def get_maintenance_stats(self) -> dict[str, Any]:
        """Get maintenance statistics."""
        return {
            **self.stats,
            "total_entries": len(self.index.entries),
            "backup_dir": str(self.backup_dir),
            "auto_backup_enabled": self.auto_backup,
        }


def create_maintainer(kb_path: Path, **kwargs) -> KnowledgeBaseMaintainer:
    """
    Factory function to create a knowledge base maintainer.

    Args:
        kb_path: Path to knowledge base directory
        **kwargs: Additional arguments for KnowledgeBaseMaintainer

    Returns:
        Configured maintainer instance
    """
    return KnowledgeBaseMaintainer(kb_path, **kwargs)


def main():
    """
    Command-line interface for maintenance tools.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Base Maintenance Tools")
    parser.add_argument("kb_path", help="Path to knowledge base directory")
    parser.add_argument("--check", action="store_true", help="Run maintenance check")
    parser.add_argument(
        "--stale-days", type=int, default=90, help="Days to consider entries stale"
    )
    parser.add_argument("--update-entry", help="Entry title to update")
    parser.add_argument("--contributor", help="Contributor name for updates")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Create maintainer
    maintainer = create_maintainer(Path(args.kb_path))

    try:
        if args.check:
            # Run maintenance check
            print("🔍 Running maintenance check...")
            report = maintainer.run_maintenance_check(stale_days=args.stale_days)

            print("\n📊 Maintenance Report:")
            print(f"   Stale entries: {len(report.stale_entries)}")
            print(f"   Validation errors: {len(report.validation_errors)}")
            print(f"   Broken links: {len(report.broken_links)}")
            print(f"   Duplicate candidates: {len(report.duplicate_candidates)}")
            print(f"   Orphaned files: {len(report.orphaned_files)}")

            if report.suggestions:
                print("\n💡 Suggestions:")
                for suggestion in report.suggestions:
                    print(f"   • {suggestion}")

        elif args.update_entry:
            # Update specific entry
            print(f"📝 Updating entry: {args.update_entry}")
            # This would need more specific update parameters
            print("Use programmatic interface for specific updates")

        else:
            print(
                "Specify --check to run maintenance check or --update-entry to update an entry"
            )

        # Show statistics
        stats = maintainer.get_maintenance_stats()
        print("\n📈 Statistics:")
        print(f"   Entries updated: {stats['entries_updated']}")
        print(f"   Backups created: {stats['backups_created']}")
        print(f"   Conflicts resolved: {stats['conflicts_resolved']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
