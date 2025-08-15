#!/usr/bin/env python3
"""
Documentation Migration Script

This script helps migrate existing documentation and markdown files
into the knowledge base format with proper frontmatter and structure.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class DocumentationMigrator:
    """Handles migration of existing documentation to knowledge base format."""

    def __init__(self, kb_path: Path):
        self.kb_path = kb_path
        self.categories_path = kb_path / "categories"
        self.migration_log = []

    def migrate_file(
        self,
        source_file: Path,
        target_category: str,
        title: str | None = None,
        tags: list[str] | None = None,
        difficulty: str = "intermediate",
        contributor: str = "Migration Script",
    ) -> Path:
        """Migrate a single file to knowledge base format."""

        # Read source file
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        content = source_file.read_text(encoding="utf-8")

        # Extract existing frontmatter if present
        existing_frontmatter, body_content = self._extract_frontmatter(content)

        # Generate title if not provided
        if not title:
            title = self._generate_title(
                source_file, body_content, existing_frontmatter
            )

        # Generate tags if not provided
        if not tags:
            tags = self._generate_tags(
                body_content, existing_frontmatter, target_category
            )

        # Create new frontmatter
        new_frontmatter = {
            "title": title,
            "category": target_category,
            "tags": tags,
            "difficulty": difficulty,
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "contributors": [contributor],
        }

        # Merge with existing frontmatter if present
        if existing_frontmatter:
            # Preserve certain fields from existing frontmatter
            preserve_fields = ["contributors", "last_updated", "difficulty"]
            for field in preserve_fields:
                if field in existing_frontmatter:
                    new_frontmatter[field] = existing_frontmatter[field]

        # Create target file path
        target_filename = self._generate_filename(title, source_file)
        target_path = self.categories_path / target_category / target_filename

        # Ensure target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Process content
        processed_content = self._process_content(body_content)

        # Create final content with frontmatter
        final_content = self._create_final_content(new_frontmatter, processed_content)

        # Write target file
        target_path.write_text(final_content, encoding="utf-8")

        # Log migration
        self.migration_log.append(
            {
                "source": str(source_file),
                "target": str(target_path),
                "title": title,
                "category": target_category,
                "tags": tags,
            }
        )

        return target_path

    def migrate_directory(
        self,
        source_dir: Path,
        target_category: str,
        recursive: bool = True,
        file_pattern: str = "*.md",
    ) -> list[Path]:
        """Migrate all files in a directory."""

        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        migrated_files = []

        # Find files to migrate
        if recursive:
            files = source_dir.rglob(file_pattern)
        else:
            files = source_dir.glob(file_pattern)

        for file_path in files:
            if file_path.is_file():
                try:
                    target_path = self.migrate_file(file_path, target_category)
                    migrated_files.append(target_path)
                    print(f"✓ Migrated: {file_path.name} -> {target_path}")
                except Exception as e:
                    print(f"❌ Failed to migrate {file_path}: {e}")

        return migrated_files

    def _extract_frontmatter(self, content: str) -> tuple[[dict[str, Any] | None], str]:
        """Extract existing frontmatter from content."""
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1))
                body = match.group(2)
                return frontmatter, body
            except yaml.YAMLError:
                # Invalid YAML, treat as regular content
                return None, content

        return None, content

    def _generate_title(
        self,
        source_file: Path,
        content: str,
        existing_frontmatter: dict[str, Any] | None,
    ) -> str:
        """Generate title from various sources."""

        # Check existing frontmatter
        if existing_frontmatter and "title" in existing_frontmatter:
            return existing_frontmatter["title"]

        # Look for first heading in content
        heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()

        # Use filename as fallback
        return source_file.stem.replace("-", " ").replace("_", " ").title()

    def _generate_tags(
        self,
        content: str,
        existing_frontmatter: dict[str, Any] | None,
        category: str,
    ) -> list[str]:
        """Generate tags based on content analysis."""

        # Check existing frontmatter
        if existing_frontmatter and "tags" in existing_frontmatter:
            existing_tags = existing_frontmatter["tags"]
            if isinstance(existing_tags, list):
                return existing_tags

        tags = [category]  # Always include category as tag

        # Common technology keywords
        tech_keywords = {
            "python": ["python", "py"],
            "mlx": ["mlx"],
            "apple-silicon": ["apple silicon", "m1", "m2", "metal"],
            "performance": ["performance", "optimization", "speed", "memory"],
            "testing": ["test", "testing", "pytest", "unittest"],
            "deployment": ["deploy", "deployment", "docker", "kubernetes"],
            "troubleshooting": ["error", "issue", "problem", "fix", "debug"],
        }

        content_lower = content.lower()

        for tag, keywords in tech_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                if tag not in tags:
                    tags.append(tag)

        # Limit to 5 tags maximum
        return tags[:5]

    def _generate_filename(self, title: str, source_file: Path) -> str:
        """Generate appropriate filename from title."""

        # Clean title for filename
        filename = re.sub(r"[^\w\s-]", "", title.lower())
        filename = re.sub(r"[-\s]+", "-", filename)
        filename = filename.strip("-")

        # Ensure it's not empty
        if not filename:
            filename = source_file.stem

        return f"{filename}.md"

    def _process_content(self, content: str) -> str:
        """Process and clean up content."""

        # Remove any existing frontmatter delimiters
        content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)

        # Fix common markdown issues
        # Ensure proper spacing around headers
        content = re.sub(r"^(#{1,6})\s*(.+)$", r"\1 \2", content, flags=re.MULTILINE)

        # Ensure proper spacing around code blocks
        content = re.sub(r"\n```", r"\n\n```", content)
        content = re.sub(r"```\n", r"```\n\n", content)

        # Clean up multiple consecutive newlines
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content.strip()

    def _create_final_content(self, frontmatter: dict[str, Any], content: str) -> str:
        """Create final content with frontmatter."""

        # Convert frontmatter to YAML
        yaml_frontmatter = yaml.dump(
            frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

        return f"---\n{yaml_frontmatter}---\n\n{content}\n"

    def create_migration_report(self, output_path: Path | None = None) -> None:
        """Create a migration report."""

        if not self.migration_log:
            print("No files were migrated")
            return

        report_content = f"""# Documentation Migration Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total files migrated: {len(self.migration_log)}

## Migrated Files

| Source | Target | Title | Category | Tags |
|--------|--------|-------|----------|------|
"""

        for entry in self.migration_log:
            tags_str = ", ".join(entry["tags"])
            report_content += f"| {entry['source']} | {entry['target']} | {entry['title']} | {entry['category']} | {tags_str} |\n"

        report_content += """
## Summary by Category

"""

        # Count by category
        category_counts = {}
        for entry in self.migration_log:
            category = entry["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

        for category, count in category_counts.items():
            report_content += f"- **{category}**: {count} files\n"

        report_content += """
## Next Steps

1. Review migrated files for accuracy
2. Update any broken internal links
3. Add cross-references between related entries
4. Run quality checks: `uv run python -m kb quality-check`
5. Rebuild search index: `uv run python -m kb rebuild-index`
"""

        if output_path:
            output_path.write_text(report_content, encoding="utf-8")
            print(f"✓ Migration report saved to: {output_path}")
        else:
            print(report_content)


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate documentation to Knowledge Base format"
    )
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=".",
        help="Path to knowledge base (default: current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Migration commands")

    # Migrate file command
    file_parser = subparsers.add_parser("file", help="Migrate a single file")
    file_parser.add_argument("source", type=Path, help="Source file to migrate")
    file_parser.add_argument("category", help="Target category")
    file_parser.add_argument(
        "--title", help="Entry title (auto-generated if not provided)"
    )
    file_parser.add_argument("--tags", nargs="+", help="Entry tags")
    file_parser.add_argument(
        "--difficulty",
        default="intermediate",
        choices=["beginner", "intermediate", "advanced"],
        help="Entry difficulty level",
    )
    file_parser.add_argument(
        "--contributor", default="Migration Script", help="Contributor name"
    )

    # Migrate directory command
    dir_parser = subparsers.add_parser(
        "directory", help="Migrate all files in a directory"
    )
    dir_parser.add_argument("source", type=Path, help="Source directory to migrate")
    dir_parser.add_argument("category", help="Target category")
    dir_parser.add_argument(
        "--recursive", action="store_true", help="Migrate files recursively"
    )
    dir_parser.add_argument(
        "--pattern", default="*.md", help="File pattern to match (default: *.md)"
    )
    dir_parser.add_argument(
        "--contributor", default="Migration Script", help="Contributor name"
    )

    # Generate report command
    report_parser = subparsers.add_parser("report", help="Generate migration report")
    report_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for report (prints to console if not provided)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    kb_path = args.kb_path.resolve()

    # Check if knowledge base exists
    if not (kb_path / ".meta").exists():
        print(f"❌ No knowledge base found at {kb_path}")
        print("Run init_knowledge_base.py first to create a knowledge base")
        sys.exit(1)

    migrator = DocumentationMigrator(kb_path)

    try:
        if args.command == "file":
            target_path = migrator.migrate_file(
                source_file=args.source,
                target_category=args.category,
                title=args.title,
                tags=args.tags,
                difficulty=args.difficulty,
                contributor=args.contributor,
            )
            print(f"✅ Successfully migrated to: {target_path}")

        elif args.command == "directory":
            migrated_files = migrator.migrate_directory(
                source_dir=args.source,
                target_category=args.category,
                recursive=args.recursive,
                file_pattern=args.pattern,
            )
            print(f"✅ Successfully migrated {len(migrated_files)} files")

        elif args.command == "report":
            migrator.create_migration_report(args.output)

        # Always show migration summary
        if migrator.migration_log:
            print("\n📊 Migration Summary:")
            print(f"   Total files: {len(migrator.migration_log)}")

            # Show category breakdown
            categories = {}
            for entry in migrator.migration_log:
                cat = entry["category"]
                categories[cat] = categories.get(cat, 0) + 1

            for category, count in categories.items():
                print(f"   {category}: {count} files")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
