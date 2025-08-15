"""
Knowledge Base Contribution Tools

This module provides utilities for creating, updating, and managing knowledge base
entries, including guided entry creation, template management, and validation.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from indexer import KnowledgeBaseIndexer
from validation import (
    validate_code_examples,
    validate_frontmatter,
    validate_markdown_content,
)

logger = logging.getLogger(__name__)


@dataclass
class ContributionConfig:
    """
    Configuration for contribution workflow.

    Attributes:
        default_contributor: Default contributor name
        auto_categorize: Whether to suggest categories automatically
        validate_on_create: Whether to validate entries on creation
        backup_existing: Whether to backup existing files before updates
        editor_command: Command to open files in editor
    """

    default_contributor: str = "Unknown"
    auto_categorize: bool = True
    validate_on_create: bool = True
    backup_existing: bool = True
    editor_command: str | None = None


class KnowledgeBaseContributor:
    """
    Tools for contributing to the knowledge base.

    This class provides utilities for creating new entries, updating existing ones,
    and managing the contribution workflow with validation and suggestions.
    """

    def __init__(self, kb_path: Path, config: ContributionConfig | None = None):
        self.kb_path = Path(kb_path)
        self.config = config or ContributionConfig()

        # Load available categories and tags for suggestions
        self._load_existing_metadata()

        # Statistics
        self.stats = {
            "entries_created": 0,
            "entries_updated": 0,
            "validation_errors": 0,
            "suggestions_provided": 0,
        }

    def _load_existing_metadata(self) -> None:
        """Load existing categories and tags for suggestions."""
        try:
            indexer = KnowledgeBaseIndexer(self.kb_path)
            index = indexer.load_index_from_file()

            if index:
                self.existing_categories = set(index.categories.keys())
                self.existing_tags = set(index.tags.keys())
            else:
                self.existing_categories = set()
                self.existing_tags = set()

        except Exception as e:
            logger.warning(f"Could not load existing metadata: {e}")
            self.existing_categories = set()
            self.existing_tags = set()

    def create_entry_interactive(self) -> Path | None:
        """
        Create a new knowledge base entry through interactive prompts.

        Returns:
            Path to created entry file, or None if cancelled
        """
        print("🚀 Creating a new knowledge base entry")
        print("=" * 50)

        try:
            # Gather basic information
            title = self._prompt_for_title()
            if not title:
                print("❌ Entry creation cancelled")
                return None

            category = self._prompt_for_category()
            tags = self._prompt_for_tags()
            difficulty = self._prompt_for_difficulty()
            entry_type = self._prompt_for_entry_type()

            # Create entry metadata
            entry_data = {
                "title": title,
                "category": category,
                "tags": tags,
                "difficulty": difficulty,
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "contributors": [self.config.default_contributor],
            }

            # Generate filename and path
            filename = self._generate_filename(title)
            file_path = self._get_entry_path(category, filename)

            # Create entry from template
            template_path = self._get_template_path(entry_type)
            entry_content = self._create_entry_from_template(template_path, entry_data)

            # Validate if enabled
            if self.config.validate_on_create:
                validation_errors = self._validate_entry_content(entry_content)
                if validation_errors:
                    print("⚠️  Validation warnings:")
                    for error in validation_errors:
                        print(f"   • {error}")

                    if not self._confirm("Continue despite warnings?"):
                        print("❌ Entry creation cancelled")
                        return None

            # Write file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(entry_content, encoding="utf-8")

            print(f"✅ Entry created: {file_path}")

            # Open in editor if configured
            if self.config.editor_command:
                self._open_in_editor(file_path)

            self.stats["entries_created"] += 1
            return file_path

        except KeyboardInterrupt:
            print("\n❌ Entry creation cancelled")
            return None
        except Exception as e:
            print(f"❌ Error creating entry: {e}")
            logger.error(f"Entry creation failed: {e}")
            return None

    def create_entry_from_template(
        self,
        title: str,
        category: str,
        tags: list[str],
        difficulty: str = "intermediate",
        entry_type: str = "standard",
        contributor: str | None = None,
    ) -> Path:
        """
        Create a new entry from template with provided metadata.

        Args:
            title: Entry title
            category: Entry category
            tags: List of tags
            difficulty: Difficulty level
            entry_type: Type of entry (standard, troubleshooting, pattern)
            contributor: Contributor name

        Returns:
            Path to created entry file

        Raises:
            ValueError: If parameters are invalid
            FileExistsError: If entry already exists
        """
        # Validate inputs
        if not title.strip():
            raise ValueError("Title cannot be empty")

        if difficulty not in ["beginner", "intermediate", "advanced"]:
            raise ValueError(f"Invalid difficulty: {difficulty}")

        # Create entry metadata
        entry_data = {
            "title": title.strip(),
            "category": category,
            "tags": tags,
            "difficulty": difficulty,
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "contributors": [contributor or self.config.default_contributor],
        }

        # Generate filename and path
        filename = self._generate_filename(title)
        file_path = self._get_entry_path(category, filename)

        if file_path.exists():
            raise FileExistsError(f"Entry already exists: {file_path}")

        # Create entry from template
        template_path = self._get_template_path(entry_type)
        entry_content = self._create_entry_from_template(template_path, entry_data)

        # Write file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(entry_content, encoding="utf-8")

        logger.info(f"Created entry: {file_path}")
        self.stats["entries_created"] += 1

        return file_path

    def suggest_category(self, title: str, content: str = "") -> list[str]:
        """
        Suggest appropriate categories for an entry.

        Args:
            title: Entry title
            content: Entry content (optional)

        Returns:
            List of suggested categories
        """
        suggestions = []
        text = (title + " " + content).lower()

        # Category keywords mapping
        category_keywords = {
            "apple-silicon": [
                "apple",
                "silicon",
                "m1",
                "m2",
                "m3",
                "metal",
                "unified",
                "memory",
            ],
            "mlx-framework": ["mlx", "framework", "array", "neural", "network", "nn"],
            "performance": [
                "performance",
                "optimization",
                "speed",
                "memory",
                "efficient",
                "fast",
            ],
            "testing": ["test", "testing", "unit", "integration", "pytest", "assert"],
            "deployment": [
                "deploy",
                "deployment",
                "production",
                "docker",
                "container",
                "ci",
                "cd",
            ],
            "troubleshooting": [
                "error",
                "fix",
                "problem",
                "issue",
                "debug",
                "troubleshoot",
                "solve",
            ],
        }

        # Score categories based on keyword matches
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score

        # Sort by score and return top suggestions
        sorted_categories = sorted(
            category_scores.items(), key=lambda x: x[1], reverse=True
        )
        suggestions = [cat for cat, score in sorted_categories[:3]]

        # Add existing categories that might be relevant
        for existing_cat in self.existing_categories:
            if existing_cat not in suggestions and any(
                word in text for word in existing_cat.split("-")
            ):
                suggestions.append(existing_cat)

        self.stats["suggestions_provided"] += 1
        return suggestions[:5]  # Return top 5 suggestions

    def suggest_tags(self, title: str, category: str, content: str = "") -> list[str]:
        """
        Suggest appropriate tags for an entry.

        Args:
            title: Entry title
            category: Entry category
            content: Entry content (optional)

        Returns:
            List of suggested tags
        """
        suggestions = []
        text = (title + " " + content).lower()

        # Common tag patterns
        tag_patterns = {
            "mlx": ["mlx"],
            "python": ["python", "py"],
            "optimization": ["optimization", "optimize", "efficient", "performance"],
            "memory": ["memory", "ram", "allocation"],
            "training": ["training", "train", "learning"],
            "inference": ["inference", "prediction", "infer"],
            "apple-silicon": ["apple", "silicon", "m1", "m2", "m3"],
            "beginner": ["beginner", "basic", "intro", "getting-started"],
            "advanced": ["advanced", "expert", "complex"],
            "tutorial": ["tutorial", "guide", "how-to", "walkthrough"],
            "pattern": ["pattern", "template", "example"],
            "error": ["error", "exception", "bug", "issue"],
            "configuration": ["config", "configuration", "setup", "settings"],
        }

        # Score tags based on pattern matches
        tag_scores = {}
        for tag, patterns in tag_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text)
            if score > 0:
                tag_scores[tag] = score

        # Sort by score and add to suggestions
        sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
        suggestions.extend([tag for tag, score in sorted_tags[:5]])

        # Add category as a tag if not already included
        if category not in suggestions:
            suggestions.append(category)

        # Add existing tags that might be relevant
        for existing_tag in self.existing_tags:
            if existing_tag not in suggestions and len(suggestions) < 8:
                if any(word in text for word in existing_tag.split("-")):
                    suggestions.append(existing_tag)

        self.stats["suggestions_provided"] += 1
        return suggestions[:8]  # Return top 8 suggestions

    def _prompt_for_title(self) -> str:
        """Prompt user for entry title."""
        while True:
            title = input("📝 Entry title: ").strip()
            if title:
                return title
            print("❌ Title cannot be empty. Please try again.")

    def _prompt_for_category(self) -> str:
        """Prompt user for entry category with suggestions."""
        print(
            f"\n📁 Available categories: {', '.join(sorted(self.existing_categories))}"
        )

        while True:
            category = input("📁 Category: ").strip().lower()
            if category:
                # Suggest similar categories if not exact match
                if category not in self.existing_categories:
                    similar = [
                        cat
                        for cat in self.existing_categories
                        if category in cat or cat in category
                    ]
                    if similar:
                        print(f"💡 Did you mean: {', '.join(similar)}?")
                        if self._confirm("Use suggested category?"):
                            return similar[0]

                return category
            print("❌ Category cannot be empty. Please try again.")

    def _prompt_for_tags(self) -> list[str]:
        """Prompt user for entry tags."""
        print(f"\n🏷️  Popular tags: {', '.join(sorted(list(self.existing_tags)[:10]))}")

        tags_input = input("🏷️  Tags (comma-separated): ").strip()
        if tags_input:
            tags = [tag.strip().lower() for tag in tags_input.split(",") if tag.strip()]
            return tags
        return []

    def _prompt_for_difficulty(self) -> str:
        """Prompt user for difficulty level."""
        difficulties = ["beginner", "intermediate", "advanced"]
        print(f"\n📊 Difficulty levels: {', '.join(difficulties)}")

        while True:
            difficulty = input("📊 Difficulty [intermediate]: ").strip().lower()
            if not difficulty:
                return "intermediate"
            if difficulty in difficulties:
                return difficulty
            print(f"❌ Invalid difficulty. Choose from: {', '.join(difficulties)}")

    def _prompt_for_entry_type(self) -> str:
        """Prompt user for entry type."""
        entry_types = {
            "1": ("standard", "Standard knowledge base entry"),
            "2": ("troubleshooting", "Problem-solution troubleshooting entry"),
            "3": ("pattern", "Code pattern documentation"),
        }

        print("\n📋 Entry types:")
        for key, (type_name, description) in entry_types.items():
            print(f"  {key}. {description}")

        while True:
            choice = input("📋 Entry type [1]: ").strip()
            if not choice:
                return "standard"
            if choice in entry_types:
                return entry_types[choice][0]
            print("❌ Invalid choice. Please select 1, 2, or 3.")

    def _confirm(self, message: str) -> bool:
        """Get yes/no confirmation from user."""
        while True:
            response = input(f"{message} (y/n): ").strip().lower()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            print("❌ Please answer 'y' or 'n'")

    def _generate_filename(self, title: str) -> str:
        """Generate filename from title."""
        # Convert to lowercase and replace spaces/special chars with hyphens
        filename = re.sub(r"[^\w\s-]", "", title.lower())
        filename = re.sub(r"[-\s]+", "-", filename)
        filename = filename.strip("-")
        return f"{filename}.md"

    def _get_entry_path(self, category: str, filename: str) -> Path:
        """Get full path for entry file."""
        return self.kb_path / "categories" / category / filename

    def _get_template_path(self, entry_type: str) -> Path:
        """Get path to template file."""
        template_mapping = {
            "standard": "entry-template.md",
            "troubleshooting": "troubleshooting-template.md",
            "pattern": "pattern-template.md",
        }

        template_filename = template_mapping.get(entry_type, "entry-template.md")
        return self.kb_path / "templates" / template_filename

    def _create_entry_from_template(
        self, template_path: Path, entry_data: dict[str, Any]
    ) -> str:
        """Create entry content from template."""
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        template_content = template_path.read_text(encoding="utf-8")

        # Replace template placeholders
        replacements = {
            "Your Entry Title Here": entry_data["title"],
            "Fixing [Specific Problem/Error]": entry_data["title"],
            "[Pattern Name] Pattern": entry_data["title"],
            "category-name": entry_data["category"],
            '["tag1", "tag2", "tag3"]': str(entry_data["tags"]),
            "beginner": entry_data["difficulty"],
            "2025-08-14": entry_data["last_updated"],
            '["Tom Mathews"]': str(entry_data["contributors"]),
        }

        content = template_content
        for old, new in replacements.items():
            content = content.replace(old, new)

        return content

    def _validate_entry_content(self, content: str) -> list[str]:
        """Validate entry content and return warnings."""
        warnings = []

        try:
            # Split into frontmatter and body
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    import yaml

                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2]

                    # Validate frontmatter
                    fm_errors = validate_frontmatter(frontmatter)
                    warnings.extend(fm_errors)

                    # Validate content
                    content_warnings = validate_markdown_content(body)
                    warnings.extend(content_warnings)

                    # Validate code examples
                    code_errors = validate_code_examples(body)
                    warnings.extend(code_errors)

        except Exception as e:
            warnings.append(f"Validation error: {e}")

        return warnings

    def _open_in_editor(self, file_path: Path) -> None:
        """Open file in configured editor."""
        if not self.config.editor_command:
            return

        try:
            import subprocess

            command = self.config.editor_command.replace("{file}", str(file_path))
            subprocess.run(command, shell=True)
        except Exception as e:
            logger.warning(f"Could not open editor: {e}")

    def get_contribution_stats(self) -> dict[str, Any]:
        """Get contribution statistics."""
        return {
            **self.stats,
            "existing_categories": len(self.existing_categories),
            "existing_tags": len(self.existing_tags),
            "kb_path": str(self.kb_path),
        }


def create_contributor(
    kb_path: Path, default_contributor: str = "Unknown", **kwargs
) -> KnowledgeBaseContributor:
    """
    Factory function to create a knowledge base contributor.

    Args:
        kb_path: Path to knowledge base directory
        default_contributor: Default contributor name
        **kwargs: Additional configuration options

    Returns:
        Configured contributor instance
    """
    config = ContributionConfig(default_contributor=default_contributor, **kwargs)
    return KnowledgeBaseContributor(kb_path, config)


def main():
    """
    Command-line interface for the contributor tools.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Base Contributor Tools")
    parser.add_argument("kb_path", help="Path to knowledge base directory")
    parser.add_argument("--contributor", default="Unknown", help="Contributor name")
    parser.add_argument(
        "--create", action="store_true", help="Create new entry interactively"
    )
    parser.add_argument("--title", help="Entry title (for non-interactive creation)")
    parser.add_argument("--category", help="Entry category")
    parser.add_argument("--tags", help="Entry tags (comma-separated)")
    parser.add_argument("--difficulty", default="intermediate", help="Entry difficulty")
    parser.add_argument("--type", default="standard", help="Entry type")
    parser.add_argument("--editor", help="Editor command to open files")

    args = parser.parse_args()

    # Create contributor
    config = ContributionConfig(
        default_contributor=args.contributor, editor_command=args.editor
    )
    contributor = KnowledgeBaseContributor(Path(args.kb_path), config)

    try:
        if args.create or not args.title:
            # Interactive creation
            file_path = contributor.create_entry_interactive()
            if file_path:
                print(f"✅ Entry created successfully: {file_path}")
            else:
                print("❌ Entry creation cancelled or failed")
        else:
            # Non-interactive creation
            tags = args.tags.split(",") if args.tags else []
            file_path = contributor.create_entry_from_template(
                title=args.title,
                category=args.category or "general",
                tags=tags,
                difficulty=args.difficulty,
                entry_type=args.type,
                contributor=args.contributor,
            )
            print(f"✅ Entry created: {file_path}")

        # Show statistics
        stats = contributor.get_contribution_stats()
        print("\n📊 Contribution Statistics:")
        print(f"   Entries created: {stats['entries_created']}")
        print(f"   Suggestions provided: {stats['suggestions_provided']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

if __name__ == "__main__":
    exit(main())
