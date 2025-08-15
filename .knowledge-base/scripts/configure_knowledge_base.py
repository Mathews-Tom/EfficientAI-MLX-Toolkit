#!/usr/bin/env python3
"""
Knowledge Base Configuration Script

This script helps configure and customize knowledge base settings,
categories, and behavior.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


class KnowledgeBaseConfigurator:
    """Handles knowledge base configuration management."""

    def __init__(self, kb_path: Path):
        self.kb_path = kb_path
        self.config_path = kb_path / ".meta" / "config.yaml"
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load existing configuration or create default."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "search": {
                "max_results": 50,
                "fuzzy_threshold": 0.6,
                "enable_full_text": True,
            },
            "quality": {
                "max_title_length": 100,
                "min_content_length": 200,
                "required_sections": ["Problem/Context", "Solution"],
                "check_code_syntax": True,
            },
            "analytics": {
                "track_usage": True,
                "track_searches": True,
                "retention_days": 365,
            },
            "categories": [
                "apple-silicon",
                "mlx-framework",
                "performance",
                "troubleshooting",
                "deployment",
                "testing",
            ],
            "difficulty_levels": ["beginner", "intermediate", "advanced"],
            "supported_extensions": [".md", ".markdown"],
            "indexing": {
                "parallel_processing": True,
                "batch_size": 100,
                "exclude_patterns": ["*.tmp", ".*", "__pycache__"],
            },
        }

    def save_config(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"✓ Configuration saved to {self.config_path}")

    def add_category(self, category: str) -> None:
        """Add a new category."""
        if category not in self.config["categories"]:
            self.config["categories"].append(category)
            # Create category directory
            category_dir = self.kb_path / "categories" / category
            category_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Added category: {category}")
        else:
            print(f"Category '{category}' already exists")

    def remove_category(self, category: str) -> None:
        """Remove a category."""
        if category in self.config["categories"]:
            self.config["categories"].remove(category)
            print(f"✓ Removed category: {category}")
            print(f"Note: Directory 'categories/{category}' still exists")
        else:
            print(f"Category '{category}' not found")

    def list_categories(self) -> None:
        """List all categories."""
        print("Current categories:")
        for i, category in enumerate(self.config["categories"], 1):
            print(f"  {i}. {category}")

    def set_search_config(
        self,
        max_results: int | None = None,
        fuzzy_threshold: float | None = None,
        enable_full_text: bool | None = None,
    ) -> None:
        """Update search configuration."""
        if max_results is not None:
            self.config["search"]["max_results"] = max_results
            print(f"✓ Set max_results to {max_results}")

        if fuzzy_threshold is not None:
            self.config["search"]["fuzzy_threshold"] = fuzzy_threshold
            print(f"✓ Set fuzzy_threshold to {fuzzy_threshold}")

        if enable_full_text is not None:
            self.config["search"]["enable_full_text"] = enable_full_text
            print(f"✓ Set enable_full_text to {enable_full_text}")

    def set_quality_config(
        self,
        max_title_length: int | None = None,
        min_content_length: int | None = None,
        check_code_syntax: bool | None = None,
    ) -> None:
        """Update quality configuration."""
        if max_title_length is not None:
            self.config["quality"]["max_title_length"] = max_title_length
            print(f"✓ Set max_title_length to {max_title_length}")

        if min_content_length is not None:
            self.config["quality"]["min_content_length"] = min_content_length
            print(f"✓ Set min_content_length to {min_content_length}")

        if check_code_syntax is not None:
            self.config["quality"]["check_code_syntax"] = check_code_syntax
            print(f"✓ Set check_code_syntax to {check_code_syntax}")

    def show_config(self) -> None:
        """Display current configuration."""
        print("Current Knowledge Base Configuration:")
        print("=" * 40)
        print(yaml.dump(self.config, default_flow_style=False, indent=2))

    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self.config = self._get_default_config()
        print("✓ Configuration reset to defaults")

    def validate_config(self) -> list[str]:
        """Validate configuration and return any issues."""
        issues = []

        # Check required sections
        required_sections = ["search", "quality", "analytics", "categories"]
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")

        # Validate categories exist as directories
        for category in self.config.get("categories", []):
            category_dir = self.kb_path / "categories" / category
            if not category_dir.exists():
                issues.append(f"Category directory missing: {category_dir}")

        # Validate numeric values
        search_config = self.config.get("search", {})
        if "max_results" in search_config and search_config["max_results"] <= 0:
            issues.append("max_results must be positive")

        if "fuzzy_threshold" in search_config:
            threshold = search_config["fuzzy_threshold"]
            if not 0 <= threshold <= 1:
                issues.append("fuzzy_threshold must be between 0 and 1")

        return issues


def main():
    """Main configuration function."""
    parser = argparse.ArgumentParser(description="Configure Knowledge Base settings")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=".",
        help="Path to knowledge base (default: current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Configuration commands")

    # Show config command
    show_parser = subparsers.add_parser("show", help="Show current configuration")

    # Add category command
    add_cat_parser = subparsers.add_parser("add-category", help="Add new category")
    add_cat_parser.add_argument("category", help="Category name to add")

    # Remove category command
    remove_cat_parser = subparsers.add_parser("remove-category", help="Remove category")
    remove_cat_parser.add_argument("category", help="Category name to remove")

    # List categories command
    list_cat_parser = subparsers.add_parser(
        "list-categories", help="List all categories"
    )

    # Search config command
    search_parser = subparsers.add_parser("search", help="Configure search settings")
    search_parser.add_argument("--max-results", type=int, help="Maximum search results")
    search_parser.add_argument(
        "--fuzzy-threshold", type=float, help="Fuzzy matching threshold (0-1)"
    )
    search_parser.add_argument(
        "--enable-full-text", type=bool, help="Enable full-text search"
    )

    # Quality config command
    quality_parser = subparsers.add_parser("quality", help="Configure quality settings")
    quality_parser.add_argument(
        "--max-title-length", type=int, help="Maximum title length"
    )
    quality_parser.add_argument(
        "--min-content-length", type=int, help="Minimum content length"
    )
    quality_parser.add_argument(
        "--check-code-syntax", type=bool, help="Enable code syntax checking"
    )

    # Reset config command
    reset_parser = subparsers.add_parser(
        "reset", help="Reset configuration to defaults"
    )

    # Validate config command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate current configuration"
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

    configurator = KnowledgeBaseConfigurator(kb_path)

    try:
        if args.command == "show":
            configurator.show_config()

        elif args.command == "add-category":
            configurator.add_category(args.category)
            configurator.save_config()

        elif args.command == "remove-category":
            configurator.remove_category(args.category)
            configurator.save_config()

        elif args.command == "list-categories":
            configurator.list_categories()

        elif args.command == "search":
            configurator.set_search_config(
                max_results=args.max_results,
                fuzzy_threshold=args.fuzzy_threshold,
                enable_full_text=args.enable_full_text,
            )
            configurator.save_config()

        elif args.command == "quality":
            configurator.set_quality_config(
                max_title_length=args.max_title_length,
                min_content_length=args.min_content_length,
                check_code_syntax=args.check_code_syntax,
            )
            configurator.save_config()

        elif args.command == "reset":
            configurator.reset_config()
            configurator.save_config()

        elif args.command == "validate":
            issues = configurator.validate_config()
            if issues:
                print("❌ Configuration issues found:")
                for issue in issues:
                    print(f"  - {issue}")
                sys.exit(1)
            else:
                print("✅ Configuration is valid")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
