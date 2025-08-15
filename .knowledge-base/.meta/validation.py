"""
Knowledge Base Entry Validation and Parsing

This module provides comprehensive validation and parsing functionality for
knowledge base entries, including frontmatter validation, content analysis,
and consistency checking.
"""

import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class ValidationError(Exception):
    """Raised when knowledge base entry validation fails."""

    pass


class ParseError(Exception):
    """Raised when parsing knowledge base entry fails."""

    pass


def validate_frontmatter(frontmatter: dict[str, Any]) -> list[str]:
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
    for field in required_fields:
        if field not in frontmatter:
            errors.append(f"Missing required field: {field}")
        elif not frontmatter[field]:
            errors.append(f"Field cannot be empty: {field}")

    # Validate title
    if "title" in frontmatter:
        title = frontmatter["title"]
        if not isinstance(title, str):
            errors.append("Title must be a string")
        elif len(title.strip()) == 0:
            errors.append("Title cannot be empty")
        elif len(title) > 200:
            errors.append("Title is too long (maximum 200 characters)")

    # Validate category format (lowercase, hyphenated)
    if "category" in frontmatter:
        category = frontmatter["category"]
        if not isinstance(category, str):
            errors.append("Category must be a string")
        elif not re.match(r"^[a-z]+(-[a-z]+)*$", category):
            errors.append(
                "Category must be lowercase with hyphens (e.g., 'apple-silicon')"
            )

    # Validate difficulty
    if "difficulty" in frontmatter:
        difficulty = frontmatter["difficulty"]
        valid_difficulties = {"beginner", "intermediate", "advanced"}
        if difficulty not in valid_difficulties:
            errors.append(
                f"Invalid difficulty '{difficulty}'. Must be one of: {', '.join(valid_difficulties)}"
            )

    # Validate tags
    if "tags" in frontmatter:
        tags = frontmatter["tags"]
        if not isinstance(tags, list):
            errors.append("Tags must be a list")
        elif not tags:
            errors.append("At least one tag is required")
        else:
            for i, tag in enumerate(tags):
                if not isinstance(tag, str):
                    errors.append(f"Tag {i+1} must be a string")
                elif not tag.strip():
                    errors.append(f"Tag {i+1} cannot be empty")
                elif len(tag) > 50:
                    errors.append(f"Tag {i+1} is too long (maximum 50 characters)")

    # Validate contributors
    if "contributors" in frontmatter:
        contributors = frontmatter["contributors"]
        if not isinstance(contributors, list):
            errors.append("Contributors must be a list")
        elif not contributors:
            errors.append("At least one contributor is required")
        else:
            for i, contributor in enumerate(contributors):
                if not isinstance(contributor, str):
                    errors.append(f"Contributor {i+1} must be a string")
                elif not contributor.strip():
                    errors.append(f"Contributor {i+1} cannot be empty")

    # Validate last_updated
    if "last_updated" in frontmatter:
        last_updated = frontmatter["last_updated"]
        if isinstance(last_updated, str):
            try:
                datetime.strptime(last_updated, "%Y-%m-%d")
            except ValueError:
                errors.append("last_updated must be in YYYY-MM-DD format")
        elif not isinstance(last_updated, datetime):
            errors.append(
                "last_updated must be a date string (YYYY-MM-DD) or datetime object"
            )

    # Validate optional usage_count
    if "usage_count" in frontmatter:
        usage_count = frontmatter["usage_count"]
        if not isinstance(usage_count, int) or usage_count < 0:
            errors.append("usage_count must be a non-negative integer")

    return errors


def parse_markdown_file(file_path: Path) -> tuple[dict[str, Any], str]:
    """
    Parse a markdown file with frontmatter.

    Args:
        file_path: Path to the markdown file

    Returns:
        Tuple of (frontmatter_dict, content_body)

    Raises:
        ParseError: If the file cannot be parsed
        FileNotFoundError: If the file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise ParseError(f"Could not read file {file_path}: {e}")

    # Check for frontmatter
    if not content.startswith("---"):
        raise ParseError(
            f"File {file_path} missing frontmatter (must start with '---')"
        )

    try:
        # Split content into frontmatter and body
        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ParseError("Invalid frontmatter format (missing closing '---')")

        frontmatter_text = parts[1].strip()
        body_content = parts[2]

        # Parse YAML frontmatter
        frontmatter = yaml.safe_load(frontmatter_text)
        if frontmatter is None:
            frontmatter = {}

        return frontmatter, body_content

    except yaml.YAMLError as e:
        raise ParseError(f"Invalid YAML in frontmatter: {e}")
    except Exception as e:
        raise ParseError(f"Error parsing file {file_path}: {e}")


def validate_markdown_content(content: str) -> list[str]:
    """
    Validate the markdown content structure and quality.

    Args:
        content: Markdown content to validate

    Returns:
        List of validation warnings/suggestions
    """
    warnings = []
    lines = content.split("\n")

    # Check for required sections
    required_sections = ["Problem/Context", "Solution/Pattern", "Code Example"]
    found_sections = []

    for line in lines:
        if line.startswith("## "):
            section_name = line[3:].strip()
            found_sections.append(section_name)

    for required in required_sections:
        if not any(required.lower() in section.lower() for section in found_sections):
            warnings.append(f"Missing recommended section: {required}")

    # Check for code blocks
    code_blocks = re.findall(r"```(\w+)?\n(.*?)```", content, re.DOTALL)
    if not code_blocks:
        warnings.append("No code examples found - consider adding practical examples")

    # Check for links
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
    for link_text, link_url in links:
        if link_url.startswith("http"):
            continue  # External links are fine
        elif link_url.startswith("../"):
            # Internal knowledge base link - could validate existence
            continue
        else:
            warnings.append(f"Potentially invalid link: {link_url}")

    # Check content length
    if len(content.strip()) < 200:
        warnings.append("Content seems quite short - consider adding more detail")

    return warnings


def validate_code_examples(content: str) -> list[str]:
    """
    Validate code examples in markdown content.

    Args:
        content: Markdown content containing code blocks

    Returns:
        List of validation errors for code examples
    """
    errors = []

    # Find all code blocks
    code_blocks = re.findall(r"```(\w+)?\n(.*?)```", content, re.DOTALL)

    for i, (language, code) in enumerate(code_blocks):
        block_num = i + 1

        if not language:
            errors.append(f"Code block {block_num}: Missing language specification")
            continue

        # Validate Python code syntax
        if language.lower() == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append(f"Code block {block_num}: Python syntax error - {e}")

        # Check for common issues
        if not code.strip():
            errors.append(f"Code block {block_num}: Empty code block")

        # Check for placeholder content
        if "TODO" in code or "FIXME" in code or "..." in code:
            errors.append(
                f"Code block {block_num}: Contains placeholder content (TODO/FIXME/...)"
            )

    return errors


def validate_entry_consistency(
    entry_data: dict[str, Any], file_path: Path
) -> list[str]:
    """
    Validate consistency between entry metadata and file location.

    Args:
        entry_data: Parsed entry frontmatter
        file_path: Path to the entry file

    Returns:
        List of consistency warnings
    """
    warnings = []

    # Check if file is in correct category directory
    if "category" in entry_data:
        category = entry_data["category"]
        expected_category_path = f"categories/{category}/"

        if expected_category_path not in str(file_path):
            warnings.append(
                f"File location doesn't match category '{category}' - consider moving to categories/{category}/"
            )

    # Check filename conventions
    filename = file_path.stem
    if "title" in entry_data:
        title = entry_data["title"]
        # Convert title to expected filename format
        expected_filename = re.sub(r"[^\w\s-]", "", title.lower())
        expected_filename = re.sub(r"[-\s]+", "-", expected_filename)

        if filename != expected_filename:
            warnings.append(
                f"Filename '{filename}' doesn't match title - consider renaming to '{expected_filename}.md'"
            )

    return warnings


def validate_knowledge_base_structure(kb_path: Path) -> dict[str, list[str]]:
    """
    Validate the overall knowledge base directory structure.

    Args:
        kb_path: Path to the .knowledge-base directory

    Returns:
        Dictionary with validation results for different aspects
    """
    results = {
        "structure_errors": [],
        "missing_files": [],
        "entry_errors": [],
        "warnings": [],
    }

    if not kb_path.exists():
        results["structure_errors"].append(
            f"Knowledge base directory not found: {kb_path}"
        )
        return results

    # Check required directories
    required_dirs = ["categories", "patterns", "templates", ".meta"]
    for dir_name in required_dirs:
        dir_path = kb_path / dir_name
        if not dir_path.exists():
            results["missing_files"].append(f"Missing required directory: {dir_name}")

    # Check required files
    required_files = ["README.md", ".meta/contribution-guide.md"]
    for file_path in required_files:
        full_path = kb_path / file_path
        if not full_path.exists():
            results["missing_files"].append(f"Missing required file: {file_path}")

    # Validate all entries
    for category_dir in (kb_path / "categories").glob("*"):
        if category_dir.is_dir():
            for entry_file in category_dir.glob("*.md"):
                if entry_file.name == "README.md":
                    continue

                try:
                    frontmatter, content = parse_markdown_file(entry_file)

                    # Validate frontmatter
                    fm_errors = validate_frontmatter(frontmatter)
                    if fm_errors:
                        results["entry_errors"].extend(
                            [f"{entry_file.name}: {error}" for error in fm_errors]
                        )

                    # Validate content
                    content_warnings = validate_markdown_content(content)
                    if content_warnings:
                        results["warnings"].extend(
                            [
                                f"{entry_file.name}: {warning}"
                                for warning in content_warnings
                            ]
                        )

                    # Validate code examples
                    code_errors = validate_code_examples(content)
                    if code_errors:
                        results["entry_errors"].extend(
                            [f"{entry_file.name}: {error}" for error in code_errors]
                        )

                    # Validate consistency
                    consistency_warnings = validate_entry_consistency(
                        frontmatter, entry_file
                    )
                    if consistency_warnings:
                        results["warnings"].extend(
                            [
                                f"{entry_file.name}: {warning}"
                                for warning in consistency_warnings
                            ]
                        )

                except Exception as e:
                    results["entry_errors"].append(
                        f"{entry_file.name}: Could not validate - {e}"
                    )

    return results


def generate_validation_report(kb_path: Path, output_file: Path | None = None) -> str:
    """
    Generate a comprehensive validation report for the knowledge base.

    Args:
        kb_path: Path to the .knowledge-base directory
        output_file: Optional path to save the report

    Returns:
        Validation report as a string
    """
    results = validate_knowledge_base_structure(kb_path)

    report_lines = [
        "# Knowledge Base Validation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Knowledge Base: {kb_path}",
        "",
    ]

    # Structure errors
    if results["structure_errors"]:
        report_lines.extend(["## ❌ Structure Errors", ""])
        for error in results["structure_errors"]:
            report_lines.append(f"- {error}")
        report_lines.append("")

    # Missing files
    if results["missing_files"]:
        report_lines.extend(["## 📁 Missing Files/Directories", ""])
        for missing in results["missing_files"]:
            report_lines.append(f"- {missing}")
        report_lines.append("")

    # Entry errors
    if results["entry_errors"]:
        report_lines.extend(["## 🐛 Entry Errors", ""])
        for error in results["entry_errors"]:
            report_lines.append(f"- {error}")
        report_lines.append("")

    # Warnings
    if results["warnings"]:
        report_lines.extend(["## ⚠️ Warnings and Suggestions", ""])
        for warning in results["warnings"]:
            report_lines.append(f"- {warning}")
        report_lines.append("")

    # Summary
    total_issues = (
        len(results["structure_errors"])
        + len(results["missing_files"])
        + len(results["entry_errors"])
    )
    if total_issues == 0:
        report_lines.extend(
            [
                "## ✅ Summary",
                "",
                "No critical issues found! The knowledge base structure is valid.",
                f"Found {len(results['warnings'])} suggestions for improvement.",
            ]
        )
    else:
        report_lines.extend(
            [
                "## 📊 Summary",
                "",
                f"Found {total_issues} issues that need attention:",
                f"- Structure errors: {len(results['structure_errors'])}",
                f"- Missing files: {len(results['missing_files'])}",
                f"- Entry errors: {len(results['entry_errors'])}",
                f"- Warnings: {len(results['warnings'])}",
            ]
        )

    report = "\n".join(report_lines)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report, encoding="utf-8")

    return report
