"""
Knowledge Base Quality Assurance Tools

This module provides automated tools for maintaining knowledge base quality,
including content validation, link checking, consistency verification,
and automated quality improvements.
"""

import ast
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml
from indexer import KnowledgeBaseIndexer
from models import KnowledgeBaseEntry, KnowledgeBaseIndex
from validation import (
    validate_code_examples,
    validate_frontmatter,
    validate_markdown_content,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """
    Represents a quality issue found in the knowledge base.

    Attributes:
        issue_type: Type of issue (broken_link, validation_error, etc.)
        severity: Severity level (low, medium, high, critical)
        entry_path: Path to the affected entry
        entry_title: Title of the affected entry
        description: Description of the issue
        location: Specific location within the entry (line number, section)
        suggested_fix: Suggested fix for the issue
        auto_fixable: Whether this issue can be automatically fixed
    """

    issue_type: str
    severity: str
    entry_path: Path
    entry_title: str
    description: str
    location: str | None = None
    suggested_fix: str | None = None
    auto_fixable: bool = False


@dataclass
class QualityReport:
    """
    Comprehensive quality assessment report.

    Attributes:
        generated_at: When the report was generated
        total_entries_checked: Number of entries analyzed
        issues_found: List of all issues found
        issues_by_type: Issues grouped by type
        issues_by_severity: Issues grouped by severity
        quality_score: Overall quality score (0-100)
        recommendations: Quality improvement recommendations
        auto_fix_summary: Summary of issues that can be auto-fixed
    """

    generated_at: datetime
    total_entries_checked: int
    issues_found: list[QualityIssue]
    issues_by_type: dict[str, list[QualityIssue]]
    issues_by_severity: dict[str, list[QualityIssue]]
    quality_score: float
    recommendations: list[str]
    auto_fix_summary: dict[str, int]


class KnowledgeBaseQualityAssurance:
    """
    Comprehensive quality assurance system for knowledge base maintenance.

    This class provides automated tools for validating content quality,
    checking links, ensuring consistency, and suggesting improvements.
    """

    def __init__(
        self,
        kb_path: Path,
        check_external_links: bool = False,
        auto_fix_enabled: bool = False,
    ):
        self.kb_path = Path(kb_path)
        self.check_external_links = check_external_links
        self.auto_fix_enabled = auto_fix_enabled

        # Load knowledge base index
        self.indexer = KnowledgeBaseIndexer(self.kb_path)
        self.index = self.indexer.load_index_from_file()
        if not self.index:
            logger.info("Building knowledge base index for quality assurance")
            self.index = self.indexer.build_index()

        # Quality thresholds
        self.quality_thresholds = {
            "min_content_length": 200,
            "max_title_length": 100,
            "min_tags": 1,
            "max_tags": 10,
            "stale_days": 180,
            "max_broken_links": 0,
        }

        # Statistics
        self.stats = {
            "total_checks_performed": 0,
            "issues_found": 0,
            "issues_auto_fixed": 0,
            "entries_processed": 0,
            "last_check_date": None,
        }

    def run_comprehensive_quality_check(self) -> QualityReport:
        """
        Run a comprehensive quality check on the entire knowledge base.

        Returns:
            QualityReport with all findings and recommendations
        """
        logger.info("Starting comprehensive quality check")

        issues_found = []
        entries_checked = 0

        try:
            # Check all entries
            for entry in self.index.entries:
                entry_issues = self._check_entry_quality(entry)
                issues_found.extend(entry_issues)
                entries_checked += 1
                self.stats["entries_processed"] += 1

            # Check cross-references and consistency
            consistency_issues = self._check_consistency()
            issues_found.extend(consistency_issues)

            # Check for orphaned files
            orphan_issues = self._check_orphaned_files()
            issues_found.extend(orphan_issues)

            # Group issues
            issues_by_type = self._group_issues_by_type(issues_found)
            issues_by_severity = self._group_issues_by_severity(issues_found)

            # Calculate quality score
            quality_score = self._calculate_quality_score(issues_found, entries_checked)

            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                issues_found, entries_checked
            )

            # Auto-fix summary
            auto_fix_summary = self._generate_auto_fix_summary(issues_found)

            # Update statistics
            self.stats["total_checks_performed"] += 1
            self.stats["issues_found"] += len(issues_found)
            self.stats["last_check_date"] = datetime.now().isoformat()

            return QualityReport(
                generated_at=datetime.now(),
                total_entries_checked=entries_checked,
                issues_found=issues_found,
                issues_by_type=issues_by_type,
                issues_by_severity=issues_by_severity,
                quality_score=quality_score,
                recommendations=recommendations,
                auto_fix_summary=auto_fix_summary,
            )

        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return QualityReport(
                generated_at=datetime.now(),
                total_entries_checked=entries_checked,
                issues_found=issues_found,
                issues_by_type={},
                issues_by_severity={},
                quality_score=0.0,
                recommendations=[],
                auto_fix_summary={},
            )

    def _check_entry_quality(self, entry: KnowledgeBaseEntry) -> list[QualityIssue]:
        """Check quality issues for a single entry."""
        issues = []

        try:
            # Check frontmatter validation
            content = entry.get_content()
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2]

                    # Validate frontmatter
                    fm_errors = validate_frontmatter(frontmatter)
                    for error in fm_errors:
                        issues.append(
                            QualityIssue(
                                issue_type="validation_error",
                                severity="medium",
                                entry_path=entry.content_path,
                                entry_title=entry.title,
                                description=f"Frontmatter validation: {error}",
                                location="frontmatter",
                                auto_fixable=False,
                            )
                        )

                    # Check content quality
                    content_issues = self._check_content_quality(entry, body)
                    issues.extend(content_issues)

                    # Check links
                    link_issues = self._check_entry_links(entry, body)
                    issues.extend(link_issues)

                    # Check code examples
                    code_issues = self._check_code_quality(entry, body)
                    issues.extend(code_issues)

                    # Check metadata quality
                    metadata_issues = self._check_metadata_quality(entry, frontmatter)
                    issues.extend(metadata_issues)

        except Exception as e:
            issues.append(
                QualityIssue(
                    issue_type="parse_error",
                    severity="high",
                    entry_path=entry.content_path,
                    entry_title=entry.title,
                    description=f"Failed to parse entry: {e}",
                    auto_fixable=False,
                )
            )

        return issues

    def _check_content_quality(
        self, entry: KnowledgeBaseEntry, content: str
    ) -> list[QualityIssue]:
        """Check content quality issues."""
        issues = []

        # Check content length
        content_length = len(content.strip())
        if content_length < self.quality_thresholds["min_content_length"]:
            issues.append(
                QualityIssue(
                    issue_type="content_too_short",
                    severity="medium",
                    entry_path=entry.content_path,
                    entry_title=entry.title,
                    description=f"Content too short: {content_length} characters (minimum: {self.quality_thresholds['min_content_length']})",
                    suggested_fix="Add more detailed explanations, examples, or sections",
                    auto_fixable=False,
                )
            )

        # Check for required sections
        required_sections = ["Problem/Context", "Solution/Pattern", "Code Example"]
        found_sections = re.findall(r"^## (.+)", content, re.MULTILINE)

        for required_section in required_sections:
            if not any(
                required_section.lower() in section.lower()
                for section in found_sections
            ):
                issues.append(
                    QualityIssue(
                        issue_type="missing_section",
                        severity="low",
                        entry_path=entry.content_path,
                        entry_title=entry.title,
                        description=f"Missing recommended section: {required_section}",
                        suggested_fix=f"Add a '{required_section}' section to improve structure",
                        auto_fixable=False,
                    )
                )

        # Check for placeholder content
        placeholders = ["TODO", "FIXME", "TBD", "[Content to be added]", "Lorem ipsum"]
        for placeholder in placeholders:
            if placeholder in content:
                issues.append(
                    QualityIssue(
                        issue_type="placeholder_content",
                        severity="high",
                        entry_path=entry.content_path,
                        entry_title=entry.title,
                        description=f"Contains placeholder content: {placeholder}",
                        suggested_fix="Replace placeholder with actual content",
                        auto_fixable=False,
                    )
                )

        # Check for spelling/grammar issues (basic)
        common_typos = {
            "teh": "the",
            "adn": "and",
            "taht": "that",
            "thier": "their",
            "recieve": "receive",
        }

        for typo, correction in common_typos.items():
            if re.search(r"\b" + typo + r"\b", content, re.IGNORECASE):
                issues.append(
                    QualityIssue(
                        issue_type="spelling_error",
                        severity="low",
                        entry_path=entry.content_path,
                        entry_title=entry.title,
                        description=f"Possible spelling error: '{typo}' should be '{correction}'",
                        suggested_fix=f"Replace '{typo}' with '{correction}'",
                        auto_fixable=True,
                    )
                )

        return issues

    def _check_entry_links(
        self, entry: KnowledgeBaseEntry, content: str
    ) -> list[QualityIssue]:
        """Check link quality and validity."""
        issues = []

        # Find all markdown links
        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)

        for link_text, link_url in links:
            # Check internal links
            if not link_url.startswith("http") and not link_url.startswith("#"):
                # Resolve relative path
                if link_url.startswith("../"):
                    target_path = (entry.content_path.parent / link_url).resolve()
                else:
                    target_path = (self.kb_path / link_url).resolve()

                if not target_path.exists():
                    issues.append(
                        QualityIssue(
                            issue_type="broken_internal_link",
                            severity="medium",
                            entry_path=entry.content_path,
                            entry_title=entry.title,
                            description=f"Broken internal link: {link_text} -> {link_url}",
                            location=f"Link: {link_text}",
                            suggested_fix="Update link path or create missing target file",
                            auto_fixable=False,
                        )
                    )

            # Check external links (if enabled)
            elif self.check_external_links and link_url.startswith("http"):
                if not self._check_external_link(link_url):
                    issues.append(
                        QualityIssue(
                            issue_type="broken_external_link",
                            severity="low",
                            entry_path=entry.content_path,
                            entry_title=entry.title,
                            description=f"Broken external link: {link_text} -> {link_url}",
                            location=f"Link: {link_text}",
                            suggested_fix="Update or remove broken external link",
                            auto_fixable=False,
                        )
                    )

        return issues

    def _check_code_quality(
        self, entry: KnowledgeBaseEntry, content: str
    ) -> list[QualityIssue]:
        """Check code example quality."""
        issues = []

        # Find code blocks
        code_blocks = re.findall(r"```(\w+)?\n(.*?)```", content, re.DOTALL)

        if not code_blocks:
            issues.append(
                QualityIssue(
                    issue_type="missing_code_examples",
                    severity="low",
                    entry_path=entry.content_path,
                    entry_title=entry.title,
                    description="No code examples found",
                    suggested_fix="Add practical code examples to illustrate concepts",
                    auto_fixable=False,
                )
            )

        for i, (language, code) in enumerate(code_blocks):
            block_num = i + 1

            # Check for language specification
            if not language:
                issues.append(
                    QualityIssue(
                        issue_type="missing_code_language",
                        severity="low",
                        entry_path=entry.content_path,
                        entry_title=entry.title,
                        description=f"Code block {block_num} missing language specification",
                        location=f"Code block {block_num}",
                        suggested_fix="Add language identifier (e.g., ```python)",
                        auto_fixable=True,
                    )
                )

            # Check Python syntax
            if language and language.lower() == "python":
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    issues.append(
                        QualityIssue(
                            issue_type="python_syntax_error",
                            severity="high",
                            entry_path=entry.content_path,
                            entry_title=entry.title,
                            description=f"Python syntax error in code block {block_num}: {e}",
                            location=f"Code block {block_num}, line {e.lineno}",
                            suggested_fix="Fix Python syntax error",
                            auto_fixable=False,
                        )
                    )

            # Check for placeholder code
            if (
                "# TODO" in code
                or "# FIXME" in code
                or "pass" in code.strip().split("\n")[-1]
            ):
                issues.append(
                    QualityIssue(
                        issue_type="incomplete_code",
                        severity="medium",
                        entry_path=entry.content_path,
                        entry_title=entry.title,
                        description=f"Code block {block_num} appears incomplete",
                        location=f"Code block {block_num}",
                        suggested_fix="Complete the code example with working implementation",
                        auto_fixable=False,
                    )
                )

        return issues

    def _check_metadata_quality(
        self, entry: KnowledgeBaseEntry, frontmatter: dict[str, Any]
    ) -> list[QualityIssue]:
        """Check metadata quality."""
        issues = []

        # Check title length
        title = frontmatter.get("title", "")
        if len(title) > self.quality_thresholds["max_title_length"]:
            issues.append(
                QualityIssue(
                    issue_type="title_too_long",
                    severity="low",
                    entry_path=entry.content_path,
                    entry_title=entry.title,
                    description=f"Title too long: {len(title)} characters (max: {self.quality_thresholds['max_title_length']})",
                    suggested_fix="Shorten the title while keeping it descriptive",
                    auto_fixable=False,
                )
            )

        # Check tags
        tags = frontmatter.get("tags", [])
        if len(tags) < self.quality_thresholds["min_tags"]:
            issues.append(
                QualityIssue(
                    issue_type="insufficient_tags",
                    severity="low",
                    entry_path=entry.content_path,
                    entry_title=entry.title,
                    description=f"Too few tags: {len(tags)} (minimum: {self.quality_thresholds['min_tags']})",
                    suggested_fix="Add more relevant tags for better discoverability",
                    auto_fixable=False,
                )
            )
        elif len(tags) > self.quality_thresholds["max_tags"]:
            issues.append(
                QualityIssue(
                    issue_type="too_many_tags",
                    severity="low",
                    entry_path=entry.content_path,
                    entry_title=entry.title,
                    description=f"Too many tags: {len(tags)} (maximum: {self.quality_thresholds['max_tags']})",
                    suggested_fix="Remove less relevant tags to focus on key topics",
                    auto_fixable=False,
                )
            )

        # Check for stale content
        last_updated = frontmatter.get("last_updated")
        if last_updated:
            try:
                if isinstance(last_updated, str):
                    update_date = datetime.strptime(last_updated, "%Y-%m-%d")
                else:
                    update_date = last_updated

                days_old = (datetime.now() - update_date).days
                if days_old > self.quality_thresholds["stale_days"]:
                    issues.append(
                        QualityIssue(
                            issue_type="stale_content",
                            severity="medium",
                            entry_path=entry.content_path,
                            entry_title=entry.title,
                            description=f"Content is {days_old} days old (threshold: {self.quality_thresholds['stale_days']} days)",
                            suggested_fix="Review and update content to ensure accuracy",
                            auto_fixable=False,
                        )
                    )
            except (ValueError, TypeError):
                issues.append(
                    QualityIssue(
                        issue_type="invalid_date_format",
                        severity="low",
                        entry_path=entry.content_path,
                        entry_title=entry.title,
                        description="Invalid date format in last_updated field",
                        suggested_fix="Use YYYY-MM-DD format for dates",
                        auto_fixable=True,
                    )
                )

        return issues

    def _check_external_link(self, url: str) -> bool:
        """Check if external link is accessible."""
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code < 400
        except Exception:
            return False

    def _check_consistency(self) -> list[QualityIssue]:
        """Check for consistency issues across entries."""
        issues = []

        # Check for duplicate titles
        title_counts = {}
        for entry in self.index.entries:
            title_lower = entry.title.lower()
            if title_lower in title_counts:
                title_counts[title_lower].append(entry)
            else:
                title_counts[title_lower] = [entry]

        for title, entries in title_counts.items():
            if len(entries) > 1:
                for entry in entries:
                    issues.append(
                        QualityIssue(
                            issue_type="duplicate_title",
                            severity="medium",
                            entry_path=entry.content_path,
                            entry_title=entry.title,
                            description=f"Duplicate title found: '{entry.title}'",
                            suggested_fix="Rename one of the entries to have a unique title",
                            auto_fixable=False,
                        )
                    )

        # Check category consistency
        category_variations = {}
        for entry in self.index.entries:
            category_lower = entry.category.lower()
            if category_lower in category_variations:
                category_variations[category_lower].add(entry.category)
            else:
                category_variations[category_lower] = {entry.category}

        for category_lower, variations in category_variations.items():
            if len(variations) > 1:
                # Multiple case variations of the same category
                most_common = max(
                    variations,
                    key=lambda cat: sum(
                        1 for e in self.index.entries if e.category == cat
                    ),
                )
                for entry in self.index.entries:
                    if (
                        entry.category.lower() == category_lower
                        and entry.category != most_common
                    ):
                        issues.append(
                            QualityIssue(
                                issue_type="category_case_inconsistency",
                                severity="low",
                                entry_path=entry.content_path,
                                entry_title=entry.title,
                                description=f"Category case inconsistency: '{entry.category}' should be '{most_common}'",
                                suggested_fix=f"Change category to '{most_common}'",
                                auto_fixable=True,
                            )
                        )

        return issues

    def _check_orphaned_files(self) -> list[QualityIssue]:
        """Check for orphaned markdown files."""
        issues = []

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
        orphaned_files = all_md_files - indexed_files

        for orphaned_file in orphaned_files:
            issues.append(
                QualityIssue(
                    issue_type="orphaned_file",
                    severity="medium",
                    entry_path=orphaned_file,
                    entry_title=orphaned_file.stem,
                    description=f"Orphaned file not in index: {orphaned_file.name}",
                    suggested_fix="Either fix the file format to be indexed or remove it",
                    auto_fixable=False,
                )
            )

        return issues

    def _group_issues_by_type(
        self, issues: list[QualityIssue]
    ) -> dict[str, list[QualityIssue]]:
        """Group issues by type."""
        grouped = {}
        for issue in issues:
            if issue.issue_type not in grouped:
                grouped[issue.issue_type] = []
            grouped[issue.issue_type].append(issue)
        return grouped

    def _group_issues_by_severity(
        self, issues: list[QualityIssue]
    ) -> dict[str, list[QualityIssue]]:
        """Group issues by severity."""
        grouped = {}
        for issue in issues:
            if issue.severity not in grouped:
                grouped[issue.severity] = []
            grouped[issue.severity].append(issue)
        return grouped

    def _calculate_quality_score(
        self, issues: list[QualityIssue], total_entries: int
    ) -> float:
        """Calculate overall quality score (0-100)."""
        if total_entries == 0:
            return 100.0

        # Weight issues by severity
        severity_weights = {"low": 1, "medium": 3, "high": 5, "critical": 10}
        total_penalty = sum(severity_weights.get(issue.severity, 1) for issue in issues)

        # Calculate score (higher penalty = lower score)
        max_possible_penalty = total_entries * 10  # Assume worst case
        score = max(0, 100 - (total_penalty / max_possible_penalty * 100))

        return round(score, 1)

    def _generate_quality_recommendations(
        self, issues: list[QualityIssue], total_entries: int
    ) -> list[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        # Group issues by type for analysis
        issues_by_type = self._group_issues_by_type(issues)

        # High-priority recommendations
        if "broken_internal_link" in issues_by_type:
            count = len(issues_by_type["broken_internal_link"])
            recommendations.append(
                f"Fix {count} broken internal links to improve navigation"
            )

        if "python_syntax_error" in issues_by_type:
            count = len(issues_by_type["python_syntax_error"])
            recommendations.append(f"Fix {count} Python syntax errors in code examples")

        if "placeholder_content" in issues_by_type:
            count = len(issues_by_type["placeholder_content"])
            recommendations.append(f"Complete {count} entries with placeholder content")

        # Content quality recommendations
        if "content_too_short" in issues_by_type:
            count = len(issues_by_type["content_too_short"])
            recommendations.append(f"Expand {count} entries with insufficient content")

        if "missing_code_examples" in issues_by_type:
            count = len(issues_by_type["missing_code_examples"])
            recommendations.append(f"Add code examples to {count} entries")

        # Consistency recommendations
        if "duplicate_title" in issues_by_type:
            count = len(issues_by_type["duplicate_title"])
            recommendations.append(f"Resolve {count} duplicate titles")

        # Maintenance recommendations
        if "stale_content" in issues_by_type:
            count = len(issues_by_type["stale_content"])
            recommendations.append(f"Review and update {count} stale entries")

        # General recommendations
        total_issues = len(issues)
        if total_issues > total_entries * 0.5:  # More than 0.5 issues per entry
            recommendations.append(
                "Consider implementing automated quality checks in CI/CD"
            )

        if not recommendations:
            recommendations.append(
                "Knowledge base quality is good - maintain current standards"
            )

        return recommendations

    def _generate_auto_fix_summary(self, issues: list[QualityIssue]) -> dict[str, int]:
        """Generate summary of auto-fixable issues."""
        auto_fixable = [issue for issue in issues if issue.auto_fixable]

        summary = {}
        for issue in auto_fixable:
            if issue.issue_type not in summary:
                summary[issue.issue_type] = 0
            summary[issue.issue_type] += 1

        return summary

    def auto_fix_issues(self, issues: list[QualityIssue]) -> dict[str, int]:
        """
        Automatically fix issues that can be auto-fixed.

        Args:
            issues: List of issues to attempt to fix

        Returns:
            Dictionary with counts of fixed issues by type
        """
        if not self.auto_fix_enabled:
            logger.warning("Auto-fix is disabled")
            return {}

        fixed_counts = {}

        for issue in issues:
            if not issue.auto_fixable:
                continue

            try:
                if issue.issue_type == "spelling_error":
                    self._fix_spelling_error(issue)
                elif issue.issue_type == "missing_code_language":
                    self._fix_missing_code_language(issue)
                elif issue.issue_type == "category_case_inconsistency":
                    self._fix_category_case(issue)
                elif issue.issue_type == "invalid_date_format":
                    self._fix_date_format(issue)

                # Update count
                if issue.issue_type not in fixed_counts:
                    fixed_counts[issue.issue_type] = 0
                fixed_counts[issue.issue_type] += 1
                self.stats["issues_auto_fixed"] += 1

            except Exception as e:
                logger.error(
                    f"Failed to auto-fix issue {issue.issue_type} in {issue.entry_title}: {e}"
                )

        return fixed_counts

    def _fix_spelling_error(self, issue: QualityIssue) -> None:
        """Fix spelling errors in content."""
        try:
            content = issue.entry_path.read_text(encoding="utf-8")

            # Extract typo and correction from description
            # Format: "Possible spelling error: 'typo' should be 'correction'"
            match = re.search(r"'([^']+)' should be '([^']+)'", issue.description)
            if match:
                typo, correction = match.groups()
                # Use word boundaries to avoid partial replacements
                pattern = r"\b" + re.escape(typo) + r"\b"
                content = re.sub(pattern, correction, content, flags=re.IGNORECASE)

                issue.entry_path.write_text(content, encoding="utf-8")
                logger.info(
                    f"Fixed spelling error '{typo}' -> '{correction}' in {issue.entry_title}"
                )

        except Exception as e:
            logger.error(f"Failed to fix spelling error: {e}")
            raise

    def _fix_missing_code_language(self, issue: QualityIssue) -> None:
        """Add language specification to code blocks."""
        try:
            content = issue.entry_path.read_text(encoding="utf-8")

            # Find code blocks without language specification
            def add_language(match):
                code_content = match.group(1)
                # Try to detect language from content
                if (
                    "import " in code_content
                    or "def " in code_content
                    or "class " in code_content
                ):
                    return f"```python\n{code_content}```"
                elif (
                    "function " in code_content
                    or "const " in code_content
                    or "let " in code_content
                ):
                    return f"```javascript\n{code_content}```"
                elif "#include" in code_content or "int main" in code_content:
                    return f"```c\n{code_content}```"
                else:
                    return f"```text\n{code_content}```"

            # Replace code blocks without language
            content = re.sub(r"```\n(.*?)```", add_language, content, flags=re.DOTALL)

            issue.entry_path.write_text(content, encoding="utf-8")
            logger.info(
                f"Added language specification to code blocks in {issue.entry_title}"
            )

        except Exception as e:
            logger.error(f"Failed to fix missing code language: {e}")
            raise

    def _fix_category_case(self, issue: QualityIssue) -> None:
        """Fix category case inconsistency."""
        try:
            content = issue.entry_path.read_text(encoding="utf-8")

            # Extract correct category from description
            match = re.search(r"should be '([^']+)'", issue.description)
            if match:
                correct_category = match.group(1)

                # Update frontmatter
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter = yaml.safe_load(parts[1])
                        frontmatter["category"] = correct_category

                        # Reconstruct content
                        new_frontmatter = yaml.dump(
                            frontmatter, default_flow_style=False
                        )
                        content = f"---\n{new_frontmatter}---{parts[2]}"

                        issue.entry_path.write_text(content, encoding="utf-8")
                        logger.info(
                            f"Fixed category case to '{correct_category}' in {issue.entry_title}"
                        )

        except Exception as e:
            logger.error(f"Failed to fix category case: {e}")
            raise

    def _fix_date_format(self, issue: QualityIssue) -> None:
        """Fix invalid date format in frontmatter."""
        try:
            content = issue.entry_path.read_text(encoding="utf-8")

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])

                    # Fix date fields
                    date_fields = ["last_updated", "created_date", "date"]
                    for field in date_fields:
                        if field in frontmatter:
                            date_value = frontmatter[field]
                            if isinstance(date_value, str):
                                # Try to parse and reformat
                                try:
                                    parsed_date = datetime.strptime(
                                        date_value, "%Y-%m-%d"
                                    )
                                    frontmatter[field] = parsed_date.strftime(
                                        "%Y-%m-%d"
                                    )
                                except ValueError:
                                    # Try other common formats
                                    for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
                                        try:
                                            parsed_date = datetime.strptime(
                                                date_value, fmt
                                            )
                                            frontmatter[field] = parsed_date.strftime(
                                                "%Y-%m-%d"
                                            )
                                            break
                                        except ValueError:
                                            continue

                    # Reconstruct content
                    new_frontmatter = yaml.dump(frontmatter, default_flow_style=False)
                    content = f"---\n{new_frontmatter}---{parts[2]}"

                    issue.entry_path.write_text(content, encoding="utf-8")
                    logger.info(f"Fixed date format in {issue.entry_title}")

        except Exception as e:
            logger.error(f"Failed to fix date format: {e}")
            raise

    def generate_quality_report_json(self, output_path: Path | None = None) -> Path:
        """
        Generate a JSON quality report.

        Args:
            output_path: Optional path for the output file

        Returns:
            Path to the generated report file
        """
        if not output_path:
            output_path = self.kb_path / ".meta" / "quality_report.json"

        try:
            report = self.run_comprehensive_quality_check()

            # Convert to serializable format
            report_data = {
                "generated_at": report.generated_at.isoformat(),
                "total_entries_checked": report.total_entries_checked,
                "quality_score": report.quality_score,
                "total_issues": len(report.issues_found),
                "issues_by_severity": {
                    severity: len(issues)
                    for severity, issues in report.issues_by_severity.items()
                },
                "issues_by_type": {
                    issue_type: len(issues)
                    for issue_type, issues in report.issues_by_type.items()
                },
                "issues": [
                    {
                        "type": issue.issue_type,
                        "severity": issue.severity,
                        "entry_title": issue.entry_title,
                        "entry_path": str(issue.entry_path),
                        "description": issue.description,
                        "location": issue.location,
                        "suggested_fix": issue.suggested_fix,
                        "auto_fixable": issue.auto_fixable,
                    }
                    for issue in report.issues_found
                ],
                "recommendations": report.recommendations,
                "auto_fix_summary": report.auto_fix_summary,
                "statistics": self.stats,
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"Generated quality report: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            raise

    def get_statistics(self) -> dict[str, Any]:
        """Get quality assurance statistics."""
        return self.stats.copy()

    def validate_single_entry(self, entry_path: Path) -> list[QualityIssue]:
        """
        Validate a single entry and return issues found.

        Args:
            entry_path: Path to the entry to validate

        Returns:
            List of quality issues found
        """
        try:
            # Find entry in index
            entry = None
            for e in self.index.entries:
                if e.content_path == entry_path:
                    entry = e
                    break

            if not entry:
                return [
                    QualityIssue(
                        issue_type="entry_not_indexed",
                        severity="medium",
                        entry_path=entry_path,
                        entry_title=entry_path.stem,
                        description="Entry not found in knowledge base index",
                        suggested_fix="Rebuild the knowledge base index",
                        auto_fixable=False,
                    )
                ]

            return self._check_entry_quality(entry)

        except Exception as e:
            logger.error(f"Failed to validate entry {entry_path}: {e}")
            return [
                QualityIssue(
                    issue_type="validation_error",
                    severity="high",
                    entry_path=entry_path,
                    entry_title=entry_path.stem,
                    description=f"Validation failed: {e}",
                    auto_fixable=False,
                )
            ]

    def get_quality_trends(self, days_back: int = 30) -> dict[str, Any]:
        """
        Get quality trends over time (placeholder for future implementation).

        Args:
            days_back: Number of days to look back

        Returns:
            Dictionary with trend information
        """
        # This would require storing historical quality data
        # For now, return current statistics
        return {
            "period_days": days_back,
            "current_stats": self.stats,
            "trend_data": "Historical tracking not yet implemented",
            "recommendations": [
                "Implement historical quality tracking",
                "Set up automated quality monitoring",
                "Create quality dashboards",
            ],
        }


# Convenience functions for common operations


def run_quality_check(
    kb_path: Path, check_external_links: bool = False
) -> QualityReport:
    """
    Quick function to run a quality check on the knowledge base.

    Args:
        kb_path: Path to the knowledge base directory
        check_external_links: Whether to check external links

    Returns:
        QualityReport with analysis results
    """
    qa = KnowledgeBaseQualityAssurance(
        kb_path, check_external_links=check_external_links
    )
    return qa.run_comprehensive_quality_check()


def auto_fix_quality_issues(kb_path: Path) -> dict[str, int]:
    """
    Quick function to automatically fix quality issues.

    Args:
        kb_path: Path to the knowledge base directory

    Returns:
        Dictionary with counts of fixed issues by type
    """
    qa = KnowledgeBaseQualityAssurance(kb_path, auto_fix_enabled=True)
    report = qa.run_comprehensive_quality_check()
    return qa.auto_fix_issues(report.issues_found)


def validate_entry(kb_path: Path, entry_path: Path) -> list[QualityIssue]:
    """
    Quick function to validate a single entry.

    Args:
        kb_path: Path to the knowledge base directory
        entry_path: Path to the entry to validate

    Returns:
        List of quality issues found
    """
    qa = KnowledgeBaseQualityAssurance(kb_path)
    return qa.validate_single_entry(entry_path)
