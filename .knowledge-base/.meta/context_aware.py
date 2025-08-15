"""
Context-Aware Knowledge Base Access

This module provides contextual access to knowledge base entries based on
current development context, including file analysis, error detection,
and workflow-based recommendations.
"""

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cross_reference import CrossReferenceAnalyzer
from models import KnowledgeBaseEntry, KnowledgeBaseIndex
from search import KnowledgeBaseSearch, SearchFilter

logger = logging.getLogger(__name__)


@dataclass
class DevelopmentContext:
    """
    Represents the current development context.

    Attributes:
        current_file: Path to the file being worked on
        file_type: Type of file (python, markdown, yaml, etc.)
        project_root: Root directory of the project
        recent_files: Recently accessed files
        current_errors: Current error messages or stack traces
        git_branch: Current git branch (if available)
        technologies: Detected technologies in use
        keywords: Extracted keywords from current context
    """

    current_file: Path | None = None
    file_type: str | None = None
    project_root: Path | None = None
    recent_files: list[Path] = None
    current_errors: list[str] = None
    git_branch: str | None = None
    technologies: set[str] = None
    keywords: set[str] = None

    def __post_init__(self):
        if self.recent_files is None:
            self.recent_files = []
        if self.current_errors is None:
            self.current_errors = []
        if self.technologies is None:
            self.technologies = set()
        if self.keywords is None:
            self.keywords = set()


@dataclass
class ContextualRecommendation:
    """
    A contextual recommendation for knowledge base content.

    Attributes:
        entry: Recommended knowledge base entry
        relevance_score: How relevant this entry is to current context
        reason: Why this entry was recommended
        context_match: What part of the context matched
        urgency: How urgent this recommendation is (low, medium, high)
    """

    entry: KnowledgeBaseEntry
    relevance_score: float
    reason: str
    context_match: str
    urgency: str = "medium"


class ContextAwareAccessor:
    """
    Provides context-aware access to knowledge base entries.

    This class analyzes the current development context and provides
    relevant knowledge base recommendations based on what the developer
    is currently working on.
    """

    def __init__(
        self,
        index: KnowledgeBaseIndex,
        searcher: KnowledgeBaseSearcher,
        cross_ref_analyzer: CrossReferenceAnalyzer | None = None,
    ):
        self.index = index
        self.searcher = searcher
        self.cross_ref_analyzer = cross_ref_analyzer

        # Context analysis patterns
        self.error_patterns = self._load_error_patterns()
        self.technology_patterns = self._load_technology_patterns()
        self.keyword_extractors = self._load_keyword_extractors()

        # Statistics
        self.stats = {
            "contexts_analyzed": 0,
            "recommendations_made": 0,
            "error_matches": 0,
            "technology_detections": 0,
            "file_analyses": 0,
        }

    def analyze_current_context(
        self,
        current_file: Path | None = None,
        project_root: Path | None = None,
        error_messages: list[str] | None = None,
    ) -> DevelopmentContext:
        """
        Analyze the current development context.

        Args:
            current_file: Path to the file currently being worked on
            project_root: Root directory of the project
            error_messages: Current error messages or stack traces

        Returns:
            DevelopmentContext with analyzed information
        """
        context = DevelopmentContext(
            current_file=current_file,
            project_root=project_root,
            current_errors=error_messages or [],
        )

        try:
            # Analyze current file
            if current_file and current_file.exists():
                context.file_type = self._detect_file_type(current_file)
                context.technologies.update(
                    self._detect_technologies_in_file(current_file)
                )
                context.keywords.update(self._extract_keywords_from_file(current_file))
                self.stats["file_analyses"] += 1

            # Analyze project structure
            if project_root and project_root.exists():
                context.technologies.update(
                    self._detect_project_technologies(project_root)
                )
                context.git_branch = self._get_git_branch(project_root)

            # Analyze error messages
            if error_messages:
                for error in error_messages:
                    context.technologies.update(
                        self._extract_technologies_from_error(error)
                    )
                    context.keywords.update(self._extract_keywords_from_error(error))
                self.stats["error_matches"] += len(error_messages)

            # Get recent files (if available)
            context.recent_files = self._get_recent_files(project_root)

            self.stats["contexts_analyzed"] += 1
            self.stats["technology_detections"] += len(context.technologies)

        except Exception as e:
            logger.error(f"Error analyzing context: {e}")

        return context

    def get_contextual_recommendations(
        self, context: DevelopmentContext, limit: int = 10
    ) -> list[ContextualRecommendation]:
        """
        Get knowledge base recommendations based on current context.

        Args:
            context: Current development context
            limit: Maximum number of recommendations

        Returns:
            List of contextual recommendations
        """
        recommendations = []

        try:
            # Error-based recommendations (highest priority)
            if context.current_errors:
                error_recs = self._get_error_based_recommendations(context)
                recommendations.extend(error_recs)

            # Technology-based recommendations
            if context.technologies:
                tech_recs = self._get_technology_based_recommendations(context)
                recommendations.extend(tech_recs)

            # File-type based recommendations
            if context.file_type:
                file_recs = self._get_file_type_recommendations(context)
                recommendations.extend(file_recs)

            # Keyword-based recommendations
            if context.keywords:
                keyword_recs = self._get_keyword_based_recommendations(context)
                recommendations.extend(keyword_recs)

            # Cross-reference based recommendations
            if self.cross_ref_analyzer:
                cross_recs = self._get_cross_reference_recommendations(context)
                recommendations.extend(cross_recs)

            # Remove duplicates and sort by relevance
            unique_recs = self._deduplicate_recommendations(recommendations)
            sorted_recs = sorted(
                unique_recs, key=lambda r: r.relevance_score, reverse=True
            )

            self.stats["recommendations_made"] += len(sorted_recs[:limit])

            return sorted_recs[:limit]

        except Exception as e:
            logger.error(f"Error getting contextual recommendations: {e}")
            return []

    def _detect_file_type(self, file_path: Path) -> str:
        """Detect the type of file based on extension and content."""
        extension = file_path.suffix.lower()

        type_mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".md": "markdown",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".sh": "shell",
            ".dockerfile": "docker",
            ".rs": "rust",
            ".go": "go",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c_header",
        }

        return type_mapping.get(extension, "unknown")

    def _detect_technologies_in_file(self, file_path: Path) -> set[str]:
        """Detect technologies used in a file."""
        technologies = set()

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Check for technology-specific patterns
            for tech, patterns in self.technology_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        technologies.add(tech)
                        break

            # Python-specific analysis
            if file_path.suffix == ".py":
                technologies.update(self._analyze_python_imports(content))

        except Exception as e:
            logger.debug(f"Error detecting technologies in {file_path}: {e}")

        return technologies

    def _detect_project_technologies(self, project_root: Path) -> set[str]:
        """Detect technologies used in the project."""
        technologies = set()

        try:
            # Check for common project files
            project_files = {
                "requirements.txt": "python",
                "pyproject.toml": "python",
                "setup.py": "python",
                "package.json": "javascript",
                "Cargo.toml": "rust",
                "go.mod": "go",
                "Dockerfile": "docker",
                "docker-compose.yml": "docker",
                "CMakeLists.txt": "cmake",
            }

            for filename, tech in project_files.items():
                if (project_root / filename).exists():
                    technologies.add(tech)

            # Check for MLX-specific indicators
            if (project_root / "requirements.txt").exists():
                req_content = (project_root / "requirements.txt").read_text()
                if "mlx" in req_content.lower():
                    technologies.add("mlx")

        except Exception as e:
            logger.debug(f"Error detecting project technologies: {e}")

        return technologies

    def _analyze_python_imports(self, content: str) -> set[str]:
        """Analyze Python imports to detect technologies."""
        technologies = set()

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        tech = self._map_import_to_technology(alias.name)
                        if tech:
                            technologies.add(tech)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        tech = self._map_import_to_technology(node.module)
                        if tech:
                            technologies.add(tech)

        except Exception as e:
            logger.debug(f"Error analyzing Python imports: {e}")

        return technologies

    def _map_import_to_technology(self, import_name: str) -> str | None:
        """Map Python import to technology."""
        import_mapping = {
            "mlx": "mlx",
            "torch": "pytorch",
            "tensorflow": "tensorflow",
            "numpy": "numpy",
            "pandas": "pandas",
            "sklearn": "scikit-learn",
            "fastapi": "fastapi",
            "flask": "flask",
            "django": "django",
            "requests": "requests",
            "asyncio": "asyncio",
        }

        # Check for exact matches or prefixes
        for prefix, tech in import_mapping.items():
            if import_name.startswith(prefix):
                return tech

        return None

    def _extract_keywords_from_file(self, file_path: Path) -> set[str]:
        """Extract relevant keywords from a file."""
        keywords = set()

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Extract function/class names (Python)
            if file_path.suffix == ".py":
                keywords.update(self._extract_python_identifiers(content))

            # Extract common technical terms
            tech_terms = re.findall(
                r"\b(?:optimization|performance|memory|training|inference|model|neural|network|gpu|cpu|batch|tensor|array|matrix|vector)\b",
                content,
                re.IGNORECASE,
            )
            keywords.update(term.lower() for term in tech_terms)

            # Extract TODO/FIXME comments
            todo_matches = re.findall(
                r"(?:TODO|FIXME|HACK|NOTE):\s*([^\n]+)", content, re.IGNORECASE
            )
            for match in todo_matches:
                keywords.update(self._extract_keywords_from_text(match))

        except Exception as e:
            logger.debug(f"Error extracting keywords from {file_path}: {e}")

        return keywords

    def _extract_python_identifiers(self, content: str) -> set[str]:
        """Extract Python function and class names."""
        identifiers = set()

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    identifiers.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    identifiers.add(node.name)

        except Exception:
            pass

        return identifiers

    def _extract_keywords_from_error(self, error_message: str) -> set[str]:
        """Extract keywords from error messages."""
        keywords = set()

        # Common error-related keywords
        error_keywords = re.findall(
            r"\b(?:error|exception|failed|missing|not found|invalid|timeout|memory|import|module|attribute|type|value|key|index|name)\b",
            error_message,
            re.IGNORECASE,
        )
        keywords.update(keyword.lower() for keyword in error_keywords)

        # Extract module/package names from import errors
        import_errors = re.findall(
            r'No module named [\'"]([^\'"]+)[\'"]', error_message
        )
        keywords.update(import_errors)

        # Extract file paths
        file_paths = re.findall(r'File "([^"]+)"', error_message)
        for path in file_paths:
            path_obj = Path(path)
            if path_obj.suffix:
                keywords.add(path_obj.suffix[1:])  # Add file extension without dot

        return keywords

    def _extract_technologies_from_error(self, error_message: str) -> set[str]:
        """Extract technology names from error messages."""
        technologies = set()

        # Check for technology mentions in error messages
        for tech, patterns in self.technology_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    technologies.add(tech)
                    break

        return technologies

    def _get_error_based_recommendations(
        self, context: DevelopmentContext
    ) -> list[ContextualRecommendation]:
        """Get recommendations based on current errors."""
        recommendations = []

        for error in context.current_errors:
            # Search for troubleshooting entries
            search_filter = SearchFilter(categories=["troubleshooting"])

            # Extract key terms from error
            error_keywords = self._extract_keywords_from_error(error)
            query = " ".join(list(error_keywords)[:5])  # Use top 5 keywords

            if query:
                results = self.searcher.search(query, filters=search_filter, limit=3)

                for result in results.results:
                    rec = ContextualRecommendation(
                        entry=result.entry,
                        relevance_score=result.relevance_score
                        * 1.2,  # Boost error-based recommendations
                        reason=f"Matches current error: {error[:50]}...",
                        context_match=f"Error keywords: {', '.join(list(error_keywords)[:3])}",
                        urgency="high",
                    )
                    recommendations.append(rec)

        return recommendations

    def _get_technology_based_recommendations(
        self, context: DevelopmentContext
    ) -> list[ContextualRecommendation]:
        """Get recommendations based on detected technologies."""
        recommendations = []

        for tech in context.technologies:
            # Search for entries related to this technology
            search_filter = SearchFilter(tags=[tech])
            results = self.searcher.search(tech, filters=search_filter, limit=2)

            for result in results.results:
                rec = ContextualRecommendation(
                    entry=result.entry,
                    relevance_score=result.relevance_score,
                    reason=f"Related to technology: {tech}",
                    context_match=f"Technology: {tech}",
                    urgency="medium",
                )
                recommendations.append(rec)

        return recommendations

    def _get_file_type_recommendations(
        self, context: DevelopmentContext
    ) -> list[ContextualRecommendation]:
        """Get recommendations based on current file type."""
        recommendations = []

        if context.file_type == "python":
            # Look for Python-related entries
            search_filter = SearchFilter(tags=["python"])
            results = self.searcher.search("python", filters=search_filter, limit=2)

            for result in results.results:
                rec = ContextualRecommendation(
                    entry=result.entry,
                    relevance_score=result.relevance_score
                    * 0.8,  # Lower priority than errors
                    reason="Related to Python development",
                    context_match=f"File type: {context.file_type}",
                    urgency="low",
                )
                recommendations.append(rec)

        return recommendations

    def _get_keyword_based_recommendations(
        self, context: DevelopmentContext
    ) -> list[ContextualRecommendation]:
        """Get recommendations based on extracted keywords."""
        recommendations = []

        # Use top keywords for search
        top_keywords = list(context.keywords)[:5]
        if top_keywords:
            query = " ".join(top_keywords)
            results = self.searcher.search(query, limit=3)

            for result in results.results:
                rec = ContextualRecommendation(
                    entry=result.entry,
                    relevance_score=result.relevance_score * 0.7,
                    reason=f"Matches keywords: {', '.join(top_keywords[:3])}",
                    context_match=f"Keywords: {', '.join(top_keywords)}",
                    urgency="low",
                )
                recommendations.append(rec)

        return recommendations

    def _get_cross_reference_recommendations(
        self, context: DevelopmentContext
    ) -> list[ContextualRecommendation]:
        """Get recommendations based on cross-references."""
        recommendations = []

        # This would be enhanced with actual cross-reference analysis
        # For now, return empty list
        return recommendations

    def _deduplicate_recommendations(
        self, recommendations: list[ContextualRecommendation]
    ) -> list[ContextualRecommendation]:
        """Remove duplicate recommendations."""
        seen_entries = set()
        unique_recs = []

        for rec in recommendations:
            if rec.entry.title not in seen_entries:
                seen_entries.add(rec.entry.title)
                unique_recs.append(rec)

        return unique_recs

    def _get_git_branch(self, project_root: Path) -> str | None:
        """Get current git branch."""
        try:
            git_head = project_root / ".git" / "HEAD"
            if git_head.exists():
                head_content = git_head.read_text().strip()
                if head_content.startswith("ref: refs/heads/"):
                    return head_content.split("/")[-1]
        except Exception:
            pass
        return None

    def _get_recent_files(self, project_root: Path | None) -> list[Path]:
        """Get recently modified files."""
        recent_files = []

        if not project_root or not project_root.exists():
            return recent_files

        try:
            # Find recently modified Python files
            for py_file in project_root.rglob("*.py"):
                if py_file.is_file():
                    recent_files.append(py_file)

            # Sort by modification time and take top 5
            recent_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return recent_files[:5]

        except Exception as e:
            logger.debug(f"Error getting recent files: {e}")
            return []

    def _extract_keywords_from_text(self, text: str) -> set[str]:
        """Extract keywords from arbitrary text."""
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out common stop words
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
        keywords = {word for word in words if len(word) >= 3 and word not in stop_words}

        return keywords

    def _load_error_patterns(self) -> dict[str, list[str]]:
        """Load patterns for matching error types."""
        return {
            "import_error": [
                r"ModuleNotFoundError",
                r"ImportError",
                r"No module named",
            ],
            "memory_error": [r"MemoryError", r"out of memory", r"memory allocation"],
            "type_error": [r"TypeError", r"type object", r"not callable"],
            "value_error": [r"ValueError", r"invalid value", r"cannot convert"],
        }

    def _load_technology_patterns(self) -> dict[str, list[str]]:
        """Load patterns for detecting technologies."""
        return {
            "mlx": [r"\bmlx\b", r"mlx\.", r"import mlx"],
            "pytorch": [r"\btorch\b", r"torch\.", r"import torch"],
            "tensorflow": [r"\btensorflow\b", r"tf\.", r"import tensorflow"],
            "numpy": [r"\bnumpy\b", r"np\.", r"import numpy"],
            "pandas": [r"\bpandas\b", r"pd\.", r"import pandas"],
            "fastapi": [r"\bfastapi\b", r"FastAPI", r"from fastapi"],
            "apple-silicon": [
                r"\bM1\b",
                r"\bM2\b",
                r"\bM3\b",
                r"Apple Silicon",
                r"arm64",
            ],
            "docker": [r"\bDocker\b", r"dockerfile", r"docker-compose"],
            "python": [r"\.py$", r"python", r"def ", r"class ", r"import "],
        }

    def _load_keyword_extractors(self) -> dict[str, str]:
        """Load keyword extraction patterns."""
        return {
            "function_names": r"def\s+(\w+)",
            "class_names": r"class\s+(\w+)",
            "variable_names": r"(\w+)\s*=",
            "comments": r"#\s*(.+)",
            "docstrings": r'"""(.+?)"""',
        }

    def get_context_stats(self) -> dict[str, Any]:
        """Get context analysis statistics."""
        return {
            **self.stats,
            "total_entries": len(self.index.entries),
            "error_patterns_loaded": len(self.error_patterns),
            "technology_patterns_loaded": len(self.technology_patterns),
        }


def create_context_aware_accessor(
    index: KnowledgeBaseIndex,
    searcher: KnowledgeBaseSearcher,
    cross_ref_analyzer: CrossReferenceAnalyzer | None = None,
) -> ContextAwareAccessor:
    """
    Factory function to create a context-aware accessor.

    Args:
        index: Knowledge base index
        searcher: Knowledge base searcher
        cross_ref_analyzer: Optional cross-reference analyzer

    Returns:
        Configured context-aware accessor
    """
    return ContextAwareAccessor(index, searcher, cross_ref_analyzer)


def main():
    """
    Command-line interface for context-aware access.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Context-Aware Knowledge Base Access")
    parser.add_argument("kb_path", help="Path to knowledge base directory")
    parser.add_argument("--file", help="Current file being worked on")
    parser.add_argument("--project", help="Project root directory")
    parser.add_argument("--error", help="Current error message")
    parser.add_argument("--limit", type=int, default=5, help="Maximum recommendations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Load knowledge base
        from indexer import KnowledgeBaseIndexer
        from search import KnowledgeBaseSearch

        indexer = KnowledgeBaseIndexer(Path(args.kb_path))
        index = indexer.load_index_from_file()

        if not index:
            print("No index found. Building index...")
            index = indexer.build_index()

        searcher = create_searcher(index)
        accessor = create_context_aware_accessor(index, searcher)

        # Analyze context
        current_file = Path(args.file) if args.file else None
        project_root = Path(args.project) if args.project else None
        errors = [args.error] if args.error else []

        context = accessor.analyze_current_context(
            current_file=current_file, project_root=project_root, error_messages=errors
        )

        print("🔍 Context Analysis:")
        print(f"   File type: {context.file_type}")
        print(
            f"   Technologies: {', '.join(context.technologies) if context.technologies else 'None detected'}"
        )
        print(
            f"   Keywords: {', '.join(list(context.keywords)[:5]) if context.keywords else 'None extracted'}"
        )
        print(f"   Errors: {len(context.current_errors)}")

        # Get recommendations
        recommendations = accessor.get_contextual_recommendations(
            context, limit=args.limit
        )

        if recommendations:
            print("\n💡 Contextual Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(
                    f"{i}. {rec.entry.title} (score: {rec.relevance_score:.2f}, {rec.urgency})"
                )
                print(f"   Reason: {rec.reason}")
                print(f"   Match: {rec.context_match}")
                print(
                    f"   Category: {rec.entry.category} | Tags: {', '.join(rec.entry.tags)}"
                )
                print()
        else:
            print("\n💡 No contextual recommendations found")

        # Show statistics
        stats = accessor.get_context_stats()
        print("📊 Statistics:")
        print(f"   Contexts analyzed: {stats['contexts_analyzed']}")
        print(f"   Recommendations made: {stats['recommendations_made']}")
        print(f"   Technologies detected: {stats['technology_detections']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
