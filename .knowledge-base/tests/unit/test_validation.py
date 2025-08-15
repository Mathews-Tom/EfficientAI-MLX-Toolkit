"""
Unit tests for Knowledge Base validation system.

Tests the validation functions including frontmatter validation,
content analysis, and consistency checking.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Add the .meta directory to Python path for imports
kb_meta_path = Path(__file__).parent.parent / ".meta"
sys.path.insert(0, str(kb_meta_path))

from validation import (
    ParseError,
    ValidationError,
    generate_validation_report,
    parse_markdown_file,
    validate_code_examples,
    validate_entry_consistency,
    validate_frontmatter,
    validate_knowledge_base_structure,
    validate_markdown_content,
)


class TestValidateFrontmatter:
    """Test cases for frontmatter validation."""

    def test_valid_frontmatter(self):
        """Test validation of valid frontmatter."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test", "example"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) == 0

    def test_missing_required_fields(self):
        """Test validation fails for missing required fields."""
        frontmatter = {
            "title": "Test Entry",
            # Missing other required fields
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("Missing required field" in error for error in errors)

    def test_empty_fields(self):
        """Test validation fails for empty fields."""
        frontmatter = {
            "title": "",
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("Field cannot be empty" in error for error in errors)

    def test_invalid_title_type(self):
        """Test validation fails for non-string title."""
        frontmatter = {
            "title": 123,  # Should be string
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("Title must be a string" in error for error in errors)

    def test_title_too_long(self):
        """Test validation fails for overly long title."""
        frontmatter = {
            "title": "x" * 201,  # Too long
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("too long" in error for error in errors)

    def test_invalid_category_format(self):
        """Test validation fails for invalid category format."""
        frontmatter = {
            "title": "Test Entry",
            "category": "Test_Category",  # Invalid format
            "tags": ["test"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("lowercase with hyphens" in error for error in errors)

    def test_invalid_difficulty(self):
        """Test validation fails for invalid difficulty."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "invalid",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("Invalid difficulty" in error for error in errors)

    def test_invalid_tags_format(self):
        """Test validation fails for invalid tags format."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": "not-a-list",  # Should be a list
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("Tags must be a list" in error for error in errors)

    def test_empty_tags_list(self):
        """Test validation fails for empty tags list."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": [],  # Empty list
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("At least one tag is required" in error for error in errors)

    def test_invalid_tag_type(self):
        """Test validation fails for non-string tags."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test", 123],  # Non-string tag
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("must be a string" in error for error in errors)

    def test_tag_too_long(self):
        """Test validation fails for overly long tags."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test", "x" * 51],  # Tag too long
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("too long" in error for error in errors)

    def test_invalid_contributors_format(self):
        """Test validation fails for invalid contributors format."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": "not-a-list",  # Should be a list
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("Contributors must be a list" in error for error in errors)

    def test_empty_contributors_list(self):
        """Test validation fails for empty contributors list."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": [],  # Empty list
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("At least one contributor is required" in error for error in errors)

    def test_invalid_date_format(self):
        """Test validation fails for invalid date format."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "intermediate",
            "last_updated": "invalid-date",
            "contributors": ["test-user"],
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("YYYY-MM-DD format" in error for error in errors)

    def test_invalid_usage_count(self):
        """Test validation fails for invalid usage count."""
        frontmatter = {
            "title": "Test Entry",
            "category": "test-category",
            "tags": ["test"],
            "difficulty": "intermediate",
            "last_updated": "2024-01-15",
            "contributors": ["test-user"],
            "usage_count": -1,  # Negative count
        }

        errors = validate_frontmatter(frontmatter)
        assert len(errors) > 0
        assert any("non-negative integer" in error for error in errors)


class TestParseMarkdownFile:
    """Test cases for markdown file parsing."""

    def test_parse_valid_file(self):
        """Test parsing a valid markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            content = """---
title: "Test Entry"
category: "test-category"
tags: ["test", "example"]
difficulty: "intermediate"
last_updated: "2024-01-15"
contributors: ["test-user"]
---

# Test Entry

This is test content.
"""
            f.write(content)
            f.flush()

            frontmatter, body = parse_markdown_file(Path(f.name))

            assert frontmatter["title"] == "Test Entry"
            assert frontmatter["category"] == "test-category"
            assert "# Test Entry" in body

            # Cleanup
            Path(f.name).unlink()

    def test_parse_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            parse_markdown_file(Path("nonexistent.md"))

    def test_parse_file_no_frontmatter(self):
        """Test error handling for file without frontmatter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            content = "# Test Entry\n\nNo frontmatter here."
            f.write(content)
            f.flush()

            with pytest.raises(ParseError, match="missing frontmatter"):
                parse_markdown_file(Path(f.name))

            # Cleanup
            Path(f.name).unlink()

    def test_parse_file_invalid_frontmatter(self):
        """Test error handling for invalid frontmatter format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            content = """---
title: "Test Entry"
# This frontmatter is missing the closing delimiter
# Test Entry
Some content here without proper frontmatter closure."""
            f.write(content)
            f.flush()

            with pytest.raises(ParseError, match="Invalid frontmatter format"):
                parse_markdown_file(Path(f.name))

            # Cleanup
            Path(f.name).unlink()

    def test_parse_file_invalid_yaml(self):
        """Test error handling for invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            content = """---
title: "Test Entry"
invalid_yaml: [unclosed list
---

# Test Entry
"""
            f.write(content)
            f.flush()

            with pytest.raises(ParseError, match="Invalid YAML"):
                parse_markdown_file(Path(f.name))

            # Cleanup
            Path(f.name).unlink()

    def test_parse_file_empty_frontmatter(self):
        """Test parsing file with empty frontmatter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            content = """---
---

# Test Entry

Content here.
"""
            f.write(content)
            f.flush()

            frontmatter, body = parse_markdown_file(Path(f.name))

            assert frontmatter == {}
            assert "# Test Entry" in body

            # Cleanup
            Path(f.name).unlink()


class TestValidateMarkdownContent:
    """Test cases for markdown content validation."""

    def test_validate_complete_content(self):
        """Test validation of complete content with all sections."""
        content = """# Test Entry

## Problem/Context
This describes the problem.

## Solution/Pattern
This describes the solution.

## Code Example
```python
def example():
    return "Hello, World!"
```

## Additional Section
More content here.
"""

        warnings = validate_markdown_content(content)
        # Should have no warnings for complete content
        assert len(warnings) == 0

    def test_validate_missing_sections(self):
        """Test validation identifies missing recommended sections."""
        content = """# Test Entry

Just some basic content without recommended sections.
"""

        warnings = validate_markdown_content(content)
        assert len(warnings) > 0
        assert any("Missing recommended section" in warning for warning in warnings)

    def test_validate_no_code_examples(self):
        """Test validation identifies missing code examples."""
        content = """# Test Entry

## Problem/Context
This describes the problem.

## Solution/Pattern
This describes the solution but has no code examples.
"""

        warnings = validate_markdown_content(content)
        assert len(warnings) > 0
        assert any("No code examples found" in warning for warning in warnings)

    def test_validate_short_content(self):
        """Test validation identifies very short content."""
        content = "# Short\n\nToo short."

        warnings = validate_markdown_content(content)
        assert len(warnings) > 0
        assert any("quite short" in warning for warning in warnings)

    def test_validate_invalid_links(self):
        """Test validation identifies potentially invalid links."""
        content = """# Test Entry

Here's a [broken link](invalid-link) that might be problematic.
And here's a [good external link](https://example.com).
And here's a [good internal link](../other-entry.md).
"""

        warnings = validate_markdown_content(content)
        assert len(warnings) > 0
        assert any("Potentially invalid link" in warning for warning in warnings)


class TestValidateCodeExamples:
    """Test cases for code example validation."""

    def test_validate_valid_python_code(self):
        """Test validation of valid Python code."""
        content = """# Test Entry

```python
def hello_world():
    print("Hello, World!")
    return True
```
"""

        errors = validate_code_examples(content)
        assert len(errors) == 0

    def test_validate_invalid_python_syntax(self):
        """Test validation identifies Python syntax errors."""
        content = """# Test Entry

```python
def invalid_function(
    # Missing closing parenthesis and colon
    print("This is invalid")
```
"""

        errors = validate_code_examples(content)
        assert len(errors) > 0
        assert any("Python syntax error" in error for error in errors)

    def test_validate_missing_language_specification(self):
        """Test validation identifies missing language specification."""
        content = """# Test Entry

```
def some_function():
    pass
```
"""

        errors = validate_code_examples(content)
        assert len(errors) > 0
        assert any("Missing language specification" in error for error in errors)

    def test_validate_empty_code_block(self):
        """Test validation identifies empty code blocks."""
        content = """# Test Entry

```python
```
"""

        errors = validate_code_examples(content)
        assert len(errors) > 0
        assert any("Empty code block" in error for error in errors)

    def test_validate_placeholder_content(self):
        """Test validation identifies placeholder content."""
        content = """# Test Entry

```python
def example():
    # TODO: Implement this function
    pass
```

```javascript
function example() {
    // FIXME: This needs work
    return ...;
}
```
"""

        errors = validate_code_examples(content)
        assert len(errors) > 0
        assert any("placeholder content" in error for error in errors)

    def test_validate_non_python_code(self):
        """Test validation handles non-Python code gracefully."""
        content = """# Test Entry

```javascript
function hello() {
    console.log("Hello, World!");
}
```

```bash
echo "Hello, World!"
```
"""

        errors = validate_code_examples(content)
        # Should not have syntax errors for non-Python code
        python_syntax_errors = [e for e in errors if "Python syntax error" in e]
        assert len(python_syntax_errors) == 0


class TestValidateEntryConsistency:
    """Test cases for entry consistency validation."""

    def test_validate_correct_category_location(self):
        """Test validation passes for correct file location."""
        entry_data = {
            "title": "Test Entry",
            "category": "apple-silicon",
        }
        file_path = Path(".knowledge-base/categories/apple-silicon/test-entry.md")

        warnings = validate_entry_consistency(entry_data, file_path)
        # Should have no warnings for correct location
        category_warnings = [w for w in warnings if "File location" in w]
        assert len(category_warnings) == 0

    def test_validate_incorrect_category_location(self):
        """Test validation identifies incorrect file location."""
        entry_data = {
            "title": "Test Entry",
            "category": "apple-silicon",
        }
        file_path = Path(".knowledge-base/categories/wrong-category/test-entry.md")

        warnings = validate_entry_consistency(entry_data, file_path)
        assert len(warnings) > 0
        assert any(
            "File location doesn't match category" in warning for warning in warnings
        )

    def test_validate_correct_filename(self):
        """Test validation passes for correct filename."""
        entry_data = {
            "title": "Test Entry Name",
        }
        file_path = Path(".knowledge-base/categories/test/test-entry-name.md")

        warnings = validate_entry_consistency(entry_data, file_path)
        # Should have no warnings for correct filename
        filename_warnings = [w for w in warnings if "Filename" in w]
        assert len(filename_warnings) == 0

    def test_validate_incorrect_filename(self):
        """Test validation identifies incorrect filename."""
        entry_data = {
            "title": "Test Entry Name",
        }
        file_path = Path(".knowledge-base/categories/test/wrong-filename.md")

        warnings = validate_entry_consistency(entry_data, file_path)
        assert len(warnings) > 0
        assert any(
            "Filename" in warning and "doesn't match title" in warning
            for warning in warnings
        )


class TestValidateKnowledgeBaseStructure:
    """Test cases for knowledge base structure validation."""

    def create_test_kb_structure(
        self, temp_dir: Path, include_errors: bool = False
    ) -> Path:
        """Create a test knowledge base structure."""
        kb_path = temp_dir / ".knowledge-base"

        # Create required directories
        (kb_path / "categories" / "test-category").mkdir(parents=True)
        (kb_path / "patterns" / "test-pattern").mkdir(parents=True)
        (kb_path / "templates").mkdir(parents=True)
        (kb_path / ".meta").mkdir(parents=True)

        # Create required files
        (kb_path / "README.md").write_text("# Knowledge Base")
        (kb_path / ".meta" / "contribution-guide.md").write_text("# Contribution Guide")

        if include_errors:
            # Create entry with errors
            invalid_entry = """---
title: "Invalid Entry"
# Missing required fields
---

# Invalid Entry

Short content.
"""
            (kb_path / "categories" / "test-category" / "invalid-entry.md").write_text(
                invalid_entry
            )
        else:
            # Create valid entry
            valid_entry = """---
title: "Valid Entry"
category: "test-category"
tags: ["test", "example"]
difficulty: "intermediate"
last_updated: "2024-01-15"
contributors: ["test-user"]
---

# Valid Entry

## Problem/Context
This describes the problem.

## Solution/Pattern
This describes the solution.

## Code Example
```python
def example():
    return "Hello, World!"
```
"""
            (kb_path / "categories" / "test-category" / "valid-entry.md").write_text(
                valid_entry
            )

        return kb_path

    def test_validate_valid_structure(self):
        """Test validation of valid knowledge base structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(
                Path(temp_dir), include_errors=False
            )

            results = validate_knowledge_base_structure(kb_path)

            assert len(results["structure_errors"]) == 0
            assert len(results["missing_files"]) == 0
            assert len(results["entry_errors"]) == 0

    def test_validate_missing_directory(self):
        """Test validation identifies missing knowledge base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / "nonexistent"

            results = validate_knowledge_base_structure(kb_path)

            assert len(results["structure_errors"]) > 0
            assert any("not found" in error for error in results["structure_errors"])

    def test_validate_missing_required_directories(self):
        """Test validation identifies missing required directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()
            # Don't create required subdirectories

            results = validate_knowledge_base_structure(kb_path)

            assert len(results["missing_files"]) > 0
            assert any(
                "Missing required directory" in missing
                for missing in results["missing_files"]
            )

    def test_validate_missing_required_files(self):
        """Test validation identifies missing required files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            (kb_path / "categories").mkdir(parents=True)
            (kb_path / "patterns").mkdir(parents=True)
            (kb_path / "templates").mkdir(parents=True)
            (kb_path / ".meta").mkdir(parents=True)
            # Don't create required files

            results = validate_knowledge_base_structure(kb_path)

            assert len(results["missing_files"]) > 0
            assert any(
                "Missing required file" in missing
                for missing in results["missing_files"]
            )

    def test_validate_entry_errors(self):
        """Test validation identifies entry errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.create_test_kb_structure(Path(temp_dir), include_errors=True)

            results = validate_knowledge_base_structure(kb_path)

            assert len(results["entry_errors"]) > 0


class TestGenerateValidationReport:
    """Test cases for validation report generation."""

    def test_generate_report_valid_kb(self):
        """Test generating report for valid knowledge base."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            (kb_path / "categories").mkdir(parents=True)
            (kb_path / "patterns").mkdir(parents=True)
            (kb_path / "templates").mkdir(parents=True)
            (kb_path / ".meta").mkdir(parents=True)
            (kb_path / "README.md").write_text("# Knowledge Base")
            (kb_path / ".meta" / "contribution-guide.md").write_text("# Guide")

            report = generate_validation_report(kb_path)

            assert "Knowledge Base Validation Report" in report
            assert "No critical issues found" in report

    def test_generate_report_with_issues(self):
        """Test generating report for knowledge base with issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()
            # Create incomplete structure

            report = generate_validation_report(kb_path)

            assert "Knowledge Base Validation Report" in report
            assert "issues that need attention" in report

    def test_generate_report_to_file(self):
        """Test generating report and saving to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / ".knowledge-base"
            kb_path.mkdir()

            output_file = Path(temp_dir) / "validation_report.md"
            report = generate_validation_report(kb_path, output_file)

            assert output_file.exists()
            saved_content = output_file.read_text()
            assert saved_content == report


if __name__ == "__main__":
    pytest.main([__file__])
