#!/usr/bin/env python3
"""
Knowledge Base Initialization Script

This script sets up a new knowledge base directory structure with all required
files, templates, and configuration.
"""

import argparse
import shutil
import sys
from inspect import cleandoc
from pathlib import Path
from typing import Optional


def create_directory_structure(kb_path: Path) -> None:
    """Create the standard knowledge base directory structure."""
    directories = [
        "categories/apple-silicon",
        "categories/mlx-framework",
        "categories/performance",
        "categories/troubleshooting",
        "categories/deployment",
        "categories/testing",
        "patterns/training-patterns",
        "patterns/inference-patterns",
        "patterns/common",
        "templates",
        ".meta",
        "docs",
        "scripts",
    ]

    for directory in directories:
        dir_path = kb_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_core_files(kb_path: Path) -> None:
    """Create core knowledge base files."""

    # Main README
    readme_content = cleandoc(
        """# Knowledge Base

        Welcome to your new Knowledge Base! This system helps you store, organize, and access development knowledge.

        ## Getting Started

        1. **Read the documentation**: Check out [docs/README.md](docs/README.md)
        2. **Search for knowledge**: `uv run python -m kb search "query"`
        3. **Create entries**: `uv run python -m kb create --title "Title" --category performance`

        ## Quick Links

        - [📚 Documentation](docs/README.md)
        - [🚀 User Guide](docs/USER_GUIDE.md)
        - [🔧 Developer Guide](docs/DEVELOPER_GUIDE.md)
        - [🚨 Troubleshooting](docs/TROUBLESHOOTING.md)

        ## Categories

        - [Apple Silicon](categories/apple-silicon/) - Apple Silicon optimizations
        - [MLX Framework](categories/mlx-framework/) - MLX patterns and usage
        - [Performance](categories/performance/) - Performance optimization
        - [Troubleshooting](categories/troubleshooting/) - Issues and solutions
        - [Deployment](categories/deployment/) - Deployment strategies
        - [Testing](categories/testing/) - Testing approaches

        ---
        *Start documenting your knowledge and make your team more effective!*
        """
    )

    (kb_path / "README.md").write_text(readme_content)
    print("✓ Created README.md")

    # CLI README
    cli_readme_content = cleandoc(
        """# Knowledge Base CLI

        This directory contains the command-line interface for the Knowledge Base system.

        ## Usage

        ```bash
        # Search for knowledge
        uv run python -m kb search "query"

        # Create new entry
        uv run python -m kb create --title "Title" --category performance

        # List entries
        uv run python -m kb list

        # Get statistics
        uv run python -m kb stats

        # Run quality checks
        uv run python -m kb quality-check

        # Rebuild search index
        uv run python -m kb rebuild-index
        ```

        ## Help

        For complete documentation, see [docs/USER_GUIDE.md](docs/USER_GUIDE.md).
        """
    )

    (kb_path / "CLI_README.md").write_text(cli_readme_content)
    print("✓ Created CLI_README.md")

    # Contribution guide
    contribution_guide = cleandoc(
        """# Knowledge Base Contribution Guide

        ## How to Contribute

        1. **Identify Knowledge**: Look for gaps or improvements needed
        2. **Choose Template**: Select appropriate template for your content
        3. **Write Content**: Follow template structure and guidelines
        4. **Validate**: Ensure proper format and working code examples
        5. **Submit**: Add to knowledge base and update index

        ## Entry Standards

        - Use clear, descriptive titles
        - Include practical code examples
        - Provide context and rationale
        - Tag appropriately for discoverability
        - Test all code examples
        - Update regularly to keep current

        ## Templates Available

        - **Standard Entry**: General knowledge entries
        - **Troubleshooting**: Problem-solution format
        - **Pattern**: Reusable code patterns

        ## Quality Guidelines

        - Follow markdown best practices
        - Include proper frontmatter
        - Use consistent formatting
        - Add cross-references to related entries
        - Ensure code examples are complete and working

        For detailed guidelines, see [docs/USER_GUIDE.md](../docs/USER_GUIDE.md).
        """
    )

    (kb_path / ".meta" / "contribution-guide.md").write_text(contribution_guide)
    print("✓ Created contribution guide")


def create_templates(kb_path: Path) -> None:
    """Create entry templates."""

    # Standard entry template
    entry_template = cleandoc(
        """---
        title: "Your Entry Title Here"
        category: "category-name"
        tags: ["tag1", "tag2", "tag3"]
        difficulty: "beginner"
        last_updated: "2025-08-14"
        contributors: ["Your Name"]
        ---

        # Your Entry Title Here

        ## Problem/Context

        Describe the problem or context this entry addresses. What challenge does it solve?

        ## Solution/Pattern

        Provide the solution or pattern. Explain the approach and why it works.

        ### Key Points

        - Important point 1
        - Important point 2
        - Important point 3

        ## Code Example

        ```python
        # Provide a complete, working code example
        import mlx.core as mx

        def example_function():
            \"\"\"Example function demonstrating the pattern.\"\"\"
            # Implementation here
            return mx.array([1, 2, 3])

        # Usage example
        result = example_function()
        print(f"Result: {result}")
        ```

        ## Best Practices

        - Best practice 1
        - Best practice 2
        - Best practice 3

        ## Common Pitfalls

        - Pitfall 1 and how to avoid it
        - Pitfall 2 and how to avoid it

        ## Related Entries

        - [Related Entry 1](../category/related-entry.md)
        - [Related Entry 2](../category/another-entry.md)

        ## External Resources

        - [Official Documentation](https://example.com)
        - [Tutorial](https://example.com/tutorial)
        - [GitHub Repository](https://github.com/example/repo)

        ---

        ## Template Instructions

        1. Replace "Your Entry Title Here" with descriptive title
        2. Update category to match directory location
        3. Add 3-5 relevant tags
        4. Set appropriate difficulty level
        5. Update last_updated date
        6. Add your name to contributors
        7. Fill in all sections with relevant content
        8. Test all code examples
        9. Remove these instructions before saving
        """
    )

    (kb_path / "templates" / "entry-template.md").write_text(entry_template)
    print("✓ Created entry template")

    # Troubleshooting template
    troubleshooting_template = cleandoc(
        """---
        title: "Fixing [Specific Problem/Error]"
        category: "troubleshooting"
        tags: ["error-type", "technology", "platform"]
        difficulty: "intermediate"
        last_updated: "2025-08-14"
        contributors: ["Your Name"]
        ---

        # Fixing [Specific Problem/Error]

        ## Problem Description

        Clearly describe the issue, including:

        - Symptoms observed
        - Error messages (exact text)
        - When the problem occurs
        - Affected systems or configurations

        ### Error Message

        ```bash
        Exact error message or stack trace here
        Include full context when possible
        ```

        ### Environment

        - **OS**: macOS 14.2 (Apple Silicon M2)
        - **Python**: 3.11.5
        - **MLX**: 0.0.8
        - **Other relevant versions**

        ## Root Cause

        Explain what causes this problem:

        - Technical explanation of the underlying issue
        - Why it happens in certain conditions
        - Common triggers or scenarios

        ## Solution

        Provide step-by-step resolution:

        ### Quick Fix

        For immediate resolution:

        ```bash
        # Commands to quickly resolve the issue
        pip install --upgrade mlx
        ```

        ### Detailed Solution

        1. **Step 1**: Detailed explanation

            ```python
            # Code example for this step
            import mlx.core as mx
            mx.set_default_device(mx.gpu)
            ```

        2. **Step 2**: Next action

            ```bash
            # Shell commands if needed
            export MLX_MEMORY_LIMIT=8GB
            ```

        3. **Step 3**: Verification

            ```python
            # How to verify the fix worked
            print(mx.default_device())
            ```

        ## Prevention

        How to avoid this problem in the future:

        - Configuration changes
        - Best practices to follow
        - Warning signs to watch for

        ## Alternative Solutions

        If the main solution doesn't work:

        ### Alternative 1: [Brief Description]

        ```python
        # Alternative approach
        ```

        ### Alternative 2: [Brief Description]

        ```bash
        # Another way to solve it
        ```

        ## Verification

        How to confirm the problem is resolved:

        - Tests to run
        - Expected output
        - Signs that indicate success

        ```python
        # Verification code
        def verify_fix():
            \"\"\"Test that the issue is resolved.\"\"\"
            # Implementation
            return True

        assert verify_fix(), "Fix verification failed"
        ```

        ## Related Issues

        - [Similar Problem](../troubleshooting/similar-problem.md)
        - [Configuration Guide](../deployment/configuration-guide.md)

        ## Additional Resources

        - [Official Documentation](https://docs.example.com/troubleshooting)
        - [Community Discussion](https://forum.example.com/topic/123)
        - [Stack Overflow](https://stackoverflow.com/questions/123456)

        ---

        ## Template Instructions

        1. Replace [Specific Problem/Error] with actual issue name
        2. Update frontmatter with appropriate tags and category
        3. Include exact error messages for searchability
        4. Provide complete environment details
        5. Test your solution before documenting it
        6. Include verification steps
        7. Link to related issues and external resources
        8. Remove these instructions before saving
        """
    )

    (kb_path / "templates" / "troubleshooting-template.md").write_text(
        troubleshooting_template
    )
    print("✓ Created troubleshooting template")

    # Pattern template
    pattern_template = cleandoc(
        """---
        title: "[Pattern Name] Pattern"
        category: "patterns"
        tags: ["pattern", "design", "architecture"]
        difficulty: "intermediate"
        last_updated: "2025-08-14"
        contributors: ["Your Name"]
        ---

        # [Pattern Name] Pattern

        ## Overview

        Brief description of what this pattern does and when to use it.

        ## Problem

        What problem does this pattern solve? What challenges does it address?

        ## Solution

        How does this pattern solve the problem? What's the core approach?

        ## Implementation

        ### Basic Structure

        ```python
        # Core pattern implementation
        class PatternExample:
            \"\"\"Example implementation of the pattern.\"\"\"

            def __init__(self):
                self.setup()

            def setup(self):
                \"\"\"Initialize pattern components.\"\"\"
                pass

            def execute(self):
                \"\"\"Execute the pattern.\"\"\"
                pass
        ```

        ### Advanced Usage

        ```python
        # More sophisticated example
        class AdvancedPatternExample(PatternExample):
            \"\"\"Advanced pattern with additional features.\"\"\"

            def __init__(self, config):
                self.config = config
                super().__init__()

            def setup(self):
                \"\"\"Advanced setup with configuration.\"\"\"
                # Implementation details
                pass
        ```

        ## Usage Examples

        ### Basic Usage

        ```python
        # Simple usage example
        pattern = PatternExample()
        result = pattern.execute()
        ```

        ### Advanced Usage

        ```python
        # Advanced usage with configuration
        config = {"option1": "value1", "option2": "value2"}
        advanced_pattern = AdvancedPatternExample(config)
        result = advanced_pattern.execute()
        ```

        ## Benefits

        - Benefit 1: Explanation
        - Benefit 2: Explanation
        - Benefit 3: Explanation

        ## Trade-offs

        - Trade-off 1: When this might not be ideal
        - Trade-off 2: Performance considerations
        - Trade-off 3: Complexity considerations

        ## When to Use

        - Scenario 1: Description
        - Scenario 2: Description
        - Scenario 3: Description

        ## When Not to Use

        - Scenario 1: Why not appropriate
        - Scenario 2: Better alternatives
        - Scenario 3: Complexity concerns

        ## Related Patterns

        - [Related Pattern 1](../patterns/related-pattern.md)
        - [Related Pattern 2](../patterns/another-pattern.md)

        ## Real-World Examples

        - Example 1: Brief description and link
        - Example 2: Brief description and link
        - Example 3: Brief description and link

        ---

        ## Template Instructions

        1. Replace [Pattern Name] with actual pattern name
        2. Provide clear problem and solution descriptions
        3. Include complete, working code examples
        4. Explain benefits and trade-offs honestly
        5. Give clear guidance on when to use/not use
        6. Link to related patterns and examples
        7. Remove these instructions before saving
        """
    )

    (kb_path / "templates" / "pattern-template.md").write_text(pattern_template)
    print("✓ Created pattern template")


def copy_documentation(kb_path: Path, source_kb_path: Optional[Path] = None) -> None:
    """Copy documentation files if available."""
    if source_kb_path and (source_kb_path / "docs").exists():
        # Copy existing documentation
        shutil.copytree(source_kb_path / "docs", kb_path / "docs", dirs_exist_ok=True)
        print("✓ Copied existing documentation")
    else:
        # Create basic documentation structure
        docs_readme = cleandoc(
            """# Knowledge Base Documentation

            Welcome to your Knowledge Base documentation.

            ## Getting Started

            This knowledge base helps you store and organize development knowledge.

            ### Basic Usage

            1. **Search**: `uv run python -m kb search "query"`
            2. **Create**: `uv run python -m kb create --title "Title"`
            3. **List**: `uv run python -m kb list`

            ### Documentation Structure

            - This file provides an overview
            - Check the main README.md for quick start
            - Use templates in templates/ directory
            - Follow the contribution guide in .meta/

            ## Next Steps

            1. Create your first entry using a template
            2. Set up regular contribution workflows
            3. Customize categories for your needs
            4. Establish quality review processes

            For more detailed documentation, consider adding:
            - User guides for your specific use cases
            - Developer documentation for customizations
            - Troubleshooting guides for common issues
            """
        )

        (kb_path / "docs" / "README.md").write_text(docs_readme)
        print("✓ Created basic documentation")


def create_configuration_files(kb_path: Path) -> None:
    """Create configuration files for the knowledge base."""

    # Configuration file
    config_content = cleandoc(
        """# Knowledge Base Configuration

        # Search Configuration
        search:
            max_results: 50
            fuzzy_threshold: 0.6
            enable_full_text: true

        # Quality Assurance
        quality:
            max_title_length: 100
            min_content_length: 200
            required_sections: ["Problem/Context", "Solution"]
            check_code_syntax: true

        # Analytics
        analytics:
            track_usage: true
            track_searches: true
            retention_days: 365

        # Categories
        categories:
            - apple-silicon
            - mlx-framework
            - performance
            - troubleshooting
            - deployment
            - testing

        # Difficulty Levels
        difficulty_levels:
            - beginner
            - intermediate
            - advanced

        # File Extensions
        supported_extensions:
            - .md
            - .markdown

        # Indexing
        indexing:
            parallel_processing: true
            batch_size: 100
            exclude_patterns:
                - "*.tmp"
                - ".*"
                - "__pycache__"
        """
    )

    (kb_path / ".meta" / "config.yaml").write_text(config_content)
    print("✓ Created configuration file")

    # Git ignore file
    gitignore_content = cleandoc(
        """# Knowledge Base specific ignores
        .meta/index.json
        .meta/cache/
        .meta/logs/
        *.tmp
        *.log

        # Python
        __pycache__/
        *.py[cod]
        *$py.class
        *.so
        .Python
        build/
        develop-eggs/
        dist/
        downloads/
        eggs/
        .eggs/
        lib/
        lib64/
        parts/
        sdist/
        var/
        wheels/
        *.egg-info/
        .installed.cfg
        *.egg

        # Virtual environments
        .env
        .venv
        env/
        venv/
        ENV/
        env.bak/
        venv.bak/

        # IDE
        .vscode/
        .idea/
        *.swp
        *.swo
        *~

        # OS
        .DS_Store
        .DS_Store?
        ._*
        .Spotlight-V100
        .Trashes
        ehthumbs.db
        Thumbs.db
        """
    )

    (kb_path / ".gitignore").write_text(gitignore_content)
    print("✓ Created .gitignore file")


def create_example_entries(kb_path: Path) -> None:
    """Create example entries to demonstrate the system."""

    # Example Apple Silicon entry
    apple_silicon_example = cleandoc(
        """---
        title: "MLX Memory Optimization for Apple Silicon"
        category: "apple-silicon"
        tags: ["mlx", "memory", "optimization", "apple-silicon"]
        difficulty: "intermediate"
        last_updated: "2025-08-14"
        contributors: ["Knowledge Base System"]
        ---

        # MLX Memory Optimization for Apple Silicon

        ## Problem/Context

        Apple Silicon devices have unified memory architecture that requires specific optimization strategies for ML workloads. Default MLX configurations may not utilize available memory efficiently.

        ## Solution/Pattern

        Implement memory-aware training and inference patterns that leverage Apple Silicon's unified memory.

        ### Key Points

        - Set appropriate memory limits for MLX
        - Use gradient checkpointing for large models
        - Implement mixed precision training
        - Monitor memory usage during training

        ## Code Example

        ```python
        import mlx.core as mx
        import mlx.nn as nn

        def configure_memory_optimization():
            \"\"\"Configure MLX for optimal Apple Silicon memory usage.\"\"\"
            # Set memory limit to 80% of available memory
            available_memory = mx.metal.get_available_memory()
            memory_limit = int(available_memory * 0.8)
            mx.metal.set_memory_limit(memory_limit)

            print(f"Memory limit set to: {memory_limit / (1024**3):.2f} GB")
            return memory_limit

        def create_memory_efficient_model(input_size, hidden_size, output_size):
            \"\"\"Create a memory-efficient model for Apple Silicon.\"\"\"
            return nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

        # Usage
        configure_memory_optimization()
        model = create_memory_efficient_model(784, 256, 10)
        ```

        ## Best Practices

        - Monitor memory usage with `mx.metal.get_memory_info()`
        - Use appropriate batch sizes for your hardware
        - Enable memory mapping for large datasets
        - Profile memory usage during development

        ## Related Entries

        - [Performance Optimization Techniques](../performance/optimization-techniques.md)
        - [MLX Training Patterns](../mlx-framework/training-patterns.md)

        ## External Resources

        - [MLX Documentation](https://ml-explore.github.io/mlx/)
        - [Apple Silicon ML Performance Guide](https://developer.apple.com/documentation/accelerate)
        """
    )

    (kb_path / "categories" / "apple-silicon" / "memory-optimization.md").write_text(
        apple_silicon_example
    )
    print("✓ Created example Apple Silicon entry")


def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(description="Initialize a new Knowledge Base")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path where to create the knowledge base (default: current directory)",
    )
    parser.add_argument(
        "--source", type=Path, help="Source knowledge base to copy documentation from"
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Create minimal structure without examples",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing knowledge base"
    )

    args = parser.parse_args()

    kb_path = Path(args.path).resolve()

    # Check if knowledge base already exists
    if (kb_path / ".meta").exists() and not args.force:
        print(f"❌ Knowledge base already exists at {kb_path}")
        print("Use --force to overwrite existing knowledge base")
        sys.exit(1)

    print(f"🚀 Initializing Knowledge Base at: {kb_path}")
    print()

    try:
        # Create directory structure
        create_directory_structure(kb_path)
        print()

        # Create core files
        create_core_files(kb_path)
        print()

        # Create templates
        create_templates(kb_path)
        print()

        # Copy or create documentation
        copy_documentation(kb_path, args.source)
        print()

        # Create configuration files
        create_configuration_files(kb_path)
        print()

        # Create example entries (unless minimal)
        if not args.minimal:
            create_example_entries(kb_path)
            print()

        print("✅ Knowledge Base initialization complete!")
        print()
        print("Next steps:")
        print("1. Read the documentation: docs/README.md")
        print("2. Create your first entry: uv run python -m kb create")
        print("3. Search for knowledge: uv run python -m kb search")
        print("4. Get help: uv run python -m kb --help")

    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
