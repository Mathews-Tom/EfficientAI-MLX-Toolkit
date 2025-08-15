# Knowledge Base User Guide

## Overview

The EfficientAI MLX Toolkit Knowledge Base is a comprehensive system for storing, organizing, and accessing development knowledge, patterns, and solutions specifically optimized for Apple Silicon AI development.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Searching for Knowledge](#searching-for-knowledge)
3. [Contributing Knowledge](#contributing-knowledge)
4. [Understanding Entry Structure](#understanding-entry-structure)
5. [Using Categories and Tags](#using-categories-and-tags)
6. [CLI Commands](#cli-commands)
7. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

- Python 3.12 or higher
- `uv` package manager installed
- Access to the EfficientAI MLX Toolkit repository

### Quick Start

1. **Navigate to the knowledge base directory:**

   ```bash
   cd .knowledge-base
   ```

2. **Search for knowledge:**

   ```bash
   uv run python -m kb search "memory optimization"
   ```

3. **List all entries:**

   ```bash
   uv run python -m kb list
   ```

4. **Get knowledge base statistics:**

   ```bash
   uv run python -m kb stats
   ```

## Searching for Knowledge

### Basic Search

Search across all knowledge base entries:

```bash
uv run python -m kb search "MLX training"
```

### Advanced Search with Filters

Filter by category:

```bash
uv run python -m kb search "optimization" --category apple-silicon
```

Filter by tags:

```bash
uv run python -m kb search "patterns" --tags mlx,performance
```

Filter by difficulty:

```bash
uv run python -m kb search "tutorial" --difficulty beginner
```

### Search Results

Search results include:

- **Title** and **Category** of the entry
- **Relevance Score** (0-100)
- **Tags** and **Difficulty Level**
- **Brief Description** or highlighted content
- **File Path** for direct access

## Contributing Knowledge

### Creating New Entries

1. **Use the CLI to create from template:**

   ```bash
   uv run python -m kb create --title "Your Entry Title" --category performance --tags optimization,mlx
   ```

2. **Manual creation:**
   - Navigate to the appropriate category directory
   - Create a new `.md` file with descriptive name
   - Use the entry template structure

### Entry Templates

The knowledge base provides several templates:

- **Standard Entry** (`entry-template.md`): General knowledge entries
- **Troubleshooting** (`troubleshooting-template.md`): Problem-solution format
- **Pattern** (`pattern-template.md`): Reusable code patterns

### Contribution Workflow

1. **Identify Knowledge Gap**: Look for missing information or improvements
2. **Choose Template**: Select appropriate template for your content type
3. **Write Content**: Follow the template structure and guidelines
4. **Validate**: Ensure proper frontmatter and formatting
5. **Test**: Verify code examples work correctly
6. **Submit**: Add to knowledge base and rebuild index

## Understanding Entry Structure

### Frontmatter

Every knowledge base entry starts with YAML frontmatter:

```yaml
---
title: "Your Entry Title"
category: "category-name"
tags: ["tag1", "tag2", "tag3"]
difficulty: "beginner|intermediate|advanced"
last_updated: "2025-08-14"
contributors: ["Your Name"]
---
```

### Content Sections

Standard entries should include:

1. **Problem/Context**: What problem does this solve?
2. **Solution/Pattern**: How to solve it
3. **Code Examples**: Practical implementation
4. **Best Practices**: Tips and recommendations
5. **Related Links**: Cross-references to other entries

### Code Examples

Always include working code examples:

```python
import mlx.core as mx
import mlx.nn as nn

def optimize_memory():
    # Set memory limit for Apple Silicon
    mx.metal.set_memory_limit(8 * 1024**3)  # 8GB
    return mx.metal.get_available_memory()
```

## Using Categories and Tags

### Categories

Organize entries by primary domain:

- **apple-silicon**: Apple Silicon specific optimizations
- **mlx-framework**: MLX framework patterns and usage
- **performance**: Performance optimization techniques
- **troubleshooting**: Common issues and solutions
- **deployment**: Deployment strategies
- **testing**: Testing approaches and frameworks

### Tags

Use tags for cross-cutting concerns:

- **Technology tags**: `mlx`, `pytorch`, `coreml`
- **Technique tags**: `optimization`, `debugging`, `profiling`
- **Level tags**: `beginner`, `advanced`, `expert`
- **Platform tags**: `m1`, `m2`, `macos`

### Tag Best Practices

- Use 3-5 relevant tags per entry
- Prefer existing tags over creating new ones
- Use lowercase, hyphenated format
- Be specific but not overly narrow

## CLI Commands

### Search Commands

```bash
# Basic search
uv run python -m kb search "query"

# Search with filters
uv run python -m kb search "query" --category apple-silicon --difficulty beginner

# Search by tags
uv run python -m kb search --tags mlx,performance
```

### List Commands

```bash
# List all entries
uv run python -m kb list

# List by category
uv run python -m kb list --category troubleshooting

# List recent entries
uv run python -m kb list --recent 10
```

### Management Commands

```bash
# Create new entry
uv run python -m kb create --title "Title" --category performance

# Rebuild search index
uv run python -m kb rebuild-index

# Run quality checks
uv run python -m kb quality-check

# Check content freshness
uv run python -m kb freshness-check

# Get statistics
uv run python -m kb stats
```

## Best Practices

### For Contributors

1. **Be Specific**: Use descriptive titles and clear problem statements
2. **Include Context**: Explain when and why to use the solution
3. **Test Code**: Ensure all code examples work correctly
4. **Update Regularly**: Keep content current with framework changes
5. **Cross-Reference**: Link to related entries and external resources

### For Content Quality

1. **Follow Templates**: Use provided templates for consistency
2. **Validate Frontmatter**: Ensure all required fields are present
3. **Use Proper Formatting**: Follow Markdown best practices
4. **Include Examples**: Always provide practical code examples
5. **Add Error Handling**: Show how to handle common errors

### For Searchability

1. **Use Keywords**: Include relevant terms in title and content
2. **Tag Appropriately**: Use relevant, existing tags
3. **Write Descriptions**: Include clear problem/solution descriptions
4. **Update Usage**: Popular entries get better search ranking

### For Maintenance

1. **Regular Reviews**: Periodically review and update entries
2. **Check Links**: Ensure internal and external links work
3. **Validate Code**: Test code examples with current versions
4. **Monitor Usage**: Track which entries are most/least used

## Getting Help

### Troubleshooting

If you encounter issues:

1. **Check the troubleshooting guide**: Look for common issues
2. **Validate your entry**: Use `uv run python -m kb quality-check`
3. **Rebuild index**: Try `uv run python -m kb rebuild-index`
4. **Check file permissions**: Ensure you can read/write to `.knowledge-base/`

### Common Issues

- **Search returns no results**: Try rebuilding the index
- **Entry not appearing**: Check frontmatter format and required fields
- **CLI commands fail**: Ensure you're in the correct directory and have dependencies installed

### Support

For additional help:

- Check the [Developer Documentation](DEVELOPER_GUIDE.md)
- Review the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Look at existing entries for examples
- Consult the contribution guide in `.meta/contribution-guide.md`

---

*This guide covers the essential features of the Knowledge Base system. For advanced usage and development information, see the Developer Guide.*
