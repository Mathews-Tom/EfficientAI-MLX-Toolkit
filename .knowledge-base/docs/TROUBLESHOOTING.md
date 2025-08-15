# Knowledge Base Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues with the EfficientAI MLX Toolkit Knowledge Base system.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Search Problems](#search-problems)
3. [Entry Creation Issues](#entry-creation-issues)
4. [CLI Command Failures](#cli-command-failures)
5. [Performance Issues](#performance-issues)
6. [File Permission Problems](#file-permission-problems)
7. [Index Corruption](#index-corruption)
8. [Validation Errors](#validation-errors)

## Installation Issues

### Problem: `uv` command not found

**Symptoms:**

```bash
bash: uv: command not found
```

**Solution:**

1. Install `uv` package manager:

   ```bash
   # macOS with Homebrew
   brew install uv
   
   # Or using pip
   pip install uv
   
   # Or using the installer
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Restart your terminal or source your shell profile:

   ```bash
   source ~/.zshrc  # or ~/.bashrc
   ```

### Problem: Python version compatibility

**Symptoms:**

```bash
error: No interpreter found for Python >=3.12
```

**Solution:**

1. Check your Python version:

   ```bash
   python3 --version
   ```

2. Install Python 3.12 or higher:

   ```bash
   # macOS with Homebrew
   brew install python@3.12
   
   # Or use pyenv
   pyenv install 3.12.0
   pyenv global 3.12.0
   ```

### Problem: Missing dependencies

**Symptoms:**

```bash
ModuleNotFoundError: No module named 'pytest'
```

**Solution:**

1. Install dependencies using uv:

   ```bash
   uv sync
   ```

2. Or install manually:

   ```bash
   uv add pytest pyyaml typer rich
   ```

## Search Problems

### Problem: Search returns no results

**Symptoms:**

- Search queries return empty results
- Known entries don't appear in search

**Diagnosis:**

```bash
# Check if index exists
ls -la .knowledge-base/.meta/index.json

# Check entry count
uv run python -m kb stats
```

**Solutions:**

1. **Rebuild the search index:**

   ```bash
   uv run python -m kb rebuild-index
   ```

2. **Check entry format:**

   ```bash
   uv run python -m kb quality-check
   ```

3. **Verify file permissions:**

   ```bash
   find .knowledge-base -name "*.md" -exec ls -l {} \;
   ```

### Problem: Search results are irrelevant

**Symptoms:**

- Search returns unrelated entries
- Poor relevance scoring

**Solutions:**

1. **Use more specific queries:**

   ```bash
   # Instead of: "optimization"
   uv run python -m kb search "MLX memory optimization"
   ```

2. **Use filters:**

   ```bash
   uv run python -m kb search "training" --category mlx-framework --difficulty intermediate
   ```

3. **Check entry tags and categories:**

   ```bash
   uv run python -m kb list --category performance
   ```

### Problem: Slow search performance

**Symptoms:**

- Search takes several seconds
- CLI commands are slow to respond

**Solutions:**

1. **Rebuild index with optimization:**

   ```bash
   uv run python -m kb rebuild-index --optimize
   ```

2. **Limit search results:**

   ```bash
   uv run python -m kb search "query" --limit 5
   ```

3. **Check knowledge base size:**

   ```bash
   find .knowledge-base -name "*.md" | wc -l
   ```

## Entry Creation Issues

### Problem: Entry template not found

**Symptoms:**

```bash
FileNotFoundError: Template not found: /path/to/template.md
```

**Solutions:**

1. **Check available templates:**

   ```bash
   ls -la .knowledge-base/templates/
   ```

2. **Use correct entry type:**

   ```bash
   # Available types: standard, troubleshooting, pattern
   uv run python -m kb create --title "Title" --category performance --entry-type standard
   ```

3. **Create missing template:**

   ```bash
   cp .knowledge-base/templates/entry-template.md .knowledge-base/templates/missing-template.md
   ```

### Problem: Invalid frontmatter format

**Symptoms:**

```bash
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Solutions:**

1. **Check YAML syntax:**

   ```yaml
   ---
   title: "Your Title"  # Use quotes for strings
   tags: ["tag1", "tag2"]  # Use array format
   difficulty: "beginner"  # Valid values: beginner, intermediate, advanced
   ---
   ```

2. **Validate entry:**

   ```bash
   uv run python -m kb quality-check --file path/to/entry.md
   ```

3. **Use template:**

   ```bash
   uv run python -m kb create --title "Title" --category category
   ```

### Problem: Entry not appearing in search

**Symptoms:**

- New entry created but not found in search
- Entry exists but not indexed

**Solutions:**

1. **Rebuild index:**

   ```bash
   uv run python -m kb rebuild-index
   ```

2. **Check entry location:**

   ```bash
   # Entries must be in categories/ directory
   find .knowledge-base/categories -name "*.md"
   ```

3. **Validate entry format:**

   ```bash
   uv run python -m kb quality-check --file path/to/entry.md
   ```

## CLI Command Failures

### Problem: Command not recognized

**Symptoms:**

```bash
python: No module named kb
```

**Solutions:**

1. **Check current directory:**

   ```bash
   pwd  # Should be in project root or .knowledge-base
   ```

2. **Use full module path:**

   ```bash
   uv run python -m knowledge_base.cli search "query"
   ```

3. **Install package in development mode:**

   ```bash
   uv add -e .
   ```

### Problem: Permission denied errors

**Symptoms:**

```bash
PermissionError: [Errno 13] Permission denied: '.knowledge-base/index.json'
```

**Solutions:**

1. **Check file permissions:**

   ```bash
   ls -la .knowledge-base/
   ```

2. **Fix permissions:**

   ```bash
   chmod -R u+rw .knowledge-base/
   ```

3. **Check directory ownership:**

   ```bash
   sudo chown -R $USER:$USER .knowledge-base/
   ```

### Problem: CLI hangs or freezes

**Symptoms:**

- Commands don't complete
- No output or error messages

**Solutions:**

1. **Check for large files:**

   ```bash
   find .knowledge-base -name "*.md" -size +1M
   ```

2. **Disable parallel processing:**

   ```bash
   uv run python -m kb rebuild-index --no-parallel
   ```

3. **Check system resources:**

   ```bash
   top  # Check CPU and memory usage
   ```

## Performance Issues

### Problem: Slow indexing

**Symptoms:**

- Index building takes very long
- High CPU usage during indexing

**Solutions:**

1. **Enable parallel processing:**

   ```bash
   uv run python -m kb rebuild-index --parallel
   ```

2. **Check file count:**

   ```bash
   find .knowledge-base -name "*.md" | wc -l
   ```

3. **Use incremental updates:**

   ```bash
   uv run python -m kb update-index  # Instead of full rebuild
   ```

### Problem: High memory usage

**Symptoms:**

- System becomes slow during operations
- Out of memory errors

**Solutions:**

1. **Process files in batches:**

   ```bash
   uv run python -m kb rebuild-index --batch-size 50
   ```

2. **Clear cache:**

   ```bash
   rm -f .knowledge-base/.meta/index.json
   uv run python -m kb rebuild-index
   ```

3. **Check for large entries:**

   ```bash
   find .knowledge-base -name "*.md" -exec wc -l {} + | sort -n
   ```

## File Permission Problems

### Problem: Cannot create files

**Symptoms:**

```bash
PermissionError: [Errno 13] Permission denied: '.knowledge-base/categories/new-entry.md'
```

**Solutions:**

1. **Check directory permissions:**

   ```bash
   ls -ld .knowledge-base/categories/
   ```

2. **Fix directory permissions:**

   ```bash
   chmod 755 .knowledge-base/categories/
   ```

3. **Create directories if missing:**

   ```bash
   mkdir -p .knowledge-base/categories/performance
   ```

### Problem: Cannot read existing files

**Symptoms:**

```bash
PermissionError: [Errno 13] Permission denied: 'entry.md'
```

**Solutions:**

1. **Fix file permissions:**

   ```bash
   chmod 644 .knowledge-base/categories/*/*.md
   ```

2. **Check file ownership:**

   ```bash
   ls -la .knowledge-base/categories/
   ```

3. **Reset permissions recursively:**

   ```bash
   find .knowledge-base -type f -name "*.md" -exec chmod 644 {} \;
   find .knowledge-base -type d -exec chmod 755 {} \;
   ```

## Index Corruption

### Problem: Corrupted index file

**Symptoms:**

```bash
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Solutions:**

1. **Remove corrupted index:**

   ```bash
   rm -f .knowledge-base/.meta/index.json
   ```

2. **Rebuild from scratch:**

   ```bash
   uv run python -m kb rebuild-index --force
   ```

3. **Backup and restore:**

   ```bash
   # Create backup
   cp .knowledge-base/.meta/index.json .knowledge-base/.meta/index.json.backup
   
   # Restore from backup
   cp .knowledge-base/.meta/index.json.backup .knowledge-base/.meta/index.json
   ```

### Problem: Inconsistent index state

**Symptoms:**

- Search finds entries that don't exist
- Missing entries that should be indexed

**Solutions:**

1. **Force complete rebuild:**

   ```bash
   uv run python -m kb rebuild-index --force --clean
   ```

2. **Validate all entries:**

   ```bash
   uv run python -m kb quality-check --all
   ```

3. **Check for orphaned files:**

   ```bash
   find .knowledge-base -name "*.md" -not -path "*/templates/*"
   ```

## Validation Errors

### Problem: Frontmatter validation fails

**Symptoms:**

```bash
ValidationError: Missing required field: title
```

**Solutions:**

1. **Check required fields:**

   ```yaml
   ---
   title: "Required"
   category: "Required"
   tags: ["Required"]
   difficulty: "Required"
   last_updated: "Required"
   contributors: ["Required"]
   ---
   ```

2. **Use validation tool:**

   ```bash
   uv run python -m kb quality-check --file entry.md
   ```

3. **Fix common issues:**

   ```yaml
   # Wrong: difficulty: easy
   # Right: difficulty: "beginner"
   
   # Wrong: tags: tag1, tag2
   # Right: tags: ["tag1", "tag2"]
   ```

### Problem: Code validation errors

**Symptoms:**

```bash
SyntaxError: invalid syntax in code block
```

**Solutions:**

1. **Check code syntax:**

   ```python
   # Ensure proper indentation
   def example():
       return True  # Correct indentation
   ```

2. **Specify language:**

   ````markdown
   ```python
   # Python code here
   ```
   ````

3. **Test code separately:**

   ```bash
   python -c "import ast; ast.parse(open('code.py').read())"
   ```

## Getting Additional Help

### Diagnostic Commands

```bash
# System information
uv --version
python3 --version

# Knowledge base status
uv run python -m kb stats
uv run python -m kb quality-check

# File system check
find .knowledge-base -type f -name "*.md" | head -5
ls -la .knowledge-base/.meta/
```

### Log Files

Check for log files in:

- `.knowledge-base/.meta/logs/`
- System logs: `/var/log/` (macOS/Linux)

### Support Resources

1. **Documentation**: Check USER_GUIDE.md and DEVELOPER_GUIDE.md
2. **Examples**: Look at existing entries in `categories/`
3. **Templates**: Use templates in `templates/` directory
4. **Tests**: Run tests to verify system integrity

### Reporting Issues

When reporting issues, include:

1. **System information:**

   ```bash
   uv --version
   python3 --version
   uname -a
   ```

2. **Error messages:** Full error output
3. **Steps to reproduce:** Exact commands used
4. **File structure:** Output of `find .knowledge-base -type f`

---

*This troubleshooting guide covers common issues. For additional help, consult the User Guide and Developer Guide.*
