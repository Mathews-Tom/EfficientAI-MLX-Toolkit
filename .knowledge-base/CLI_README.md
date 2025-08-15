# Knowledge Base CLI

A modern, beautiful command-line interface for managing the EfficientAI-MLX-Toolkit knowledge base, built with Typer and Rich for an enhanced user experience. Fully integrated with UV package management.

## Features

- 🎨 **Beautiful UI**: Rich tables, panels, and colored output
- ⚡ **Fast**: Built with modern Python tools (Typer + Rich)
- 🔍 **Powerful Search**: Fuzzy search with relevance scoring
- 📊 **Rich Reports**: Visual quality and freshness reports
- 🛠️ **Interactive Mode**: Guided entry creation
- 📈 **Statistics**: Comprehensive knowledge base analytics

## Installation

### Prerequisites

1. **Install UV** (if not already installed):
   ```bash
   # Using pip
   pip install uv
   
   # Or using homebrew on macOS
   brew install uv
   ```

2. **Install dependencies and the CLI**:
   ```bash
   # Install dependencies
   uv sync
   
   # Install the package in development mode
   uv pip install -e .
   ```

## Usage

All CLI commands are run with `uv run kb`:

```bash
# Show help
uv run kb --help

# Show available commands
uv run kb
```

## Commands

### Entry Management

#### Create Entry
```bash
# Basic creation
uv run kb create "MLX Memory Optimization" --category apple-silicon --tags mlx,memory

# Interactive mode (recommended for new users)
uv run kb create "My Entry" --interactive

# With all options
uv run kb create "Advanced Pattern" \
  --category patterns \
  --tags python,mlx,performance \
  --difficulty advanced \
  --contributor "Tom Mathews" \
  --template pattern
```

#### List Entries
```bash
# List all entries (beautiful table format)
uv run kb list

# Filter by category
uv run kb list --category apple-silicon

# Filter by tags
uv run kb list --tags mlx,performance

# Sort and limit
uv run kb list --sort date --limit 10
```

### Search

```bash
# Fuzzy search (default)
uv run kb search "memory optimization"

# Filter by category
uv run kb search "optimization" --category apple-silicon

# Limit results
uv run kb search "troubleshooting" --limit 5
```

### Maintenance

#### Rebuild Index
```bash
uv run kb rebuild-index
```

#### Quality Check
```bash
# Basic quality check with visual report
uv run kb quality-check

# Auto-fix issues
uv run kb quality-check --auto-fix

# Save detailed report
uv run kb quality-check --output quality_report.json
```

#### Freshness Check
```bash
# Check content freshness with visual report
uv run kb freshness

# Save detailed report
uv run kb freshness --output freshness_report.json
```

### Statistics

```bash
# Show beautiful statistics dashboard
uv run kb stats
```

## Interactive Mode

For new users, interactive mode provides guided entry creation:

```bash
uv run kb create "My Entry" --interactive
```

This will prompt you for:
- Entry title
- Category (with available options shown)
- Tags (with popular tags suggested)
- Difficulty level
- Contributor name

## Visual Features

The CLI provides rich visual feedback:

- **Progress Bars**: For long operations like indexing
- **Colored Output**: Status indicators and severity levels
- **Tables**: Beautiful formatting for lists and statistics
- **Panels**: Organized information display
- **Emojis**: Visual command indicators

## Examples

### Daily Workflow

```bash
# Check knowledge base health
uv run kb stats

# Check for stale content
uv run kb freshness

# Run quality checks with auto-fix
uv run kb quality-check --auto-fix

# Search for specific topics
uv run kb search "MLX training" --category apple-silicon

# Create new entry interactively
uv run kb create "New Discovery" --interactive
```

### Maintenance Workflow

```bash
# Comprehensive quality assessment
uv run kb quality-check --check-external-links --output quality_report.json

# Content freshness analysis
uv run kb freshness --output freshness_report.json

# Rebuild index after changes
uv run kb rebuild-index

# View overall statistics
uv run kb stats
```

## Architecture

The CLI is now simplified to a single file:

- `knowledge_base/cli.py` - Main CLI implementation using Typer and Rich
- Entry point defined in `pyproject.toml` as `kb = "knowledge_base.cli:main"`

All import issues have been resolved by converting the `.meta` modules to use absolute imports.

## Troubleshooting

### Common Issues

1. **UV Not Found**
   ```bash
   # Install UV first
   pip install uv
   # or
   brew install uv
   ```

2. **Dependencies Not Installed**
   ```bash
   uv sync
   uv pip install -e .
   ```

3. **Command Not Found**
   ```bash
   # Make sure you're in the project root
   cd /path/to/EfficientAI-MLX-Toolkit
   uv run kb --help
   ```

4. **Index Not Found**
   ```bash
   uv run kb rebuild-index
   ```

### Debug Mode

The CLI provides detailed error messages and stack traces when issues occur. All operations show progress indicators and clear status messages.

## Contributing

When contributing to the knowledge base:

1. Use the interactive CLI for new entries:
   ```bash
   uv run kb create "Your Entry" --interactive
   ```

2. Run quality checks before committing:
   ```bash
   uv run kb quality-check --auto-fix
   ```

3. Check for freshness issues:
   ```bash
   uv run kb freshness
   ```

4. Update the index:
   ```bash
   uv run kb rebuild-index
   ```

The CLI makes knowledge base management both efficient and enjoyable with its modern, visual interface!