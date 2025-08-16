# Knowledge Base CLI

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