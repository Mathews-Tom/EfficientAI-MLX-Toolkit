# Knowledge Base Scripts

This directory contains setup, configuration, and maintenance scripts for the Knowledge Base system.

## Scripts Overview

### 🚀 Setup and Initialization

#### `init_knowledge_base.py`

Initializes a new knowledge base with complete directory structure, templates, and configuration.

```bash
# Initialize new knowledge base
uv run python init_knowledge_base.py

# Initialize with custom path
uv run python init_knowledge_base.py /path/to/kb

# Minimal setup without examples
uv run python init_knowledge_base.py --minimal

# Force overwrite existing KB
uv run python init_knowledge_base.py --force
```

#### `kb-setup.sh`

Convenient shell wrapper for all knowledge base operations.

```bash
# Initialize knowledge base
./kb-setup.sh init

# Show status
./kb-setup.sh status

# Run maintenance
./kb-setup.sh maintenance full

# Run tests
./kb-setup.sh test
```

### ⚙️ Configuration Management

#### `configure_knowledge_base.py`

Manages knowledge base configuration settings, categories, and behavior.

```bash
# Show current configuration
uv run python configure_knowledge_base.py show

# Add new category
uv run python configure_knowledge_base.py add-category machine-learning

# Configure search settings
uv run python configure_knowledge_base.py search --max-results 100 --fuzzy-threshold 0.7

# Configure quality settings
uv run python configure_knowledge_base.py quality --max-title-length 120

# Validate configuration
uv run python configure_knowledge_base.py validate
```

### 📄 Documentation Migration

#### `migrate_documentation.py`

Migrates existing documentation and markdown files to knowledge base format.

```bash
# Migrate single file
uv run python migrate_documentation.py file README.md troubleshooting --title "Setup Guide"

# Migrate entire directory
uv run python migrate_documentation.py directory docs/ performance --recursive

# Generate migration report
uv run python migrate_documentation.py report --output migration_report.md
```

### 🔧 Maintenance

#### `maintenance.py`

Performs regular maintenance tasks including quality checks, index optimization, and cleanup.

```bash
# Run full maintenance
uv run python maintenance.py full

# Rebuild search index
uv run python maintenance.py index --force

# Run quality checks
uv run python maintenance.py quality

# Check content freshness
uv run python maintenance.py freshness

# Clean up temporary files
uv run python maintenance.py cleanup

# Show maintenance status
uv run python maintenance.py status
```

## Usage Patterns

### Initial Setup

1. **Initialize knowledge base:**

   ```bash
   ./kb-setup.sh init
   ```

2. **Configure for your needs:**

   ```bash
   ./kb-setup.sh configure add-category your-domain
   ./kb-setup.sh configure search --max-results 50
   ```

3. **Migrate existing documentation:**

   ```bash
   ./kb-setup.sh migrate directory old-docs/ your-domain --recursive
   ```

### Regular Maintenance

1. **Weekly maintenance:**

   ```bash
   ./kb-setup.sh maintenance full
   ```

2. **Check status:**

   ```bash
   ./kb-setup.sh status
   ```

3. **Run tests:**

   ```bash
   ./kb-setup.sh test
   ```

### Configuration Updates

1. **Add new categories:**

   ```bash
   ./kb-setup.sh configure add-category new-category
   ```

2. **Update search settings:**

   ```bash
   ./kb-setup.sh configure search --max-results 100
   ```

3. **Validate configuration:**

   ```bash
   ./kb-setup.sh configure validate
   ```

## Script Dependencies

### Python Requirements

- Python 3.12 or higher
- PyYAML for configuration management
- Standard library modules (pathlib, argparse, json, etc.)

### System Requirements

- `uv` package manager (auto-installed by kb-setup.sh)
- Unix-like system for shell scripts (macOS, Linux)
- Write permissions in knowledge base directory

### Knowledge Base Modules

Scripts import from the `.meta` directory:

- `indexer.py` - Search indexing
- `quality_assurance.py` - Quality checks
- `freshness_tracker.py` - Content freshness
- `cross_reference.py` - Cross-reference validation
- `analytics.py` - Usage analytics
- `reporting.py` - Report generation

## Error Handling

### Common Issues

1. **Module import errors:**
   - Ensure you're running from the knowledge base directory
   - Check that `.meta` directory exists and contains required modules

2. **Permission errors:**
   - Ensure write permissions in knowledge base directory
   - Check file ownership and permissions

3. **Configuration errors:**
   - Use `configure validate` to check configuration
   - Reset to defaults with `configure reset`

### Troubleshooting

1. **Check system requirements:**

   ```bash
   ./kb-setup.sh init  # Checks and installs requirements
   ```

2. **Validate setup:**

   ```bash
   ./kb-setup.sh status
   ./kb-setup.sh test
   ```

3. **Reset if needed:**

   ```bash
   ./kb-setup.sh init --force
   ```

## Customization

### Adding New Scripts

1. Create script in this directory
2. Follow naming convention: `action_knowledge_base.py`
3. Add to `kb-setup.sh` if needed
4. Update this README

### Extending Functionality

1. **Configuration options:** Modify `configure_knowledge_base.py`
2. **Migration patterns:** Extend `migrate_documentation.py`
3. **Maintenance tasks:** Add to `maintenance.py`
4. **Shell commands:** Update `kb-setup.sh`

## Best Practices

### Script Development

- Use argparse for command-line interfaces
- Include comprehensive error handling
- Log important operations
- Provide helpful output messages
- Follow existing code patterns

### Maintenance Schedule

- **Daily:** Status checks
- **Weekly:** Full maintenance
- **Monthly:** Configuration review
- **Quarterly:** Migration of new documentation

### Security Considerations

- Scripts only operate within knowledge base directory
- No external network access required
- File permissions respected
- No sensitive data in logs

---

For more information, see the main [Knowledge Base Documentation](../docs/README.md).
