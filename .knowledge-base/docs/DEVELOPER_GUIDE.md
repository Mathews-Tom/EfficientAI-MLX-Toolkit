# Knowledge Base Developer Guide

## Overview

This guide provides technical documentation for developers who want to extend, maintain, or integrate with the EfficientAI MLX Toolkit Knowledge Base system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Data Models](#data-models)
4. [API Reference](#api-reference)
5. [Extending the System](#extending-the-system)
6. [Testing](#testing)
7. [Performance Considerations](#performance-considerations)
8. [Maintenance Tasks](#maintenance-tasks)

## Architecture Overview

### System Design

The knowledge base follows a modular architecture with clear separation of concerns:

```
.knowledge-base/
├── .meta/                    # Core system modules
│   ├── models.py            # Data models and validation
│   ├── indexer.py           # Indexing and file processing
│   ├── search.py            # Search functionality
│   ├── contributor.py       # Entry creation and management
│   ├── quality_assurance.py # Quality checking
│   ├── freshness_tracker.py # Content freshness analysis
│   ├── cross_reference.py   # Cross-reference management
│   ├── maintenance.py       # Automated maintenance
│   ├── analytics.py         # Usage analytics
│   └── reporting.py         # Report generation
├── categories/              # Knowledge entries by category
├── patterns/               # Reusable code patterns
├── templates/              # Entry templates
└── tests/                  # Comprehensive test suite
```

### Data Flow

1. **Entry Creation**: Contributors create entries using templates
2. **Indexing**: System scans and indexes all entries
3. **Search**: Users query the indexed knowledge base
4. **Analytics**: System tracks usage and generates insights
5. **Maintenance**: Automated quality checks and updates

## Core Components

### Models (`models.py`)

Defines the core data structures:

- `KnowledgeBaseEntry`: Individual knowledge entries
- `KnowledgeBaseIndex`: Collection of entries with search capabilities
- `SearchResult`: Search result with relevance scoring
- `SearchFilter`: Query filtering options

### Indexer (`indexer.py`)

Handles knowledge base indexing:

- Scans directory structure for entries
- Parses frontmatter and content
- Builds searchable indexes
- Supports incremental updates

### Search (`search.py`)

Provides search functionality:

- Text-based search with relevance scoring
- Category and tag filtering
- Fuzzy matching and suggestions
- Usage tracking and analytics

### Quality Assurance (`quality_assurance.py`)

Ensures content quality:

- Validates frontmatter format
- Checks code syntax
- Identifies broken links
- Detects duplicate content

## Data Models

### KnowledgeBaseEntry

```python
@dataclass
class KnowledgeBaseEntry:
    title: str
    category: str
    tags: List[str]
    difficulty: str
    last_updated: datetime
    contributors: List[str]
    file_path: Path
    usage_count: int = 0

    def get_content(self) -> str:
        """Get the full content of the entry."""

    def update_usage(self) -> None:
        """Increment usage counter."""

    def is_stale(self, days: int = 180) -> bool:
        """Check if entry is stale."""
```

### KnowledgeBaseIndex

```python
@dataclass
class KnowledgeBaseIndex:
    entries: List[KnowledgeBaseEntry]
    categories: Dict[str, List[KnowledgeBaseEntry]]
    tags: Dict[str, List[KnowledgeBaseEntry]]
    full_text_index: Dict[str, Set[int]]

    def search(self, query: str, filters: SearchFilter = None) -> SearchResults:
        """Search entries with optional filtering."""

    def add_entry(self, entry: KnowledgeBaseEntry) -> None:
        """Add entry to index."""

    def remove_entry(self, entry_path: Path) -> None:
        """Remove entry from index."""
```

## API Reference

### Core Classes

#### KnowledgeBaseIndexer

```python
class KnowledgeBaseIndexer:
    def __init__(self, kb_path: Path, enable_parallel: bool = True):
        """Initialize indexer with knowledge base path."""

    def build_index(self) -> KnowledgeBaseIndex:
        """Build complete index from scratch."""

    def incremental_update(self) -> Optional[KnowledgeBaseIndex]:
        """Update index with only changed files."""

    def save_index(self, index: KnowledgeBaseIndex) -> None:
        """Save index to disk."""

    def load_index(self) -> Optional[KnowledgeBaseIndex]:
        """Load index from disk."""
```

#### KnowledgeBaseSearcher

```python
class KnowledgeBaseSearcher:
    def __init__(self, index: KnowledgeBaseIndex):
        """Initialize searcher with index."""

    def search(self, query: str, filters: SearchFilter = None,
               sort_by: str = "relevance", limit: int = 10) -> SearchResults:
        """Search knowledge base with filtering and sorting."""

    def suggest_similar_entries(self, entry: KnowledgeBaseEntry,
                               limit: int = 5) -> List[KnowledgeBaseEntry]:
        """Find similar entries based on content."""

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search usage statistics."""
```

#### KnowledgeBaseContributor

```python
class KnowledgeBaseContributor:
    def __init__(self, kb_path: Path):
        """Initialize contributor with knowledge base path."""

    def create_entry_from_template(self, title: str, category: str,
                                  tags: List[str], difficulty: str,
                                  contributor: str, entry_type: str = "standard") -> Path:
        """Create new entry from template."""

    def validate_entry(self, entry_path: Path) -> List[ValidationError]:
        """Validate entry format and content."""

    def update_entry_metadata(self, entry_path: Path, **kwargs) -> None:
        """Update entry frontmatter."""
```

### Utility Functions

#### Validation

```python
def validate_frontmatter(frontmatter: Dict[str, Any]) -> List[ValidationError]:
    """Validate entry frontmatter format."""

def validate_markdown_content(content: str) -> List[ValidationError]:
    """Validate markdown content structure."""

def validate_code_examples(content: str) -> List[ValidationError]:
    """Validate code blocks for syntax errors."""
```

#### File Operations

```python
def parse_markdown_file(file_path: Path) -> Tuple[Dict[str, Any], str]:
    """Parse markdown file into frontmatter and content."""

def create_entry_from_template(template_path: Path, entry_data: Dict[str, Any]) -> str:
    """Generate entry content from template."""
```

## Extending the System

### Adding New Entry Types

1. **Create Template**: Add new template in `templates/` directory
2. **Update Contributor**: Modify `contributor.py` to handle new type
3. **Add Validation**: Extend validation rules if needed
4. **Update Tests**: Add tests for new entry type

Example:
```python
# In contributor.py
def _get_template_path(self, entry_type: str) -> Path:
    template_mapping = {
        "standard": "entry-template.md",
        "troubleshooting": "troubleshooting-template.md",
        "pattern": "pattern-template.md",
        "tutorial": "tutorial-template.md",  # New type
    }
    return self.kb_path / "templates" / template_mapping[entry_type]
```

### Adding Search Features

1. **Extend SearchFilter**: Add new filter options
2. **Update Search Logic**: Modify search algorithms
3. **Add Sorting Options**: Implement new sorting methods
4. **Update CLI**: Add command-line options

Example:
```python
# In search.py
@dataclass
class SearchFilter:
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    difficulty: Optional[List[str]] = None
    contributors: Optional[List[str]] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    min_usage: Optional[int] = None
    language: Optional[str] = None  # New filter
```

### Custom Analytics

1. **Extend Analytics Class**: Add new metrics
2. **Update Tracking**: Modify usage tracking
3. **Add Reports**: Create new report types
4. **Update CLI**: Add reporting commands

Example:
```python
# In analytics.py
class KnowledgeBaseAnalytics:
    def get_language_usage_stats(self) -> Dict[str, int]:
        """Get statistics by programming language."""

    def get_contributor_activity(self) -> Dict[str, Dict[str, int]]:
        """Get contributor activity metrics."""
```

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_models.py      # Data model tests
│   ├── test_indexer.py     # Indexing tests
│   ├── test_search.py      # Search functionality tests
│   └── test_validation.py  # Validation tests
├── integration/            # Integration tests
│   ├── test_workflows.py   # End-to-end workflow tests
│   └── test_cli_integration.py  # CLI integration tests
└── e2e/                   # End-to-end tests
    └── test_complete_workflows.py  # Full system tests
```

### Running Tests

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test categories
uv run python -m pytest tests/unit/ -v
uv run python -m pytest tests/integration/ -v
uv run python -m pytest tests/e2e/ -v

# Run with coverage
uv run python -m pytest tests/ --cov=.meta --cov-report=html
```

### Writing Tests

Follow these patterns:

```python
class TestKnowledgeBaseEntry:
    def test_entry_creation_valid(self):
        """Test creating valid entry."""
        entry = KnowledgeBaseEntry(
            title="Test Entry",
            category="testing",
            tags=["test", "example"],
            difficulty="beginner",
            last_updated=datetime.now(),
            contributors=["test-user"],
            file_path=Path("test.md")
        )
        assert entry.title == "Test Entry"
        assert entry.usage_count == 0
```

## Performance Considerations

### Indexing Performance

- **Parallel Processing**: Enable for large knowledge bases
- **Incremental Updates**: Use for frequent changes
- **Caching**: Index is cached to disk for faster startup

```python
# Enable parallel processing for large knowledge bases
indexer = KnowledgeBaseIndexer(kb_path, enable_parallel=True)

# Use incremental updates for better performance
updated_index = indexer.incremental_update()
```

### Search Performance

- **Full-Text Index**: Pre-built for fast text search
- **Category/Tag Indexes**: O(1) lookup for filtered searches
- **Result Limiting**: Configurable result limits

```python
# Optimize search with filters
results = searcher.search(
    query="optimization",
    filters=SearchFilter(categories=["performance"]),
    limit=5  # Limit results for better performance
)
```

### Memory Usage

- **Lazy Loading**: Content loaded on demand
- **Index Compression**: Efficient storage of search indexes
- **Cleanup**: Regular cleanup of unused data

## Maintenance Tasks

### Regular Maintenance

1. **Quality Checks**: Run weekly quality assessments
2. **Freshness Analysis**: Monthly content freshness reviews
3. **Index Optimization**: Periodic index rebuilding
4. **Usage Analytics**: Regular usage pattern analysis

### Automated Tasks

```python
# Quality assurance
qa = KnowledgeBaseQualityAssurance(kb_path)
report = qa.run_comprehensive_quality_check()

# Freshness tracking
tracker = ContentFreshnessTracker(kb_path)
freshness_report = tracker.analyze_content_freshness()

# Cross-reference validation
cross_ref = CrossReferenceAnalyzer(kb_path)
cross_ref_report = cross_ref.analyze_cross_references()
```

### Monitoring

Monitor these metrics:

- **Entry Count**: Total number of entries
- **Category Distribution**: Entries per category
- **Quality Score**: Overall quality metrics
- **Search Performance**: Query response times
- **Usage Patterns**: Popular entries and search terms

### Backup and Recovery

1. **Regular Backups**: Backup entire `.knowledge-base/` directory
2. **Index Recovery**: Rebuild indexes from source files
3. **Version Control**: Use git for change tracking
4. **Export Options**: Export to various formats for backup

---

*This developer guide provides the technical foundation for working with the Knowledge Base system. For user-facing documentation, see the User Guide.*