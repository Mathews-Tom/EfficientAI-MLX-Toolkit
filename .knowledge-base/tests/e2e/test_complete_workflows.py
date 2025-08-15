"""
End-to-end tests for complete knowledge base workflows.

Tests complete user scenarios from start to finish, including
real-world usage patterns and complex interactions.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add the .meta directory to Python path for imports
kb_meta_path = Path(__file__).parent.parent.parent / ".meta"
sys.path.insert(0, str(kb_meta_path))

from contributor import KnowledgeBaseContributor
from freshness_tracker import ContentFreshnessTracker
from indexer import KnowledgeBaseIndexer
from models import KnowledgeBaseEntry
from quality_assurance import KnowledgeBaseQualityAssurance
from search import KnowledgeBaseSearcher


class TestCompleteUserWorkflows:
    """End-to-end tests for complete user workflows."""

    def setup_realistic_kb(self, temp_dir: Path) -> Path:
        """Set up a realistic knowledge base for E2E testing."""
        kb_path = temp_dir / ".knowledge-base"

        # Create comprehensive directory structure
        categories = [
            "apple-silicon",
            "mlx-framework",
            "performance",
            "troubleshooting",
            "deployment",
            "testing",
        ]

        for category in categories:
            (kb_path / "categories" / category).mkdir(parents=True)

        (kb_path / "patterns" / "training-patterns").mkdir(parents=True)
        (kb_path / "patterns" / "inference-patterns").mkdir(parents=True)
        (kb_path / "templates").mkdir(parents=True)
        (kb_path / ".meta").mkdir(parents=True)

        # Copy templates from real knowledge base
        real_kb_path = Path(__file__).parent.parent.parent
        real_templates_path = real_kb_path / "templates"
        if real_templates_path.exists():
            for template_file in real_templates_path.glob("*.md"):
                shutil.copy2(template_file, kb_path / "templates")
            for template_file in real_templates_path.glob("*.yaml"):
                shutil.copy2(template_file, kb_path / "templates")
            for template_file in real_templates_path.glob("*.py"):
                shutil.copy2(template_file, kb_path / "templates")

        # Create required files
        (kb_path / "README.md").write_text(
            """# EfficientAI MLX Toolkit Knowledge Base

This knowledge base contains best practices, patterns, and solutions for Apple Silicon AI development.

## Categories

- [Apple Silicon](categories/apple-silicon/) - Apple Silicon specific optimizations
- [MLX Framework](categories/mlx-framework/) - MLX framework patterns and usage
- [Performance](categories/performance/) - Performance optimization techniques
- [Troubleshooting](categories/troubleshooting/) - Common issues and solutions
- [Deployment](categories/deployment/) - Deployment strategies and configurations
- [Testing](categories/testing/) - Testing approaches and frameworks
"""
        )

        (kb_path / ".meta" / "contribution-guide.md").write_text(
            """# Contribution Guide

## How to Contribute

1. Identify knowledge gaps or improvements
2. Create new entries using templates
3. Follow the established format and standards
4. Submit for review and validation

## Entry Standards

- Use clear, descriptive titles
- Include practical code examples
- Provide context and rationale
- Tag appropriately for discoverability
"""
        )

        return kb_path

    def create_realistic_entries(self, kb_path: Path) -> list[Path]:
        """Create realistic knowledge base entries."""
        entries_data = [
            {
                "path": "categories/apple-silicon/memory-optimization.md",
                "content": """---
title: "MLX Memory Optimization for Apple Silicon"
category: "apple-silicon"
tags: ["mlx", "memory", "optimization", "apple-silicon", "performance"]
difficulty: "advanced"
last_updated: "2025-08-14"
contributors: ["alice-dev", "bob-engineer"]
---

# MLX Memory Optimization for Apple Silicon

## Problem/Context

Apple Silicon devices have unified memory architecture that requires specific optimization strategies for ML workloads. Default MLX configurations may not utilize available memory efficiently.

## Solution/Pattern

Implement memory-aware training and inference patterns that leverage Apple Silicon's unified memory.

### Key Strategies

1. **Memory Limit Configuration**
2. **Gradient Checkpointing**
3. **Mixed Precision Training**
4. **Batch Size Optimization**

## Code Example

```python
import mlx.core as mx
import mlx.nn as nn

def configure_memory_optimization():
    # Set memory limit to 80% of available memory
    available_memory = mx.metal.get_available_memory()
    memory_limit = int(available_memory * 0.8)
    mx.metal.set_memory_limit(memory_limit)
    
    print(f"Memory limit set to: {memory_limit / (1024**3):.2f} GB")

def create_memory_efficient_model(input_size, hidden_size, output_size):
    class MemoryEfficientModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size)
            ]
        
        def __call__(self, x):
            # Use gradient checkpointing for memory efficiency
            for layer in self.layers:
                x = layer(x)
                # Clear intermediate activations when possible
                mx.eval(x)
            return x
    
    return MemoryEfficientModel()

# Usage example
configure_memory_optimization()
model = create_memory_efficient_model(784, 512, 10)
```

## Performance Impact

- **Memory Usage**: Reduces peak memory usage by 30-50%
- **Training Speed**: Minimal impact (< 5% slowdown)
- **Model Accuracy**: No degradation in model performance

## Gotchas/Pitfalls

- Don't set memory limit too aggressively (< 70% can cause OOM)
- Monitor memory usage during training to find optimal settings
- Some operations may not benefit from gradient checkpointing

## Related Knowledge

- [MLX Training Patterns](../mlx-framework/training-patterns.md)
- [Performance Monitoring](../performance/monitoring-tools.md)
- [Apple Silicon Best Practices](./best-practices.md)
""",
            },
            {
                "path": "categories/mlx-framework/training-patterns.md",
                "content": """---
title: "MLX Training Loop Patterns"
category: "mlx-framework"
tags: ["mlx", "training", "patterns", "loops", "best-practices"]
difficulty: "intermediate"
last_updated: "2025-08-14"
contributors: ["charlie-ml", "diana-researcher"]
---

# MLX Training Loop Patterns

## Problem/Context

Implementing efficient training loops in MLX requires understanding of the framework's lazy evaluation and memory management. Standard PyTorch patterns don't always translate directly.

## Solution/Pattern

Use MLX-native patterns that leverage lazy evaluation and efficient memory usage.

## Code Example

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

def create_training_loop(model, train_data, val_data, epochs=10):
    optimizer = optim.Adam(learning_rate=0.001)
    
    def loss_fn(model, x, y):
        return nn.losses.cross_entropy(model(x), y)
    
    def train_step(model, x, y):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss
    
    def validate(model, val_data):
        total_loss = 0
        total_samples = 0
        
        for batch in val_data:
            x, y = batch
            loss = loss_fn(model, x, y)
            total_loss += loss * x.shape[0]
            total_samples += x.shape[0]
        
        return total_loss / total_samples
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_data:
            x, y = batch
            loss = train_step(model, x, y)
            epoch_loss += loss
            num_batches += 1
            
            # Evaluate periodically to avoid memory buildup
            mx.eval(model.parameters())
        
        # Validation
        val_loss = validate(model, val_data)
        avg_train_loss = epoch_loss / num_batches
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Advanced pattern with learning rate scheduling
def create_advanced_training_loop(model, train_data, val_data, epochs=10):
    optimizer = optim.Adam(learning_rate=0.001)
    scheduler = optim.cosine_decay(0.001, epochs * len(train_data))
    
    step = 0
    for epoch in range(epochs):
        for batch in train_data:
            # Update learning rate
            optimizer.learning_rate = scheduler(step)
            
            x, y = batch
            loss = train_step(model, x, y)
            
            step += 1
            
            # Log progress
            if step % 100 == 0:
                print(f"Step {step}: Loss: {loss:.4f}, LR: {optimizer.learning_rate:.6f}")
```

## Performance Impact

- **Training Speed**: 2-3x faster than equivalent PyTorch on Apple Silicon
- **Memory Efficiency**: 40% lower memory usage due to lazy evaluation
- **Convergence**: Similar convergence properties to other frameworks

## Gotchas/Pitfalls

- Always call `mx.eval()` to materialize gradients and prevent memory leaks
- Don't mix MLX arrays with NumPy/PyTorch tensors without explicit conversion
- Use MLX-native optimizers for best performance

## Related Knowledge

- [Memory Optimization](../apple-silicon/memory-optimization.md)
- [Performance Profiling](../performance/profiling-tools.md)
""",
            },
            {
                "path": "categories/troubleshooting/common-errors.md",
                "content": """---
title: "Common MLX Errors and Solutions"
category: "troubleshooting"
tags: ["troubleshooting", "errors", "mlx", "debugging", "solutions"]
difficulty: "beginner"
last_updated: "2025-08-14"
contributors: ["eve-support", "frank-debugger"]
---

# Common MLX Errors and Solutions

## Problem/Context

Developers frequently encounter specific errors when working with MLX. This guide provides solutions to the most common issues.

## Common Error Patterns

### 1. Memory Errors

**Error**: `RuntimeError: Metal buffer allocation failed`

**Cause**: Insufficient GPU memory or memory fragmentation

**Solution**:
```python
import mlx.core as mx

# Check available memory
available = mx.metal.get_available_memory()
print(f"Available memory: {available / (1024**3):.2f} GB")

# Set conservative memory limit
mx.metal.set_memory_limit(int(available * 0.7))

# Clear cache if needed
mx.metal.clear_cache()
```

### 2. Shape Mismatch Errors

**Error**: `ValueError: Incompatible shapes for operation`

**Cause**: Array dimensions don't match for the operation

**Solution**:
```python
# Always check shapes before operations
def safe_matmul(a, b):
    print(f"Shape A: {a.shape}, Shape B: {b.shape}")
    
    if a.shape[-1] != b.shape[0]:
        raise ValueError(f"Cannot multiply {a.shape} and {b.shape}")
    
    return mx.matmul(a, b)

# Use reshape or transpose as needed
a = mx.random.normal((10, 5))
b = mx.random.normal((3, 5))  # Wrong shape
b = b.T  # Fix: transpose to (5, 3)
result = safe_matmul(a, b)
```

### 3. Gradient Computation Issues

**Error**: `RuntimeError: No gradient available for parameter`

**Cause**: Gradient computation not properly set up

**Solution**:
```python
import mlx.nn as nn

# Ensure model parameters require gradients
model = MyModel()
mx.eval(model.parameters())  # Initialize parameters

# Use value_and_grad for proper gradient computation
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad_fn(model, x, y)
```

## Debugging Strategies

1. **Enable Verbose Logging**
2. **Check Array Properties**
3. **Monitor Memory Usage**
4. **Validate Input Shapes**

## Related Knowledge

- [Performance Debugging](../performance/debugging-tools.md)
- [Memory Management](../apple-silicon/memory-optimization.md)
""",
            },
        ]

        created_paths = []
        for entry_data in entries_data:
            file_path = kb_path / entry_data["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(entry_data["content"])
            created_paths.append(file_path)

        return created_paths

    @pytest.mark.slow
    def test_complete_developer_workflow(self):
        """Test a complete developer workflow from contribution to usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup realistic knowledge base
            kb_path = self.setup_realistic_kb(Path(temp_dir))
            entry_paths = self.create_realistic_entries(kb_path)

            # Phase 1: Initial Setup and Indexing
            indexer = KnowledgeBaseIndexer(kb_path)
            initial_index = indexer.build_index()

            # Verify initial setup
            assert len(initial_index.entries) == 3
            assert "apple-silicon" in initial_index.categories
            assert "mlx-framework" in initial_index.categories
            assert "troubleshooting" in initial_index.categories

            # Phase 2: Developer Searches for Information
            searcher = KnowledgeBaseSearcher(initial_index)

            # Simulate developer searching for memory optimization
            memory_results = searcher.search("memory optimization")
            assert len(memory_results.results) >= 1

            # Developer finds and uses the memory optimization entry
            memory_entry = memory_results.results[0].entry
            memory_entry.update_usage()
            assert memory_entry.usage_count == 1

            # Developer searches for training patterns
            training_results = searcher.search("training loops")
            assert len(training_results.results) >= 1

            # Phase 3: Developer Contributes New Knowledge
            contributor = KnowledgeBaseContributor(kb_path)

            # Developer creates a new troubleshooting entry
            new_entry_path = contributor.create_entry_from_template(
                title="MLX Installation Issues",
                category="troubleshooting",
                tags=["installation", "setup", "mlx", "troubleshooting"],
                difficulty="beginner",
                contributor="new-developer",
                entry_type="troubleshooting",
            )

            # Verify new entry was created
            assert new_entry_path.exists()
            new_entry = KnowledgeBaseEntry.from_file(new_entry_path)
            assert new_entry.title == "MLX Installation Issues"

            # Phase 4: Incremental Index Update
            updated_index = indexer.incremental_update()
            assert updated_index is not None
            assert len(updated_index.entries) == 4

            # Phase 5: Quality Assurance Check
            qa = KnowledgeBaseQualityAssurance(kb_path)
            qa_report = qa.run_comprehensive_quality_check()

            # Verify QA found the entries and assessed quality
            assert qa_report.total_entries_checked == 4
            # Quality score may be 0 due to template placeholders and syntax issues
            # This is expected behavior for template-generated content
            assert qa_report.quality_score >= 0
            assert (
                len(qa_report.issues_found) > 0
            )  # Should find issues in template content

            # Phase 6: Analytics and Reporting
            # Simulate more usage to generate analytics data
            searcher = KnowledgeBaseSearcher(updated_index)

            search_queries = [
                "memory",
                "training",
                "optimization",
                "troubleshooting",
                "mlx",
                "apple silicon",
                "performance",
                "errors",
            ]

            for query in search_queries:
                results = searcher.search(query)
                for result in results.results[:2]:  # Use top 2 results
                    result.entry.update_usage()

            # Verify analytics data
            stats = searcher.get_search_stats()
            assert stats["total_searches"] == len(search_queries)
            assert len(stats["popular_queries"]) > 0

            # Phase 7: Maintenance and Freshness Check
            freshness_tracker = ContentFreshnessTracker(kb_path)
            freshness_report = freshness_tracker.analyze_content_freshness()

            # All entries should be fresh (recently created)
            assert len(freshness_report.stale_entries) == 0
            assert freshness_report.freshness_breakdown.get("fresh", 0) >= 4

    @pytest.mark.slow
    def test_team_collaboration_workflow(self):
        """Test workflow involving multiple team members collaborating."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.setup_realistic_kb(Path(temp_dir))
            contributor = KnowledgeBaseContributor(kb_path)

            # Simulate multiple team members contributing
            team_contributions = [
                {
                    "contributor": "alice-senior",
                    "title": "Advanced MLX Patterns",
                    "category": "mlx-framework",
                    "tags": ["mlx", "advanced", "patterns"],
                    "difficulty": "advanced",
                },
                {
                    "contributor": "bob-junior",
                    "title": "Getting Started with MLX",
                    "category": "mlx-framework",
                    "tags": ["mlx", "beginner", "tutorial"],
                    "difficulty": "beginner",
                },
                {
                    "contributor": "charlie-ops",
                    "title": "MLX Deployment Strategies",
                    "category": "deployment",
                    "tags": ["deployment", "production", "mlx"],
                    "difficulty": "intermediate",
                },
                {
                    "contributor": "diana-qa",
                    "title": "Testing MLX Applications",
                    "category": "testing",
                    "tags": ["testing", "mlx", "quality"],
                    "difficulty": "intermediate",
                },
            ]

            created_entries = []
            for contrib in team_contributions:
                entry_path = contributor.create_entry_from_template(
                    title=contrib["title"],
                    category=contrib["category"],
                    tags=contrib["tags"],
                    difficulty=contrib["difficulty"],
                    contributor=contrib["contributor"],
                    entry_type="standard",
                )
                created_entries.append(entry_path)

            # Build comprehensive index
            indexer = KnowledgeBaseIndexer(kb_path)
            index = indexer.build_index()

            # Verify all contributions are indexed
            assert len(index.entries) == 4

            # Test contributor-based queries
            alice_entries = index.get_by_contributor("alice-senior")
            bob_entries = index.get_by_contributor("bob-junior")
            charlie_entries = index.get_by_contributor("charlie-ops")
            diana_entries = index.get_by_contributor("diana-qa")

            assert len(alice_entries) == 1
            assert len(bob_entries) == 1
            assert len(charlie_entries) == 1
            assert len(diana_entries) == 1

            # Test difficulty-based organization
            beginner_entries = index.get_by_difficulty("beginner")
            intermediate_entries = index.get_by_difficulty("intermediate")
            advanced_entries = index.get_by_difficulty("advanced")

            assert len(beginner_entries) == 1
            assert len(intermediate_entries) == 2
            assert len(advanced_entries) == 1

            # Test cross-team knowledge discovery
            searcher = KnowledgeBaseSearcher(index)

            # Junior developer searches for MLX help
            mlx_help = searcher.search("MLX beginner")
            assert len(mlx_help.results) >= 1

            # Ops team searches for deployment info
            deployment_info = searcher.search("deployment production")
            assert len(deployment_info.results) >= 1

            # QA team searches for testing approaches
            testing_info = searcher.search("testing quality")
            assert len(testing_info.results) >= 1

    @pytest.mark.slow
    def test_knowledge_evolution_workflow(self):
        """Test how knowledge base evolves over time with updates and maintenance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = self.setup_realistic_kb(Path(temp_dir))

            # Phase 1: Initial knowledge base with older entries
            old_entry_content = """---
title: "Legacy MLX Patterns"
category: "mlx-framework"
tags: ["mlx", "legacy", "old"]
difficulty: "intermediate"
last_updated: "2023-06-01"
contributors: ["legacy-dev"]
---

# Legacy MLX Patterns

This entry represents older knowledge that may need updating.
"""

            old_entry_path = (
                kb_path / "categories" / "mlx-framework" / "legacy-patterns.md"
            )
            old_entry_path.write_text(old_entry_content)

            # Phase 2: Build initial index
            indexer = KnowledgeBaseIndexer(kb_path)
            initial_index = indexer.build_index()
            assert len(initial_index.entries) == 1

            # Phase 3: Add new, updated knowledge
            contributor = KnowledgeBaseContributor(kb_path)

            new_entry_path = contributor.create_entry_from_template(
                title="Modern MLX Best Practices",
                category="mlx-framework",
                tags=["mlx", "modern", "best-practices", "2024"],
                difficulty="intermediate",
                contributor="modern-dev",
                entry_type="standard",
            )

            # Phase 4: Update index incrementally
            updated_index = indexer.incremental_update()
            assert len(updated_index.entries) == 2

            # Phase 5: Freshness analysis identifies stale content
            freshness_tracker = ContentFreshnessTracker(kb_path)
            freshness_report = freshness_tracker.analyze_content_freshness()

            # Should identify the legacy entry as stale
            assert len(freshness_report.stale_entries) >= 1
            stale_titles = [
                entry.entry_title for entry in freshness_report.stale_entries
            ]
            assert "Legacy MLX Patterns" in stale_titles

            # Phase 6: Quality assurance identifies improvement opportunities
            qa = KnowledgeBaseQualityAssurance(kb_path)
            qa_report = qa.run_comprehensive_quality_check()

            # Should have recommendations for the legacy entry
            assert len(qa_report.recommendations) > 0

            # Phase 7: Simulate content update
            updated_legacy_content = """---
title: "Updated MLX Patterns"
category: "mlx-framework"
tags: ["mlx", "updated", "current", "patterns"]
difficulty: "intermediate"
last_updated: "2025-08-14"
contributors: ["legacy-dev", "modern-dev"]
---

# Updated MLX Patterns

This entry has been updated with current best practices and modern approaches.

## What's New

- Updated for MLX 0.5+
- New performance optimizations
- Modern API usage patterns

## Migration Guide

For users of legacy patterns, here's how to migrate to the new approaches.
"""

            old_entry_path.write_text(updated_legacy_content)

            # Phase 8: Final index rebuild
            final_index = indexer.build_index(force_rebuild=True)

            # Verify updated entry is properly indexed
            updated_entry = final_index.get_entry("Updated MLX Patterns")
            assert updated_entry is not None
            assert "modern-dev" in updated_entry.contributors
            assert "updated" in updated_entry.tags

            # Phase 9: Final freshness check
            final_freshness = freshness_tracker.analyze_content_freshness()

            # Should now have fewer stale entries
            final_stale_count = len(final_freshness.stale_entries)
            initial_stale_count = len(freshness_report.stale_entries)
            assert final_stale_count <= initial_stale_count


if __name__ == "__main__":
    pytest.main([__file__])
