---
inclusion: always
---

# Development Standards for EfficientAI-MLX-Toolkit

## Code Quality Standards

### Type Hints and Documentation

- Use type hints for all function parameters and return values
- Provide comprehensive docstrings following Google style
- Include examples in docstrings for complex functions
- Document Apple Silicon-specific optimizations and requirements

### Testing Requirements

- Write unit tests for all utility functions
- Include integration tests for cross-project compatibility
- Test Apple Silicon optimizations with mock hardware when needed
- Provide performance regression tests

### Project Structure

- Follow the standardized project structure for all individual projects
- Use consistent naming conventions across all projects
- Organize code into logical modules (models, training, inference, utils)
- Maintain clear separation between project-specific and shared utilities

## Performance Standards

### Benchmarking Requirements

- Include performance benchmarks for all optimization techniques
- Compare against baseline implementations
- Measure memory usage, training time, and inference speed
- Provide Apple Silicon-specific performance metrics

### Memory Optimization

- Implement memory-efficient training strategies
- Use gradient accumulation for larger effective batch sizes
- Provide memory usage monitoring and reporting
- Optimize for Apple Silicon's unified memory architecture

## Documentation Standards

### README Requirements

- Include clear installation instructions using uv
- Provide usage examples with pathlib-based file operations
- Document Apple Silicon-specific requirements and optimizations
- Include performance benchmarks and comparisons

### API Documentation

- Document all public APIs with clear examples
- Include parameter descriptions and expected types
- Provide usage examples for common workflows
- Document hardware requirements and optimizations

## Example Code Structure

```python
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Apple Silicon-optimized model trainer.
    
    This class provides efficient training capabilities optimized for
    Apple Silicon hardware using MLX framework.
    
    Args:
        model_path: Path to model configuration file
        data_path: Path to training data directory
        optimization_level: Level of Apple Silicon optimization (1-3)
    
    Example:
        >>> trainer = ModelTrainer(
        ...     model_path=Path("models/config.yaml"),
        ...     data_path=Path("data/training"),
        ...     optimization_level=2
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model_path: Path,
        data_path: Path,
        optimization_level: int = 2
    ) -> None:
        self.model_path = model_path
        self.data_path = data_path
        self.optimization_level = optimization_level
        
    def train(self) -> Dict[str, float]:
        """Train the model with Apple Silicon optimizations.
        
        Returns:
            Dictionary containing training metrics including loss,
            accuracy, and performance statistics.
        """
        # Implementation here
        pass
```
