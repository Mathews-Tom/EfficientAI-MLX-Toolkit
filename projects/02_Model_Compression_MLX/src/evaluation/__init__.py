"""
Evaluation module for compressed models.

Provides evaluation tools for:
- Accuracy assessment
- Performance measurement  
- Compression analysis
- Comprehensive reporting
"""

from .evaluator import CompressionEvaluator

__all__ = [
    "CompressionEvaluator",
]