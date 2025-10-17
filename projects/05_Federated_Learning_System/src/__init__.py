"""Federated Learning System for Lightweight Models.

This package implements a privacy-preserving distributed learning system optimized
for Apple Silicon and edge devices.

Components:
    - server: Federated server architecture
    - client: Client implementation
    - privacy: Differential privacy and secure aggregation
    - communication: Efficient communication protocols
    - aggregation: Model aggregation strategies
"""

__version__ = "0.1.0"
__author__ = "AetherForge"

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

__all__ = [
    "__version__",
    "__author__",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
]
