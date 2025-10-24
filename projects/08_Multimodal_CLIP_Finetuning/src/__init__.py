"""
Multimodal CLIP Fine-tuning Framework.

Domain-specific CLIP fine-tuning with PyTorch MPS optimization for specialized
image-text understanding (medical, industrial, scientific domains).
"""

from config import CLIPFinetuningConfig
from device_manager import DeviceManager
from model import CLIPFinetuningController

__version__ = "0.1.0"

__all__ = [
    "CLIPFinetuningConfig",
    "CLIPFinetuningController",
    "DeviceManager",
]
