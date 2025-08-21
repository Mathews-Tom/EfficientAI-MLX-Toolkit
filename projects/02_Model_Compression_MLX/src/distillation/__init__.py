"""
Knowledge distillation module for model compression.

Provides distillation capabilities for:
- Teacher-student training
- CPU deployment optimization
- Knowledge transfer techniques
"""

from .distiller import KnowledgeDistiller, TeacherStudentPair

__all__ = [
    "KnowledgeDistiller",
    "TeacherStudentPair",
]