"""Utility functions for mellowgate mathematical operations."""

from .functions import softmax
from .statistics import sample_gumbel

__all__ = ["sample_gumbel", "softmax"]
