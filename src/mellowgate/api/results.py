"""Data structures for storing and managing experimental results.

This module provides containers for organizing results from gradient estimation
experiments. The ResultsContainer class offers a structured way to store
gradients, metadata, and experimental parameters, making it easy to analyze
and compare different estimation methods.

The container supports adding results incrementally and provides convenient
access patterns for downstream analysis and visualization.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import jax.numpy as jnp


@dataclass
class ResultsContainer:
    """Container for storing experiment results and metadata."""

    theta_values: jnp.ndarray
    gradient_estimates: Dict[str, Dict[str, jnp.ndarray]]
    sampled_points: Optional[Dict[str, jnp.ndarray]] = None
    expectation_values: Optional[jnp.ndarray] = None
    discrete_distributions: Optional[jnp.ndarray] = None

    def add_sampled_points(
        self, estimator_name: str, sampled_points: jnp.ndarray
    ) -> None:
        """Add sampled points for a specific estimator."""
        if self.sampled_points is None:
            self.sampled_points = {}
        self.sampled_points[estimator_name] = sampled_points

    def add_expectation_values(self, expectation_values: jnp.ndarray) -> None:
        """Add expectation values."""
        self.expectation_values = expectation_values

    def add_discrete_distributions(self, distributions: jnp.ndarray) -> None:
        """Add discrete distributions."""
        self.discrete_distributions = distributions
