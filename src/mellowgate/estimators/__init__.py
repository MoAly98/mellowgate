"""Gradient estimation methods for discrete optimization problems.

This module implements three main approaches for estimating gradients in discrete
optimization problems where exact gradients may not be available:

1. Finite Differences: Approximates gradients using numerical differentiation
2. REINFORCE: Uses policy gradient methods with optional baseline reduction
3. Gumbel-Softmax: Provides differentiable relaxation of discrete sampling

Each estimator is designed to work with DiscreteProblem instances and supports
vectorized operations for efficient computation across multiple parameter values.
The estimators handle the fundamental challenge of computing gradients through
discrete sampling operations.
"""

from .finite_difference import FiniteDifferenceConfig, finite_difference_gradient
from .gumbel_softmax import GumbelSoftmaxConfig, gumbel_softmax_gradient
from .reinforce import ReinforceConfig, ReinforceState, reinforce_gradient

__all__ = [
    "FiniteDifferenceConfig",
    "GumbelSoftmaxConfig",
    "ReinforceConfig",
    "ReinforceState",
    "finite_difference_gradient",
    "gumbel_softmax_gradient",
    "reinforce_gradient",
]
