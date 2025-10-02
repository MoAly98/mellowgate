"""Mellowgate: A library for differentiable discrete optimization.

This library provides tools for gradient estimation in discrete optimization
problems using various stochastic gradient estimators including finite differences,
REINFORCE, and Gumbel-Softmax.
"""

# Configure JAX for float64 precision (must be done before other imports)
from . import config  # This enables x64 precision
from .api.estimators import (
    FiniteDifferenceConfig,
    GumbelSoftmaxConfig,
    ReinforceConfig,
    ReinforceState,
    finite_difference_gradient,
    gumbel_softmax_gradient,
    reinforce_gradient,
)
from .api.experiments import Sweep, run_parameter_sweep

# Expose main API classes and functions
from .api.functions import Branch, DiscreteProblem, LogitsModel

# Expose plotting utilities
from .plots.metrics import (
    plot_bias_variance_mse_analysis,
    plot_computational_time_analysis,
    plot_gradient_estimates_vs_truth,
)

# Expose utility functions
from .utils.functions import softmax
from .utils.statistics import sample_gumbel
