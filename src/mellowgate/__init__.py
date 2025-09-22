"""Mellowgate: A library for differentiable discrete optimization.

This library provides tools for gradient estimation in discrete optimization
problems using various stochastic gradient estimators including finite differences,
REINFORCE, and Gumbel-Softmax.
"""

# Expose main API classes and functions
from .api.functions import DiscreteProblem, Branch, LogitsModel
from .api.estimators import (
    finite_difference_gradient,
    FiniteDifferenceConfig,
    reinforce_gradient,
    ReinforceConfig,
    ReinforceState,
    gumbel_softmax_gradient,
    GumbelSoftmaxConfig,
)
from .api.experiments import run_parameter_sweep, Sweep

# Expose plotting utilities
from .plots.metrics import (
    plot_gradient_estimates_vs_truth,
    plot_bias_variance_mse_analysis,
    plot_computational_time_analysis,
)

# Expose utility functions
from .utils.functions import softmax
from .utils.statistics import sample_gumbel
