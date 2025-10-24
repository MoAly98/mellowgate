"""Mellowgate: A library for differentiable discrete optimization.

This library provides tools for gradient estimation in discrete optimization
problems using various stochastic gradient estimators including finite differences,
REINFORCE, and Gumbel-Softmax.
"""

# Configure JAX for float64 precision (must be done before other imports)
from . import config
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
from .api.functions import Branch, DiscreteProblem, LogitsModel
from .plots.metrics import (
    plot_bias_variance_mse_analysis,
    plot_computational_time_analysis,
    plot_gradient_estimates_vs_truth,
)
from .utils.functions import softmax
from .utils.statistics import sample_gumbel

__all__ = [
    "Branch",
    "DiscreteProblem",
    "FiniteDifferenceConfig",
    "GumbelSoftmaxConfig",
    "LogitsModel",
    "ReinforceConfig",
    "ReinforceState",
    "Sweep",
    "config",
    "finite_difference_gradient",
    "gumbel_softmax_gradient",
    "plot_bias_variance_mse_analysis",
    "plot_computational_time_analysis",
    "plot_gradient_estimates_vs_truth",
    "reinforce_gradient",
    "run_parameter_sweep",
    "sample_gumbel",
    "softmax",
]
