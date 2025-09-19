# Expose main API classes and functions
from .api.functions import DiscreteProblem, Branch, LogitsModel
from .api.estimators import fd_gradient, FDConfig, reinforce_gradient, ReinforceConfig, ReinforceState, gs_gradient, GSConfig
from .api.experiments import run_sweep, Sweep

# Expose plotting utilities
from .plots.metrics import plot_means_vs_true, plot_bias_var_mse, plot_time

# Expose utility functions
from .utils.functions import softmax
from .utils.statistics import sample_gumbel
