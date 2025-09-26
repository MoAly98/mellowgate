# example_toy.py
import cProfile
import pstats
from io import StringIO

import numpy as np

from mellowgate.api.estimators import (
    FiniteDifferenceConfig,
    GumbelSoftmaxConfig,
    ReinforceConfig,
    ReinforceState,
)
from mellowgate.api.experiments import Sweep, run_parameter_sweep
from mellowgate.api.functions import Branch, DiscreteProblem, LogitsModel
from mellowgate.plots.functions import plot_combined_overlay
from mellowgate.plots.metrics import (
    plot_bias_variance_mse_analysis,
    plot_computational_time_analysis,
    plot_gradient_estimates_vs_truth,
)
from mellowgate.utils.outputs import OutputManager

# Initialise output manager for this script
output_manager = OutputManager(base_directory="outputs")

# branches
branches = [
    Branch(
        function=lambda th: float(np.cos(th)),
        derivative_function=lambda th: float(-np.sin(th)),
    ),
    Branch(
        function=lambda th: float(np.sin(th)),
        derivative_function=lambda th: float(np.cos(th)),
    ),
    Branch(
        function=lambda th: float(np.tanh(th)),
        derivative_function=lambda th: float(1 - np.tanh(th) ** 2),
    ),
]
alpha = np.array([-1.0, 0.0, 1.0])


# Define a custom softmax probability function
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return np.array(exp_logits / np.sum(exp_logits))


# Define a custom Bernoulli sampling function
def bernoulli_sampling(probabilities):
    return np.random.choice(len(probabilities), p=probabilities)


# Update logits model to use softmax and Bernoulli sampling
logits_model = LogitsModel(
    logits_function=lambda th: alpha * th,
    logits_derivative_function=lambda th: alpha,
    probability_function=softmax,  # Use softmax for probabilities
)

prob = DiscreteProblem(
    branches=branches,
    logits_model=logits_model,
    # sampling_function=bernoulli_sampling,  # Use Bernoulli sampling
)

thetas = np.linspace(-2.5, 2.5, 21)
sweep = Sweep(
    theta_values=thetas,
    num_repetitions=1000,
    estimator_configs={
        "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=10000)},
        "reinforce": {
            "cfg": ReinforceConfig(num_samples=10000, use_baseline=True),
            "state": ReinforceState(),
        },
        "gs": {
            "cfg": GumbelSoftmaxConfig(
                temperature=0.5, num_samples=10000, use_straight_through_estimator=True
            )
        },
    },
)
# Profile the code execution
pr = cProfile.Profile()
pr.enable()

# Run parameter sweep and get results for each estimator
results_dict = run_parameter_sweep(prob, sweep)

# Use updated plotting functions
plot_gradient_estimates_vs_truth(
    results_dict, prob.compute_exact_gradient, output_manager=output_manager
)
plot_bias_variance_mse_analysis(
    results_dict, prob.compute_exact_gradient, output_manager=output_manager
)
plot_computational_time_analysis(results_dict, output_manager=output_manager)

# Generate combined overlay plot for all estimators
plot_combined_overlay(
    results_dict,
    output_manager=output_manager,
)

# Disable profiling and print stats
pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s)
ps.strip_dirs().sort_stats("cumulative").print_stats(10)
print(s.getvalue())
