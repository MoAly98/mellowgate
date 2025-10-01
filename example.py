# example_toy.py
import cProfile
import pstats
from io import StringIO

import jax
import jax.numpy as jnp

from mellowgate.api.estimators import (
    FiniteDifferenceConfig,
    GumbelSoftmaxConfig,
    ReinforceConfig,
    ReinforceState,
)
from mellowgate.api.experiments import Sweep, run_parameter_sweep
from mellowgate.api.functions import Bound, Branch, DiscreteProblem, LogitsModel

# from mellowgate.plots.functions import plot_combined_overlay
from mellowgate.plots.metrics import (
    plot_bias_variance_mse_analysis,
    plot_computational_time_analysis,
    plot_gradient_estimates_vs_truth,
)
from mellowgate.utils.outputs import OutputManager

# Initialise output manager for this script
output_manager = OutputManager(base_directory="outputs")


branches = [
    Branch(
        function=lambda th: jnp.cos(th),
        derivative_function=lambda th: -jnp.sin(th),
        threshold=(None, Bound(0, inclusive=False)),
    ),
    Branch(
        function=lambda th: jnp.sin(th),
        derivative_function=lambda th: jnp.cos(th),
        threshold=(Bound(0, inclusive=True), None),
    ),
]

alpha = jnp.array([-1.0, 1.0])


# Define a custom softmax probability function
def softmax(logits):
    """Custom softmax that handles both 1D and 2D inputs."""
    if logits.ndim == 1:
        # Single theta case: logits shape (num_branches,)
        exp_logits = jnp.exp(logits - jnp.max(logits))
        return exp_logits / jnp.sum(exp_logits)
    else:
        # Multiple theta case: logits shape (num_branches, num_theta)
        exp_logits = jnp.exp(logits - jnp.max(logits, axis=0, keepdims=True))
        return exp_logits / jnp.sum(exp_logits, axis=0, keepdims=True)


def sigmoid(logits):
    exp_logits = jnp.exp(-logits)
    return 1 / (1 + exp_logits)


# Define a custom Bernoulli sampling function using JAX
def bernoulli_sampling(probabilities, key=None):
    """Sample from a discrete distribution using JAX random.

    Args:
        probabilities: Array of probabilities for each branch
        key: JAX random key (if None, uses a default key)

    Returns:
        int: Index of the sampled branch
    """
    if key is None:
        key = jax.random.PRNGKey(0)  # Default key for reproducibility

    return jax.random.choice(key, len(probabilities), p=probabilities)


# Update logits model to use softmax and Bernoulli sampling
logits_model = LogitsModel(
    logits_function=lambda th: alpha[:, jnp.newaxis]
    * th,  # Broadcasting: (2, 1) * (N,) -> (2, N)
    logits_derivative_function=lambda th: alpha[:, jnp.newaxis]
    * jnp.ones_like(th),  # (2, 1) * (N,) -> (2, N)
    probability_function=sigmoid,  # Use softmax for probabilities
)

prob = DiscreteProblem(
    branches=branches,
    logits_model=logits_model,
    # sampling_function=bernoulli_sampling,  # Use Bernoulli sampling
)

thetas = jnp.linspace(-2.5, 2.5, 100)
sweep = Sweep(
    theta_values=thetas,
    num_repetitions=100,
    estimator_configs={
        "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=10000)},
        "reinforce": {
            "cfg": ReinforceConfig(num_samples=10000, use_baseline=True),
            "state": ReinforceState(),
        },
        "gs": {
            "cfg": GumbelSoftmaxConfig(
                temperature=0.01, num_samples=10000, use_straight_through_estimator=True
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
# plot_combined_overlay(
#     results_dict,
#     output_manager=output_manager,
# )

# Disable profiling and print stats
pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s)
ps.strip_dirs().sort_stats("cumulative").print_stats(10)
print(s.getvalue())
