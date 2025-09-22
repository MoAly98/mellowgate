# example_toy.py
import numpy as np
from mellowgate.api.functions import Branch, LogitsModel, DiscreteProblem
from mellowgate.api.estimators import (
    FiniteDifferenceConfig,
    ReinforceConfig,
    ReinforceState,
    GumbelSoftmaxConfig,
)
from mellowgate.plots.metrics import (
    plot_gradient_estimates_vs_truth,
    plot_bias_variance_mse_analysis,
    plot_computational_time_analysis,
)
from mellowgate.api.experiments import Sweep, run_parameter_sweep
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
logits_model = LogitsModel(
    logits_function=lambda th: alpha * th, logits_derivative_function=lambda th: alpha
)

prob = DiscreteProblem(branches=branches, logits_model=logits_model)

thetas = np.linspace(-2.5, 2.5, 21)
sweep = Sweep(
    theta_values=thetas,
    num_repetitions=1000,
    estimator_configs={
        "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=20000)},
        "reinforce": {
            "cfg": ReinforceConfig(num_samples=20000, use_baseline=True),
            "state": ReinforceState(),
        },
        "gs": {
            "cfg": GumbelSoftmaxConfig(
                temperature=0.5, num_samples=800, use_straight_through_estimator=True
            )
        },
    },
)

results = run_parameter_sweep(prob, sweep)

# plotting (uses exact gradient if available)
plot_gradient_estimates_vs_truth(
    results, prob.compute_exact_gradient, output_manager=output_manager
)
plot_bias_variance_mse_analysis(
    results, prob.compute_exact_gradient, output_manager=output_manager
)
plot_computational_time_analysis(results, output_manager=output_manager)
