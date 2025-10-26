# Quickstart

This quickstart shows how to turn a discrete decision into a gradient-friendly
workflow. You will define a small problem, pick a few estimators, and compare
their outputs. No prior experience with automatic differentiation is required;
we explain each step in plain language before showing the code. All examples run on
CPU or GPU with JAX.

## 1. Define the problem

We begin by describing the alternatives in our discrete decision. In mellowgate
this is a {class}`mellowgate.core.DiscreteProblem`. Think of it as:

- **Branches**: functions you could choose (for example two different payoffs).
- **Optional derivatives**: how each branch changes as the parameter changes.
- **Logits model**: a small function that turns the parameter into probabilities.

We work with a single scalar parameter {math}`\theta`; multivariate cases follow
the same pattern.

```python
import jax.numpy as jnp
from mellowgate.core import Branch, DiscreteProblem, LogitsModel

branches = [
    Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th),
    Branch(function=lambda th: th**3, derivative_function=lambda th: 3 * th**2),
]
logits_model = LogitsModel(logits_function=lambda th: jnp.array([th, -th]))

problem = DiscreteProblem(branches=branches, logits_model=logits_model)
```

By providing derivatives the problem can compute exact gradients for validation,
which is handy when verifying estimator accuracy.

## 2. Configure estimators

Mellowgate offers three estimators:

- **Finite differences (`fd`)**: numerical derivative using small perturbations.
- **REINFORCE (`reinforce`)**: score-function estimator that works even when
  branches lack derivatives.
- **Gumbel-Softmax (`gs`)**: differentiable relaxation that samples a soft choice.

You configure each estimator by filling out the corresponding dataclass.

```python
from mellowgate.estimators import (
    FiniteDifferenceConfig,
    ReinforceConfig,
    ReinforceState,
    GumbelSoftmaxConfig,
)

estimator_configs = {
    "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=100)},
    "gs": {"cfg": GumbelSoftmaxConfig(temperature=0.5, num_samples=100)},
    "reinforce": {
        "cfg": ReinforceConfig(num_samples=500, use_baseline=True),
        "state": ReinforceState(),
    },
}
```

## 3. Run a sweep

Use the {class}`mellowgate.experiments.Sweep` dataclass to bundle the theta grid
and run counts. Pass it to {func}`mellowgate.experiments.run_parameter_sweep`
along with the problem and estimator configs.

```python
import jax.numpy as jnp
from mellowgate.experiments import Sweep, run_parameter_sweep

sweep = Sweep(
    theta_values=jnp.linspace(0.5, 1.5, 5),
    num_repetitions=3,
    estimator_configs=estimator_configs,
)

results = run_parameter_sweep(problem, sweep)
```

The function returns a dictionary keyed by estimator name. Each value is a
{class}`mellowgate.results.ResultsContainer` with gradient statistics, cached
function evaluations, and sampled indices.

## 4. Inspect the results

```python
fd = results["fd"].gradient_estimates["fd"]
print("θ:", fd["theta"])
print("Mean gradient:", fd["mean"])
print("Standard deviation:", fd["std"])

# Compare against the analytic gradient when available
exact = problem.compute_exact_gradient(sweep.theta_values)
print("Analytic gradient:", exact)
```

::::{grid} 1
:gutter: 3

:::{grid-item-card} Next steps
:class-header: bg-info text-white
- Swap in your own branch functions or logits model.
- Increase `num_repetitions` to tighten the Monte Carlo error bars.
- Capture `ResultsContainer.expectation_values` for downstream analysis.
:::

:::{grid-item-card} Troubleshooting
:class-header: bg-dark text-white
If an estimator diverges, confirm that the branch functions are numerically
stable and that the logits model normalises to a valid probability simplex. The
`tests/estimators/` suite contains minimal reproductions for each estimator.
:::

:::{grid-item-card} Where to next?
:class-header: bg-light
Continue to the {doc}`tutorial` for a complete workflow that prepares plots and
diagnostics from the same API.
:::

::::
