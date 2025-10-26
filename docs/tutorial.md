# Tutorial: Analysing a discrete control problem

This tutorial walks through a complete workflow. We start from the story ("a
controller that has to pick between two modes"), show how to encode that story
in mellowgate, run the estimators, and end with plots and simple diagnostics.
You only need the quickstart concepts: discrete problems, estimator configs, and
parameter sweeps.

## Scenario

Consider a controller that picks between two actuation modes depending on an
angular parameter {math}`\theta`. We model the payoff of each mode with simple
polynomials and allow the logits to favour one mode as {math}`\theta` grows.

```python
import jax
import jax.numpy as jnp

from mellowgate.core import Branch, Bound, DiscreteProblem, LogitsModel
from mellowgate.estimators import (
    FiniteDifferenceConfig,
    GumbelSoftmaxConfig,
    ReinforceConfig,
    ReinforceState,
)
from mellowgate.experiments import Sweep, run_parameter_sweep

jax.config.update("jax_enable_x64", True)

branches = [
    Branch(
        function=lambda th: 0.3 * th**3 - th,
        derivative_function=lambda th: 0.9 * th**2 - 1,
        threshold=(None, Bound(1.5, inclusive=True)),
    ),
    Branch(
        function=lambda th: -0.25 * th**3 + 2 * th + 1,
        derivative_function=lambda th: -0.75 * th**2 + 2,
        threshold=(Bound(0.2, inclusive=False), None),
    ),
]
logits_model = LogitsModel(logits_function=lambda th: jnp.vstack([th, -0.6 * th]))
problem = DiscreteProblem(branches=branches, logits_model=logits_model)
```

Thresholds ensure each branch only contributes when valid; outside the bounds
the payoff is treated as NaN and ignored during expectation calculations.

## Configure estimators

We benchmark all three estimators with moderate sample counts. REINFORCE uses a
baseline to stabilise variance.

```python
configs = {
    "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=200)},
    "gs": {"cfg": GumbelSoftmaxConfig(temperature=0.7, num_samples=200)},
    "reinforce": {
        "cfg": ReinforceConfig(num_samples=800, use_baseline=True),
        "state": ReinforceState(),
    },
}
```

## Run the sweep

We examine {math}`\theta` in the range [-1.5, 2.0] with four repetitions to
estimate per-theta variability.

```python
theta_grid = jnp.linspace(-1.5, 2.0, 25)
sweep = Sweep(theta_values=theta_grid, num_repetitions=4, estimator_configs=configs)
results = run_parameter_sweep(problem, sweep)
```

## Post-process results

Use the cached expectation values and sampled indices to build diagnostics.

```python
import matplotlib.pyplot as plt

fd = results["fd"].gradient_estimates["fd"]
gs = results["gs"].gradient_estimates["gs"]
rf = results["reinforce"].gradient_estimates["reinforce"]

exact = problem.compute_exact_gradient(theta_grid)
expectation = results["fd"].expectation_values

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(theta_grid, exact, label="Exact", linewidth=2)
ax[0].plot(theta_grid, fd["mean"], label="FD", linestyle="--")
ax[0].plot(theta_grid, gs["mean"], label="Gumbel-Softmax", linestyle="-.")
ax[0].plot(theta_grid, rf["mean"], label="REINFORCE", linestyle=":")
ax[0].set_ylabel("Gradient estimate")
ax[0].legend()

ax[1].plot(theta_grid, expectation, color="tab:purple")
ax[1].set_ylabel("Expected payoff")
ax[1].set_xlabel(r"$\theta$")
ax[1].set_title("Cached expectation values from the sweep")
plt.tight_layout()
```

::::{admonition} Interpreting the curves
:class: tip

- Finite differences typically tracks the exact gradient closely when the
  step size balances bias and variance.
- Gumbel-Softmax may smooth sharp transitions but remains unbiased in the
  limit of many samples and low temperatures.
- REINFORCE can exhibit higher variance; increases in `num_samples` plus a tuned
  baseline mitigate the effect.
:::

## Working with sampled points

The sweep stores the shared branch samples generated ahead of each repetition.
This enables downstream analyses without rerunning expensive Monte Carlo loops.

```python
samples = results["fd"].sampled_points["sampled_branch_indices"]
# samples has shape (num_theta, num_samples)
branch_zero_rate = (samples == 0).mean(axis=1)
```

Combine this with the logits model to check whether the sampling distribution
aligns with expectations or to visualise mode switching behaviour.

## Persisting artefacts

For reproducible studies, serialise the {class}`ResultsContainer` directly:

```python
import pickle

with open("fd_results.pkl", "wb") as handle:
    pickle.dump(results["fd"], handle)
```

Because the container stores plain JAX arrays and dictionaries it can also be
converted to `dataclasses.asdict` or written with NumPy’s `np.savez`.

## Summary

You now have a complete pipeline that:

1. Defines a discrete control problem with thresholds.
2. Benchmarks finite differences, Gumbel-Softmax, and REINFORCE estimators.
3. Reuses cached expectations and samples for analysis.
4. Produces publication-quality figures with minimal glue code.

The same pattern scales to more complex problems because the high-level utilities keeps the
interfaces stable and explicit.
