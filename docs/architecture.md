# Architecture

Mellowgate is organised around a few small modules that cooperate to evaluate
discrete optimisation problems. At a glance:

- `mellowgate.core` describes the problem (branches, logits, thresholds).
- `mellowgate.estimators` provides gradient estimators.
- `mellowgate.experiments` runs batched sweeps over parameter grids.
- `mellowgate.results` packages gradient statistics and cached values.
- `mellowgate.utils` supplies supporting math and IO helpers.

Each part keeps a narrow, explicit interface so mixing and matching components
feels straightforward.

## Problem definition (`mellowgate.core`)

- {class}`mellowgate.core.DiscreteProblem` encapsulates the functions, optional
  derivatives, and logits model that determine branch probabilities.
- {class}`mellowgate.core.Branch` stores the callable and optional thresholds.
- {class}`mellowgate.core.LogitsModel` separates probability generation from
  problem definition, making it easy to reuse policies across experiments.
- Utility helpers validate custom sampling functions and precompute expected
  values.

Guiding principle: you describe the discrete landscape once and reuse it across
all estimators.

## Gradient estimators (`mellowgate.estimators`)

- {func}`mellowgate.estimators.finite_difference_gradient` implements common
  random number finite differences controlled by
  {class}`mellowgate.estimators.FiniteDifferenceConfig`.
- {func}`mellowgate.estimators.reinforce_gradient` uses score-function gradients
  with a running baseline stored in {class}`mellowgate.estimators.ReinforceState`.
- {func}`mellowgate.estimators.gumbel_softmax_gradient` draws relaxed samples via
  {class}`mellowgate.estimators.GumbelSoftmaxConfig`.

Each estimator accepts the same problem description and array of theta values,
returning JAX arrays that can be composed or stacked inside jit/vmap pipelines.

## Experiments (`mellowgate.experiments`)

- {class}`mellowgate.experiments.Sweep` bundles theta grids, repetition counts,
  and estimator configs.
- {func}`mellowgate.experiments.run_parameter_sweep` coordinates vectorised
  calls to estimators, caches expectation values, and records timing metadata.
- Internal helpers share Monte Carlo samples across estimators to minimise noise
  and provide deterministic behaviour for repeatability.

The design emphasises batch execution: all theta values are processed in one
pass, and repeated sweeps reuse the same random keys.

## Results (`mellowgate.results`)

- {class}`mellowgate.results.ResultsContainer` stores gradient statistics along
  with optional sampled points, expectation values, and discrete distributions.
- Mutator methods (`add_sampled_points`, `add_expectation_values`,
  `add_discrete_distributions`) enable incremental enrichment, which is useful
  if you derive extra diagnostics after the sweep.

Results containers are lightweight dataclasses intended for downstream plotting,
serialization, or further analysis.

## Utilities (`mellowgate.utils`)

- `mellowgate.utils.functions` contains numerically stable helpers such as
  `softmax`.
- `mellowgate.utils.statistics` provides sampling primitives used by estimators.
- Additional modules house common plotting routines (under `mellowgate.plots`)
  and filesystem helpers (`mellowgate.utils.outputs.OutputManager`).

## Extending the API

To plug in a new estimator:

1. Implement the estimator as a function mirroring the signature of the existing
   ones (`(problem, theta_values, config, state?) -> jnp.ndarray`).
2. Add a configuration dataclass with clear defaults.
3. Register the estimator inside `run_parameter_sweep` by recognising a new key.
4. Return statistics in the same dictionary shape so
   {class}`ResultsContainer` continues to work unchanged.

Because each layer focuses on a single responsibility, extending mellowgate is
usually just a matter of slotting new code into the relevant module without
surprising the rest of the stack.
