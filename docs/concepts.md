# Concepts: Maths behind mellowgate

This note introduces the mathematical ideas that motivate mellowgate: how a
discrete system becomes stochastic, how expectations give rise to gradients, and
why the bundled estimators look the way they do. Each section finishes with the
relevant API entry point so you can move from equations to code.

## Discrete choices as random variables

Assume a collection of branch functions
{math}`f_k(\theta)` indexed by {math}`k = 1, \ldots, K`. We pick a branch at
random using probabilities {math}`p_k(\theta)` with {math}`\sum_k p_k(\theta) = 1`
and {math}`p_k(\theta) \ge 0`. The stochastic payoff is the random variable

```{math}
X(\theta) = f_{Z(\theta)}(\theta),
```

where {math}`Z(\theta)` is a discrete random variable taking value {math}`k`
with probability {math}`p_k(\theta)`. The expected payoff, called the deterministic
relaxation in mellowgate, is

```{math}
\mathbb{E}[X(\theta)] = \sum_{k=1}^K p_k(\theta) f_k(\theta).
```

In the library this structure is packaged as
{class}`mellowgate.core.DiscreteProblem`, which gathers the branch functions and
logits model into a single object.

## From deterministic models to stochastic ones

To produce {math}`p_k(\theta)` we pass raw logits {math}`\ell_k(\theta)` through
the probability function. By default mellowgate uses a softmax:

```{math}
p_k(\theta) = \frac{\exp(\ell_k(\theta))}{\sum_{j=1}^K \exp(\ell_j(\theta))}.
```

You can override the probability function if you already have a normalised
distribution. The logits can encode thresholds, temperature schedules, or learned
policies. In code you supply them via {class}`mellowgate.core.LogitsModel`, then build the
problem:

```python
from mellowgate.core import Branch, DiscreteProblem, LogitsModel

branches = [
    Branch(function=lambda th: th**2),
    Branch(function=lambda th: th**3 + 1),
]
logits = LogitsModel(logits_function=lambda th: jnp.stack([th, -th]))
problem = DiscreteProblem(branches=branches, logits_model=logits)
```

Calling `problem.compute_expected_value(theta)` evaluates the expectation above,
while `problem.sample_branch` draws real samples of {math}`Z(\theta)`.

:::{admonition} Insight: probability maps beyond softmax
:class: tip

The default softmax is convenient, but you can pass any callable that returns
non-negative values summing to one. For a binary choice a sigmoid works fine; a
stick-breaking transform is another option when you want asymmetric control.
:::

:::{admonition} Insight: from hard cuts to soft cuts
:class: tip

Each branch can stand for a version of an analysis cut. Turning the decision
into a probability lets you see how downstream quantities respond when the cut
is softened or tightened, without touching later stages in the pipeline.
:::

## Gradients of expectations

Our goal is {math}`\nabla_\theta \mathbb{E}[X(\theta)]`. When branch
derivatives exist, mellowgate can compute the exact gradient via
{math}`d/d\theta` of the expectation:

```{math}
\frac{d}{d\theta} \mathbb{E}[X(\theta)]
  = \sum_k \left( p_k(\theta) \frac{d}{d\theta} f_k(\theta)
                + f_k(\theta) \frac{d}{d\theta} p_k(\theta) \right).
```

The second term is often written using the log-derivative trick:

```{math}
\frac{d}{d\theta} p_k(\theta)
  = p_k(\theta) \frac{d}{d\theta} \log p_k(\theta).
```

This identity underpins REINFORCE-style estimators.
{meth}`mellowgate.core.DiscreteProblem.compute_exact_gradient` implements the
closed-form expression when all derivatives are available; otherwise we fall
back to stochastic estimators.

## Estimator formulas

### Finite differences

Central differences approximate the derivative using mirrored perturbations:

```{math}
\frac{d}{d\theta} \mathbb{E}[X(\theta)]
  \approx \frac{\mathbb{E}[X(\theta + h)] - \mathbb{E}[X(\theta - h)]}{2h}.
```

Using the same random key for both evaluations (common random numbers) reduces
variance. {func}`mellowgate.estimators.finite_difference_gradient` implements
this with a user-controlled step size {math}`h` and sample count.

The estimator used in mellowgate is

```{math}
\hat{g}_\text{fd}(\theta)
  = \frac{1}{2 h N} \sum_{k=1}^{N}
    \Bigl(X^{(k)}(\theta + h) - X^{(k)}(\theta - h)\Bigr),
```

with {math}`N` samples per side. Reusing the same randomness for both {math}`X`
terms lowers variance. The bias scales with {math}`h^2`, while the variance
scales roughly with {math}`1/(h^2 N)`.

:::{admonition} Insight: choosing the step size
:class: tip

Plot the finite-difference estimate for a simple function (for example
{math}`\sin \theta`) while sweeping {math}`h`. You will see the curve flatten
once numerical noise dominates, which signals that the step size is too small.
:::

### REINFORCE (score-function)

When branch derivatives are not available we differentiate the sampling
procedure:

```{math}
\frac{d}{d\theta} \mathbb{E}[X(\theta)]
  = \mathbb{E}\left[ X(\theta) \frac{d}{d\theta} \log p_{Z(\theta)}(\theta) \right].
```

A baseline {math}`b(\theta)` can be subtracted to reduce variance without
changing the expectation. {func}`mellowgate.estimators.reinforce_gradient`
handles the score-function estimator and uses {class}`ReinforceState` to track a
moving baseline.

The estimator returned by mellowgate is

```{math}
\hat{g}_\text{reinforce}(\theta)
  = \frac{1}{N}\sum_{k=1}^{N}
    \Bigl(f(x^{(k)}) - b_t\Bigr)
    \nabla_\theta \log p_{x^{(k)}}(\theta)
    + \frac{1}{N} \sum_{k=1}^{N} \nabla_\theta f(x^{(k)}),
```

where the second term only appears when branch derivatives exist. The baseline
inside {class}`ReinforceState` follows the exponential moving average

```{math}
b_t = \gamma b_{t-1} + (1 - \gamma)\bar{f}_t,
```

with {math}`\gamma` the momentum parameter and {math}`\bar{f}_t` the batch mean
of {math}`f(x^{(k)})`.

:::{admonition} Insight: keeping variance in check
:class: tip

Keeping a moving average baseline centred on the recent function values
dramatically lowers gradient noise, especially when the payoff changes sharply
with {math}`\theta`.
:::

### Gumbel-Softmax (relaxation)

#### The Gumbel-Max trick

Let {math}`g_k` be i.i.d. samples from the standard Gumbel distribution whose
probability density function is {math}`f(g) = \exp(-(g + \exp(-g)))` and
cumulative distribution function is
{math}`F(g) = \exp(-\exp(-g))`. Define

```{math}
k^\star = \arg\max_i \left(\log p_i(\theta) + g_i\right).
```

Then {math}`\Pr[k^\star = k] = p_k(\theta)` for every {math}`k`. A short proof
comes from conditioning on {math}`g_k = t` and integrating over the remaining
noise:

```{math}
\Pr[k^\star = k]
 = \int_{-\infty}^{\infty}
    \prod_{j \ne k} \Pr\!\left[g_j < t + \log p_k - \log p_j\right]
    f(t)\, dt
 = \int_{-\infty}^{\infty}
    \prod_{j \ne k} F\!\left(t + \log p_k - \log p_j\right)
    f(t)\, dt
 = p_k.
```

This is the **Gumbel-Max trick**: adding Gumbel noise to the logits turns the
argmax into an exact categorical sample.

#### Relaxing the argmax

To make the expression differentiable we replace the argmax with a tempered
softmax:

```{math}
y_k = \frac{\exp((\log p_k(\theta) + g_k)/\tau)}{\sum_j \exp((\log p_j(\theta) + g_j)/\tau)}.
```

As {math}`\tau \to 0`, {math}`y` approaches a one-hot vector that chooses
{math}`k^\star`. Larger temperatures smooth the vector and give stable
gradients. {func}`mellowgate.estimators.gumbel_softmax_gradient` wraps this
reparameterisation and lets you pick the temperature and sample count.

The relaxed payoff is

```{math}
X_\text{gs}(\theta, g)
  = \sum_{k=1}^{K} y_k(\theta, g) f_k(\theta),
```

with gradients estimated via

```{math}
\hat{g}_\text{gs}(\theta)
  = \frac{1}{N} \sum_{k=1}^{N} \nabla_\theta X_\text{gs}(\theta, g^{(k)}).
```

Lower temperatures make {math}`y` almost discrete but increase variance. Higher
temperatures smooth the landscape and introduce a small bias that often helps
optimisation.

:::{admonition} Insight: more on Gumbel-Softmax
:class: tip

A longer derivation walks through how the Gumbel-Max trick emerges from the
log-survival function of the Gumbel distribution and then swaps the argmax for a
softmax. Any standard reference on the concrete distribution will reproduce the
same steps.
:::

## Linking maths to code

Once you understand the maths, the API mirrors the equations closely:

```python
import jax.numpy as jnp
from mellowgate.estimators import (
    FiniteDifferenceConfig,
    ReinforceConfig,
    ReinforceState,
    GumbelSoftmaxConfig,
)
from mellowgate.experiments import Sweep, run_parameter_sweep

theta_values = jnp.linspace(-2.0, 2.0, 21)
configs = {
    "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=200)},
    "reinforce": {
        "cfg": ReinforceConfig(num_samples=600, use_baseline=True),
        "state": ReinforceState(),
    },
    "gs": {"cfg": GumbelSoftmaxConfig(temperature=0.7, num_samples=300)},
}

sweep = Sweep(theta_values=theta_values, num_repetitions=3, estimator_configs=configs)
results = run_parameter_sweep(problem, sweep)
```

From here you can access `results["fd"].gradient_estimates["fd"]["mean"]`,
compare against `problem.compute_exact_gradient(theta_values)`, and plot or
persist the statistics as described in the quickstart and tutorial.

Understanding the underlying maths helps you tune estimator parameters, choose
appropriate baselines or temperatures, and interpret the trade-offs each method
makes.
