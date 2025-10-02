# mellowgate

A Python library for gradient estimation in discrete optimization problems. This package provides differentiable relaxations of discrete decisions using JAX, implementing three main gradient estimation methods: finite differences, REINFORCE, and Gumbel-Softmax for efficient discrete decision making.

## Installation

### For Users

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MoAly98/mellowgate.git
   cd mellowgate
   ```

2. **Install from source**:
   ```bash
   pip install -e .
   ```

### For Developers

This project uses [pixi](https://pixi.sh/) for dependency management and development environment setup.

1. **Install pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Set up the development environment**:
   ```bash
   make setup-dev
   ```
   This will:
   - Install all dependencies via pixi
   - Set up pre-commit hooks for code quality
   - Install the package in editable mode

## Development Workflow

### Quick Start Commands

The project provides a comprehensive Makefile for common development tasks:

```bash
make help          # Show all available commands
make setup-dev     # Set up development environment (run this first!)
make format        # Format code with black and isort
make lint          # Run all linting checks (black, isort, flake8)
make test          # Run unit tests with pytest
make pre-commit    # Run pre-commit hooks on all files
make example       # Run the example script
make clean         # Clean up build artifacts
```

### Code Quality Standards

This project enforces code quality standards:

- **Formatting**: [Black](https://black.readthedocs.io/) (88 character line length)
- **Import Sorting**: [isort](https://pycqa.github.io/isort/) (black-compatible profile)
- **Linting**: [flake8](https://flake8.pycqa.org/) with strict rules
- **Numerical Computing**: [JAX](https://jax.readthedocs.io/) for high-performance array operations with automatic differentiation

### Pre-commit Hooks

Pre-commit hooks automatically run on every commit to ensure code quality:

```bash
# Install hooks (done automatically by make setup-dev)
pixi run pre-commit install

# Run hooks manually on all files
make pre-commit

# Run specific hook
pixi run pre-commit run black
```

### Development Guidelines

#### Code Style
- Use descriptive variable names (no single-letter variables except for standard math notation)
- Add comprehensive docstrings to all public functions and classes
- Follow Google-style docstring format
- Use type annotations for all function parameters and return values
- Keep functions focused and modular

#### Testing
- Add unit tests for all new functionality using pytest
- Ensure all tests pass before submitting changes: `make test`
- Use descriptive test names and clear test organization
- Test both success cases and error conditions
- Use fixtures for common test setup

#### Documentation
- Update docstrings when changing function signatures
- Add examples to docstrings for complex functions
- Update README.md for significant changes

### Continuous Integration

The CI pipeline runs on every push and pull request:

- **Linting**: Checks code formatting, import sorting, and flake8 compliance
- **Testing**: Unit tests with pytest
- **Build Validation**: Ensures package can be built and installed

### Test Coverage

The test suite covers core functionality including:
- Mathematical operations and numerical stability
- API components and workflows
- Edge cases and error handling

Run `make test-cov` to generate coverage reports.

### Troubleshooting

#### Common Issues

1. **Import errors**: Make sure you've run `make setup-dev` and the package is installed in editable mode
2. **Linting failures**: Run `make format` to auto-fix most formatting issues
3. **Pre-commit failures**: The hooks will auto-fix most issues, then re-commit

#### Environment Reset

If you encounter environment issues:

```bash
# Clean up
make clean

# Reset pixi environment
pixi clean
pixi install

# Reinstall pre-commit hooks
pixi run pre-commit install
```

### Project Structure

```
mellowgate/
├── src/mellowgate/           # Main package source code
│   ├── api/                  # Core API modules
│   │   ├── estimators.py     # Gradient estimation algorithms
│   │   ├── experiments.py    # Experiment orchestration utilities
│   │   ├── functions.py      # Problem definition classes
│   │   └── results.py        # Results data structures
│   ├── plots/                # Visualization utilities
│   ├── utils/                # Mathematical utility functions
│   ├── logging/              # Logging configuration
│   └── config.py             # JAX configuration and precision settings
├── tests/                    # Comprehensive unit tests (pytest)
├── learning/                 # Tutorial and example notebooks
├── example.py                # Main example script
├── pyproject.toml           # Project metadata and tool config
├── pixi.toml                # Pixi dependency management
├── .pre-commit-config.yaml  # Pre-commit hook configuration
├── .flake8                  # Flake8 linting configuration
├── .github/workflows/       # CI/CD pipelines
└── Makefile                 # Development convenience commands
```

### Adding New Features

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement changes**:
   - Follow code style guidelines
   - Add comprehensive docstrings
   - Use descriptive variable names

3. **Test locally**:
   ```bash
   make lint          # Check code quality
   make test          # Run tests
   make example       # Test functionality
   ```

4. **Submit pull request**:
   - Ensure CI passes
   - Add clear description of changes
   - Reference any related issues


## Running Tests

Run tests using the provided commands:

```bash
# Run all tests
make test

# Run tests with coverage report
make test-cov

# Run tests directly with pytest
pytest tests/ -v

# Run specific test file
pytest tests/test_functions.py -v
```

## Example Usage

For a complete working example, run:
```bash
make example
```

## Mathematical Formulation

### Key Concepts and Symbols
- **Discrete Optimization Problem**: A problem where the goal is to optimize a function over a set of discrete choices.
  - $x$: Discrete choices.
  - $\theta$: Parameter controlling the optimization.
  - $f(x; \theta)$: Function value for choice $x$ and parameter $\theta$.
  - $\pi(x | \theta)$: Probability of selecting choice $x$ given $\theta$.
  - $a(x)$: Logits, which are unnormalized log probabilities used to compute $\pi(x | \theta)$.

### Formulations

1. **Expected Value**:
   The expected value of the function over discrete choices is:
   $$
   \mathbb{E}[f(x; \theta)] = \sum_{i=1}^N \pi(x_i | \theta) f(x_i; \theta),
   $$
   where $N$ is the number of discrete choices.

2. **Gradient of Expected Value**:
   The gradient of the expected value with respect to $\theta" is:
   $$
   \nabla_\theta \mathbb{E}[f(x; \theta)] = \sum_{i=1}^N \nabla_\theta \pi(x_i | \theta) f(x_i; \theta) + \pi(x_i | \theta) \nabla_\theta f(x_i; \theta).
   $$

3. **Stochastic Sampling**:
   Samples are drawn from the probability distribution $\pi(x | \theta)$ using a user-defined sampling function $S$:
   $$
   x \sim S(\pi(x | \theta)).
   $$

4. **Gradient Estimation Methods**:
   - **Finite Differences**:
     $$
     \nabla_\theta \mathbb{E}[f(x; \theta)] \approx \frac{\mathbb{E}[f(\theta + \epsilon)] - \mathbb{E}[f(\theta - \epsilon)]}{2\epsilon},
     $$
     where $\epsilon" is a small perturbation.
   - **REINFORCE**:
     $$
     \nabla_\theta \mathbb{E}[f(x; \theta)] \approx \mathbb{E}\left[f(x) \cdot \nabla_\theta \log \pi(x | \theta)\right].
     $$
   - **Gumbel-Softmax**:
     The Gumbel-Softmax method introduces a continuous relaxation of discrete sampling by adding Gumbel noise $g(x)$ to the logits $a(x)$ and applying the softmax function:
     $$
     \pi(x | \theta) = \text{softmax}\left(\frac{a(x) + g(x)}{\tau}\right),
     $$
     where $\tau" is the temperature parameter controlling the sharpness of the distribution. As $\tau \to 0$, the distribution becomes one-hot, approximating discrete sampling.

     - **Straight-Through Estimator (STE)**: To enable backpropagation through the discrete sampling process, the STE method is used. During the forward pass, discrete samples are drawn, but during the backward pass, gradients are computed as if the softmax relaxation was used:
       $$
       \nabla_\theta \mathbb{E}[f(x; \theta)] \approx \nabla_\theta f(x) + f(x) \cdot \nabla_\theta \pi(x | \theta).
       $$

### Notes on Choosing Logits

- **Interpretation**: Logits represent unnormalized scores for each branch. Higher logits correspond to higher probabilities after normalization.
- **Design**: Logits should reflect the relative importance or likelihood of each branch. For example:
  - If a branch is more likely for larger $\theta$, assign it a higher $\alpha_k(\theta)$.
  - Ensure logits are scaled appropriately to avoid numerical instability in the softmax computation.
- **Normalization**: The softmax function is a common choice to convert logits into probabilities that sum to 1. Other methods such as sigmoid, temperature-scaled softmax, or sparsemax can also be used depending on the problem requirements. Each method has its own advantages and should be chosen based on the desired properties of the probability distribution.

---

### Example: Sigmoid-Bernoulli Problem

Consider a discrete optimization problem with two branches:
$$
\begin{aligned}
    f(\theta) &= \begin{cases}
        \cos(\theta), & \text{if } k = 0, \\
        \sin(\theta), & \text{if } k = 1.
    \end{cases}
\end{aligned}
$$

The probability of selecting branch $k$ is computed using the sigmoid function:
$$
\pi(k | \theta) = \sigma(\alpha_k(\theta)) = \frac{1}{1 + e^{-\alpha_k(\theta)}}.
$$

Where the logits $\alpha_k(\theta)$ are defined as:
$$
\alpha_k(\theta) = \begin{cases}
    -\theta, & \text{if } k = 0, \\
    \theta, & \text{if } k = 1.
\end{cases}
$$

Samples are drawn from the Bernoulli distribution:
$$
\begin{aligned}
    k \sim \text{Ber}(k; \pi(k | \theta)).
\end{aligned}
$$

The expected value of the function is:
$$
\mathbb{E}[f(\theta)] = \sum_{k \in \{0, 1\}} \pi(k | \theta) f_k(\theta).
$$

The analytical expected value is:
$$
\mathbb{E}[f(\theta)] = \sigma(-\theta) \cdot \cos(\theta) + \sigma(\theta) \cdot \sin(\theta).
$$
---

### Example: Sigmoid-Bernoulli Problem with Three Branches

#### Mathematical Formulation

Consider a discrete optimization problem with three branches:
$$
\begin{aligned}
    f(\theta) &= \begin{cases}
        \sin(\theta), & \text{if } k = 0, \\
        \cos(\theta), & \text{if } k = 1, \\
        \tanh(\theta), & \text{if } k = 2.
    \end{cases}
\end{aligned}
$$

The logits $\alpha_k(\theta)$ are defined as:
$$
\alpha_k(\theta) = \begin{cases}
    -\theta, & \text{if } k = 0, \\
    \theta, & \text{if } k = 1, \\
    2\theta, & \text{if } k = 2.
\end{cases}
$$

The probabilities $\pi(k | \theta)$ are computed using the softmax function to ensure they sum to 1:
$$
\pi(k | \theta) = \frac{\exp(\alpha_k(\theta))}{\sum_{j=0}^{2} \exp(\alpha_j(\theta))}.
$$

Samples are drawn from the categorical distribution:
$$
\begin{aligned}
    k \sim \text{Cat}(k; \pi(k | \theta)).
\end{aligned}
$$

The expected value of the function is:
$$
\mathbb{E}[f(\theta)] = \sum_{k \in \{0, 1, 2\}} \pi(k | \theta) f_k(\theta).
$$
---

## mellowgate API Overview

The `mellowgate` library provides a Python API for gradient estimation in discrete optimization problems. Below is a brief explanation of the API, supplemented with examples.

### Core Concepts

1. **Discrete Problem Definition**:
   - A discrete optimization problem is defined using the `DiscreteProblem` class, which combines:
     - A set of branches, each with a function $f_i(\theta)$ and its derivative $f_i'(\theta)$.
     - A logits model that defines the logits $\alpha(\theta)$ and their derivatives $\alpha'(\theta)$.
     - A customizable probability function to compute probabilities from logits (e.g., softmax or sigmoid).
     - A sampling function to draw samples from the probability distribution (e.g., Bernoulli).

   ```python
   import jax.numpy as jnp
   from mellowgate.api.functions import DiscreteProblem, Branch, LogitsModel

   # Define branches with JAX-compatible functions
   branches = [
       Branch(
           function=lambda th: jnp.cos(th),
           derivative_function=lambda th: -jnp.sin(th),
       ),
       Branch(
           function=lambda th: jnp.sin(th),
           derivative_function=lambda th: jnp.cos(th),
       )
   ]

   # Define a custom sigmoid probability function
   def sigmoid(logits):
       return 1 / (1 + jnp.exp(-logits))

   # Define a custom categorical sampling function using JAX
   def categorical_sampling(probabilities, key=None):
       import jax.random
       if key is None:
           key = jax.random.PRNGKey(0)
       return jax.random.categorical(key, jnp.log(probabilities))

   # Create a logits model with sigmoid probabilities
   logits_model = LogitsModel(
       logits_function=lambda th: jnp.array([-th, th]),
       logits_derivative_function=lambda th: jnp.array([-1.0, 1.0]),
       probability_function=sigmoid,
   )

   # Define the discrete problem with JAX-compatible sampling
   problem = DiscreteProblem(
       branches=branches,
       logits_model=logits_model,
       sampling_function=categorical_sampling,
   )
   ```

2. **Gradient Estimation Methods**:
   - The library implements various gradient estimation methods, configurable per problem instance:
     - **Finite Differences**: Approximates gradients by perturbing parameters.
     - **REINFORCE**: Uses Monte Carlo sampling and policy gradients.
     - **Gumbel-Softmax**: Provides a continuous relaxation of discrete samples.

   ```python
   from mellowgate.api.estimators import FiniteDifferenceConfig, ReinforceConfig, GumbelSoftmaxConfig

   fd_config = FiniteDifferenceConfig(step_size=1e-3, num_samples=20000)
   reinforce_config = ReinforceConfig(num_samples=20000, use_baseline=True)
   gs_config = GumbelSoftmaxConfig(temperature=0.5, num_samples=800)
   ```

3. **Running Experiments**:
   - Use the `run_parameter_sweep` function to evaluate gradient estimators over a range of parameter values.

   ```python
   import jax.numpy as jnp
   from mellowgate.api.experiments import Sweep, run_parameter_sweep

   sweep = Sweep(
       theta_values=jnp.linspace(-2.5, 2.5, 21),
       num_repetitions=1000,
       estimator_configs={
           "fd": {"cfg": fd_config},
           "reinforce": {"cfg": reinforce_config},
           "gs": {"cfg": gs_config},
       },
   )

   results = run_parameter_sweep(problem, sweep)
   ```

4. **Visualization**:
   - The library provides utilities for visualizing gradient estimates, bias-variance trade-offs, and computational time.

   ```python
   from mellowgate.plots.metrics import (
       plot_gradient_estimates_vs_truth,
       plot_bias_variance_mse_analysis,
       plot_computational_time_analysis,
   )

   plot_gradient_estimates_vs_truth(results, problem.compute_exact_gradient)
   plot_bias_variance_mse_analysis(results, problem.compute_exact_gradient)
   plot_computational_time_analysis(results)
   ```

### Example Workflow

1. Define the discrete problem.
2. Configure gradient estimators.
3. Run experiments over a range of parameters.
4. Visualize the results.

For a complete example, see `example.py` in the repository.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Set up development environment with `make setup-dev`
3. Create a feature branch
4. Make your changes following our guidelines
5. Ensure all checks pass with `make lint` and `make pre-commit`
6. Submit a pull request

## License

[Add license information here]

### Notes on Normalizing Logits

Logits can be normalized using various methods depending on the requirements of the problem. While the softmax function is a common choice, it is not the only option. Below are some alternatives:

- **Softmax Function**:
  - Converts logits into probabilities that sum to 1.
  - Formula:
    $$
    \pi(k | \theta) = \frac{\exp(\alpha_k)}{\sum_j \exp(\alpha_j)}.
    $$
  - Commonly used in multi-class classification problems.

- **Sigmoid Function**:
  - Normalizes logits independently, producing probabilities for each branch without ensuring they sum to 1.
  - Formula:
    $$
    \sigma(\alpha) = \frac{1}{1 + e^{-\alpha}}.
    $$
  - Useful for binary classification or independent probabilities.

- **Temperature-Scaled Softmax**:
  - A variant of softmax where logits are divided by a temperature parameter $\tau$ before normalization.
  - Formula:
    $$
    \pi(k | \theta) = \frac{\exp(\alpha_k / \tau)}{\sum_j \exp(\alpha_j / \tau)}.
    $$
  - Lower $\tau$ sharpens the distribution, while higher $\tau$ smoothens it.

- **Sparsemax**:
  - Produces sparse probability distributions, where some probabilities are exactly zero.
  - Useful in applications like attention mechanisms where sparsity is desired.

- **Custom Normalization**:
  - In some cases, a custom normalization function may be designed to meet specific requirements of the problem.

Each method has its own advantages and is suited for different scenarios. The choice of normalization depends on the nature of the discrete optimization problem and the desired properties of the probability distribution.
