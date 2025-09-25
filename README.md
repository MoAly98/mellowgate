# mellowgate

A Python library for gradient estimation in discrete optimization problems. This package provides various gradient estimation methods including finite differences, REINFORCE, and Gumbel-Softmax for differentiable discrete decision making.

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
make pre-commit    # Run pre-commit hooks on all files
make example       # Run the example script
make clean         # Clean up build artifacts
```

### Code Quality Standards

This project enforces code quality standards:

- **Formatting**: [Black](https://black.readthedocs.io/) (88 character line length)
- **Import Sorting**: [isort](https://pycqa.github.io/isort/) (black-compatible profile)
- **Linting**: [flake8](https://flake8.pycqa.org/) with strict rules
<!-- - **Type Checking**: [MyPy](https://mypy.readthedocs.io/) (disabled by default, ready to enable) -->

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
- Add unit tests for all new functionality (when test framework is set up)
- Ensure all tests pass before submitting changes
- Aim for high test coverage

#### Documentation
- Update docstrings when changing function signatures
- Add examples to docstrings for complex functions
- Update README.md for significant changes

### Continuous Integration

The CI pipeline runs on every push and pull request:

- **Linting**: Checks code formatting, import sorting, and flake8 compliance
- **Type Checking**: MyPy type checking (when enabled)
- **Testing**: Unit tests with pytest (when implemented)
- **Build Validation**: Ensures package can be built and installed

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
│   │   ├── experiments.py    # Experiment running utilities
│   │   └── functions.py      # Problem definition classes
│   ├── plots/                # Visualization utilities
│   ├── utils/                # Utility functions
│   └── logging/              # Logging configuration
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
   make example       # Test functionality
   make pre-commit    # Run all hooks
   ```

4. **Submit pull request**:
   - Ensure CI passes
   - Add clear description of changes
   - Reference any related issues


## Example Usage

For a complete working example, run:
```bash
make example
```

## mellowgate API Overview

The `mellowgate` library provides a Python API for gradient estimation in discrete optimization problems. Below is a brief explanation of the API, supplemented with mathematical formulations.

### Core Concepts

1. **Discrete Problem Definition**:
   - A discrete optimization problem is defined using the `DiscreteProblem` class, which combines:
     - A set of branches, each with a function $f_i(\theta)$ and its derivative $f_i'(\theta)$.
     - A logits model that defines the logits $\alpha(\theta)$ and their derivatives $\alpha'(\theta)$.
     - A customizable probability function to compute probabilities from logits (e.g., softmax or sigmoid).
     - A sampling function to draw samples from the probability distribution (e.g., Bernoulli).

   ```python
   from mellowgate.api.functions import DiscreteProblem, Branch, LogitsModel

   # Define branches
   branches = [
       Branch(
           function=lambda th: float(np.cos(th)),
           derivative_function=lambda th: float(-np.sin(th)),
       ),
       Branch(
           function=lambda th: float(np.sin(th)),
           derivative_function=lambda th: float(np.cos(th)),
       )
   ]

   # Define a custom sigmoid probability function
   def sigmoid(logits):
       return 1 / (1 + np.exp(-logits))

   # Define a custom Bernoulli sampling function
   def bernoulli_sampling(probabilities):
       return np.random.choice(len(probabilities), p=probabilities)

   # Create a logits model with sigmoid probabilities
   logits_model = LogitsModel(
       logits_function=lambda th: alpha * th,
       logits_derivative_function=lambda th: alpha,
       probability_function=sigmoid,  # Use sigmoid for probabilities
   )

   # Define the discrete problem with Bernoulli sampling
   problem = DiscreteProblem(
       branches=branches,
       logits_model=logits_model,
       sampling_function=bernoulli_sampling,  # Use Bernoulli sampling
   )
   ```

2. **Mathematical Formulation**:
   - The library is designed to solve discrete optimization problems by estimating gradients through stochastic relaxations. Below is the updated mathematical formulation:

     1. **Discrete Optimization Problem**:
       - Given a parameter $\theta$, the goal is to optimize the expected value of a function $f(x; \theta)$ over discrete choices $x \in \{x_1, x_2, \dots, x_N\}$:
         $$
         \mathbb{E}[f(x; \theta)] = \sum_{i=1}^N \pi(x_i | \theta) f(x_i; \theta),
         $$
         where $\pi(x_i | \theta)$ is the probability of selecting $x_i$.

     2. **Stochastic Sampling**:
       - To compute stochastic values, samples are drawn from the probability distribution $\pi(x_i | \theta)$ using a user-defined sampling function $S$:
         $$
         x \sim S(\pi(x_i | \theta)).
         $$
       - For example:
         - **Bernoulli Sampling**: $S(\pi) = \text{Bernoulli}(\pi)$

     3. **Gradient Estimation**:
       - The gradient of the objective with respect to $\theta$ is:
         $$
         \nabla_\theta \mathbb{E}[f(x; \theta)] = \sum_{i=1}^N \nabla_\theta \pi(x_i | \theta) f(x_i; \theta) + \pi(x_i | \theta) \nabla_\theta f(x_i; \theta).
         $$
       - This gradient is estimated using methods like finite differences, REINFORCE, and Gumbel-Softmax.

     4. **Stochastic Relaxation**:
       - To enable gradient-based optimization, stochastic relaxations like Gumbel-Softmax introduce continuous approximations to discrete choices, allowing backpropagation through $\pi(x_i | \theta)$.

   ```python
   from mellowgate.api.estimators import FiniteDifferenceConfig, ReinforceConfig, GumbelSoftmaxConfig

   fd_config = FiniteDifferenceConfig(step_size=1e-3, num_samples=20000)
   reinforce_config = ReinforceConfig(num_samples=20000, use_baseline=True)
   gs_config = GumbelSoftmaxConfig(temperature=0.5, num_samples=800)
   ```

4. **Running Experiments**:
   - Use the `run_parameter_sweep` function to evaluate gradient estimators over a range of parameter values.

   ```python
   from mellowgate.api.experiments import Sweep, run_parameter_sweep

   sweep = Sweep(
       theta_values=np.linspace(-2.5, 2.5, 21),
       num_repetitions=1000,
       estimator_configs={
           "fd": {"cfg": fd_config},
           "reinforce": {"cfg": reinforce_config},
           "gs": {"cfg": gs_config},
       },
   )

   results = run_parameter_sweep(problem, sweep)
   ```

5. **Visualization**:
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
