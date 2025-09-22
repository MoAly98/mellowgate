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

This project enforces strict code quality standards:

- **Formatting**: [Black](https://black.readthedocs.io/) (88 character line length)
- **Import Sorting**: [isort](https://pycqa.github.io/isort/) (black-compatible profile)
- **Linting**: [flake8](https://flake8.pycqa.org/) with strict rules
- **Type Checking**: [MyPy](https://mypy.readthedocs.io/) (disabled by default, ready to enable)

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

### Performance Considerations

- Use NumPy arrays for numerical computations
- Prefer vectorized operations over loops
- Profile code for performance bottlenecks when needed
- Consider memory usage for large-scale experiments

## Example Usage

```python
import numpy as np
from mellowgate.api.functions import DiscreteProblem, LogitsModel
from mellowgate.api.estimators import (
    finite_difference_gradient,
    reinforce_gradient,
    gumbel_softmax_gradient,
    FiniteDifferenceConfig,
    ReinforceConfig,
    GumbelSoftmaxConfig,
    ReinforceState
)

# Define your discrete optimization problem
# See example.py for a complete working example
```

For a complete working example, run:
```bash
make example
```

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
