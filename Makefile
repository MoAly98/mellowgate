# Mellowgate Development Makefile
.PHONY: help install lint format test test-cov clean pre-commit setup-dev

# Default target
help:
	@echo "Mellowgate Development Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup-dev    Install development dependencies and pre-commit hooks"
	@echo "  install      Install package in editable mode"
	@echo ""
	@echo "Code Quality:"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run all linting checks (black, isort, flake8)"
	@echo "  pre-commit   Run pre-commit hooks on all files"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests with pytest"
	@echo "  test-cov     Run tests with coverage report"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        Clean up build artifacts and cache files"
	@echo "  example      Run the example script"

# Development setup
setup-dev:
	@echo "Setting up development environment..."
	pixi install
	pixi run pre-commit install
	@echo "✅ Development environment setup complete!"

install:
	@echo "Installing package in editable mode..."
	pixi run pip install -e .
	@echo "✅ Package installed!"

# Code formatting
format:
	@echo "Formatting code with black..."
	pixi run black src/ example.py
	@echo "Sorting imports with isort..."
	pixi run isort src/ example.py
	@echo "✅ Code formatting complete!"

# Linting
lint:
	@echo "Checking code formatting with black..."
	pixi run black --check --diff src/ example.py
	@echo "Checking import sorting with isort..."
	pixi run isort --check-only --diff src/ example.py
	@echo "Linting with flake8..."
	pixi run flake8 src/ example.py
	@echo "✅ All linting checks passed!"

# Pre-commit
pre-commit:
	@echo "Running pre-commit hooks on all files..."
	pixi run pre-commit run --all-files
	@echo "✅ Pre-commit checks complete!"

# Testing
test:
	@echo "Running tests with pytest..."
	pixi run pytest tests/ -v
	@echo "✅ Tests completed!"

test-cov:
	@echo "Running tests with coverage..."
	pixi run pytest tests/ -v --cov=mellowgate --cov-report=term-missing --cov-report=xml
	@echo "✅ Tests with coverage completed! Check coverage.xml for detailed report."

# Example
example:
	@echo "Running example script..."
	pixi run python example.py
	@echo "✅ Example completed!"

# Cleanup
clean:
	@echo "Cleaning up build artifacts and cache files..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Cleanup complete!"

# Type checking (placeholder - uncomment when ready)
# type-check:
# 	@echo "Type checking with mypy..."
# 	pixi run mypy src/mellowgate/
# 	@echo "✅ Type checking complete!"
