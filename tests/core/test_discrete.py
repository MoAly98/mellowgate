"""Tests for mellowgate.core.discrete module."""

import jax.numpy as jnp
import pytest

from mellowgate.core import Branch, DiscreteProblem, LogitsModel


@pytest.fixture
def simple_problem():
    """Simple two-branch problem for testing."""
    branches = [
        Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th),
        Branch(function=lambda th: th**3, derivative_function=lambda th: 3 * th**2),
    ]
    logits_model = LogitsModel(logits_function=lambda th: jnp.array([th, -th]))
    return DiscreteProblem(branches=branches, logits_model=logits_model)


class TestDiscreteProblemBasics:
    """Test basic DiscreteProblem functionality."""

    def test_num_branches_property(self, simple_problem):
        """Test num_branches property."""
        assert simple_problem.num_branches == 2

    def test_compute_probabilities(self, simple_problem):
        """Test probability computation."""
        theta = jnp.array([0.0, 1.0])
        probs = simple_problem.compute_probabilities(theta)

        assert probs.shape == (2, 2)
        assert jnp.all(jnp.isfinite(probs))
        # Probabilities should sum to 1
        assert jnp.allclose(jnp.sum(probs, axis=0), 1.0)

    def test_compute_function_values(self, simple_problem):
        """Test function value computation."""
        theta = jnp.array([0.0, 1.0, 2.0])
        values = simple_problem.compute_function_values(theta)

        assert values.shape == (2, 3)
        # Check specific values: branch 1 is th^2, branch 2 is th^3
        assert jnp.allclose(values[0, :], jnp.array([0.0, 1.0, 4.0]))
        assert jnp.allclose(values[1, :], jnp.array([0.0, 1.0, 8.0]))

    def test_compute_expected_value(self, simple_problem):
        """Test expected value computation."""
        theta = jnp.array([1.0])
        expected = simple_problem.compute_expected_value(theta)

        assert expected.shape == (1,)
        assert jnp.isfinite(expected)

    def test_compute_exact_gradient(self, simple_problem):
        """Test exact gradient computation."""
        theta = jnp.array([1.0])
        gradient = simple_problem.compute_exact_gradient(theta)

        assert gradient is not None
        assert gradient.shape == (1,)
        assert jnp.isfinite(gradient)


class TestAnalyticalSolutions:
    """Test DiscreteProblem against hand-calculated analytical solutions."""

    def test_uniform_constant_branches_zero_gradient(self):
        """Test gradient is zero for constant branches with uniform probabilities.

        Problem:
        - f₁(θ) = 1, f₂(θ) = 2
        - π₁ = π₂ = 0.5 (uniform)
        - E[f] = 1.5 (constant)
        - dE/dθ = 0
        """
        branches = [
            Branch(
                function=lambda th: jnp.ones_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
            Branch(
                function=lambda th: 2.0 * jnp.ones_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
        ]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array(
                [jnp.zeros_like(th), jnp.zeros_like(th)]
            )
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([0.0, 1.0, -1.0])
        expected = problem.compute_expected_value(theta)
        gradient = problem.compute_exact_gradient(theta)

        # Expected value should be 1.5 everywhere
        assert jnp.allclose(expected, 1.5)
        # Gradient should be exactly 0
        assert jnp.allclose(gradient, 0.0, atol=1e-10)

    def test_pure_probability_gradient(self):
        """Test gradient from probability changes only.

        Problem:
        - f₁(θ) = 1, f₂(θ) = 0
        - logits = [2θ, 0]
        - At θ=0: dE/dθ = 0.5
        """
        branches = [
            Branch(
                function=lambda th: jnp.ones_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
            Branch(
                function=lambda th: jnp.zeros_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
        ]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([2.0 * th, jnp.zeros_like(th)])
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([0.0])
        expected = problem.compute_expected_value(theta)
        gradient = problem.compute_exact_gradient(theta)

        # E[f] = π₁ = 0.5 at θ=0
        assert jnp.allclose(expected, 0.5)
        # dE/dθ = 0.5 at θ=0
        assert jnp.allclose(gradient, 0.5, atol=1e-10)

    def test_pure_pathwise_gradient(self):
        """Test gradient from function derivatives only.

        Problem:
        - f₁(θ) = θ, f₂(θ) = θ
        - π₁ = π₂ = 0.5 (uniform)
        - E[f] = θ
        - dE/dθ = 1
        """
        branches = [
            Branch(
                function=lambda th: th,
                derivative_function=lambda th: jnp.ones_like(th),
            ),
            Branch(
                function=lambda th: th,
                derivative_function=lambda th: jnp.ones_like(th),
            ),
        ]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array(
                [jnp.zeros_like(th), jnp.zeros_like(th)]
            )
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([-1.0, 0.0, 1.0])
        expected = problem.compute_expected_value(theta)
        gradient = problem.compute_exact_gradient(theta)

        # E[f] = θ
        assert jnp.allclose(expected, theta)
        # dE/dθ = 1 everywhere
        assert jnp.allclose(gradient, 1.0, atol=1e-10)

    def test_combined_gradient_components(self):
        """Test gradient with both probability and pathwise components.

        Problem:
        - f₁(θ) = θ, f₂(θ) = 0
        - logits = [θ, 0]
        - At θ=0: dE/dθ = 0.5 (pathwise term = 0.5, probability term = 0)
        """
        branches = [
            Branch(
                function=lambda th: th,
                derivative_function=lambda th: jnp.ones_like(th),
            ),
            Branch(
                function=lambda th: jnp.zeros_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
        ]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([th, jnp.zeros_like(th)])
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([0.0])
        gradient = problem.compute_exact_gradient(theta)

        # dE/dθ = 0.5 at θ=0
        assert jnp.allclose(gradient, 0.5, atol=1e-10)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_branch_problem(self):
        """Test problem with only one branch."""
        branches = [
            Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th)
        ]
        logits_model = LogitsModel(logits_function=lambda th: jnp.array([th]))
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])
        probs = problem.compute_probabilities(theta)

        assert probs.shape == (1, 1)
        assert jnp.allclose(probs, 1.0)

    def test_exact_gradient_missing_derivatives(self):
        """Test exact gradient returns None when derivatives missing."""
        branches = [
            Branch(function=lambda th: th**2),  # No derivative
            Branch(function=lambda th: th**3),
        ]
        logits_model = LogitsModel(logits_function=lambda th: jnp.array([th, -th]))
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])
        gradient = problem.compute_exact_gradient(theta)

        assert gradient is None
