"""Tests for mellowgate.estimators.gumbel_softmax module."""

import jax
import jax.numpy as jnp
import pytest

from mellowgate.core import Branch, DiscreteProblem, LogitsModel
from mellowgate.estimators import (
    GumbelSoftmaxConfig,
    gumbel_softmax_gradient,
)

# Enable x64 precision for numerical accuracy in tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_problem():
    """Simple problem for testing."""
    branches = [
        Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th),
        Branch(function=lambda th: th**3, derivative_function=lambda th: 3 * th**2),
    ]
    logits_model = LogitsModel(logits_function=lambda th: jnp.array([th, -th]))
    return DiscreteProblem(branches=branches, logits_model=logits_model)


class TestGumbelSoftmaxConfig:
    """Test GumbelSoftmaxConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GumbelSoftmaxConfig()
        assert config.temperature == 0.5
        assert config.num_samples == 1000
        assert config.use_straight_through_estimator is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GumbelSoftmaxConfig(
            temperature=0.1, num_samples=500, use_straight_through_estimator=True
        )
        assert config.temperature == 0.1
        assert config.num_samples == 500
        assert config.use_straight_through_estimator is True


class TestGumbelSoftmaxGradient:
    """Test Gumbel-Softmax gradient estimation."""

    def test_basic_gradient_computation(self, simple_problem):
        """Test basic gradient computation."""
        theta = jnp.array([1.0])
        config = GumbelSoftmaxConfig(temperature=0.5, num_samples=1000)
        gradient = gumbel_softmax_gradient(simple_problem, theta, config)

        assert gradient is not None
        assert jnp.isfinite(gradient)

    def test_matches_exact_gradient(self, simple_problem):
        """Test Gumbel-Softmax gradient matches exact gradient."""
        theta = jnp.array([1.0])
        exact_grad = simple_problem.compute_exact_gradient(theta)

        config = GumbelSoftmaxConfig(temperature=0.1, num_samples=10000)
        gs_grad = gumbel_softmax_gradient(simple_problem, theta, config)

        # Gumbel-Softmax should be reasonably close to exact
        relative_error = jnp.abs((gs_grad - exact_grad) / exact_grad)
        assert relative_error < 0.1

    def test_zero_gradient_for_constant_function(self):
        """Test gradient is near zero for constant expected value."""
        branches = [
            Branch(
                function=lambda th: 5 * jnp.ones_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
            Branch(
                function=lambda th: 5 * jnp.ones_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
        ]
        logits_model = LogitsModel(logits_function=lambda th: jnp.array([th, -th]))
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([0.5])
        config = GumbelSoftmaxConfig(temperature=0.5, num_samples=5000)
        gradient = gumbel_softmax_gradient(problem, theta, config)

        assert jnp.allclose(gradient, 0.0, atol=0.05)


class TestAnalyticalCorrectness:
    """Test Gumbel-Softmax against problems with known analytical solutions."""

    @pytest.fixture
    def linear_problem(self):
        """Identity function with uniform probabilities: E[f]=θ, dE/dθ=1."""
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
        return DiscreteProblem(branches=branches, logits_model=logits_model)

    @pytest.fixture
    def binary_problem(self):
        """Binary choice: f₁=1, f₂=0, logits=[2θ,0]. At θ=0: dE/dθ=0.5."""
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
        return DiscreteProblem(branches=branches, logits_model=logits_model)

    @pytest.fixture
    def increasing_problem(self):
        """Problem where E[f] increases with θ."""
        branches = [
            Branch(
                function=lambda th: th**2,
                derivative_function=lambda th: 2 * th,
            ),
            Branch(
                function=lambda th: 2 * th**2,
                derivative_function=lambda th: 4 * th,
            ),
        ]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array(
                [jnp.zeros_like(th), jnp.zeros_like(th)]
            )
        )
        return DiscreteProblem(branches=branches, logits_model=logits_model)

    def test_linear_function_gradient_one(self, linear_problem):
        """Gumbel-Softmax should produce gradient near 1 for linear problem."""
        theta = jnp.array([0.0])
        exact_grad = linear_problem.compute_exact_gradient(theta)
        assert jnp.allclose(exact_grad, 1.0, atol=1e-10)

        config = GumbelSoftmaxConfig(temperature=0.1, num_samples=10000)
        gs_grad = gumbel_softmax_gradient(linear_problem, theta, config)
        assert jnp.allclose(gs_grad, 1.0, atol=0.05)

    def test_binary_choice_gradient_half(self, binary_problem):
        """Gumbel-Softmax should produce gradient near 0.5 at θ=0.

        Analytical solution:
        - π₁ = exp(2θ)/(exp(2θ)+1)
        - dπ₁/dθ at θ=0 = 2/(2²) = 0.5
        - dE/dθ = dπ₁/dθ × (f₁-f₂) = 0.5 × 1 = 0.5
        """
        theta = jnp.array([0.0])
        exact_grad = binary_problem.compute_exact_gradient(theta)
        assert jnp.allclose(exact_grad, 0.5, atol=1e-10)

        config = GumbelSoftmaxConfig(temperature=0.1, num_samples=10000)
        gs_grad = gumbel_softmax_gradient(binary_problem, theta, config)
        assert jnp.allclose(gs_grad, 0.5, atol=0.05)

    def test_gradient_sign_correctness(self, increasing_problem):
        """Gumbel-Softmax should produce positive gradient for increasing function."""
        theta = jnp.array([1.0])
        exact_grad = increasing_problem.compute_exact_gradient(theta)
        assert exact_grad > 0

        config = GumbelSoftmaxConfig(temperature=0.5, num_samples=5000)
        gs_grad = gumbel_softmax_gradient(increasing_problem, theta, config)
        assert gs_grad > 0
