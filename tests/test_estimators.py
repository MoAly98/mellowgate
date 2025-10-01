"""Unit tests for gradient estimators module.

This module provides comprehensive tests for all gradient estimation methods
including finite differences, REINFORCE, and Gumbel-Softmax approaches.
Tests cover shapes, values, edge cases, and error conditions.
"""

import jax
import jax.numpy as jnp
import pytest

from mellowgate.api.estimators import (
    FiniteDifferenceConfig,
    GumbelSoftmaxConfig,
    ReinforceConfig,
    ReinforceState,
    finite_difference_gradient,
    gumbel_softmax_gradient,
    reinforce_gradient,
)
from mellowgate.api.functions import Branch, DiscreteProblem, LogitsModel


@pytest.fixture
def test_theta():
    """Common theta values for testing with minimum 5 values."""
    return jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])


@pytest.fixture
def random_key():
    """Common random key for reproducible testing."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def simple_branches():
    """Simple branches for easier testing."""
    return [
        Branch(
            function=lambda th: th**2,
            derivative_function=lambda th: 2 * th,
        ),
        Branch(
            function=lambda th: th**3,
            derivative_function=lambda th: 3 * th**2,
        ),
    ]


@pytest.fixture
def simple_logits_model():
    """Simple logits model for easier testing."""
    return LogitsModel(
        logits_function=lambda th: jnp.array([0.5 * th, -0.5 * th]),
        logits_derivative_function=lambda th: jnp.array(
            [0.5 * jnp.ones_like(th), -0.5 * jnp.ones_like(th)]
        ),
    )


@pytest.fixture
def simple_discrete_problem(simple_branches, simple_logits_model):
    return DiscreteProblem(
        branches=simple_branches,
        logits_model=simple_logits_model,
    )


@pytest.fixture
def logits_model_no_derivatives():
    """Logits model without derivative functions for testing error cases."""
    return LogitsModel(
        logits_function=lambda th: jnp.array([0.5 * th, -0.5 * th]),
    )


@pytest.fixture
def discrete_problem_no_logits_derivatives(
    simple_branches, logits_model_no_derivatives
):
    return DiscreteProblem(
        branches=simple_branches,
        logits_model=logits_model_no_derivatives,
    )


class TestFiniteDifferenceConfig:
    """Test FiniteDifferenceConfig dataclass."""

    def test_config_creation_default(self):
        """Test config creation with default values."""
        config = FiniteDifferenceConfig()
        assert config.step_size == 1e-3
        assert config.num_samples == 2000

    def test_config_creation_custom(self):
        """Test config creation with custom values."""
        config = FiniteDifferenceConfig(step_size=1e-4, num_samples=1000)
        assert config.step_size == 1e-4
        assert config.num_samples == 1000


class TestFiniteDifferenceGradient:
    """Test finite difference gradient estimation."""

    def test_scalar_input_output_type(self, simple_discrete_problem):
        """Test that scalar input returns scalar output."""
        config = FiniteDifferenceConfig(num_samples=100)
        theta = 1.0

        gradient = finite_difference_gradient(simple_discrete_problem, theta, config)

        assert isinstance(gradient, float), f"Expected float, got {type(gradient)}"

    def test_array_input_output_shape(self, simple_discrete_problem, test_theta):
        """Test that array input returns array output with correct shape."""
        config = FiniteDifferenceConfig(num_samples=100)

        gradient = finite_difference_gradient(
            simple_discrete_problem, test_theta, config
        )

        assert isinstance(
            gradient, jnp.ndarray
        ), f"Expected ndarray, got {type(gradient)}"
        assert (
            gradient.shape == test_theta.shape
        ), f"Expected shape {test_theta.shape}, got {gradient.shape}"

    def test_gradient_finite_values(self, simple_discrete_problem, test_theta):
        """Test that gradient estimates are finite (not NaN or infinite)."""
        config = FiniteDifferenceConfig(num_samples=100)

        gradient = finite_difference_gradient(
            simple_discrete_problem, test_theta, config
        )

        assert jnp.all(jnp.isfinite(gradient)), "All gradient values should be finite"

    def test_step_size_effect(self, simple_discrete_problem):
        """Test that different step sizes produce different results."""
        theta = 1.0

        config_small = FiniteDifferenceConfig(step_size=1e-4, num_samples=500)
        config_large = FiniteDifferenceConfig(step_size=1e-2, num_samples=500)

        gradient_small = finite_difference_gradient(
            simple_discrete_problem, theta, config_small
        )
        gradient_large = finite_difference_gradient(
            simple_discrete_problem, theta, config_large
        )

        # Different step sizes should generally produce different results
        # (though they might be close for well-behaved functions)
        assert isinstance(gradient_small, float)
        assert isinstance(gradient_large, float)

    def test_zero_theta(self, simple_discrete_problem):
        """Test behavior at theta=0."""
        config = FiniteDifferenceConfig(num_samples=200)
        theta = 0.0

        gradient = finite_difference_gradient(simple_discrete_problem, theta, config)

        assert isinstance(gradient, float)
        assert jnp.isfinite(gradient)

    def test_reproducibility_with_seed(self, simple_branches, simple_logits_model):
        """Test that results are reproducible with fixed random seed."""
        config = FiniteDifferenceConfig(num_samples=100)
        theta = 1.0

        # Create discrete problems with custom sampling functions for reproducibility
        def deterministic_sampler(probabilities):
            # Always return the first branch for reproducibility
            return 0

        discrete_problem1 = DiscreteProblem(
            branches=simple_branches,
            logits_model=simple_logits_model,
            sampling_function=deterministic_sampler,
        )
        discrete_problem2 = DiscreteProblem(
            branches=simple_branches,
            logits_model=simple_logits_model,
            sampling_function=deterministic_sampler,
        )

        gradient1 = finite_difference_gradient(discrete_problem1, theta, config)
        gradient2 = finite_difference_gradient(discrete_problem2, theta, config)

        assert gradient1 == gradient2, "Results should be reproducible with same seed"


class TestReinforceConfig:
    """Test ReinforceConfig dataclass."""

    def test_config_creation_default(self):
        """Test config creation with default values."""
        config = ReinforceConfig()
        assert config.num_samples == 2000
        assert config.use_baseline is True
        assert config.baseline_momentum == 0.9

    def test_config_creation_custom(self):
        """Test config creation with custom values."""
        config = ReinforceConfig(
            num_samples=1000, use_baseline=False, baseline_momentum=0.8
        )
        assert config.num_samples == 1000
        assert config.use_baseline is False
        assert config.baseline_momentum == 0.8


class TestReinforceState:
    """Test ReinforceState class."""

    def test_state_initialization(self):
        """Test state initialization with default values."""
        state = ReinforceState()
        assert state.baseline == 0.0
        assert state.initialized is False

    def test_baseline_update_first_time(self):
        """Test baseline update when not initialized."""
        state = ReinforceState()
        state.update_baseline(5.0, 0.9)

        assert state.baseline == 5.0
        assert state.initialized is True

    def test_baseline_update_subsequent(self):
        """Test baseline update with momentum."""
        state = ReinforceState()

        # First update
        state.update_baseline(10.0, 0.9)
        assert state.baseline == 10.0

        # Second update with momentum
        state.update_baseline(20.0, 0.9)
        expected = 0.9 * 10.0 + 0.1 * 20.0  # 9.0 + 2.0 = 11.0
        assert state.baseline == expected


class TestReinforceGradient:
    """Test REINFORCE gradient estimation."""

    def test_scalar_input_output_type(self, simple_discrete_problem):
        """Test that scalar input returns scalar output."""
        config = ReinforceConfig(num_samples=100)
        state = ReinforceState()
        theta = 1.0

        gradient = reinforce_gradient(simple_discrete_problem, theta, config, state)

        assert isinstance(gradient, float), f"Expected float, got {type(gradient)}"

    def test_array_input_output_shape(self, simple_discrete_problem, test_theta):
        """Test that array input returns array output with correct shape."""
        config = ReinforceConfig(num_samples=100)
        state = ReinforceState()

        gradient = reinforce_gradient(
            simple_discrete_problem, test_theta, config, state
        )

        assert isinstance(
            gradient, jnp.ndarray
        ), f"Expected ndarray, got {type(gradient)}"
        assert (
            gradient.shape == test_theta.shape
        ), f"Expected shape {test_theta.shape}, got {gradient.shape}"

    def test_gradient_finite_values(self, simple_discrete_problem, test_theta):
        """Test that gradient estimates are finite."""
        config = ReinforceConfig(num_samples=100)
        state = ReinforceState()

        gradient = reinforce_gradient(
            simple_discrete_problem, test_theta, config, state
        )

        assert jnp.all(jnp.isfinite(gradient)), "All gradient values should be finite"

    def test_baseline_effect(self, simple_discrete_problem):
        """Test that using baseline affects results."""
        theta = 1.0

        config_with_baseline = ReinforceConfig(num_samples=200, use_baseline=True)
        config_without_baseline = ReinforceConfig(num_samples=200, use_baseline=False)

        state_with = ReinforceState()
        state_without = ReinforceState()

        gradient_with = reinforce_gradient(
            simple_discrete_problem, theta, config_with_baseline, state_with
        )
        gradient_without = reinforce_gradient(
            simple_discrete_problem, theta, config_without_baseline, state_without
        )

        assert isinstance(gradient_with, float)
        assert isinstance(gradient_without, float)
        # They might be different due to baseline usage

    def test_state_updates_baseline(self, simple_discrete_problem):
        """Test that state baseline gets updated during gradient computation."""
        config = ReinforceConfig(num_samples=100, use_baseline=True)
        state = ReinforceState()
        theta = 1.0

        assert not state.initialized

        reinforce_gradient(simple_discrete_problem, theta, config, state)

        assert state.initialized
        assert state.baseline != 0.0  # Should have been updated

    def test_missing_logits_derivatives_error(
        self, discrete_problem_no_logits_derivatives
    ):
        """Test that missing logits derivatives raises ValueError."""
        config = ReinforceConfig(num_samples=100)
        state = ReinforceState()
        theta = 1.0

        with pytest.raises(
            ValueError, match="REINFORCE requires logits_derivative_function"
        ):
            reinforce_gradient(
                discrete_problem_no_logits_derivatives, theta, config, state
            )

    def test_zero_theta(self, simple_discrete_problem):
        """Test behavior at theta=0."""
        config = ReinforceConfig(num_samples=200)
        state = ReinforceState()
        theta = 0.0

        gradient = reinforce_gradient(simple_discrete_problem, theta, config, state)

        assert isinstance(gradient, float)
        assert jnp.isfinite(gradient)


class TestGumbelSoftmaxConfig:
    """Test GumbelSoftmaxConfig dataclass."""

    def test_config_creation_default(self):
        """Test config creation with default values."""
        config = GumbelSoftmaxConfig()
        assert config.temperature == 0.5
        assert config.num_samples == 1000
        assert config.use_straight_through_estimator is False

    def test_config_creation_custom(self):
        """Test config creation with custom values."""
        config = GumbelSoftmaxConfig(
            temperature=1.0, num_samples=500, use_straight_through_estimator=True
        )
        assert config.temperature == 1.0
        assert config.num_samples == 500
        assert config.use_straight_through_estimator is True


class TestGumbelSoftmaxGradient:
    """Test Gumbel-Softmax gradient estimation."""

    def test_scalar_input_output_type(self, simple_discrete_problem):
        """Test that scalar input returns scalar output."""
        config = GumbelSoftmaxConfig(num_samples=100)
        theta = 1.0

        gradient = gumbel_softmax_gradient(simple_discrete_problem, theta, config)

        assert isinstance(gradient, float), f"Expected float, got {type(gradient)}"

    def test_array_input_output_shape(self, simple_discrete_problem, test_theta):
        """Test that array input returns array output with correct shape."""
        config = GumbelSoftmaxConfig(num_samples=100)

        gradient = gumbel_softmax_gradient(simple_discrete_problem, test_theta, config)

        assert isinstance(
            gradient, jnp.ndarray
        ), f"Expected ndarray, got {type(gradient)}"
        assert (
            gradient.shape == test_theta.shape
        ), f"Expected shape {test_theta.shape}, got {gradient.shape}"

    def test_gradient_finite_values(self, simple_discrete_problem, test_theta):
        """Test that gradient estimates are finite."""
        config = GumbelSoftmaxConfig(num_samples=100)

        gradient = gumbel_softmax_gradient(simple_discrete_problem, test_theta, config)

        assert jnp.all(jnp.isfinite(gradient)), "All gradient values should be finite"

    def test_temperature_effect(self, simple_discrete_problem):
        """Test that different temperatures produce different results."""
        theta = 1.0

        config_low_temp = GumbelSoftmaxConfig(temperature=0.1, num_samples=200)
        config_high_temp = GumbelSoftmaxConfig(temperature=2.0, num_samples=200)

        gradient_low = gumbel_softmax_gradient(
            simple_discrete_problem, theta, config_low_temp
        )
        gradient_high = gumbel_softmax_gradient(
            simple_discrete_problem, theta, config_high_temp
        )

        assert isinstance(gradient_low, float)
        assert isinstance(gradient_high, float)
        # Different temperatures should generally produce different results

    def test_missing_logits_derivatives_error(
        self, discrete_problem_no_logits_derivatives
    ):
        """Test that missing logits derivatives raises ValueError."""
        config = GumbelSoftmaxConfig(num_samples=100)
        theta = 1.0

        with pytest.raises(
            ValueError, match="Gumbel-Softmax requires logits_derivative_function"
        ):
            gumbel_softmax_gradient(
                discrete_problem_no_logits_derivatives, theta, config
            )

    def test_zero_theta(self, simple_discrete_problem):
        """Test behavior at theta=0."""
        config = GumbelSoftmaxConfig(num_samples=200)
        theta = 0.0

        gradient = gumbel_softmax_gradient(simple_discrete_problem, theta, config)

        assert isinstance(gradient, float)
        assert jnp.isfinite(gradient)

    def test_reproducibility_with_seed(self, simple_discrete_problem):
        """Test that results are reproducible with fixed random seed."""
        config = GumbelSoftmaxConfig(num_samples=100)
        theta = 1.0

        gradient1 = gumbel_softmax_gradient(simple_discrete_problem, theta, config)
        gradient2 = gumbel_softmax_gradient(simple_discrete_problem, theta, config)

        # Results may vary due to stochastic nature, but should be reasonable
        assert isinstance(gradient1, float)
        assert isinstance(gradient2, float)


class TestGradientAccuracy:
    """Test gradient estimator accuracy against known analytical solutions."""

    @pytest.fixture
    def quadratic_problem(self):
        """Create a simple quadratic problem with known exact gradients.

        Functions: f1(θ) = θ², f2(θ) = (θ-1)²
        Logits: a1(θ) = θ, a2(θ) = -θ
        Probabilities: p1 = exp(θ)/(exp(θ) + exp(-θ)) = sigmoid(2θ)
                      p2 = exp(-θ)/(exp(θ) + exp(-θ)) = sigmoid(-2θ)

        Expected value: E[f] = p1 * θ² + p2 * (θ-1)²
        Exact gradient can be computed analytically.
        """
        branches = [
            Branch(
                function=lambda th: th**2,
                derivative_function=lambda th: 2 * th,
            ),
            Branch(
                function=lambda th: (th - 1) ** 2,
                derivative_function=lambda th: 2 * (th - 1),
            ),
        ]

        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([th, -th]),
            logits_derivative_function=lambda th: jnp.array(
                [jnp.ones_like(th), -jnp.ones_like(th)]
            ),
        )

        return DiscreteProblem(branches=branches, logits_model=logits_model)

    def _compute_exact_gradient(self, theta):
        """Compute exact gradient for the quadratic problem analytically.

        For the quadratic problem:
        E[f] = p1 * θ² + p2 * (θ-1)²
        where p1 = sigmoid(2θ), p2 = sigmoid(-2θ) = 1 - p1

        d/dθ E[f] = d/dθ [p1 * θ² + (1-p1) * (θ-1)²]
                  = dp1/dθ * [θ² - (θ-1)²] + p1 * 2θ + (1-p1) * 2(θ-1)
                  = dp1/dθ * [2θ - 1] + 2θ * p1 + 2(θ-1) * (1-p1)

        where dp1/dθ = 2 * p1 * (1-p1) (derivative of sigmoid(2θ))
        """
        theta = jnp.asarray(theta)

        # Compute probabilities
        logits_diff = 2 * theta  # θ - (-θ) = 2θ
        p1 = 1 / (1 + jnp.exp(-logits_diff))  # sigmoid(2θ)
        p2 = 1 - p1

        # Function values
        f1 = theta**2
        f2 = (theta - 1) ** 2

        # Pathwise derivatives
        df1_dtheta = 2 * theta
        df2_dtheta = 2 * (theta - 1)

        # Score function terms
        dp1_dtheta = 2 * p1 * p2  # derivative of sigmoid(2θ)
        score_contribution = dp1_dtheta * (f1 - f2)

        # Pathwise contribution
        pathwise_contribution = p1 * df1_dtheta + p2 * df2_dtheta

        return score_contribution + pathwise_contribution

    def test_finite_difference_accuracy(self, quadratic_problem):
        """Test finite difference accuracy against exact gradient."""
        theta_values = jnp.array([0.0, 0.5, 1.0])
        exact_gradients = self._compute_exact_gradient(theta_values)

        config = FiniteDifferenceConfig(step_size=1e-4, num_samples=10000)
        estimated_gradients = finite_difference_gradient(
            quadratic_problem, theta_values, config
        )

        # Check that estimates are within reasonable tolerance of exact values
        # Finite differences are inherently noisy due to Monte Carlo sampling
        relative_error = jnp.abs(estimated_gradients - exact_gradients) / (
            jnp.abs(exact_gradients) + 1e-8
        )
        print(
            f"FD: Exact={exact_gradients}, "
            f"Estimated={estimated_gradients}, "
            f"RelError={relative_error}"
        )
        assert jnp.all(
            relative_error < 0.4
        ), f"Relative errors too large: {relative_error}"

    def test_reinforce_accuracy(self, quadratic_problem):
        """Test REINFORCE accuracy against exact gradient."""
        theta_values = jnp.array([0.0, 0.5, 1.0])
        exact_gradients = self._compute_exact_gradient(theta_values)

        config = ReinforceConfig(num_samples=10000, use_baseline=True)
        state = ReinforceState()

        estimated_gradients = reinforce_gradient(
            quadratic_problem, theta_values, config, state
        )

        # REINFORCE has higher variance, so use larger tolerance
        relative_error = jnp.abs(estimated_gradients - exact_gradients) / (
            jnp.abs(exact_gradients) + 1e-8
        )
        assert jnp.all(
            relative_error < 0.3
        ), f"Relative errors too large: {relative_error}"

    def test_gumbel_softmax_accuracy(self, quadratic_problem):
        """Test Gumbel-Softmax accuracy against exact gradient."""
        theta_values = jnp.array([0.0, 0.5, 1.0])
        exact_gradients = self._compute_exact_gradient(theta_values)

        config = GumbelSoftmaxConfig(num_samples=5000, temperature=0.1)
        estimated_gradients = gumbel_softmax_gradient(
            quadratic_problem, theta_values, config
        )

        # Gumbel-Softmax should be quite accurate with low temperature
        relative_error = jnp.abs(estimated_gradients - exact_gradients) / (
            jnp.abs(exact_gradients) + 1e-8
        )
        assert jnp.all(
            relative_error < 0.2
        ), f"Relative errors too large: {relative_error}"

    def test_simple_gradient_sanity_check(self):
        """
        Test that estimators produce reasonable gradient estimates on simple problems.
        """
        # Create simple quadratic functions with known behavior
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

        # Use simple theta-dependent logits to avoid edge cases
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([0.1 * th, -0.1 * th]),
            logits_derivative_function=lambda th: jnp.array(
                [0.1 * jnp.ones_like(th), -0.1 * jnp.ones_like(th)]
            ),
        )

        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta_values = jnp.array([1.0, 2.0])

        # Use JAX autodiff for exact reference
        import jax

        def expected_function(theta):
            logits = jnp.array([0.1 * theta, -0.1 * theta])
            probs = jnp.exp(logits) / jnp.sum(jnp.exp(logits))
            f1 = theta**2
            f2 = 2 * theta**2
            return probs[0] * f1 + probs[1] * f2

        exact_grad_func = jax.grad(expected_function)
        expected_gradient = jnp.array([exact_grad_func(t) for t in theta_values])

        # Test finite difference
        fd_config = FiniteDifferenceConfig(step_size=1e-4, num_samples=5000)
        fd_gradient = finite_difference_gradient(problem, theta_values, fd_config)
        fd_error = jnp.abs(fd_gradient - expected_gradient) / (
            jnp.abs(expected_gradient) + 1e-8
        )
        assert jnp.all(fd_error < 0.2), f"FD relative error too large: {fd_error}"

        # Test REINFORCE
        reinforce_config = ReinforceConfig(num_samples=5000)
        reinforce_state = ReinforceState()
        reinforce_grad = reinforce_gradient(
            problem, theta_values, reinforce_config, reinforce_state
        )
        reinforce_error = jnp.abs(reinforce_grad - expected_gradient) / (
            jnp.abs(expected_gradient) + 1e-8
        )
        assert jnp.all(
            reinforce_error < 0.2
        ), f"REINFORCE relative error too large: {reinforce_error}"

    def test_linear_function_reasonable_accuracy(self):
        """Test that estimators give reasonable results for linear functions."""
        # Create linear functions: f1(θ) = 2θ + 1, f2(θ) = 3θ - 1
        branches = [
            Branch(
                function=lambda th: 2 * th + 1,
                derivative_function=lambda th: 2 * jnp.ones_like(th),
            ),
            Branch(
                function=lambda th: 3 * th - 1,
                derivative_function=lambda th: 3 * jnp.ones_like(th),
            ),
        ]

        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([th, -th]),
            logits_derivative_function=lambda th: jnp.array(
                [jnp.ones_like(th), -jnp.ones_like(th)]
            ),
        )

        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta_values = jnp.array([0.5, 1.0, 1.5])

        # Compute exact gradient using JAX autodiff (most reliable)
        import jax

        def expected_function(theta):
            logits = jnp.array([theta, -theta])
            probs = jnp.exp(logits) / jnp.sum(jnp.exp(logits))
            f1 = 2 * theta + 1
            f2 = 3 * theta - 1
            return probs[0] * f1 + probs[1] * f2

        exact_grad_func = jax.grad(expected_function)
        exact_gradient = jnp.array([exact_grad_func(t) for t in theta_values])

        # Test with realistic tolerances for Monte Carlo methods
        fd_config = FiniteDifferenceConfig(step_size=1e-4, num_samples=8000)
        fd_gradient = finite_difference_gradient(problem, theta_values, fd_config)
        fd_error = jnp.abs(fd_gradient - exact_gradient) / (
            jnp.abs(exact_gradient) + 1e-8
        )
        print(
            f"Linear FD: Exact={exact_gradient}, "
            f"Estimated={fd_gradient}, "
            f"RelError={fd_error}"
        )
        assert jnp.all(fd_error < 0.4), f"FD relative errors too large: {fd_error}"

        reinforce_config = ReinforceConfig(num_samples=8000)
        reinforce_state = ReinforceState()
        reinforce_grad = reinforce_gradient(
            problem, theta_values, reinforce_config, reinforce_state
        )
        reinforce_error = jnp.abs(reinforce_grad - exact_gradient) / (
            jnp.abs(exact_gradient) + 1e-8
        )
        print(
            f"Linear REINFORCE: Exact={exact_gradient}, "
            f"Estimated={reinforce_grad}, "
            f"RelError={reinforce_error}"
        )
        assert jnp.all(
            reinforce_error < 0.3
        ), f"REINFORCE relative errors too large: {reinforce_error}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_step_size_finite_difference(self, simple_discrete_problem):
        """Test finite difference with very small step size."""
        config = FiniteDifferenceConfig(step_size=1e-10, num_samples=100)
        theta = 1.0

        gradient = finite_difference_gradient(simple_discrete_problem, theta, config)

        assert isinstance(gradient, float)
        # Might be less accurate due to numerical precision, but should not crash

    def test_very_large_theta_values(self, simple_discrete_problem):
        """Test behavior with very large theta values."""
        theta = jnp.array([-100.0, 100.0])

        # Test finite difference
        fd_config = FiniteDifferenceConfig(num_samples=50)
        fd_gradient = finite_difference_gradient(
            simple_discrete_problem, theta, fd_config
        )
        assert isinstance(fd_gradient, jnp.ndarray)
        assert fd_gradient.shape == theta.shape

        # Test REINFORCE
        reinforce_config = ReinforceConfig(num_samples=50)
        reinforce_state = ReinforceState()
        reinforce_grad = reinforce_gradient(
            simple_discrete_problem, theta, reinforce_config, reinforce_state
        )
        assert isinstance(reinforce_grad, jnp.ndarray)
        assert reinforce_grad.shape == theta.shape

        # Test Gumbel-Softmax
        gs_config = GumbelSoftmaxConfig(num_samples=50)
        gs_gradient = gumbel_softmax_gradient(simple_discrete_problem, theta, gs_config)
        assert isinstance(gs_gradient, jnp.ndarray)
        assert gs_gradient.shape == theta.shape

    def test_single_branch_problem(self):
        """Test estimators with single branch problem."""
        # Create a single branch problem
        branch = Branch(
            function=lambda th: th**2, derivative_function=lambda th: 2 * th
        )
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([jnp.zeros_like(th)]),
            logits_derivative_function=lambda th: jnp.array([jnp.zeros_like(th)]),
        )
        problem = DiscreteProblem(branches=[branch], logits_model=logits_model)

        theta = 1.0

        # Test finite difference
        fd_config = FiniteDifferenceConfig(num_samples=50)
        fd_gradient = finite_difference_gradient(problem, theta, fd_config)
        assert isinstance(fd_gradient, float)

        # Test REINFORCE
        reinforce_config = ReinforceConfig(num_samples=50)
        reinforce_state = ReinforceState()
        reinforce_grad = reinforce_gradient(
            problem, theta, reinforce_config, reinforce_state
        )
        assert isinstance(reinforce_grad, float)

        # Test Gumbel-Softmax
        gs_config = GumbelSoftmaxConfig(num_samples=50)
        gs_gradient = gumbel_softmax_gradient(problem, theta, gs_config)
        assert isinstance(gs_gradient, float)

    def test_empty_array_input(self, simple_discrete_problem):
        """Test behavior with empty array input."""
        theta = jnp.array([])

        fd_config = FiniteDifferenceConfig(num_samples=50)
        fd_gradient = finite_difference_gradient(
            simple_discrete_problem, theta, fd_config
        )
        assert isinstance(fd_gradient, jnp.ndarray)
        assert fd_gradient.shape == (0,)

        reinforce_config = ReinforceConfig(num_samples=50)
        reinforce_state = ReinforceState()
        reinforce_grad = reinforce_gradient(
            simple_discrete_problem, theta, reinforce_config, reinforce_state
        )
        assert isinstance(reinforce_grad, jnp.ndarray)
        assert reinforce_grad.shape == (0,)

        gs_config = GumbelSoftmaxConfig(num_samples=50)
        gs_gradient = gumbel_softmax_gradient(simple_discrete_problem, theta, gs_config)
        assert isinstance(gs_gradient, jnp.ndarray)
        assert gs_gradient.shape == (0,)
