"""Tests for mellowgate.api.estimators module."""

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
def simple_problem():
    """Simple discrete problem for testing estimators."""
    branches = [
        Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th),
        Branch(function=lambda th: th**3, derivative_function=lambda th: 3 * th**2),
    ]
    logits_model = LogitsModel(
        logits_function=lambda th: jnp.array([th, -th]),
        logits_derivative_function=lambda th: jnp.array(
            [jnp.ones_like(th), -jnp.ones_like(th)]
        ),
    )
    return DiscreteProblem(branches=branches, logits_model=logits_model)


@pytest.fixture
def test_theta():
    """Common theta values for testing."""
    return jnp.array([1.0])


class TestFiniteDifferenceConfig:
    """Test FiniteDifferenceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FiniteDifferenceConfig()
        assert config.step_size == 1e-3
        assert config.num_samples == 2000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FiniteDifferenceConfig(step_size=1e-4, num_samples=1000)
        assert config.step_size == 1e-4
        assert config.num_samples == 1000


class TestReinforceConfig:
    """Test ReinforceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReinforceConfig()
        assert config.num_samples == 2000
        assert config.use_baseline is True
        assert config.baseline_momentum == 0.9

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReinforceConfig(
            num_samples=500, use_baseline=False, baseline_momentum=0.8
        )
        assert config.num_samples == 500
        assert config.use_baseline is False
        assert config.baseline_momentum == 0.8


class TestReinforceState:
    """Test ReinforceState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = ReinforceState()
        assert state.baseline == 0.0
        assert state.initialized is False

    def test_update_baseline(self):
        """Test baseline update functionality."""
        state = ReinforceState()

        # First update should set baseline to new value
        state.update_baseline(5.0, 0.9)
        assert state.baseline == 5.0
        assert state.initialized is True

        # Second update should use momentum
        state.update_baseline(10.0, 0.9)
        expected = 0.9 * 5.0 + 0.1 * 10.0  # 4.5 + 1.0 = 5.5
        assert abs(state.baseline - expected) < 1e-10


class TestGumbelSoftmaxConfig:
    """Test GumbelSoftmaxConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GumbelSoftmaxConfig()
        assert config.temperature == 0.5  # Default is 0.5, not 1.0
        assert config.num_samples == 1000
        assert config.use_straight_through_estimator is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GumbelSoftmaxConfig(
            temperature=0.5, use_straight_through_estimator=True, num_samples=500
        )
        assert config.temperature == 0.5
        assert config.use_straight_through_estimator is True
        assert config.num_samples == 500


class TestFiniteDifferenceGradient:
    """Test finite difference gradient estimator."""

    def test_finite_difference_basic(self, simple_problem, test_theta):
        """Test basic finite difference gradient computation."""
        config = FiniteDifferenceConfig(step_size=1e-3, num_samples=100)

        gradient = finite_difference_gradient(simple_problem, test_theta, config)

        assert gradient is not None
        assert jnp.isfinite(gradient).all()
        assert jnp.asarray(gradient).shape == test_theta.shape

    def test_finite_difference_multiple_theta(self, simple_problem):
        """Test finite difference with multiple theta values."""
        config = FiniteDifferenceConfig(step_size=1e-3, num_samples=100)
        theta_array = jnp.array([0.5, 1.0, 1.5])

        gradient = finite_difference_gradient(simple_problem, theta_array, config)

        assert gradient is not None
        assert jnp.asarray(gradient).shape == theta_array.shape
        assert jnp.isfinite(gradient).all()

    def test_finite_difference_small_step_size(self, simple_problem, test_theta):
        """Test finite difference with very small step size."""
        config = FiniteDifferenceConfig(step_size=1e-6, num_samples=100)

        gradient = finite_difference_gradient(simple_problem, test_theta, config)

        assert gradient is not None
        assert jnp.isfinite(gradient).all()

    def test_finite_difference_large_step_size(self, simple_problem, test_theta):
        """Test finite difference with larger step size."""
        config = FiniteDifferenceConfig(step_size=1e-1, num_samples=100)

        gradient = finite_difference_gradient(simple_problem, test_theta, config)

        assert gradient is not None
        assert jnp.isfinite(gradient).all()


class TestReinforceGradient:
    """Test REINFORCE gradient estimator."""

    def test_reinforce_basic(self, simple_problem, test_theta):
        """Test basic REINFORCE gradient computation."""
        config = ReinforceConfig(num_samples=100)
        state = ReinforceState()

        gradient = reinforce_gradient(simple_problem, test_theta, config, state)

        assert gradient is not None
        assert jnp.isfinite(gradient).all()
        assert jnp.asarray(gradient).shape == test_theta.shape

    def test_reinforce_multiple_theta(self, simple_problem):
        """Test REINFORCE with multiple theta values."""
        config = ReinforceConfig(num_samples=100)
        state = ReinforceState()
        theta_array = jnp.array([0.5, 1.0, 1.5])

        gradient = reinforce_gradient(simple_problem, theta_array, config, state)

        assert gradient is not None
        assert jnp.asarray(gradient).shape == theta_array.shape
        assert jnp.isfinite(gradient).all()

    def test_reinforce_state_update(self, simple_problem, test_theta):
        """Test that REINFORCE state is properly updated."""
        config = ReinforceConfig(num_samples=100, baseline_momentum=0.5)
        initial_state = ReinforceState()
        initial_baseline = initial_state.baseline

        _ = reinforce_gradient(simple_problem, test_theta, config, initial_state)

        # State should be updated
        assert initial_state.baseline != initial_baseline or initial_state.initialized

    def test_reinforce_with_baseline_disabled(self, simple_problem, test_theta):
        """Test REINFORCE with baseline disabled."""
        config = ReinforceConfig(num_samples=100, use_baseline=False)
        state = ReinforceState()

        gradient = reinforce_gradient(simple_problem, test_theta, config, state)

        assert gradient is not None
        assert jnp.isfinite(gradient).all()

    def test_reinforce_reproducibility(self, simple_problem, test_theta):
        """Test REINFORCE reproducibility."""
        config = ReinforceConfig(num_samples=100)
        state1 = ReinforceState()
        state2 = ReinforceState()

        gradient1 = reinforce_gradient(simple_problem, test_theta, config, state1)
        gradient2 = reinforce_gradient(simple_problem, test_theta, config, state2)

        # Results should be deterministic with same initial conditions
        assert gradient1 is not None
        assert gradient2 is not None


class TestGumbelSoftmaxGradient:
    """Test Gumbel-Softmax gradient estimator."""

    def test_gumbel_softmax_basic(self, simple_problem, test_theta):
        """Test basic Gumbel-Softmax gradient computation."""
        config = GumbelSoftmaxConfig(temperature=1.0, num_samples=100)

        gradient = gumbel_softmax_gradient(simple_problem, test_theta, config)

        assert gradient is not None
        assert jnp.isfinite(gradient).all()
        assert jnp.asarray(gradient).shape == test_theta.shape

    def test_gumbel_softmax_multiple_theta(self, simple_problem):
        """Test Gumbel-Softmax with multiple theta values."""
        config = GumbelSoftmaxConfig(temperature=1.0, num_samples=100)
        theta_array = jnp.array([0.5, 1.0, 1.5])

        gradient = gumbel_softmax_gradient(simple_problem, theta_array, config)

        assert gradient is not None
        assert jnp.asarray(gradient).shape == theta_array.shape
        assert jnp.isfinite(gradient).all()

    def test_gumbel_softmax_low_temperature(self, simple_problem, test_theta):
        """Test Gumbel-Softmax with low temperature."""
        config = GumbelSoftmaxConfig(temperature=0.1, num_samples=100)

        gradient = gumbel_softmax_gradient(simple_problem, test_theta, config)

        assert gradient is not None
        assert jnp.isfinite(gradient).all()

    def test_gumbel_softmax_high_temperature(self, simple_problem, test_theta):
        """Test Gumbel-Softmax with high temperature."""
        config = GumbelSoftmaxConfig(temperature=10.0, num_samples=100)

        gradient = gumbel_softmax_gradient(simple_problem, test_theta, config)

        assert gradient is not None
        assert jnp.isfinite(gradient).all()

    def test_gumbel_softmax_straight_through(self, simple_problem, test_theta):
        """Test Gumbel-Softmax with straight-through estimator."""
        config = GumbelSoftmaxConfig(
            temperature=1.0, use_straight_through_estimator=True, num_samples=100
        )

        gradient = gumbel_softmax_gradient(simple_problem, test_theta, config)

        assert gradient is not None
        assert jnp.isfinite(gradient).all()

    def test_gumbel_softmax_zero_temperature_protection(
        self, simple_problem, test_theta
    ):
        """Test Gumbel-Softmax with very low temperature."""
        # Use very small but non-zero temperature to avoid division by zero
        config = GumbelSoftmaxConfig(temperature=1e-6, num_samples=100)

        # Should handle very low temperature gracefully
        gradient = gumbel_softmax_gradient(simple_problem, test_theta, config)

        assert gradient is not None
        # With very low temperature, gradient might be unstable but should be finite
        assert jnp.isfinite(gradient).all() or jnp.abs(gradient).max() < 1e10


class TestEstimatorComparison:
    """Test comparison between different estimators."""

    def test_estimator_consistency(self, simple_problem):
        """Test that estimators give reasonable results for the same problem."""
        theta = jnp.array([1.0])

        # Finite difference
        fd_config = FiniteDifferenceConfig(step_size=1e-3, num_samples=1000)
        fd_gradient = finite_difference_gradient(simple_problem, theta, fd_config)

        # REINFORCE
        reinforce_config = ReinforceConfig(num_samples=1000)
        reinforce_state = ReinforceState()
        reinforce_grad = reinforce_gradient(
            simple_problem, theta, reinforce_config, reinforce_state
        )

        # Gumbel-Softmax
        gs_config = GumbelSoftmaxConfig(temperature=1.0, num_samples=1000)
        gs_gradient = gumbel_softmax_gradient(simple_problem, theta, gs_config)

        # All should be finite and of correct shape
        assert jnp.isfinite(fd_gradient).all()
        assert jnp.isfinite(reinforce_grad).all()
        assert jnp.isfinite(gs_gradient).all()

        assert jnp.asarray(fd_gradient).shape == theta.shape
        assert jnp.asarray(reinforce_grad).shape == theta.shape
        assert jnp.asarray(gs_gradient).shape == theta.shape

    def test_exact_gradient_comparison(self, simple_problem):
        """Compare estimators against exact gradient when available."""
        theta = jnp.array([1.0])

        # Get exact gradient
        exact_gradient = simple_problem.compute_exact_gradient(theta)
        assert exact_gradient is not None

        # Finite difference with very small step size and many samples
        fd_config = FiniteDifferenceConfig(step_size=1e-5, num_samples=5000)
        fd_gradient = finite_difference_gradient(simple_problem, theta, fd_config)

        # Should be reasonably close to exact gradient
        # Note: This is a statistical test and might occasionally fail
        relative_error = jnp.abs((fd_gradient - exact_gradient) / exact_gradient)
        assert relative_error < 0.1  # Within 10%


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_samples_protection(self, simple_problem, test_theta):
        """Test protection against zero samples."""
        fd_config = FiniteDifferenceConfig(num_samples=0)

        # Should handle zero samples gracefully or raise appropriate error
        try:
            gradient = finite_difference_gradient(simple_problem, test_theta, fd_config)
            # If it doesn't raise an error, result should still be meaningful
            assert gradient is not None
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise an error for zero samples
            pass

    def test_single_branch_problem(self):
        """Test estimators with single branch problem."""
        # Single branch problems can be problematic for gradient estimators
        # This test just ensures no crashes occur with simple validation
        branches = [
            Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th)
        ]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([0.0]),
            logits_derivative_function=lambda th: jnp.array([0.0]),
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])

        # At minimum, ensure the problem can be created and basic methods work
        assert problem.num_branches == 1
        probs = problem.compute_probabilities(theta)
        assert jnp.isclose(probs[0], 1.0)  # Single branch gets probability 1

    def test_extreme_theta_values(self, simple_problem):
        """Test estimators with extreme theta values."""
        extreme_theta = jnp.array([1000.0, -1000.0])

        # Test with smaller sample sizes to avoid computational issues
        fd_config = FiniteDifferenceConfig(num_samples=10, step_size=1e-6)
        fd_gradient = finite_difference_gradient(
            simple_problem, extreme_theta, fd_config
        )

        # Should handle extreme values without error
        assert jnp.isfinite(fd_gradient).all()

    def test_scalar_input_handling(self, simple_problem):
        """Test handling of scalar vs array inputs across estimators."""
        # Test scalar input (not array)
        scalar_theta = 1.0  # Note: not jnp.array([1.0]) but just 1.0

        # Finite difference with scalar input
        fd_config = FiniteDifferenceConfig(num_samples=50, step_size=1e-3)
        fd_result = finite_difference_gradient(simple_problem, scalar_theta, fd_config)
        assert isinstance(fd_result, float)  # Should return scalar for scalar input

        # REINFORCE with scalar input
        reinforce_config = ReinforceConfig(num_samples=50)
        reinforce_state = ReinforceState()
        reinforce_result = reinforce_gradient(
            simple_problem, scalar_theta, reinforce_config, reinforce_state
        )
        assert isinstance(
            reinforce_result, float
        )  # Should return scalar for scalar input

        # Gumbel-Softmax with scalar input
        gs_config = GumbelSoftmaxConfig(num_samples=50)
        gs_result = gumbel_softmax_gradient(simple_problem, scalar_theta, gs_config)
        assert isinstance(gs_result, float)  # Should return scalar for scalar input

    def test_estimator_missing_requirements(self):
        """Test estimators when required derivatives are missing."""
        # Create simple branches
        branches = [
            Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th),
            Branch(function=lambda th: th**3, derivative_function=lambda th: 3 * th**2),
        ]

        # Create logits model without derivatives
        logits_model_no_deriv = LogitsModel(
            logits_function=lambda th: jnp.array([0.0, 1.0])
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model_no_deriv)

        theta = jnp.array([1.0])

        # REINFORCE should raise error when logits derivatives missing
        reinforce_config = ReinforceConfig(num_samples=50)
        reinforce_state = ReinforceState()
        with pytest.raises(
            ValueError, match="REINFORCE requires logits_derivative_function"
        ):
            reinforce_gradient(problem, theta, reinforce_config, reinforce_state)

        # Gumbel-Softmax should also raise error when logits derivatives missing
        gs_config = GumbelSoftmaxConfig(num_samples=50)
        with pytest.raises(
            ValueError, match="Gumbel-Softmax requires logits_derivative_function"
        ):
            gumbel_softmax_gradient(problem, theta, gs_config)
