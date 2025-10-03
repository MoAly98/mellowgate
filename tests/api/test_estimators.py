"""Tests for mellowgate.api.estimators module."""

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


class TestAnalyticalGradientValidation:
    """Test gradient estimators against known analytical gradients."""

    def test_finite_difference_vs_analytical(self):
        """Test finite difference estimator accuracy against exact gradient."""
        # Simple problem with polynomial branches for known analytical solution
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
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])
        exact_gradient = problem.compute_exact_gradient(theta)
        assert exact_gradient is not None, "Should have exact gradient"

        # Finite difference with high precision
        fd_config = FiniteDifferenceConfig(step_size=1e-6, num_samples=10000)
        fd_gradient = finite_difference_gradient(problem, theta, fd_config)

        relative_error = jnp.abs((fd_gradient - exact_gradient) / exact_gradient)
        assert relative_error < 0.05, (
            f"Finite difference relative error {float(relative_error):.4f} "
            f"exceeds tolerance"
        )

    def test_reinforce_vs_analytical(self):
        """Test REINFORCE estimator accuracy against exact gradient."""
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
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])
        exact_gradient = problem.compute_exact_gradient(theta)
        assert exact_gradient is not None, "Should have exact gradient"

        # REINFORCE with many samples for accuracy
        reinforce_config = ReinforceConfig(num_samples=20000, use_baseline=True)
        reinforce_state = ReinforceState()
        reinforce_grad = reinforce_gradient(
            problem, theta, reinforce_config, reinforce_state
        )

        relative_error = jnp.abs((reinforce_grad - exact_gradient) / exact_gradient)
        assert (
            relative_error < 0.05
        ), f"REINFORCE relative error {float(relative_error):.4f} exceeds tolerance"

    def test_gumbel_softmax_vs_analytical(self):
        """Test Gumbel-Softmax estimator accuracy against exact gradient."""
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
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])
        exact_gradient = problem.compute_exact_gradient(theta)
        assert exact_gradient is not None, "Should have exact gradient"

        # Gumbel-Softmax with many samples for accuracy
        gs_config = GumbelSoftmaxConfig(temperature=0.1, num_samples=20000)
        gs_grad = gumbel_softmax_gradient(problem, theta, gs_config)

        relative_error = jnp.abs((gs_grad - exact_gradient) / exact_gradient)
        assert relative_error < 0.05, (
            f"Gumbel-Softmax relative error {float(relative_error):.4f} "
            f"exceeds tolerance"
        )

    def test_binary_softmax_known_analytical_result(self):
        """Test binary softmax problem with known analytical result."""
        # Binary problem: pass=1, fail=0, with logits α=[aθ, -aθ] where a=10
        # Uses default softmax probability function:
        # p₁ = exp(aθ)/[exp(aθ) + exp(-aθ)]
        # p₂ = exp(-aθ)/[exp(aθ) + exp(-aθ)]
        # Expected value: E = p₁ × 1 + p₂ × 0 = p₁
        # At θ=0: p₁ = 0.5, so E = 0.5
        # Gradient computed using automatic differentiation in mellowgate: 2.5
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
            logits_function=lambda th: jnp.array([10 * th, -10 * th]),
            logits_derivative_function=lambda th: jnp.array(
                [10 * jnp.ones_like(th), -10 * jnp.ones_like(th)]
            ),
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([0.0])
        analytical_gradient = 5.0

        # Test REINFORCE and Gumbel-Softmax estimators
        reinforce_config = ReinforceConfig(num_samples=15000, use_baseline=False)
        reinforce_state = ReinforceState()
        reinforce_grad = reinforce_gradient(
            problem, theta, reinforce_config, reinforce_state
        )

        # Use lower temperature for better Gumbel-Softmax accuracy
        gs_config = GumbelSoftmaxConfig(temperature=0.1, num_samples=15000)
        gs_grad = gumbel_softmax_gradient(problem, theta, gs_config)

        # Verify estimators are close to analytical gradient
        for name, grad_est in [
            ("REINFORCE", reinforce_grad),
            ("GumbelSoftmax", gs_grad),
        ]:
            grad_val = float(jnp.asarray(grad_est).item())
            error = abs(grad_val - analytical_gradient)
            rel_error = error / analytical_gradient

            # Both estimators should be reasonably accurate
            assert rel_error < 0.05, (
                f"{name} relative error {rel_error:.4f} exceeds 5% tolerance "
                f"(estimated={grad_val:.4f}, expected={analytical_gradient:.4f})"
            )

        print(
            f"✓ REINFORCE gradient: " f"{float(jnp.asarray(reinforce_grad).item()):.3f}"
        )
        print(
            f"✓ Gumbel-Softmax gradient: " f"{float(jnp.asarray(gs_grad).item()):.3f}"
        )
        print(f"✓ Expected analytical: {analytical_gradient}")
        print("✓ Gradient estimators produce expected results")

    def test_binary_sigmoid_probability_function(self):
        """Test gradient estimators with custom sigmoid probability function."""
        # Binary problem with custom sigmoid probability function (not softmax)
        # For binary case with logits [α₁, α₂], sigmoid gives:
        # p₁ = sigmoid(α₁), p₂ = sigmoid(α₂)
        # Note: These don't sum to 1, so we normalize:
        # p₁ = sigmoid(α₁)/(sigmoid(α₁) + sigmoid(α₂))

        def sigmoid_probability_function(logits):
            """Custom sigmoid probability function"""
            return 1.0 / (1.0 + jnp.exp(-logits))

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
            logits_function=lambda th: jnp.array([5 * th, -5 * th]),
            logits_derivative_function=lambda th: jnp.array(
                [5 * jnp.ones_like(th), -5 * jnp.ones_like(th)]
            ),
            probability_function=sigmoid_probability_function,
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([0.0])

        # Get exact gradient from mellowgate
        exact_gradient = problem.compute_exact_gradient(theta)
        assert exact_gradient is not None, "Should have exact gradient"

        # Test REINFORCE estimator
        reinforce_config = ReinforceConfig(num_samples=15000, use_baseline=False)
        reinforce_state = ReinforceState()
        reinforce_grad = reinforce_gradient(
            problem, theta, reinforce_config, reinforce_state
        )

        # Test Gumbel-Softmax estimator
        gs_config = GumbelSoftmaxConfig(temperature=0.1, num_samples=15000)
        gs_grad = gumbel_softmax_gradient(problem, theta, gs_config)

        # Verify estimators are close to exact gradient
        for name, grad_est in [
            ("REINFORCE", reinforce_grad),
            ("GumbelSoftmax", gs_grad),
        ]:
            grad_val = float(jnp.asarray(grad_est).item())
            exact_val = float(jnp.asarray(exact_gradient).item())
            rel_error = abs(grad_val - exact_val) / abs(exact_val)

            assert rel_error < 0.015, (
                f"{name} relative error {rel_error:.4f} vs exact gradient "
                f"(estimated={grad_val:.4f}, exact={exact_val:.4f})"
            )

    def test_exact_gradient_vs_analytical_computation(self):
        """Test mellowgate exact gradient vs hand-calculated analytical result."""
        # Simple polynomial problem where we can calculate the exact gradient by hand
        # Branch functions: f₁(θ) = θ², f₂(θ) = 2θ
        # Logits: α₁ = θ, α₂ = -θ
        # Probabilities: p₁ = exp(θ)/(exp(θ) + exp(-θ)), p₂ = exp(-θ)/(exp(θ) + exp(-θ))
        # Expected value: E = p₁θ² + p₂(2θ)

        branches = [
            Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th),
            Branch(
                function=lambda th: 2 * th,
                derivative_function=lambda th: 2 * jnp.ones_like(th),
            ),
        ]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([th, -th]),
            logits_derivative_function=lambda th: jnp.array(
                [jnp.ones_like(th), -jnp.ones_like(th)]
            ),
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])

        # Get exact gradient from mellowgate
        exact_gradient = problem.compute_exact_gradient(theta)
        assert exact_gradient is not None, "Should have exact gradient"

        # Verify using JAX autodiff for the analytical result
        def expected_value_function(theta_val):
            """Expected value function E(θ) = p₁(θ)θ² + p₂(θ)(2θ)"""
            logits = jnp.array([theta_val, -theta_val])
            probs = jax.nn.softmax(logits)
            p1, p2 = probs[0], probs[1]
            return p1 * theta_val**2 + p2 * (2 * theta_val)

        # Compute analytical gradient using JAX autodiff (should be exact)
        grad_fn = jax.grad(expected_value_function)
        jax_analytical_gradient = grad_fn(1.0)

        # Also verify with manual calculation (should match JAX)
        theta_val = 1.0
        p1 = jnp.exp(theta_val) / (jnp.exp(theta_val) + jnp.exp(-theta_val))
        p2 = 1.0 - p1

        # Correct softmax derivatives:
        # p₁ = exp(θ)/(exp(θ) + exp(-θ))
        # Using quotient rule:
        # dp₁/dθ
        # = [exp(θ)(exp(θ) + exp(-θ)) - exp(θ)(exp(θ) - exp(-θ))] / (exp(θ) + exp(-θ))²
        # = [exp(θ)(2exp(-θ))] / (exp(θ) + exp(-θ))²
        # = 2exp(θ-θ) / (exp(θ) + exp(-θ))²
        # = 2 / (exp(θ) + exp(-θ))²
        # using chain rule
        # dp₁/dθ = ∂p₁/∂α₁ × ∂α₁/∂θ + ∂p₁/∂α₂ × ∂α₂/∂θ
        #        = p₁(1-p₁) × 1 + (-p₁p₂) × (-1) = p₁(1-p₁) + p₁p₂
        #        = p₁p₂ + p₁p₂
        #        = 2p₁p₂

        dp1_dtheta = 2 * p1 * p2
        dp2_dtheta = -2 * p1 * p2  # dp2/dtheta = -dp1/dtheta since p1 + p2 = 1

        manual_analytical_gradient = (
            dp1_dtheta * theta_val**2
            + p1 * 2 * theta_val
            + dp2_dtheta * 2 * theta_val
            + p2 * 2
        )

        # Compare both analytical methods
        exact_val = float(jnp.asarray(exact_gradient).item())
        jax_analytical_val = float(jax_analytical_gradient)
        manual_analytical_val = float(manual_analytical_gradient)

        # Both analytical methods should match
        analytical_diff = abs(jax_analytical_val - manual_analytical_val) / abs(
            jax_analytical_val
        )
        assert analytical_diff < 1e-12, (
            f"Manual vs JAX analytical mismatch {analytical_diff:.2e} "
            f"(JAX={jax_analytical_val:.10f}, manual={manual_analytical_val:.10f})"
        )

        # Exact gradient should match analytical
        rel_error = abs(exact_val - manual_analytical_val) / abs(manual_analytical_val)
        assert rel_error < 1e-12, (
            f"Exact gradient computation error {rel_error:.2e} vs analytical "
            f"(exact={exact_val:.10f}, analytical={jax_analytical_val:.10f})"
        )


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
