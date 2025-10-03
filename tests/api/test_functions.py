"""Tests for mellowgate.api.functions module."""

import jax
import jax.numpy as jnp
import pytest

from mellowgate.api.functions import Bound, Branch, DiscreteProblem, LogitsModel


@pytest.fixture
def test_theta():
    """Common theta values for testing with minimum 5 values."""
    return jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])


@pytest.fixture
def simple_branches():
    """Simple branches for easier testing."""
    return [
        Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th),
        Branch(function=lambda th: th**3, derivative_function=lambda th: 3 * th**2),
    ]


@pytest.fixture
def example_logits_model():
    """Example logits model for testing."""
    return LogitsModel(
        logits_function=lambda th: jnp.array([th, -th]),
        logits_derivative_function=lambda th: jnp.array(
            [jnp.ones_like(th), -jnp.ones_like(th)]
        ),
    )


class TestBound:
    """Test the Bound dataclass."""

    def test_bound_creation_default(self):
        """Test creating a Bound with default parameters."""
        bound = Bound(value=1.0)
        assert bound.value == 1.0
        assert bound.inclusive is True

    def test_bound_creation_exclusive(self):
        """Test creating a Bound with explicit parameters."""
        bound = Bound(value=2.5, inclusive=False)
        assert bound.value == 2.5
        assert bound.inclusive is False


class TestBranch:
    """Test the Branch dataclass."""

    def test_branch_creation_minimal(self):
        """Test creating a Branch with minimal parameters."""

        def func(th):
            return th**2

        branch = Branch(function=func)
        assert branch.function is func
        assert branch.derivative_function is None
        assert branch.threshold == (None, None)

    def test_branch_with_derivative(self):
        """Test creating a Branch with derivative function."""

        def func(th):
            return th**2

        def deriv(th):
            return 2 * th

        branch = Branch(function=func, derivative_function=deriv)
        assert branch.function is func
        assert branch.derivative_function is deriv

    def test_branch_with_threshold(self):
        """Test creating a Branch with threshold bounds."""

        def func(th):
            return th**2

        threshold = (Bound(0, inclusive=True), Bound(5, inclusive=False))
        branch = Branch(function=func, threshold=threshold)
        assert branch.threshold == threshold


class TestLogitsModel:
    """Test the LogitsModel dataclass."""

    def test_logits_model_creation(self):
        """Test creating a LogitsModel."""

        def logits_func(th):
            return jnp.array([th, -th])

        model = LogitsModel(logits_function=logits_func)
        assert model.logits_function is logits_func
        assert model.logits_derivative_function is None

    def test_logits_model_with_derivatives(self):
        """Test creating a LogitsModel with derivatives."""

        def logits_func(th):
            return jnp.array([th, -th])

        def deriv_func(th):
            return jnp.array([jnp.ones_like(th), -jnp.ones_like(th)])

        model = LogitsModel(
            logits_function=logits_func, logits_derivative_function=deriv_func
        )
        assert model.logits_derivative_function is deriv_func

    def test_logits_model_with_custom_probability_function(self):
        """Test LogitsModel with custom probability function."""

        def logits_func(th):
            return jnp.array([th, -th])

        def prob_func(logits):
            return jnp.array([0.3, 0.7])

        model = LogitsModel(logits_function=logits_func, probability_function=prob_func)
        assert model.probability_function is prob_func


class TestDiscreteProblem:
    """Test the DiscreteProblem class."""

    def test_num_branches_property(self, simple_branches, example_logits_model):
        """Test the num_branches property."""
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=example_logits_model
        )
        assert problem.num_branches == 2

    def test_compute_probabilities(
        self, simple_branches, example_logits_model, test_theta
    ):
        """Test probability computation."""
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=example_logits_model
        )
        probabilities = problem.compute_probabilities(test_theta)

        # Check shape and properties
        assert probabilities.shape == (2, 5)
        assert jnp.allclose(probabilities.sum(axis=0), 1.0)
        assert jnp.all(probabilities >= 0)

    def test_compute_function_values(
        self, simple_branches, example_logits_model, test_theta
    ):
        """Test function value computation."""
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=example_logits_model
        )
        values = problem.compute_function_values(test_theta)

        # Check shape
        assert values.shape == (2, 5)
        # Check actual values for th^2 and th^3
        expected_0 = test_theta**2
        expected_1 = test_theta**3
        assert jnp.allclose(values[0], expected_0)
        assert jnp.allclose(values[1], expected_1)

    def test_compute_expected_value(
        self, simple_branches, example_logits_model, test_theta
    ):
        """Test expected value computation."""
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=example_logits_model
        )
        expected_val = problem.compute_expected_value(test_theta)

        # Check shape
        assert expected_val.shape == (5,)
        # Values should be finite
        assert jnp.all(jnp.isfinite(expected_val))

    def test_compute_exact_gradient(
        self, simple_branches, example_logits_model, test_theta
    ):
        """Test exact gradient computation."""
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=example_logits_model
        )
        gradient = problem.compute_exact_gradient(test_theta)

        # Should return gradient array
        assert gradient is not None
        assert jnp.asarray(gradient).shape == (5,)
        assert jnp.all(jnp.isfinite(gradient))

    def test_compute_exact_gradient_missing_logits_derivatives(
        self, simple_branches, example_logits_model
    ):
        """Test exact gradient computation when logits derivatives are missing."""
        # Create model without logits derivatives
        logits_model_no_deriv = LogitsModel(
            logits_function=example_logits_model.logits_function
        )
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=logits_model_no_deriv
        )

        theta = jnp.array([1.0])

        # Should return None when no logits derivatives available
        gradient = problem.compute_exact_gradient(theta)
        assert gradient is None

    def test_input_validation_errors(self, simple_branches, example_logits_model):
        """Test input validation in core methods."""
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=example_logits_model
        )

        # Test non-array input to compute_function_values_deterministic
        with pytest.raises(ValueError, match="Theta must be a numpy array"):
            problem.compute_function_values_deterministic([1.0, 2.0])  # type: ignore

        # Test empty theta arrays
        empty_theta = jnp.array([])
        result = problem.compute_function_values_deterministic(empty_theta)
        assert result.shape == (len(simple_branches), 0)

    def test_threshold_conditions_coverage(self, simple_branches, example_logits_model):
        """Test threshold condition checking with various boundary cases."""
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=example_logits_model
        )

        # Test non-array input to generate_threshold_conditions
        with pytest.raises(ValueError, match="Theta must be a numpy array"):
            problem.generate_threshold_conditions([1.0, 2.0], None)  # type: ignore

        # Test None threshold
        theta = jnp.array([1.0, 2.0])
        result = problem.generate_threshold_conditions(theta, None)
        assert jnp.all(result)

        # Test (None, None) threshold
        result = problem.generate_threshold_conditions(theta, (None, None))
        assert jnp.all(result)

        # Test lower bound conditions
        lower_inclusive = Bound(1.5, inclusive=True)
        result = problem.generate_threshold_conditions(theta, (lower_inclusive, None))
        assert not result[0] and result[1]  # 1.0 < 1.5, 2.0 >= 1.5

        lower_exclusive = Bound(1.5, inclusive=False)
        result = problem.generate_threshold_conditions(theta, (lower_exclusive, None))
        assert not result[0] and result[1]  # 1.0 <= 1.5, 2.0 > 1.5

        # Test upper bound conditions
        upper_inclusive = Bound(1.5, inclusive=True)
        result = problem.generate_threshold_conditions(theta, (None, upper_inclusive))
        assert result[0] and not result[1]  # 1.0 < 1.5, 2.0 >= 1.5

        upper_exclusive = Bound(1.5, inclusive=False)
        result = problem.generate_threshold_conditions(theta, (None, upper_exclusive))
        assert result[0] and not result[1]  # 1.0 <= 1.5, 2.0 > 1.5

    def test_exact_gradient_missing_derivatives(self, example_logits_model):
        """Test exact gradient when function derivatives are missing."""
        # Create branches without derivative functions
        branches_no_deriv = [
            Branch(function=lambda th: th**2),  # No derivative
            Branch(function=lambda th: th**3),  # No derivative
        ]
        problem = DiscreteProblem(
            branches=branches_no_deriv, logits_model=example_logits_model
        )

        theta = jnp.array([1.0])

        # Should return None when function derivatives are not available
        gradient = problem.compute_exact_gradient(theta)
        assert gradient is None

    def test_threshold_aware_function_evaluation(self):
        """Test threshold-aware function value computation."""
        # Create branches with thresholds
        branches_with_thresholds = [
            Branch(
                function=lambda th: th**2,
                threshold=(Bound(0.0, inclusive=True), Bound(2.0, inclusive=False)),
            ),
            Branch(
                function=lambda th: th**3, threshold=(Bound(1.0, inclusive=False), None)
            ),
        ]
        logits_model = LogitsModel(logits_function=lambda th: jnp.array([0.0, 1.0]))
        problem = DiscreteProblem(
            branches=branches_with_thresholds, logits_model=logits_model
        )

        theta = jnp.array([-1.0, 0.5, 1.5, 3.0])

        # Test deterministic computation with thresholds
        values = problem.compute_function_values_deterministic(theta)

        # First branch: active for theta in [0.0, 2.0)
        # Should have NaN for -1.0 and 3.0, valid values for 0.5 and 1.5
        assert jnp.isnan(values[0, 0])  # -1.0 not in range
        assert jnp.isfinite(values[0, 1])  # 0.5 in range
        assert jnp.isfinite(values[0, 2])  # 1.5 in range
        assert jnp.isnan(values[0, 3])  # 3.0 not in range

        # Second branch: active for theta > 1.0
        # Should have NaN for -1.0, 0.5, valid for 1.5, 3.0
        assert jnp.isnan(values[1, 0])  # -1.0 not in range
        assert jnp.isnan(values[1, 1])  # 0.5 not in range
        assert jnp.isfinite(values[1, 2])  # 1.5 in range
        assert jnp.isfinite(values[1, 3])  # 3.0 in range

    def test_sample_branch(self, simple_branches, example_logits_model):
        """Test branch sampling."""
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=example_logits_model
        )
        theta = jnp.array([1.0])
        key = jax.random.PRNGKey(42)

        samples = problem.sample_branch(theta, num_samples=10, key=key)
        assert samples.shape == (1, 10)  # (num_theta, num_samples)
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < len(simple_branches))

    def test_compute_stochastic_values(self, simple_branches, example_logits_model):
        """Test stochastic value computation."""
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=example_logits_model
        )
        theta = jnp.array([1.0])
        key = jax.random.PRNGKey(123)

        stochastic_vals = problem.compute_stochastic_values(
            theta, key=key, num_samples=5
        )
        assert stochastic_vals.shape == (1, 5)  # (num_theta, num_samples)
        assert jnp.all(jnp.isfinite(stochastic_vals))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_branch_problem(self):
        """Test problem with single branch."""
        branches = [
            Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th)
        ]
        logits_model = LogitsModel(logits_function=lambda th: jnp.array([0.0]))
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])
        probs = problem.compute_probabilities(theta)
        assert probs.shape == (1,)  # Single branch for single theta
        assert jnp.isclose(probs[0], 1.0)

    def test_zero_theta(self, simple_branches, example_logits_model):
        """Test with zero theta values."""
        problem = DiscreteProblem(
            branches=simple_branches, logits_model=example_logits_model
        )
        theta = jnp.array([0.0])

        probs = problem.compute_probabilities(theta)
        values = problem.compute_function_values(theta)
        expected = problem.compute_expected_value(theta)

        assert jnp.all(jnp.isfinite(probs))
        assert jnp.all(jnp.isfinite(values))
        assert jnp.all(jnp.isfinite(expected))

    def test_probability_validation_error_single_theta(self):
        """Test probability validation error for single theta."""

        def bad_prob_func(logits):
            return jnp.array([0.3, 0.4])  # Sum = 0.7, not 1.0

        logits_model = LogitsModel(logits_function=lambda th: jnp.array([1.0, 2.0]))
        logits_model.probability_function = bad_prob_func

        branches = [Branch(function=lambda th: th), Branch(function=lambda th: 2 * th)]
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])
        with pytest.raises(ValueError, match="Probabilities do not sum to 1"):
            problem.compute_probabilities(theta)

    def test_probability_validation_error_multiple_theta(self):
        """Test probability validation error for multiple theta."""

        def bad_logits_func(th):
            return jnp.array([10.0 * jnp.ones_like(th), -10.0 * jnp.ones_like(th)])

        def bad_prob_func(logits):
            num_theta = logits.shape[1] if logits.ndim > 1 else 1
            return jnp.array([[0.3] * num_theta, [0.4] * num_theta])  # Sum = 0.7

        logits_model = LogitsModel(logits_function=bad_logits_func)
        logits_model.probability_function = bad_prob_func

        branches = [Branch(function=lambda th: th), Branch(function=lambda th: 2 * th)]
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Probabilities do not sum to 1"):
            problem.compute_probabilities(theta)

    def test_scalar_function_result_handling(self):
        """Test scalar result handling in compute_function_values."""

        def scalar_function(th):
            return jnp.array(42.0)  # Always returns scalar as JAX array

        branches = [Branch(function=scalar_function)]
        logits_model = LogitsModel(logits_function=lambda th: jnp.array([0.0]))
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])
        func_vals = problem.compute_function_values(theta)
        assert func_vals.shape == (1, 1)
        assert func_vals[0, 0] == 42.0

    def test_scalar_gradient_return(self):
        """Test scalar return from gradient computation."""
        branches = [
            Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th)
        ]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([0.0]),
            logits_derivative_function=lambda th: jnp.array([0.0]),
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = 1.5  # Pure scalar, not array
        gradient = problem.compute_exact_gradient(theta)
        assert isinstance(gradient, float)
        assert gradient == 3.0  # 2 * 1.5

    def test_custom_sampling_function(self):
        """Test custom sampling function path."""

        def custom_sampler(probs, key):
            return jnp.asarray(0)  # Always pick first branch

        branches = [Branch(function=lambda th: th), Branch(function=lambda th: 2 * th)]
        logits_model = LogitsModel(logits_function=lambda th: jnp.array([0.0, 1.0]))
        problem = DiscreteProblem(
            branches=branches,
            logits_model=logits_model,
            sampling_function=custom_sampler,
        )

        theta = jnp.array([1.0])
        key = jax.random.PRNGKey(42)
        samples = problem.sample_branch(theta, num_samples=5, key=key)
        assert jnp.all(samples == 0)

    def test_jax_categorical_single_theta(self):
        """Test JAX categorical sampling with 1D probabilities."""
        branches = [Branch(function=lambda th: th), Branch(function=lambda th: th**2)]
        logits_model = LogitsModel(logits_function=lambda th: jnp.array([1.0, 2.0]))
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([0.5])
        key = jax.random.PRNGKey(42)

        # Test with larger sample size to check statistical properties
        samples = problem.sample_branch(theta, num_samples=10000, key=key)
        assert samples.shape == (10000,)
        assert jnp.all((samples >= 0) & (samples < 2))

        # Test relative frequencies match expected probabilities
        expected_probs = problem.compute_probabilities(theta)
        observed_freq_0 = jnp.mean(samples == 0)
        observed_freq_1 = jnp.mean(samples == 1)

        # With 10k samples, should be within ~1% of expected (99% confidence)
        assert jnp.abs(observed_freq_0 - expected_probs[0]) < 0.01
        assert jnp.abs(observed_freq_1 - expected_probs[1]) < 0.01

    def test_jax_categorical_no_key_provided(self):
        """Test JAX categorical sampling with no key provided (uses default)."""
        branches = [Branch(function=lambda th: th), Branch(function=lambda th: th**2)]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([0.0, jnp.log(3.0)])
        )  # 1:3 ratio
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([0.5])
        samples = problem.sample_branch(theta, num_samples=8000)  # No key provided
        assert samples.shape == (8000,)
        assert jnp.all((samples >= 0) & (samples < 2))

        # Test that sampling respects expected probabilities (1:3 ratio → 0.25:0.75)
        expected_probs = problem.compute_probabilities(theta)
        observed_freq_0 = jnp.mean(samples == 0)
        observed_freq_1 = jnp.mean(samples == 1)

        # Should be approximately 0.25 and 0.75 respectively
        assert jnp.abs(observed_freq_0 - expected_probs[0]) < 0.02
        assert jnp.abs(observed_freq_1 - expected_probs[1]) < 0.02

    def test_categorical_sampling_statistical_properties(self):
        """Test that categorical sampling produces statistically correct frequencies."""
        branches = [
            Branch(function=lambda th: th),
            Branch(function=lambda th: 2 * th),
            Branch(function=lambda th: 3 * th),
        ]
        # Create unequal probabilities: logits [0, ln(2), ln(4)] → probs [1/7, 2/7, 4/7]
        # For single theta, return 1D array
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([0.0, jnp.log(2.0), jnp.log(4.0)])
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])
        key = jax.random.PRNGKey(12345)

        # Large sample for reliable statistics
        samples = problem.sample_branch(theta, num_samples=14000, key=key)

        # Check sample validity - for single theta with 1D logits, samples should be 1D
        assert samples.shape == (14000,)
        assert jnp.all((samples >= 0) & (samples < 3))

        # Check statistical frequencies
        expected_probs = problem.compute_probabilities(theta)
        for i in range(3):
            observed_freq = jnp.mean(samples == i)
            expected_freq = expected_probs[i]
            # Allow 1.5% tolerance for statistical variation
            assert (
                jnp.abs(observed_freq - expected_freq) < 0.015
            ), f"Branch {i}: observed {observed_freq:.3f}, expected {expected_freq:.3f}"

    def test_1d_logits_handling(self):
        """Test 1D logits handling in probability computation."""
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([1.0, 2.0])  # Fixed 1D array
        )

        branches = [Branch(function=lambda th: th), Branch(function=lambda th: 2 * th)]
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([0.5])
        probabilities = problem.compute_probabilities(theta)
        assert probabilities.shape[0] == 2

    def test_unreachable_1d_function_values_path(self):
        """Test the theoretically unreachable 1D function values path using
        monkey patching."""

        def simple_func(th):
            return jnp.array(5.0)

        branches = [Branch(function=simple_func)]
        logits_model = LogitsModel(logits_function=lambda th: jnp.array([0.0]))
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([2.0])
        key = jax.random.PRNGKey(999)

        # Monkey patch to force 1D function values
        def mock_1d_function_values(theta):
            return jnp.array([5.0])  # Shape (1,) instead of (1, 1)

        original_method = problem.compute_function_values
        problem.compute_function_values = mock_1d_function_values

        try:
            stochastic_vals = problem.compute_stochastic_values(
                theta, key=key, num_samples=3
            )
            assert stochastic_vals.shape == (3,)
        finally:
            problem.compute_function_values = original_method


class TestKnownAnalyticalResults:
    """Test against known analytical results for validation."""

    def test_simple_quadratic_problem(self):
        """Test simple quadratic problem with known analytical solution."""
        branches = [
            Branch(function=lambda th: th**2, derivative_function=lambda th: 2 * th),
            Branch(
                function=lambda th: 2 * th**2, derivative_function=lambda th: 4 * th
            ),
        ]

        # Equal probabilities: logits = [0, ln(1)] = [0, 0]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array(
                [jnp.zeros_like(th), jnp.zeros_like(th)]
            ),
            logits_derivative_function=lambda th: jnp.array(
                [jnp.zeros_like(th), jnp.zeros_like(th)]
            ),
        )

        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([2.0])

        # Expected probabilities: [0.5, 0.5]
        probs = problem.compute_probabilities(theta)
        expected_probs = jnp.array([[0.5], [0.5]])
        assert jnp.allclose(probs, expected_probs, rtol=1e-6)

        # Expected function values: [4.0, 8.0]
        func_vals = problem.compute_function_values(theta)
        expected_func_vals = jnp.array([[4.0], [8.0]])
        assert jnp.allclose(func_vals, expected_func_vals)

        # Expected value: 0.5*4 + 0.5*8 = 6.0
        expected_val = problem.compute_expected_value(theta)
        assert jnp.allclose(expected_val, jnp.array([6.0]), rtol=1e-6)

        # Expected gradient: d/dtheta[0.5*theta^2 + 0.5*2*theta^2]
        # = d/dtheta[1.5*theta^2] = 3*theta
        gradient = problem.compute_exact_gradient(theta)
        assert gradient is not None
        expected_gradient = 3.0 * 2.0  # 6.0
        assert jnp.allclose(gradient, jnp.array([expected_gradient]), rtol=1e-6)

    def test_constant_functions_analytical(self):
        """Test with constant functions for easy analytical verification."""
        branches = [
            Branch(
                function=lambda th: 3.0 * jnp.ones_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
            Branch(
                function=lambda th: 7.0 * jnp.ones_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
        ]

        # Logits: [0, ln(3)] so probabilities are [1/4, 3/4]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array(
                [jnp.zeros_like(th), jnp.log(3.0) * jnp.ones_like(th)]
            ),
            logits_derivative_function=lambda th: jnp.array(
                [jnp.zeros_like(th), jnp.zeros_like(th)]
            ),
        )

        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta = jnp.array([1.0])

        # Expected probabilities: [1/4, 3/4]
        probs = problem.compute_probabilities(theta)
        expected_probs = jnp.array([[0.25], [0.75]])
        assert jnp.allclose(probs, expected_probs, rtol=1e-6)

        # Expected value: 0.25*3 + 0.75*7 = 0.75 + 5.25 = 6.0
        expected_val = problem.compute_expected_value(theta)
        assert jnp.allclose(expected_val, jnp.array([6.0]), rtol=1e-6)

        # Expected gradient: 0 (constant functions, constant probabilities)
        gradient = problem.compute_exact_gradient(theta)
        assert gradient is not None
        assert jnp.allclose(gradient, jnp.array([0.0]), atol=1e-6)

    def test_exact_gradient_with_softmax_analytical(self):
        """Test exact gradient with softmax probability function against
        known analytical result.

        Uses constant functions f1=1, f2=2 with linear logits α1=θ, α2=0.
        Expected: E[f] = 2 - sigmoid(θ)
        so dE/dθ = -sigmoid'(θ) = -sigmoid(θ)(1-sigmoid(θ))
        At θ=0: gradient = -0.5 * 0.5 = -0.25
        """
        # Constant branches: f1(θ) = 1, f2(θ) = 2
        branches = [
            Branch(
                function=lambda th: jnp.ones_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
            Branch(
                function=lambda th: 2 * jnp.ones_like(th),
                derivative_function=lambda th: jnp.zeros_like(th),
            ),
        ]

        # Linear logits: α1(θ) = θ, α2(θ) = 0 → softmax gives sigmoid probabilities
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([th, jnp.zeros_like(th)]),
            logits_derivative_function=lambda th: jnp.array(
                [jnp.ones_like(th), jnp.zeros_like(th)]
            ),
        )

        problem = DiscreteProblem(branches=branches, logits_model=logits_model)
        theta = jnp.array([0.0])

        # Compute gradient
        gradient = problem.compute_exact_gradient(theta)

        # Verify shape and type
        assert gradient is not None, "Gradient should not be None"
        assert isinstance(
            gradient, jnp.ndarray
        ), "Gradient should be ndarray for array input"
        assert gradient.shape == theta.shape, "Gradient shape should match input"

        # Verify exact analytical value:
        # -sigmoid(0) * (1 - sigmoid(0)) = -0.5 * 0.5 = -0.25
        expected_gradient = -0.25
        assert jnp.allclose(
            gradient, jnp.array([expected_gradient]), atol=1e-12
        ), f"Expected gradient {expected_gradient}, got {gradient[0]}"

    def test_exact_gradient_with_sigmoid_analytical(self):
        """Test exact gradient with custom sigmoid probability function against
        known analytical result.

        Uses linear functions f1=θ, f2=0 with custom binary sigmoid probabilities.
        Custom sigmoid: p1 = 1/(1+exp(-2θ)), p2 = 1/(1+exp(2θ))
        Expected: E[f] = θ * p1(θ), so dE/dθ = p1(θ) + θ * dp1/dθ
        At θ=0: p1=0.5, dp1/dθ=1, so gradient = 0.5 + 0*1 = 0.5
        """
        # Linear branches: f1(θ) = θ, f2(θ) = 0
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

        # Custom binary sigmoid probability function
        def sigmoid(logits):
            """Binary sigmoid ensuring probabilities sum to 1."""
            return 1.0 / (1 + jnp.exp(-logits))

        # Symmetric logits: α1(θ) = θ, α2(θ) = -θ
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([th, -th]),
            logits_derivative_function=lambda th: jnp.array(
                [jnp.ones_like(th), -jnp.ones_like(th)]
            ),
            probability_function=sigmoid,
        )

        problem = DiscreteProblem(branches=branches, logits_model=logits_model)
        theta = jnp.array([0.0])

        # Verify probabilities are correct at θ=0
        probs = problem.compute_probabilities(theta)
        assert jnp.allclose(probs, jnp.array([[0.5], [0.5]]), atol=1e-12)

        # Compute gradient
        gradient = problem.compute_exact_gradient(theta)

        # Verify shape and type
        assert gradient is not None, "Gradient should not be None"
        assert isinstance(
            gradient, jnp.ndarray
        ), "Gradient should be ndarray for array input"
        assert gradient.shape == theta.shape, "Gradient shape should match input"

        # Verify exact analytical value: p1(0) + 0 * dp1/dθ|θ=0 = 0.5 + 0 = 0.5
        expected_gradient = 0.5
        assert jnp.allclose(
            gradient, jnp.array([expected_gradient]), atol=1e-12
        ), f"Expected gradient {expected_gradient}, got {gradient[0]}"

    def test_exact_gradient_with_trigonometric_functions_analytical(self):
        """Test exact gradient with trigonometric functions and custom sigmoid
        probabilities.

        Uses trigonometric branches: f1(θ) = sin(θ), f2(θ) = cos(θ)
        with custom sigmoid probabilities.
        Custom sigmoid: p1 = 1/(1+exp(-2θ)), p2 = 1/(1+exp(2θ))
        Expected: E[f] = sin(θ)*p1(θ) + cos(θ)*p2(θ)
        At θ=π/6: sin(π/6)=0.5, cos(π/6)=√3/2, p1≈0.731, p2≈0.269
        Expected gradient requires chain rule with sigmoid derivatives.
        """
        # Trigonometric branches: f1(θ) = sin(θ), f2(θ) = cos(θ)
        branches = [
            Branch(
                function=lambda th: jnp.sin(th),
                derivative_function=lambda th: jnp.cos(th),
            ),
            Branch(
                function=lambda th: jnp.cos(th),
                derivative_function=lambda th: -jnp.sin(th),
            ),
        ]

        # Custom binary sigmoid probability function
        def binary_sigmoid(logits):
            """Binary sigmoid ensuring probabilities sum to 1."""
            return 1.0 / (1 + jnp.exp(-logits))

        # Asymmetric logits: α1(θ) = 2θ, α2(θ) = -2θ → sigmoid probabilities
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([2 * th, -2 * th]),
            logits_derivative_function=lambda th: jnp.array(
                [2 * jnp.ones_like(th), -2 * jnp.ones_like(th)]
            ),
            probability_function=binary_sigmoid,
        )

        problem = DiscreteProblem(branches=branches, logits_model=logits_model)
        theta = jnp.array([jnp.pi / 6])  # θ = π/6 = 30 degrees

        # At θ=π/6: p1 = 1/(1+exp(-π/3)) ≈ 0.731, p2 ≈ 0.269
        probs = problem.compute_probabilities(theta)
        expected_p1 = 1.0 / (1 + jnp.exp(-jnp.pi / 3))
        expected_p2 = 1.0 - expected_p1
        assert jnp.allclose(
            probs, jnp.array([[expected_p1], [expected_p2]]), atol=1e-12
        )

        # Verify expected value: E[f] = sin(π/6)*p1 + cos(π/6)*p2 = 0.5*p1 + (√3/2)*p2
        expected_val = problem.compute_expected_value(theta)
        analytical_expected_val = 0.5 * expected_p1 + (jnp.sqrt(3) / 2) * expected_p2
        assert jnp.allclose(
            expected_val, jnp.array([analytical_expected_val]), atol=1e-12
        )

        # Compute gradient
        gradient = problem.compute_exact_gradient(theta)

        # Verify shape and type
        assert gradient is not None, "Gradient should not be None"
        assert isinstance(
            gradient, jnp.ndarray
        ), "Gradient should be ndarray for array input"
        assert gradient.shape == theta.shape, "Gradient shape should match input"

        # Analytical gradient calculation using chain rule:
        # dE/dθ = d/dθ[sin(θ)*p1(θ) + cos(θ)*p2(θ)]
        # = cos(θ)*p1(θ) + sin(θ)*dp1/dθ - sin(θ)*p2(θ) + cos(θ)*dp2/dθ
        # where dp1/dθ = 2*p1(θ)*(1-p1(θ)) and dp2/dθ = -dp1/dθ
        sin_val = 0.5  # sin(π/6)
        cos_val = jnp.sqrt(3) / 2  # cos(π/6)
        dp1_dtheta = 2 * expected_p1 * (1 - expected_p1)
        dp2_dtheta = -dp1_dtheta

        expected_gradient = (
            cos_val * expected_p1
            + sin_val * dp1_dtheta
            + (-sin_val) * expected_p2
            + cos_val * dp2_dtheta
        )

        assert jnp.allclose(
            gradient, jnp.array([expected_gradient]), atol=1e-10
        ), f"Expected gradient {expected_gradient}, got {gradient[0]}"
