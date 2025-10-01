"""Unit tests for experiments module.

This module provides comprehensive tests for the experimental framework
including Sweep configuration and run_parameter_sweep function.
Tests cover shapes, values, edge cases, and error conditions.
"""

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from mellowgate.api.estimators import (
    FiniteDifferenceConfig,
    GumbelSoftmaxConfig,
    ReinforceConfig,
    ReinforceState,
)
from mellowgate.api.experiments import Sweep, run_parameter_sweep
from mellowgate.api.functions import Branch, DiscreteProblem, LogitsModel
from mellowgate.api.results import ResultsContainer


@pytest.fixture
def test_theta_values():
    """Common theta values for testing."""
    return jnp.linspace(-2.0, 2.0, 5)


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
    """Simple discrete problem for testing."""
    return DiscreteProblem(
        branches=simple_branches,
        logits_model=simple_logits_model,
    )


@pytest.fixture
def fd_estimator_config():
    """Finite difference estimator configuration."""
    return {"fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=100)}}


@pytest.fixture
def reinforce_estimator_config():
    """REINFORCE estimator configuration."""
    return {
        "reinforce": {
            "cfg": ReinforceConfig(num_samples=100),
            "state": ReinforceState(),
        }
    }


@pytest.fixture
def gumbel_softmax_estimator_config():
    """Gumbel-Softmax estimator configuration."""
    return {"gs": {"cfg": GumbelSoftmaxConfig(temperature=0.5, num_samples=100)}}


@pytest.fixture
def all_estimator_configs():
    """All estimator configurations combined."""
    return {
        "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=50)},
        "reinforce": {
            "cfg": ReinforceConfig(num_samples=50),
            "state": ReinforceState(),
        },
        "gs": {"cfg": GumbelSoftmaxConfig(temperature=0.5, num_samples=50)},
    }


class TestSweep:
    """Test Sweep dataclass."""

    def test_sweep_creation_minimal(self, test_theta_values):
        """Test Sweep creation with minimal parameters."""
        sweep = Sweep(theta_values=test_theta_values)

        assert jnp.array_equal(sweep.theta_values, test_theta_values)
        assert sweep.num_repetitions == 200  # default value
        assert sweep.estimator_configs is None

    def test_sweep_creation_full(self, test_theta_values, fd_estimator_config):
        """Test Sweep creation with all parameters."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=50,
            estimator_configs=fd_estimator_config,
        )

        assert jnp.array_equal(sweep.theta_values, test_theta_values)
        assert sweep.num_repetitions == 50
        assert sweep.estimator_configs == fd_estimator_config

    def test_sweep_theta_values_shape(self):
        """Test Sweep with different theta value shapes."""
        # Single value
        theta_single = jnp.array([1.0])
        sweep = Sweep(theta_values=theta_single)
        assert sweep.theta_values.shape == (1,)

        # Multiple values
        theta_multiple = jnp.array([-1.0, 0.0, 1.0])
        sweep = Sweep(theta_values=theta_multiple)
        assert sweep.theta_values.shape == (3,)

    def test_sweep_estimator_configs_structure(self, test_theta_values):
        """Test that estimator configs have correct structure."""
        configs = {
            "fd": {"cfg": FiniteDifferenceConfig()},
            "reinforce": {"cfg": ReinforceConfig(), "state": ReinforceState()},
        }

        sweep = Sweep(theta_values=test_theta_values, estimator_configs=configs)

        assert sweep.estimator_configs is not None
        assert "fd" in sweep.estimator_configs
        assert "reinforce" in sweep.estimator_configs
        assert "cfg" in sweep.estimator_configs["fd"]
        assert "cfg" in sweep.estimator_configs["reinforce"]
        assert "state" in sweep.estimator_configs["reinforce"]


class TestRunParameterSweep:
    """Test run_parameter_sweep function."""

    def test_sweep_none_estimator_configs(
        self, simple_discrete_problem, test_theta_values
    ):
        """Test sweep with None estimator configs returns empty dict."""
        sweep = Sweep(theta_values=test_theta_values, estimator_configs=None)

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        assert results == {}

    def test_sweep_finite_difference_only(
        self, simple_discrete_problem, test_theta_values, fd_estimator_config
    ):
        """Test sweep with only finite difference estimator."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=10,  # Small for faster testing
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        assert len(results) == 1
        assert "fd" in results
        assert isinstance(results["fd"], ResultsContainer)

    def test_sweep_reinforce_only(
        self, simple_discrete_problem, test_theta_values, reinforce_estimator_config
    ):
        """Test sweep with only REINFORCE estimator."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=10,
            estimator_configs=reinforce_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        assert len(results) == 1
        assert "reinforce" in results
        assert isinstance(results["reinforce"], ResultsContainer)

    def test_sweep_gumbel_softmax_only(
        self,
        simple_discrete_problem,
        test_theta_values,
        gumbel_softmax_estimator_config,
    ):
        """Test sweep with only Gumbel-Softmax estimator."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=10,
            estimator_configs=gumbel_softmax_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        assert len(results) == 1
        assert "gs" in results
        assert isinstance(results["gs"], ResultsContainer)

    def test_sweep_all_estimators(
        self, simple_discrete_problem, test_theta_values, all_estimator_configs
    ):
        """Test sweep with all estimators."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=10,
            estimator_configs=all_estimator_configs,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        assert len(results) == 3
        assert "fd" in results
        assert "reinforce" in results
        assert "gs" in results

        for estimator_name, result in results.items():
            assert isinstance(result, ResultsContainer)

    def test_results_container_structure(
        self, simple_discrete_problem, test_theta_values, fd_estimator_config
    ):
        """Test that ResultsContainer has correct structure and values."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=5,
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        result = results["fd"]

        # Test gradient_estimates structure
        assert "fd" in result.gradient_estimates
        grad_data = result.gradient_estimates["fd"]
        assert "theta" in grad_data
        assert "mean" in grad_data
        assert "std" in grad_data
        assert "time" in grad_data

        # Test shapes
        assert grad_data["theta"].shape == test_theta_values.shape
        assert grad_data["mean"].shape == test_theta_values.shape
        assert grad_data["std"].shape == test_theta_values.shape
        assert grad_data["time"].shape == test_theta_values.shape

        # Test other attributes
        assert jnp.array_equal(result.theta_values, test_theta_values)
        assert result.expectation_values is not None
        assert result.expectation_values.shape == test_theta_values.shape
        assert result.discrete_distributions is not None
        assert result.discrete_distributions.shape == (
            2,
            len(test_theta_values),
        )  # 2 branches
        assert result.sampled_points is not None
        assert "sampled_branch_indices" in result.sampled_points

    def test_gradient_estimates_statistical_properties(
        self, simple_discrete_problem, test_theta_values, fd_estimator_config
    ):
        """Test that gradient estimates have reasonable statistical properties."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=20,  # Enough for meaningful statistics
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        grad_data = results["fd"].gradient_estimates["fd"]

        # All values should be finite
        assert jnp.all(jnp.isfinite(grad_data["mean"]))
        assert jnp.all(jnp.isfinite(grad_data["std"]))
        assert jnp.all(jnp.isfinite(grad_data["time"]))

        # Standard deviations should be non-negative
        assert jnp.all(grad_data["std"] >= 0)

        # Times should be positive
        assert jnp.all(grad_data["time"] > 0)

    def test_sampled_branch_indices_shape(
        self, simple_discrete_problem, test_theta_values, fd_estimator_config
    ):
        """Test that sampled branch indices have correct shape."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=10,
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        assert results["fd"].sampled_points is not None
        sampled_indices = results["fd"].sampled_points["sampled_branch_indices"]

        # Shape should be (num_theta_values, num_repetitions)
        expected_shape = (len(test_theta_values), sweep.num_repetitions)
        assert sampled_indices.shape == expected_shape

        # All indices should be valid (0 or 1 for 2-branch problem)
        assert jnp.all((sampled_indices >= 0) & (sampled_indices < 2))

    def test_unknown_estimator_error(self, simple_discrete_problem, test_theta_values):
        """Test that unknown estimator raises ValueError."""
        unknown_config = {"unknown_estimator": {"cfg": {"some": "config"}}}

        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=5,
            estimator_configs=unknown_config,
        )

        with pytest.raises(ValueError, match="Unknown estimator: unknown_estimator"):
            run_parameter_sweep(simple_discrete_problem, sweep)

    def test_reinforce_wrong_state_type_error(
        self, simple_discrete_problem, test_theta_values
    ):
        """Test that REINFORCE with wrong state type raises TypeError."""
        wrong_state_config = {
            "reinforce": {
                "cfg": ReinforceConfig(num_samples=50),
                "state": "wrong_type",  # Should be ReinforceState
            }
        }

        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=5,
            estimator_configs=wrong_state_config,
        )

        with pytest.raises(
            TypeError, match="State for 'reinforce' must be a ReinforceState instance"
        ):
            run_parameter_sweep(simple_discrete_problem, sweep)

    def test_single_theta_value(self, simple_discrete_problem, fd_estimator_config):
        """Test sweep with single theta value."""
        theta_single = jnp.array([1.0])
        sweep = Sweep(
            theta_values=theta_single,
            num_repetitions=5,
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        grad_data = results["fd"].gradient_estimates["fd"]

        # All arrays should have shape (1,)
        assert grad_data["theta"].shape == (1,)
        assert grad_data["mean"].shape == (1,)
        assert grad_data["std"].shape == (1,)
        assert grad_data["time"].shape == (1,)

    def test_empty_theta_values(self, simple_discrete_problem, fd_estimator_config):
        """Test sweep with empty theta values."""
        theta_empty = jnp.array([])
        sweep = Sweep(
            theta_values=theta_empty,
            num_repetitions=5,
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        grad_data = results["fd"].gradient_estimates["fd"]

        # All arrays should have shape (0,)
        assert grad_data["theta"].shape == (0,)
        assert grad_data["mean"].shape == (0,)
        assert grad_data["std"].shape == (0,)
        assert grad_data["time"].shape == (0,)

    def test_timing_information(
        self, simple_discrete_problem, test_theta_values, fd_estimator_config
    ):
        """Test that timing information is recorded correctly."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=5,
            estimator_configs=fd_estimator_config,
        )

        # Mock time.time to control timing
        with patch("mellowgate.api.experiments.time.time") as mock_time:
            mock_time.side_effect = [0.0, 10.0]  # 10 second duration

            results = run_parameter_sweep(simple_discrete_problem, sweep)

        grad_data = results["fd"].gradient_estimates["fd"]

        # Time per theta should be total_time / num_theta
        expected_time_per_theta = 10.0 / len(test_theta_values)
        jnp.allclose(grad_data["time"], expected_time_per_theta)

    def test_reproducibility_with_seed(
        self,
        simple_branches,
        simple_logits_model,
        test_theta_values,
        fd_estimator_config,
    ):
        """Test that results are reproducible with fixed random seed."""

        # Create discrete problems with deterministic sampling for reproducibility
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

        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=10,
            estimator_configs=fd_estimator_config,
        )

        results1 = run_parameter_sweep(discrete_problem1, sweep)
        results2 = run_parameter_sweep(discrete_problem2, sweep)

        grad_data1 = results1["fd"].gradient_estimates["fd"]
        grad_data2 = results2["fd"].gradient_estimates["fd"]

        assert jnp.array_equal(grad_data1["mean"], grad_data2["mean"])
        assert jnp.array_equal(grad_data1["std"], grad_data2["std"])

    def test_expectation_values_consistency(
        self, simple_discrete_problem, test_theta_values, fd_estimator_config
    ):
        """Test that expectation values are consistent with manual calculation."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=5,
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        # Manually compute expected values
        probabilities = simple_discrete_problem.compute_probabilities(test_theta_values)
        function_values = simple_discrete_problem.compute_function_values(
            test_theta_values
        )
        expected_manual = jnp.sum(probabilities * function_values, axis=0)

        assert results["fd"].expectation_values is not None
        assert jnp.allclose(results["fd"].expectation_values, expected_manual)

    def test_discrete_distributions_shape(
        self, simple_discrete_problem, test_theta_values, fd_estimator_config
    ):
        """Test that discrete distributions have correct shape."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=5,
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        discrete_distributions = results["fd"].discrete_distributions
        assert discrete_distributions is not None

        # Shape should be (num_branches, num_theta_values)
        expected_shape = (simple_discrete_problem.num_branches, len(test_theta_values))
        assert discrete_distributions.shape == expected_shape


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_repetitions(
        self, simple_discrete_problem, test_theta_values, fd_estimator_config
    ):
        """Test sweep with zero repetitions."""
        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=0,
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        grad_data = results["fd"].gradient_estimates["fd"]

        # Should still return arrays of correct shape, but values may be NaN or 0
        assert grad_data["theta"].shape == test_theta_values.shape
        assert grad_data["mean"].shape == test_theta_values.shape
        assert grad_data["std"].shape == test_theta_values.shape

    def test_large_theta_range(self, simple_discrete_problem, fd_estimator_config):
        """Test sweep with large theta range."""
        theta_large = jnp.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        sweep = Sweep(
            theta_values=theta_large,
            num_repetitions=5,
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        grad_data = results["fd"].gradient_estimates["fd"]

        # Should handle large values without crashing
        assert jnp.all(jnp.isfinite(grad_data["mean"]))
        assert jnp.all(jnp.isfinite(grad_data["std"]))

    def test_single_branch_problem(self, test_theta_values, fd_estimator_config):
        """Test sweep with single branch discrete problem."""
        # Create single branch problem
        branch = Branch(
            function=lambda th: th**2, derivative_function=lambda th: 2 * th
        )
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([jnp.zeros_like(th)]),
            logits_derivative_function=lambda th: jnp.array([jnp.zeros_like(th)]),
        )
        problem = DiscreteProblem(branches=[branch], logits_model=logits_model)

        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=5,
            estimator_configs=fd_estimator_config,
        )

        results = run_parameter_sweep(problem, sweep)

        # Should handle single branch case
        assert len(results) == 1
        assert "fd" in results

        # Discrete distributions should have shape (1, num_theta)
        assert results["fd"].discrete_distributions is not None
        assert results["fd"].discrete_distributions.shape == (1, len(test_theta_values))

    def test_multiple_estimators_different_configs(
        self, simple_discrete_problem, test_theta_values
    ):
        """Test sweep with multiple estimators having different configurations."""
        mixed_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-4, num_samples=50)},
            "reinforce": {
                "cfg": ReinforceConfig(num_samples=50, use_baseline=True),
                "state": ReinforceState(),
            },
            "gs": {"cfg": GumbelSoftmaxConfig(temperature=0.5, num_samples=50)},
        }

        sweep = Sweep(
            theta_values=test_theta_values,
            num_repetitions=5,
            estimator_configs=mixed_configs,
        )

        results = run_parameter_sweep(simple_discrete_problem, sweep)

        assert len(results) == 3
        for estimator_name in mixed_configs.keys():
            assert estimator_name in results
            assert isinstance(results[estimator_name], ResultsContainer)
