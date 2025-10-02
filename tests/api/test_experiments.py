"""Tests for mellowgate.api.experiments module."""

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
def simple_problem():
    """Simple discrete problem for testing experiments."""
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


class TestSweep:
    """Test the Sweep dataclass."""

    def test_sweep_creation_minimal(self):
        """Test creating a Sweep with minimal parameters."""
        theta_values = jnp.array([0.0, 1.0, 2.0])
        sweep = Sweep(theta_values=theta_values)

        assert jnp.array_equal(sweep.theta_values, theta_values)
        assert sweep.num_repetitions == 200
        assert sweep.estimator_configs is None

    def test_sweep_creation_full(self):
        """Test creating a Sweep with all parameters."""
        theta_values = jnp.array([0.0, 1.0, 2.0])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=100)}
        }

        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=5,
            estimator_configs=estimator_configs,
        )

        assert jnp.array_equal(sweep.theta_values, theta_values)
        assert sweep.num_repetitions == 5
        assert sweep.estimator_configs is not None
        assert "fd" in sweep.estimator_configs
        assert isinstance(sweep.estimator_configs["fd"]["cfg"], FiniteDifferenceConfig)

    def test_sweep_estimator_configs_validation(self):
        """Test different estimator config formats."""
        theta_values = jnp.array([1.0])

        # Test multiple estimators
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig()},
            "reinforce": {"cfg": ReinforceConfig()},
            "gs": {"cfg": GumbelSoftmaxConfig()},
        }

        sweep = Sweep(theta_values=theta_values, estimator_configs=estimator_configs)
        assert sweep.estimator_configs is not None
        assert len(sweep.estimator_configs) == 3


class TestRunParameterSweep:
    """Test the run_parameter_sweep function."""

    def test_parameter_sweep_basic(self, simple_problem):
        """Test basic parameter sweep execution."""
        theta_values = jnp.array([0.5, 1.0])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=50)}
        }

        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=2,
            estimator_configs=estimator_configs,
        )

        results = run_parameter_sweep(simple_problem, sweep)

        assert isinstance(results, dict)
        assert "fd" in results
        assert isinstance(results["fd"], ResultsContainer)
        assert results["fd"].gradient_estimates["fd"]["mean"].shape == (2,)
        assert jnp.all(jnp.isfinite(results["fd"].gradient_estimates["fd"]["mean"]))

    def test_parameter_sweep_multiple_estimators(self, simple_problem):
        """Test parameter sweep with multiple estimators."""
        theta_values = jnp.array([1.0])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=50)},
            "gs": {"cfg": GumbelSoftmaxConfig(temperature=1.0, num_samples=50)},
        }

        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=1,
            estimator_configs=estimator_configs,
        )

        results = run_parameter_sweep(simple_problem, sweep)

        assert isinstance(results, dict)
        assert "fd" in results
        assert "gs" in results
        assert isinstance(results["fd"], ResultsContainer)
        assert isinstance(results["gs"], ResultsContainer)
        assert results["fd"].gradient_estimates["fd"]["mean"].shape == (1,)
        assert results["gs"].gradient_estimates["gs"]["mean"].shape == (1,)

    def test_parameter_sweep_with_reinforce(self, simple_problem):
        """Test parameter sweep with REINFORCE estimator."""
        theta_values = jnp.array([1.0])
        estimator_configs = {
            "reinforce": {
                "cfg": ReinforceConfig(num_samples=50),
                "state": ReinforceState(),
            },
        }

        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=1,
            estimator_configs=estimator_configs,
        )

        results = run_parameter_sweep(simple_problem, sweep)

        assert isinstance(results, dict)
        assert "reinforce" in results
        assert isinstance(results["reinforce"], ResultsContainer)
        assert results["reinforce"].gradient_estimates["reinforce"]["mean"].shape == (
            1,
        )
        assert jnp.all(
            jnp.isfinite(results["reinforce"].gradient_estimates["reinforce"]["mean"])
        )

    def test_parameter_sweep_empty_estimators(self, simple_problem):
        """Test parameter sweep with no estimators."""
        theta_values = jnp.array([1.0])

        sweep = Sweep(theta_values=theta_values, estimator_configs={})

        results = run_parameter_sweep(simple_problem, sweep)

        assert isinstance(results, dict)
        assert len(results) == 0

    def test_parameter_sweep_multiple_repetitions(self, simple_problem):
        """Test parameter sweep with multiple repetitions."""
        theta_values = jnp.array([0.5, 1.0, 1.5])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=30)}
        }

        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=3,
            estimator_configs=estimator_configs,
        )

        results = run_parameter_sweep(simple_problem, sweep)

        assert results["fd"].gradient_estimates["fd"]["mean"].shape == (3,)
        assert results["fd"].gradient_estimates["fd"]["std"].shape == (3,)
        assert jnp.all(jnp.isfinite(results["fd"].gradient_estimates["fd"]["mean"]))

    def test_parameter_sweep_vectorization(self, simple_problem):
        """Test that parameter sweep properly vectorizes computation."""
        # Test with larger arrays to ensure vectorized operations work
        theta_values = jnp.linspace(-1.0, 1.0, 10)
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=30)}
        }

        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=2,
            estimator_configs=estimator_configs,
        )

        results = run_parameter_sweep(simple_problem, sweep)

        assert results["fd"].gradient_estimates["fd"]["mean"].shape == (10,)
        assert jnp.all(jnp.isfinite(results["fd"].gradient_estimates["fd"]["mean"]))

    def test_parameter_sweep_results_structure(self, simple_problem):
        """Test that results have expected structure."""
        theta_values = jnp.array([1.0, 2.0])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=30)}
        }

        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=1,
            estimator_configs=estimator_configs,
        )

        results = run_parameter_sweep(simple_problem, sweep)

        # Should have theta_values and expectation_values
        fd_results = results["fd"]
        assert jnp.array_equal(fd_results.theta_values, theta_values)
        assert fd_results.expectation_values is not None
        assert fd_results.expectation_values.shape == (2,)

    def test_parameter_sweep_no_exact_gradients(self):
        """Test parameter sweep results structure."""
        # Create problem without logits derivatives (multi-branch to avoid shape issues)
        branches = [
            Branch(function=lambda th: th**2),
            Branch(function=lambda th: th**3),
        ]
        logits_model = LogitsModel(
            logits_function=lambda th: jnp.array([0.0, 1.0]),
            logits_derivative_function=lambda th: jnp.array(
                [0.0, 0.0]
            ),  # Add derivatives to avoid issues
        )
        problem = DiscreteProblem(branches=branches, logits_model=logits_model)

        theta_values = jnp.array([1.0, 2.0])  # Use multiple theta values
        estimator_configs = {
            "gs": {
                "cfg": GumbelSoftmaxConfig(num_samples=30)
            }  # Use Gumbel instead of FD
        }

        sweep = Sweep(theta_values=theta_values, estimator_configs=estimator_configs)

        results = run_parameter_sweep(problem, sweep)

        # Should work without exact gradients
        assert "gs" in results
        assert isinstance(results["gs"], ResultsContainer)


class TestResultsContainer:
    """Test the ResultsContainer functionality."""

    def test_results_container_access(self, simple_problem):
        """Test accessing results from ResultsContainer."""
        theta_values = jnp.array([1.0])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=30)}
        }

        sweep = Sweep(theta_values=theta_values, estimator_configs=estimator_configs)

        results = run_parameter_sweep(simple_problem, sweep)

        # Test that we get dict of results
        assert isinstance(results, dict)
        assert "fd" in results

        fd_results = results["fd"]
        assert hasattr(fd_results, "gradient_estimates")
        assert hasattr(fd_results, "theta_values")

        # Test that theta_values match
        assert jnp.array_equal(fd_results.theta_values, theta_values)

    def test_results_container_structure(self, simple_problem):
        """Test the structure of results in ResultsContainer."""
        theta_values = jnp.array([0.5, 1.0])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=30)},
            "gs": {"cfg": GumbelSoftmaxConfig(temperature=1.0, num_samples=30)},
        }

        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=2,
            estimator_configs=estimator_configs,
        )

        results = run_parameter_sweep(simple_problem, sweep)

        # Check that each estimator has the expected structure
        for estimator_name in ["fd", "gs"]:
            assert estimator_name in results
            estimator_results = results[estimator_name]

            # Should have gradient_estimates dict
            assert hasattr(estimator_results, "gradient_estimates")
            assert estimator_name in estimator_results.gradient_estimates

            # Should have mean and std
            grad_data = estimator_results.gradient_estimates[estimator_name]
            assert "mean" in grad_data
            assert "std" in grad_data
            assert grad_data["mean"].shape == (2,)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_theta_value(self, simple_problem):
        """Test parameter sweep with single theta value."""
        theta_values = jnp.array([1.0])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=30)}
        }

        sweep = Sweep(theta_values=theta_values, estimator_configs=estimator_configs)
        results = run_parameter_sweep(simple_problem, sweep)

        assert results["fd"].gradient_estimates["fd"]["mean"].shape == (1,)

    def test_single_repetition(self, simple_problem):
        """Test parameter sweep with single repetition."""
        theta_values = jnp.array([0.5, 1.0])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=30)}
        }

        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=1,
            estimator_configs=estimator_configs,
        )

        results = run_parameter_sweep(simple_problem, sweep)

        assert results["fd"].gradient_estimates["fd"]["mean"].shape == (2,)

    def test_zero_theta_values(self, simple_problem):
        """Test parameter sweep with theta=0."""
        theta_values = jnp.array([0.0])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=30)}
        }

        sweep = Sweep(theta_values=theta_values, estimator_configs=estimator_configs)
        results = run_parameter_sweep(simple_problem, sweep)

        assert jnp.all(jnp.isfinite(results["fd"].gradient_estimates["fd"]["mean"]))

    def test_negative_theta_values(self, simple_problem):
        """Test parameter sweep with negative theta values."""
        theta_values = jnp.array([-1.0, -0.5])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=30)}
        }

        sweep = Sweep(theta_values=theta_values, estimator_configs=estimator_configs)
        results = run_parameter_sweep(simple_problem, sweep)

        assert jnp.all(jnp.isfinite(results["fd"].gradient_estimates["fd"]["mean"]))


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""

    def test_large_theta_array(self, simple_problem):
        """Test parameter sweep with large theta array."""
        theta_values = jnp.linspace(-2.0, 2.0, 50)  # Larger array
        estimator_configs = {
            "fd": {
                "cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=20)
            }  # Fewer samples for speed
        }

        sweep = Sweep(theta_values=theta_values, estimator_configs=estimator_configs)
        results = run_parameter_sweep(simple_problem, sweep)

        assert results["fd"].gradient_estimates["fd"]["mean"].shape == (50,)
        assert jnp.all(jnp.isfinite(results["fd"].gradient_estimates["fd"]["mean"]))

    def test_many_repetitions(self, simple_problem):
        """Test parameter sweep with many repetitions."""
        theta_values = jnp.array([1.0])
        estimator_configs = {
            "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3, num_samples=20)}
        }

        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=10,  # Many repetitions
            estimator_configs=estimator_configs,
        )

        results = run_parameter_sweep(simple_problem, sweep)

        assert results["fd"].gradient_estimates["fd"]["mean"].shape == (1,)
        assert jnp.all(jnp.isfinite(results["fd"].gradient_estimates["fd"]["mean"]))

    def test_invalid_estimator_configs(self, simple_problem):
        """Test error handling for invalid estimator configurations."""
        theta_values = jnp.array([1.0])

        # Test unknown estimator
        estimator_configs = {"unknown_estimator": {"cfg": FiniteDifferenceConfig()}}

        sweep = Sweep(theta_values=theta_values, estimator_configs=estimator_configs)

        with pytest.raises(ValueError, match="Unknown estimator: unknown_estimator"):
            run_parameter_sweep(simple_problem, sweep)

        # Test REINFORCE with wrong state type
        estimator_configs = {
            "reinforce": {
                "cfg": ReinforceConfig(),
                "state": "not_a_reinforce_state",  # Wrong type
            }
        }

        sweep = Sweep(theta_values=theta_values, estimator_configs=estimator_configs)

        with pytest.raises(
            TypeError, match="State for 'reinforce' must be a ReinforceState instance"
        ):
            run_parameter_sweep(simple_problem, sweep)

    def test_none_estimator_configs(self, simple_problem):
        """Test None estimator_configs returns empty dict."""
        theta_values = jnp.array([1.0])

        # Create sweep with None estimator_configs
        sweep = Sweep(
            theta_values=theta_values, num_repetitions=5, estimator_configs=None
        )

        results = run_parameter_sweep(simple_problem, sweep)
        assert results == {}

    def test_zero_repetitions_edge_case(self, simple_problem):
        """Test zero repetitions case."""
        theta_values = jnp.array([1.0, 2.0])

        # Test with zero repetitions - this should trigger the zero case in
        # _compute_sweep_statistics
        sweep = Sweep(
            theta_values=theta_values,
            num_repetitions=0,  # Zero repetitions
            estimator_configs={"fd": {"cfg": FiniteDifferenceConfig()}},
        )

        results = run_parameter_sweep(simple_problem, sweep)

        # Should handle gracefully with NaN values
        assert "fd" in results
        fd_results = results["fd"]
        assert jnp.isnan(fd_results.gradient_estimates["fd"]["mean"]).all()
        assert jnp.isnan(fd_results.gradient_estimates["fd"]["std"]).all()

        # Also test the internal function directly to ensure coverage
        from mellowgate.api.experiments import _compute_sweep_statistics

        # Test with empty gradient samples array
        empty_samples = jnp.empty((0, 2))  # 0 repetitions, 2 theta values
        mean, std = _compute_sweep_statistics(empty_samples, 0)
        assert jnp.isnan(mean).all()
        assert jnp.isnan(std).all()
        assert mean.shape == (2,)
        assert std.shape == (2,)

    def test_empty_theta_values_edge_case(self, simple_problem):
        """Test empty theta values case."""
        # Test with empty theta values
        sweep = Sweep(
            theta_values=jnp.array([]),  # Empty array
            num_repetitions=5,
            estimator_configs={"fd": {"cfg": FiniteDifferenceConfig()}},
        )

        results = run_parameter_sweep(simple_problem, sweep)

        # Should handle empty arrays gracefully
        assert "fd" in results
        fd_results = results["fd"]
        assert len(fd_results.gradient_estimates["fd"]["mean"]) == 0
        assert len(fd_results.gradient_estimates["fd"]["time"]) == 0
