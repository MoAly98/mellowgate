"""Tests for mellowgate.api.results module."""

import jax.numpy as jnp

from mellowgate.api.results import ResultsContainer


class TestResultsContainer:
    """Test the ResultsContainer dataclass."""

    def test_results_container_creation_minimal(self):
        """Test creating a ResultsContainer with minimal parameters."""
        theta_values = jnp.array([1.0, 2.0])
        gradient_estimates = {
            "fd": {
                "theta": theta_values,
                "mean": jnp.array([0.5, 1.0]),
                "std": jnp.array([0.1, 0.2]),
                "time": jnp.array([0.001, 0.002]),
            }
        }

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        assert jnp.array_equal(container.theta_values, theta_values)
        assert "fd" in container.gradient_estimates
        assert container.sampled_points is None
        assert container.expectation_values is None
        assert container.discrete_distributions is None

    def test_results_container_creation_full(self):
        """Test creating a ResultsContainer with all parameters."""
        theta_values = jnp.array([1.0, 2.0])
        gradient_estimates = {
            "fd": {
                "theta": theta_values,
                "mean": jnp.array([0.5, 1.0]),
                "std": jnp.array([0.1, 0.2]),
                "time": jnp.array([0.001, 0.002]),
            }
        }
        sampled_points = {"samples": jnp.array([0, 1, 0, 1])}
        expectation_values = jnp.array([1.5, 2.5])
        discrete_distributions = jnp.array([[0.5, 0.5], [0.3, 0.7]])

        container = ResultsContainer(
            theta_values=theta_values,
            gradient_estimates=gradient_estimates,
            sampled_points=sampled_points,
            expectation_values=expectation_values,
            discrete_distributions=discrete_distributions,
        )

        assert jnp.array_equal(container.theta_values, theta_values)
        assert container.sampled_points is not None
        assert container.expectation_values is not None
        assert container.discrete_distributions is not None
        assert jnp.array_equal(container.expectation_values, expectation_values)
        assert jnp.array_equal(container.discrete_distributions, discrete_distributions)

    def test_add_sampled_points(self):
        """Test adding sampled points to a container."""
        theta_values = jnp.array([1.0])
        gradient_estimates = {"fd": {"mean": jnp.array([0.5])}}

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        # Initially no sampled points
        assert container.sampled_points is None

        # Add sampled points
        samples = jnp.array([0, 1, 0])
        container.add_sampled_points("reinforce", samples)

        assert container.sampled_points is not None
        assert "reinforce" in container.sampled_points
        assert jnp.array_equal(container.sampled_points["reinforce"], samples)

    def test_add_sampled_points_multiple_estimators(self):
        """Test adding sampled points for multiple estimators."""
        theta_values = jnp.array([1.0])
        gradient_estimates = {"fd": {"mean": jnp.array([0.5])}}

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        # Add sampled points for first estimator
        samples1 = jnp.array([0, 1, 0])
        container.add_sampled_points("reinforce", samples1)

        # Add sampled points for second estimator
        samples2 = jnp.array([1, 0, 1])
        container.add_sampled_points("gumbel", samples2)

        assert container.sampled_points is not None
        assert len(container.sampled_points) == 2
        assert jnp.array_equal(container.sampled_points["reinforce"], samples1)
        assert jnp.array_equal(container.sampled_points["gumbel"], samples2)

    def test_add_expectation_values(self):
        """Test adding expectation values to a container."""
        theta_values = jnp.array([1.0, 2.0])
        gradient_estimates = {"fd": {"mean": jnp.array([0.5, 1.0])}}

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        # Initially no expectation values
        assert container.expectation_values is None

        # Add expectation values
        expectations = jnp.array([1.5, 2.5])
        container.add_expectation_values(expectations)

        assert container.expectation_values is not None
        assert jnp.array_equal(container.expectation_values, expectations)

    def test_add_discrete_distributions(self):
        """Test adding discrete distributions to a container."""
        theta_values = jnp.array([1.0, 2.0])
        gradient_estimates = {"fd": {"mean": jnp.array([0.5, 1.0])}}

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        # Initially no discrete distributions
        assert container.discrete_distributions is None

        # Add discrete distributions
        distributions = jnp.array([[0.5, 0.5], [0.3, 0.7]])
        container.add_discrete_distributions(distributions)

        assert container.discrete_distributions is not None
        assert jnp.array_equal(container.discrete_distributions, distributions)

    def test_gradient_estimates_structure(self):
        """Test the structure of gradient estimates."""
        theta_values = jnp.array([1.0, 2.0, 3.0])
        gradient_estimates = {
            "finite_diff": {
                "theta": theta_values,
                "mean": jnp.array([0.5, 1.0, 1.5]),
                "std": jnp.array([0.1, 0.15, 0.2]),
                "time": jnp.array([0.001, 0.002, 0.003]),
            },
            "reinforce": {
                "theta": theta_values,
                "mean": jnp.array([0.48, 0.95, 1.52]),
                "std": jnp.array([0.2, 0.25, 0.3]),
                "time": jnp.array([0.01, 0.015, 0.02]),
            },
        }

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        # Check that both estimators are present
        assert "finite_diff" in container.gradient_estimates
        assert "reinforce" in container.gradient_estimates

        # Check structure of each estimator's results
        for estimator_name, results in container.gradient_estimates.items():
            assert "theta" in results
            assert "mean" in results
            assert "std" in results
            assert "time" in results

            # Check shapes match
            assert results["theta"].shape == theta_values.shape
            assert results["mean"].shape == theta_values.shape
            assert results["std"].shape == theta_values.shape
            assert results["time"].shape == theta_values.shape

    def test_empty_gradient_estimates(self):
        """Test container with empty gradient estimates."""
        theta_values = jnp.array([1.0])
        gradient_estimates = {}

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        assert len(container.gradient_estimates) == 0
        assert jnp.array_equal(container.theta_values, theta_values)

    def test_single_theta_single_estimator(self):
        """Test container with single theta value and single estimator."""
        theta_values = jnp.array([2.5])
        gradient_estimates = {
            "gumbel_softmax": {
                "theta": theta_values,
                "mean": jnp.array([1.2]),
                "std": jnp.array([0.05]),
                "time": jnp.array([0.008]),
            }
        }

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        assert container.gradient_estimates["gumbel_softmax"]["mean"].shape == (1,)
        assert container.gradient_estimates["gumbel_softmax"]["mean"][0] == 1.2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_length_arrays(self):
        """Test container with zero-length arrays."""
        theta_values = jnp.array([])
        gradient_estimates = {
            "fd": {
                "theta": theta_values,
                "mean": jnp.array([]),
                "std": jnp.array([]),
                "time": jnp.array([]),
            }
        }

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        assert container.theta_values.shape == (0,)
        assert container.gradient_estimates["fd"]["mean"].shape == (0,)

    def test_large_arrays(self):
        """Test container with large arrays."""
        n_theta = 1000
        theta_values = jnp.linspace(-5, 5, n_theta)
        gradient_estimates = {
            "fd": {
                "theta": theta_values,
                "mean": jnp.ones(n_theta) * 0.5,
                "std": jnp.ones(n_theta) * 0.1,
                "time": jnp.ones(n_theta) * 0.001,
            }
        }

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        assert container.theta_values.shape == (n_theta,)
        assert container.gradient_estimates["fd"]["mean"].shape == (n_theta,)
        assert jnp.all(container.gradient_estimates["fd"]["mean"] == 0.5)

    def test_multidimensional_arrays(self):
        """Test container with multidimensional auxiliary arrays."""
        theta_values = jnp.array([1.0, 2.0])
        gradient_estimates = {"fd": {"mean": jnp.array([0.5, 1.0])}}

        # 2D discrete distributions (num_branches x num_theta)
        discrete_distributions = jnp.array([[0.6, 0.3], [0.4, 0.7]])

        container = ResultsContainer(
            theta_values=theta_values,
            gradient_estimates=gradient_estimates,
            discrete_distributions=discrete_distributions,
        )

        assert container.discrete_distributions is not None
        assert container.discrete_distributions.shape == (2, 2)
        assert jnp.array_equal(container.discrete_distributions, discrete_distributions)

    def test_overwriting_existing_data(self):
        """Test overwriting existing sampled points and other data."""
        theta_values = jnp.array([1.0])
        gradient_estimates = {"fd": {"mean": jnp.array([0.5])}}

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        # Add initial data
        initial_samples = jnp.array([0, 1])
        container.add_sampled_points("test", initial_samples)

        initial_expectations = jnp.array([1.0])
        container.add_expectation_values(initial_expectations)

        # Overwrite with new data
        new_samples = jnp.array([1, 0, 1])
        container.add_sampled_points("test", new_samples)

        new_expectations = jnp.array([2.0])
        container.add_expectation_values(new_expectations)

        # Check that data was overwritten
        assert container.sampled_points is not None
        assert container.expectation_values is not None
        assert jnp.array_equal(container.sampled_points["test"], new_samples)
        assert jnp.array_equal(container.expectation_values, new_expectations)


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_gradient_estimates_consistency(self):
        """Test that all arrays in gradient estimates have consistent shapes."""
        theta_values = jnp.array([1.0, 2.0, 3.0])

        # All arrays should have the same length as theta_values
        gradient_estimates = {
            "estimator1": {
                "theta": theta_values,
                "mean": jnp.array([0.5, 1.0, 1.5]),
                "std": jnp.array([0.1, 0.2, 0.3]),
                "time": jnp.array([0.001, 0.002, 0.003]),
            }
        }

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        estimator_data = container.gradient_estimates["estimator1"]
        base_shape = theta_values.shape

        assert estimator_data["theta"].shape == base_shape
        assert estimator_data["mean"].shape == base_shape
        assert estimator_data["std"].shape == base_shape
        assert estimator_data["time"].shape == base_shape

    def test_nan_and_inf_handling(self):
        """Test container handles NaN and Inf values."""
        theta_values = jnp.array([1.0, 2.0])
        gradient_estimates = {
            "test": {
                "theta": theta_values,
                "mean": jnp.array([jnp.nan, jnp.inf]),
                "std": jnp.array([0.1, jnp.nan]),
                "time": jnp.array([jnp.inf, 0.002]),
            }
        }

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        # Container should accept these values without error
        assert container.gradient_estimates["test"]["mean"].shape == (2,)
        assert jnp.isnan(container.gradient_estimates["test"]["mean"][0])
        assert jnp.isinf(container.gradient_estimates["test"]["mean"][1])
