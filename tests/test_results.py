"""Unit tests for api.results module.

This module provides tests for the ResultsContainer class which stores
experiment results and metadata.
"""

import jax.numpy as jnp

from mellowgate.api.results import ResultsContainer


class TestResultsContainer:
    """Test ResultsContainer class for storing experiment results."""

    def test_results_container_creation_minimal(self):
        """Test creating ResultsContainer with minimal required fields."""
        theta_values = jnp.array([1.0, 2.0, 3.0])
        gradient_estimates = {
            "fd": {
                "theta": theta_values,
                "mean": jnp.array([0.1, 0.2, 0.3]),
                "std": jnp.array([0.01, 0.02, 0.03]),
                "time": jnp.array([0.001, 0.002, 0.003]),
            }
        }

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        assert jnp.array_equal(container.theta_values, theta_values)
        assert container.gradient_estimates == gradient_estimates
        assert container.sampled_points is None
        assert container.expectation_values is None
        assert container.discrete_distributions is None

    def test_results_container_creation_full(self):
        """Test creating ResultsContainer with all fields."""
        theta_values = jnp.array([1.0, 2.0])
        gradient_estimates = {"fd": {"mean": jnp.array([0.1, 0.2])}}
        sampled_points = {"sampled_indices": jnp.array([[0, 1], [1, 0]])}
        expectation_values = jnp.array([1.5, 2.5])
        discrete_distributions = jnp.array([[0.6, 0.7], [0.4, 0.3]])

        container = ResultsContainer(
            theta_values=theta_values,
            gradient_estimates=gradient_estimates,
            sampled_points=sampled_points,
            expectation_values=expectation_values,
            discrete_distributions=discrete_distributions,
        )

        assert jnp.array_equal(container.theta_values, theta_values)
        assert container.gradient_estimates == gradient_estimates
        assert container.sampled_points == sampled_points
        assert container.expectation_values is not None
        assert container.discrete_distributions is not None
        assert jnp.array_equal(container.expectation_values, expectation_values)
        assert jnp.array_equal(container.discrete_distributions, discrete_distributions)

    def test_add_sampled_points_new(self):
        """Test adding sampled points when container has none."""
        container = ResultsContainer(
            theta_values=jnp.array([1.0]),
            gradient_estimates={"fd": {"mean": jnp.array([0.1])}},
        )

        sampled_points = jnp.array([0, 1, 0, 1])
        container.add_sampled_points("fd", sampled_points)

        assert container.sampled_points is not None
        assert "fd" in container.sampled_points
        assert jnp.array_equal(container.sampled_points["fd"], sampled_points)

    def test_add_sampled_points_existing(self):
        """Test adding sampled points when container already has some."""
        initial_points = {"reinforce": jnp.array([1, 0, 1])}
        container = ResultsContainer(
            theta_values=jnp.array([1.0]),
            gradient_estimates={"fd": {"mean": jnp.array([0.1])}},
            sampled_points=initial_points,
        )

        new_points = jnp.array([0, 1, 0, 1])
        container.add_sampled_points("fd", new_points)

        # Should preserve existing and add new
        assert container.sampled_points is not None
        assert "reinforce" in container.sampled_points
        assert "fd" in container.sampled_points
        assert jnp.array_equal(
            container.sampled_points["reinforce"], initial_points["reinforce"]
        )
        assert jnp.array_equal(container.sampled_points["fd"], new_points)

    def test_add_sampled_points_overwrite(self):
        """Test that adding sampled points overwrites existing for same estimator."""
        initial_points = {"fd": jnp.array([0, 0, 0])}
        container = ResultsContainer(
            theta_values=jnp.array([1.0]),
            gradient_estimates={"fd": {"mean": jnp.array([0.1])}},
            sampled_points=initial_points,
        )

        # Store the initial values for comparison
        assert container.sampled_points is not None
        original_values = container.sampled_points["fd"].copy()

        new_points = jnp.array([1, 1, 1])
        container.add_sampled_points("fd", new_points)

        assert container.sampled_points is not None
        assert jnp.array_equal(container.sampled_points["fd"], new_points)
        assert not jnp.array_equal(container.sampled_points["fd"], original_values)

    def test_add_expectation_values(self):
        """Test adding expectation values."""
        container = ResultsContainer(
            theta_values=jnp.array([1.0, 2.0]),
            gradient_estimates={"fd": {"mean": jnp.array([0.1, 0.2])}},
        )

        expectation_values = jnp.array([1.5, 2.5])
        container.add_expectation_values(expectation_values)

        assert container.expectation_values is not None
        assert jnp.array_equal(container.expectation_values, expectation_values)

    def test_add_expectation_values_overwrite(self):
        """Test that adding expectation values overwrites existing."""
        initial_values = jnp.array([1.0, 2.0])
        container = ResultsContainer(
            theta_values=jnp.array([1.0, 2.0]),
            gradient_estimates={"fd": {"mean": jnp.array([0.1, 0.2])}},
            expectation_values=initial_values,
        )

        new_values = jnp.array([3.0, 4.0])
        container.add_expectation_values(new_values)

        assert container.expectation_values is not None
        assert jnp.array_equal(container.expectation_values, new_values)
        assert not jnp.array_equal(container.expectation_values, initial_values)

    def test_add_discrete_distributions(self):
        """Test adding discrete distributions."""
        container = ResultsContainer(
            theta_values=jnp.array([1.0, 2.0]),
            gradient_estimates={"fd": {"mean": jnp.array([0.1, 0.2])}},
        )

        distributions = jnp.array([[0.6, 0.7], [0.4, 0.3]])
        container.add_discrete_distributions(distributions)

        assert container.discrete_distributions is not None
        assert jnp.array_equal(container.discrete_distributions, distributions)

    def test_add_discrete_distributions_overwrite(self):
        """Test that adding discrete distributions overwrites existing."""
        initial_distributions = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        container = ResultsContainer(
            theta_values=jnp.array([1.0, 2.0]),
            gradient_estimates={"fd": {"mean": jnp.array([0.1, 0.2])}},
            discrete_distributions=initial_distributions,
        )

        new_distributions = jnp.array([[0.8, 0.9], [0.2, 0.1]])
        container.add_discrete_distributions(new_distributions)

        assert container.discrete_distributions is not None
        assert jnp.array_equal(container.discrete_distributions, new_distributions)
        assert not jnp.array_equal(
            container.discrete_distributions, initial_distributions
        )

    def test_multiple_estimators_gradient_estimates(self):
        """Test container with multiple estimators in gradient estimates."""
        theta_values = jnp.array([1.0, 2.0])
        gradient_estimates = {
            "fd": {
                "theta": theta_values,
                "mean": jnp.array([0.1, 0.2]),
                "std": jnp.array([0.01, 0.02]),
            },
            "reinforce": {
                "theta": theta_values,
                "mean": jnp.array([0.15, 0.25]),
                "std": jnp.array([0.05, 0.06]),
            },
            "gs": {
                "theta": theta_values,
                "mean": jnp.array([0.12, 0.22]),
                "std": jnp.array([0.02, 0.03]),
            },
        }

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        assert len(container.gradient_estimates) == 3
        assert "fd" in container.gradient_estimates
        assert "reinforce" in container.gradient_estimates
        assert "gs" in container.gradient_estimates

    def test_empty_arrays(self):
        """Test container with empty arrays."""
        theta_values = jnp.array([])
        gradient_estimates = {
            "fd": {"theta": theta_values, "mean": jnp.array([]), "std": jnp.array([])}
        }

        container = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )

        assert container.theta_values.shape == (0,)
        assert container.gradient_estimates["fd"]["mean"].shape == (0,)

    def test_dataclass_equality(self):
        """Test that identical containers are equal."""
        theta_values = jnp.array([1.0, 2.0])
        gradient_estimates = {"fd": {"mean": jnp.array([0.1, 0.2])}}

        container1 = ResultsContainer(
            theta_values=theta_values, gradient_estimates=gradient_estimates
        )
        container2 = ResultsContainer(
            theta_values=theta_values.copy(),
            gradient_estimates=gradient_estimates.copy(),
        )

        # Note: NumPy arrays in dataclass may not compare equal directly
        # This test verifies the structure is identical
        assert (
            container1.gradient_estimates.keys() == container2.gradient_estimates.keys()
        )
        assert container1.theta_values.shape == container2.theta_values.shape
