"""Visualization functions for discrete optimization problems.

This module provides plotting utilities for visualizing discrete optimization
problems, their probability distributions, and function behaviors. The functions
help users understand the structure of their problems and visualize how
probabilities and function values change across parameter spaces.

The plotting functions integrate with matplotlib and are designed to work
seamlessly with the core DiscreteProblem class.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from mellowgate.api.results import ResultsContainer
from mellowgate.utils.outputs import OutputManager


def plot_combined_overlay(
    results_dict: dict[str, ResultsContainer], output_manager: OutputManager
) -> None:
    """Overlay sampled points, expectation values, and discrete distributions
    for all estimators.

    Creates combined overlay plots for each estimator showing the relationship between
    expectation values, discrete distributions, and sampling results. Optimized for
    vectorized data processing.

    Args:
        results_dict: Dictionary mapping estimator names to ResultsContainer objects.
        output_manager: OutputManager instance for handling file paths.
    """

    for estimator_name, results in results_dict.items():
        # Generate output path specific to the estimator
        output_path = str(
            output_manager.base_directory / f"combined_overlay_{estimator_name}.pdf"
        )

        plt.figure(figsize=(10, 6))

        # Plot expectation values
        if results.expectation_values is not None:
            expectation_values = jnp.squeeze(results.expectation_values)  # Ensure 1D
            plt.plot(
                results.theta_values,
                expectation_values,
                label=f"{estimator_name} Expectation Value",
                linewidth=2,
            )

        # Plot discrete distributions
        if results.discrete_distributions is not None:
            discrete_distributions = jnp.squeeze(
                results.discrete_distributions
            )  # Ensure 1D
            plt.plot(
                results.theta_values,
                discrete_distributions,
                label=f"{estimator_name} Discrete Distribution",
                linestyle="--",
                linewidth=1.5,
            )

        # Plot sampled points
        if results.sampled_points:
            sampled_branch_indices = results.sampled_points.get(
                "sampled_branch_indices", []
            )

            # Handle vectorized sampled indices
            if len(sampled_branch_indices) > 0:
                # Convert to array and handle different shapes
                sampled_indices_array = jnp.asarray(sampled_branch_indices)

                if sampled_indices_array.ndim == 2:
                    # Shape: (num_theta, num_samples) - take mean or first sample
                    sampled_for_plot = sampled_indices_array[
                        :, 0
                    ]  # Take first sample for each theta
                    thetas_for_plot = results.theta_values
                else:
                    # Shape: (num_samples,) - original behavior
                    sampled_for_plot = sampled_indices_array
                    thetas_for_plot = jnp.linspace(
                        jnp.min(results.theta_values),
                        jnp.max(results.theta_values),
                        len(sampled_for_plot),
                    )

                plt.scatter(
                    thetas_for_plot,
                    sampled_for_plot,
                    alpha=0.3,
                    s=10,
                    label=f"{estimator_name} Sampled Points",
                )

        plt.xlabel("Theta")
        plt.ylabel("Values")
        plt.title(f"Combined Overlay - {estimator_name}")
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
