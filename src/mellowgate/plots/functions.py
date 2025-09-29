import matplotlib.pyplot as plt
import numpy as np

from mellowgate.api.results import ResultsContainer
from mellowgate.utils.outputs import OutputManager


def plot_combined_overlay(
    results_dict: dict[str, ResultsContainer], output_manager: OutputManager
) -> None:
    """Overlay sampled points, expectation values, and
    discrete distributions for all estimators."""

    for estimator_name, results in results_dict.items():
        # Generate output path specific to the estimator
        output_path = str(
            output_manager.base_directory / f"combined_overlay_{estimator_name}.pdf"
        )

        plt.figure(figsize=(10, 6))

        # Plot expectation values
        if results.expectation_values is not None:
            plt.plot(
                results.theta_values,
                results.expectation_values,
                label=f"{estimator_name} Expectation Value",
                linewidth=2,
            )

        # Plot discrete distributions
        if results.discrete_distributions is not None:
            plt.plot(
                results.theta_values,
                results.discrete_distributions,
                label=f"{estimator_name} Discrete Distribution",
                linestyle="--",
                linewidth=1.5,
            )

        # Plot sampled points
        if results.sampled_points:
            sampled_branch_indices = results.sampled_points.get(
                "sampled_branch_indices", []
            )
            thetas = np.linspace(
                np.min(results.theta_values),
                np.max(results.theta_values),
                len(sampled_branch_indices),
            )
            plt.scatter(
                thetas,
                sampled_branch_indices,
                alpha=0.3,
                s=10,
                label=f"{estimator_name} Sampled Points",
            )

        plt.xlabel("Theta")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
