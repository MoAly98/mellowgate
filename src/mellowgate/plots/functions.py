import matplotlib.pyplot as plt

from mellowgate.api.results import ResultsContainer
from mellowgate.utils.outputs import OutputManager


def plot_combined_overlay(
    results_dict: dict[str, ResultsContainer], output_manager: OutputManager
) -> None:
    """Overlay sampled points, expectation values, and
    discrete distributions for all estimators."""
    # Correct output path generation
    output_path = str(output_manager.base_directory / "combined_overlay.pdf")

    plt.figure(figsize=(10, 6))

    for estimator_name, results in results_dict.items():
        # Plot expectation values
        if results.expectation_values is not None:
            # Ensure expectation_values is treated as a 1D array
            if len(results.expectation_values.shape) == 1:
                plt.plot(
                    results.theta_values,
                    results.expectation_values,
                    label=f"{estimator_name} Expectation Value",
                    linewidth=2,
                )
            else:
                for i, expectation_row in enumerate(results.expectation_values):
                    plt.plot(
                        results.theta_values,
                        expectation_row,
                        label=f"{estimator_name} Expectation Value (Estimator {i})",
                        linewidth=2,
                    )

        # Plot discrete distributions
        if results.discrete_distributions is not None:
            for branch_name, distribution in results.discrete_distributions.items():
                plt.plot(
                    results.theta_values,
                    distribution,
                    label=f"{estimator_name} Discrete Distribution ({branch_name})",
                    linestyle="--",
                    linewidth=1.5,
                )

        # Plot sampled points
        if results.sampled_points:
            mean_sampled_points = results.sampled_points.get("mean_sampled_points", [])
            for i, theta in enumerate(results.theta_values):
                plt.scatter(
                    [theta],
                    [mean_sampled_points[i]],
                    alpha=0.3,
                    s=10,
                    label=f"{estimator_name} Sampled Points" if i == 0 else None,
                )

    plt.xlabel("Theta")
    plt.ylabel("Values")
    plt.title(
        "Overlay of Sampled Points, Expectation Values, and Discrete Distributions"
    )
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
