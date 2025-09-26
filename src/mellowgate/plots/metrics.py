"""Visualization tools for gradient estimation experiment results.

This module provides plotting functions for analyzing and visualizing the
performance of different gradient estimation methods, including accuracy
comparisons, bias-variance decomposition, and timing analysis.
"""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from mellowgate.api.results import ResultsContainer
from mellowgate.logging import logger
from mellowgate.utils.outputs import OutputManager


def plot_gradient_estimates_vs_truth(
    results_dict: dict[str, ResultsContainer],
    true_gradient_function: Callable[[float], Optional[float]],
    output_manager: OutputManager,
    plot_title: Optional[str] = None,
    output_filename: Optional[str] = None,
    output_subdirectory: Optional[str] = None,
) -> None:
    """Plot gradient estimate means with error bars against true gradients
        for all estimators.

    Creates a visualization comparing the estimated gradients from different
    methods against the true analytical gradient, with error bars showing
    the standard deviation across repetitions.

    Args:
        results_dict: Dictionary mapping estimator names to ResultsContainer
                       objects containing experiment results.
        true_gradient_function: Function that computes the true analytical
                               gradient for a given theta value. May return
                               None if unavailable.
        output_manager: OutputManager instance for handling file paths.
        plot_title: Optional custom title for the plot.
        output_filename: Optional custom filename for the saved plot.
        output_subdirectory: Optional subdirectory within the output manager's base.

    Examples:
        >>> plot_gradient_estimates_vs_truth(
        ...     results, problem.compute_exact_gradient, output_manager
        ... )
    """
    plot_title = plot_title or "Gradient estimates vs analytical truth"
    output_filename = output_filename or "gradient_estimates_vs_truth.pdf"
    output_subdirectory = output_subdirectory or "metrics"

    plt.figure(figsize=(7, 4))

    # Plot each estimator's results
    for estimator_name, results in results_dict.items():
        for estimator_name, estimator_results in results.gradient_estimates.items():
            theta_values = results.theta_values
            mean_estimates = estimator_results["mean"]
            std_estimates = estimator_results["std"]

            # Ensure mean and std estimates are 1D arrays
            mean_estimates = np.squeeze(mean_estimates)
            std_estimates = np.squeeze(std_estimates)

            # Plot mean with error bars
            plt.plot(
                theta_values, mean_estimates, marker="o", label=f"{estimator_name}"
            )
            plt.fill_between(
                theta_values,
                mean_estimates - std_estimates,
                mean_estimates + std_estimates,
                alpha=0.15,
            )

    # Plot true gradient if available
    true_gradients = np.array(
        [true_gradient_function(float(theta)) for theta in results.theta_values]
    )

    if None in true_gradients:
        logger.warning(
            "Some true gradient values are missing. Skipping analytical gradient plot."
        )
    else:
        plt.plot(
            results.theta_values,
            true_gradients,
            "r--",
            linewidth=2,
            label="analytical gradient",
        )

    # Format and save plot
    plt.xlabel("theta")
    plt.ylabel("gradient estimate")
    plt.title(plot_title)
    plt.legend(frameon=False)
    plt.grid(alpha=0.4)
    plt.tight_layout()

    output_path = output_manager.get_path(output_subdirectory, filename=output_filename)
    logger.info(f"Saving gradient comparison plot to {output_path}")
    plt.savefig(output_path)
    plt.close()


def plot_bias_variance_mse_analysis(
    results_dict: dict[str, ResultsContainer],
    true_gradient_function: Callable[[float], Optional[float]],
    output_manager: OutputManager,
    plot_title: Optional[str] = None,
    output_filename: Optional[str] = None,
    output_subdirectory: Optional[str] = None,
) -> None:
    """Plot bias, variance, and MSE decomposition for all estimators.

    Creates a log-scale visualization showing the bias-variance-MSE tradeoff
    for different gradient estimation methods. This helps understand the
    sources of error in each estimator.

    Args:
        results_dict: Dictionary mapping estimator names to ResultsContainer
                       objects containing experiment results.
        true_gradient_function: Function that computes the true analytical gradient.
        output_manager: OutputManager instance for handling file paths.
        plot_title: Optional custom title for the plot.
        output_filename: Optional custom filename for the saved plot.
        output_subdirectory: Optional subdirectory within the output manager's base.

    Notes:
        - Bias = |estimated_mean - true_gradient|
        - Variance = standard_deviation²
        - MSE = bias² + variance
    """
    plot_title = plot_title or "Bias-Variance-MSE decomposition"
    output_filename = output_filename or "bias_variance_mse.pdf"
    output_subdirectory = output_subdirectory or "metrics"

    plt.figure(figsize=(7, 4))

    for estimator_name, results in results_dict.items():
        for estimator_name, estimator_results in results.gradient_estimates.items():
            theta_values = results.theta_values
            mean_estimates = estimator_results["mean"]
            std_estimates = estimator_results["std"]

            # Compute true gradients
            true_gradients = np.array(
                [true_gradient_function(float(theta)) for theta in theta_values]
            )

            if None in true_gradients:
                logger.warning(
                    f"Some true gradient values are missing for {estimator_name}. "
                    f"Skipping bias/variance/MSE analysis."
                )
                continue

            # Ensure mean and std estimates are 1D arrays
            mean_estimates = np.squeeze(mean_estimates)
            std_estimates = np.squeeze(std_estimates)

            # Compute bias, variance, and MSE
            bias_values = np.abs(mean_estimates - true_gradients)  # type: ignore
            variance_values = std_estimates**2
            mse_values = bias_values**2 + variance_values

            # Ensure computed arrays are 1D
            bias_values = np.squeeze(bias_values)
            variance_values = np.squeeze(variance_values)
            mse_values = np.squeeze(mse_values)

            # Plot on log scale
            plt.semilogy(theta_values, bias_values, label=f"{estimator_name} bias")
            plt.semilogy(
                theta_values, std_estimates, "--", label=f"{estimator_name} std"
            )
            plt.semilogy(theta_values, mse_values, ":", label=f"{estimator_name} MSE")

    # Format and save plot
    plt.xlabel("theta")
    plt.ylabel("magnitude (log scale)")
    plt.title(plot_title)
    plt.legend(frameon=False, ncol=3)
    plt.grid(alpha=0.4)
    plt.tight_layout()

    output_path = output_manager.get_path(output_subdirectory, filename=output_filename)
    logger.info(f"Saving bias-variance-MSE analysis to {output_path}")
    plt.savefig(output_path)
    plt.close()


def plot_computational_time_analysis(
    results_dict: dict[str, ResultsContainer],
    output_manager: OutputManager,
    plot_title: Optional[str] = None,
    output_filename: Optional[str] = None,
    output_subdirectory: Optional[str] = None,
) -> None:
    """Plot computational time comparison across all gradient estimators."""
    plot_title = plot_title or "Computational time per theta value"
    output_filename = output_filename or "computational_time.pdf"
    output_subdirectory = output_subdirectory or "metrics"

    plt.figure(figsize=(7, 4))

    for estimator_name, results in results_dict.items():
        for estimator_name, estimator_results in results.gradient_estimates.items():
            theta_values = results.theta_values
            time_values = estimator_results["time"]

            # Ensure time values are 1D arrays
            time_values = np.squeeze(time_values)

            plt.semilogy(
                theta_values, time_values, marker="o", label=f"{estimator_name}"
            )

    # Format and save plot
    plt.xlabel("theta")
    plt.ylabel("computation time (seconds, log scale)")
    plt.title(plot_title)
    plt.legend(frameon=False)
    plt.grid(alpha=0.4)
    plt.tight_layout()

    output_path = output_manager.get_path(output_subdirectory, filename=output_filename)
    logger.info(f"Saving computational time analysis to {output_path}")
    plt.savefig(output_path)
    plt.close()
