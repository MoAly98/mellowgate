"""Visualization tools for gradient estimation performance metrics.

This module provides plotting utilities for analyzing and comparing the
performance of different gradient estimation methods. The functions help
visualize convergence properties, variance characteristics, and comparative
performance across different parameter ranges.

The visualizations support experimental analysis and help researchers understand
the trade-offs between different estimation approaches.
"""

from typing import Callable, Optional, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt

from mellowgate.api.results import ResultsContainer
from mellowgate.logging import logger
from mellowgate.utils.outputs import OutputManager


def plot_gradient_estimates_vs_truth(
    results_dict: dict[str, ResultsContainer],
    true_gradient_function: Callable[
        [Union[float, jnp.ndarray]], Union[Optional[float], Optional[jnp.ndarray]]
    ],
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
        true_gradient_function: Vectorized function that computes the true analytical
                               gradient for given theta value(s). May return
                               None if unavailable. Supports both scalar and
                               array inputs.
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

    # Get theta values from first result (they should be the same for all estimators)
    first_result = next(iter(results_dict.values()))
    theta_values = first_result.theta_values

    # Define distinct colors for each estimator
    import matplotlib.cm as cm

    tab10_cmap = cm.get_cmap("tab10")
    colors = [tab10_cmap(i) for i in range(10)]  # Get 10 distinct colors from tab10

    # Plot each estimator's results
    color_idx = 0
    for estimator_name, results in results_dict.items():
        for estimator_name, estimator_results in results.gradient_estimates.items():
            mean_estimates = estimator_results["mean"]
            std_estimates = estimator_results["std"]

            # Ensure mean and std estimates are 1D arrays
            mean_estimates = jnp.squeeze(mean_estimates)
            std_estimates = jnp.squeeze(std_estimates)

            # Get color for this estimator
            color = colors[color_idx % len(colors)]

            # Plot mean with error bars
            plt.plot(
                theta_values,
                mean_estimates,
                marker="o",
                color=color,
                label=f"{estimator_name}",
                markersize=3,
                linewidth=1.5,
            )
            plt.fill_between(
                theta_values,
                mean_estimates - std_estimates,
                mean_estimates + std_estimates,
                color=color,
                alpha=0.15,
            )

            color_idx += 1

    # Plot true gradient if available - use vectorized call
    try:
        true_gradients = true_gradient_function(theta_values)

        if true_gradients is None:
            logger.warning(
                "True gradient function returned None. Skipping analytical "
                "gradient plot."
            )
        else:
            # Handle both scalar and array returns
            if jnp.isscalar(true_gradients):
                true_gradients = jnp.full_like(theta_values, true_gradients)
            else:
                true_gradients = jnp.squeeze(true_gradients)

            plt.plot(
                theta_values,
                true_gradients,
                "r--",
                linewidth=1,
                label="analytical gradient",
            )
    except Exception as e:
        logger.warning(
            f"Could not compute true gradients: {e}. Skipping analytical gradient plot."
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
    true_gradient_function: Callable[
        [Union[float, jnp.ndarray]], Union[Optional[float], Optional[jnp.ndarray]]
    ],
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
        true_gradient_function: Vectorized function that computes the true analytical
                               gradient. Supports both scalar and array inputs.
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

    # Get theta values from first result
    first_result = next(iter(results_dict.values()))
    theta_values = first_result.theta_values

    # Compute true gradients once using vectorized call
    try:
        true_gradients = true_gradient_function(theta_values)

        if true_gradients is None:
            logger.warning(
                "True gradient function returned None. Skipping "
                "bias/variance/MSE analysis."
            )
            return

        # Handle both scalar and array returns
        if jnp.isscalar(true_gradients):
            true_gradients = jnp.full_like(theta_values, true_gradients)
        else:
            true_gradients = jnp.squeeze(true_gradients)

    except Exception as e:
        logger.warning(
            f"Could not compute true gradients: {e}. Skipping "
            f"bias/variance/MSE analysis."
        )
        return

    # Define distinct colors and line styles for each estimator and metric
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    tab10_cmap = cm.get_cmap("tab10")
    base_colors = [
        tab10_cmap(i) for i in range(10)
    ]  # Get 10 distinct base colors from tab10

    def create_color_variations(base_color):
        """Create 3 visually distinct variations of the same base color."""
        # Convert to HSV to manipulate saturation and value
        hsv = mcolors.rgb_to_hsv(base_color[:3])

        # Create variations: dark, medium, light
        dark = mcolors.hsv_to_rgb(
            [hsv[0], min(1.0, hsv[1] * 1.2), hsv[2] * 0.7]
        )  # More saturated, darker
        medium = base_color[:3]  # Original color
        light = mcolors.hsv_to_rgb(
            [hsv[0], hsv[1] * 0.6, min(1.0, hsv[2] * 1.2)]
        )  # Less saturated, lighter

        return {
            "bias": dark,  # Darkest for bias
            "variance": medium,  # Medium for variance
            "mse": light,  # Lightest for MSE
        }

    line_styles = {
        "bias": "-",  # Solid line for bias
        "variance": "--",  # Dashed line for variance
        "mse": ":",  # Dotted line for MSE
    }

    color_idx = 0
    for estimator_name, results in results_dict.items():
        for estimator_name, estimator_results in results.gradient_estimates.items():
            mean_estimates = estimator_results["mean"]
            std_estimates = estimator_results["std"]

            # Ensure mean and std estimates are 1D arrays
            mean_estimates = jnp.squeeze(mean_estimates)
            std_estimates = jnp.squeeze(std_estimates)

            # Compute bias, variance, and MSE
            bias_values = jnp.abs(mean_estimates - true_gradients)
            variance_values = std_estimates**2
            mse_values = bias_values**2 + variance_values

            # Ensure computed arrays are 1D
            bias_values = jnp.squeeze(bias_values)
            variance_values = jnp.squeeze(variance_values)
            mse_values = jnp.squeeze(mse_values)

            # Get base color for this estimator and create variations
            base_color = base_colors[color_idx % len(base_colors)]
            color_variations = create_color_variations(base_color)

            # Plot on log scale (add small epsilon to avoid log(0))
            epsilon = 1e-10
            plt.semilogy(
                theta_values,
                bias_values + epsilon,
                linestyle=line_styles["bias"],
                color=color_variations["bias"],
                linewidth=1.5,
                label=f"{estimator_name} bias",
            )
            plt.semilogy(
                theta_values,
                variance_values + epsilon,
                linestyle=line_styles["variance"],
                color=color_variations["variance"],
                linewidth=1.5,
                label=f"{estimator_name} variance",
            )
            plt.semilogy(
                theta_values,
                mse_values + epsilon,
                linestyle=line_styles["mse"],
                color=color_variations["mse"],
                linewidth=1.5,
                label=f"{estimator_name} MSE",
            )

            color_idx += 1

    # Format and save plot
    plt.xlabel("theta")
    plt.ylabel("magnitude (log scale)")
    plt.title(plot_title)
    plt.grid(alpha=0.4)

    # Place legend below the x-axis label
    plt.legend(frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.2), loc="upper center")
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

    # Define distinct colors for each estimator
    import matplotlib.cm as cm

    tab10_cmap = cm.get_cmap("tab10")
    colors = [tab10_cmap(i) for i in range(10)]  # Get 10 distinct colors from tab10

    color_idx = 0
    for estimator_name, results in results_dict.items():
        for estimator_name, estimator_results in results.gradient_estimates.items():
            theta_values = results.theta_values
            time_values = estimator_results["time"]

            # Ensure time values are 1D arrays
            time_values = jnp.squeeze(time_values)

            # Get color for this estimator
            color = colors[color_idx % len(colors)]

            plt.semilogy(
                theta_values,
                time_values,
                marker="o",
                color=color,
                label=f"{estimator_name}",
                markersize=4,
                linewidth=1.5,
            )

            color_idx += 1

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
