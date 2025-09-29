"""Experimental framework for running gradient estimation sweeps.

This module provides tools for conducting systematic experiments with different
gradient estimation methods across parameter ranges, collecting statistics,
and timing information for performance analysis.

All operations are fully vectorized to efficiently process multiple parameter
values simultaneously, dramatically improving computational performance.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from mellowgate.api.estimators import (
    ReinforceState,
    finite_difference_gradient,
    gumbel_softmax_gradient,
    reinforce_gradient,
)
from mellowgate.api.functions import DiscreteProblem
from mellowgate.api.results import ResultsContainer
from mellowgate.logging import logger

ArrayType = np.ndarray


@dataclass
class Sweep:
    """Configuration for a parameter sweep experiment.

    Defines the experimental setup for testing gradient estimators across
    a range of parameter values with multiple repetitions for statistical
    analysis.

    Attributes:
        theta_values: 1D array of theta parameter values to test.
        num_repetitions: Number of repetitions per theta value for computing
                        mean and standard deviation statistics.
        estimator_configs: Dictionary mapping estimator names to their
                          configurations. Each entry should contain a 'cfg'
                          key with the estimator configuration, and optionally
                          a 'state' key for stateful estimators like REINFORCE.

    Examples:
        >>> import numpy as np
        >>> from mellowgate.api.estimators import FiniteDifferenceConfig
        >>>
        >>> sweep = Sweep(
        ...     theta_values=np.linspace(-2, 2, 10),
        ...     num_repetitions=100,
        ...     estimator_configs={
        ...         "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3)}
        ...     }
        ... )
    """

    theta_values: ArrayType
    num_repetitions: int = 200
    estimator_configs: Optional[Dict[str, Dict[str, Any]]] = None


def run_parameter_sweep(
    discrete_problem: DiscreteProblem, sweep_config: Sweep
) -> Dict[str, ResultsContainer]:
    """Execute a parameter sweep experiment with multiple gradient estimators.

    Fully vectorized version that processes all theta values at once for
    maximum computational efficiency.

    Args:
        discrete_problem: The discrete optimization problem to analyze.
        sweep_config: Configuration defining the experimental parameters.

    Returns:
        Dict[str, ResultsContainer]: A dictionary where each key is an estimator name,
        and the value is its corresponding ResultsContainer.
    """
    if sweep_config.estimator_configs is None:
        logger.error("Sweep estimator_configs is None. Nothing to run.")
        return {}

    logger.info(
        "Starting vectorized sweep over estimators: %s",
        list(sweep_config.estimator_configs.keys()),
    )

    results_containers = {}

    # Compute discrete distributions for all theta values at once
    discrete_distribution = discrete_problem.compute_function_values_deterministic(
        sweep_config.theta_values
    )

    for estimator_name, estimator_spec in sweep_config.estimator_configs.items():
        logger.info(f"Running vectorized estimator: {estimator_name}")

        estimator_config = estimator_spec["cfg"]
        estimator_state = estimator_spec.get("state", None)

        # Start timing for the entire batch
        start_time = time.time()

        # Vectorized computation: run multiple repetitions using the vectorized API
        gradient_samples = []

        for rep in range(sweep_config.num_repetitions):
            # Determine estimator type from name
            if estimator_name == "fd":
                # Process all theta values at once
                rep_gradients = finite_difference_gradient(
                    discrete_problem, sweep_config.theta_values, estimator_config
                )
            elif estimator_name == "reinforce":
                if not isinstance(estimator_state, ReinforceState):
                    logger.error(
                        "State for 'reinforce' must be a ReinforceState instance."
                    )
                    raise TypeError(
                        "State for 'reinforce' must be a ReinforceState instance."
                    )
                # Process all theta values at once
                rep_gradients = reinforce_gradient(
                    discrete_problem,
                    sweep_config.theta_values,
                    estimator_config,
                    estimator_state,
                )
            elif estimator_name == "gs":
                # Process all theta values at once
                rep_gradients = gumbel_softmax_gradient(
                    discrete_problem, sweep_config.theta_values, estimator_config
                )
            else:
                logger.error(f"Unknown estimator: {estimator_name}")
                raise ValueError(f"Unknown estimator: {estimator_name}")

            gradient_samples.append(rep_gradients)

        # Convert to numpy array: shape (num_repetitions, num_theta_values)
        if len(gradient_samples) > 0:
            gradient_samples = np.array(gradient_samples)
            # Compute statistics across repetitions (axis=0)
            sample_mean = gradient_samples.mean(axis=0)  # Mean across repetitions
            sample_std = gradient_samples.std(axis=0, ddof=1)  # Std across repetitions
        else:
            # Handle zero repetitions case
            theta_shape = sweep_config.theta_values.shape
            sample_mean = np.full(theta_shape, np.nan)
            sample_std = np.full(theta_shape, np.nan)

        # Record timing for the entire batch
        elapsed_time = time.time() - start_time

        # Compute timing per theta value for backwards compatibility
        if len(sweep_config.theta_values) > 0:
            times_array = np.full(
                len(sweep_config.theta_values),
                elapsed_time / len(sweep_config.theta_values),
            )
        else:
            times_array = np.array([])

        # Log total time for the current estimator
        logger.info(
            f"Finished vectorized estimator: {estimator_name}, "
            f"Total Time: {elapsed_time:.2f}s"
        )

        # Compute sampled branch indices for each theta value (vectorized)
        sampled_branch_indices = discrete_problem.sample_branch(
            sweep_config.theta_values, num_samples=sweep_config.num_repetitions
        )

        # Compute expectation values of the function itself (vectorized)
        expectation_values = discrete_problem.compute_expected_value(
            sweep_config.theta_values
        )

        # Store the sampled indices in ResultsContainer
        sampled_points = {"sampled_branch_indices": sampled_branch_indices}

        # Store results for this estimator
        results_containers[estimator_name] = ResultsContainer(
            gradient_estimates={
                estimator_name: {
                    "theta": sweep_config.theta_values.copy(),
                    "mean": sample_mean,
                    "std": sample_std,
                    "time": times_array,
                }
            },
            theta_values=sweep_config.theta_values,
            expectation_values=expectation_values,
            discrete_distributions=discrete_distribution,
            sampled_points=sampled_points,
        )

    logger.info("Vectorized sweep complete.")
    return results_containers
