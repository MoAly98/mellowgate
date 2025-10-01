"""Experimental framework for running parameter sweeps and comparing estimators.

This module provides tools for systematically evaluating gradient estimation
methods across different parameter values and configurations. The main components
allow users to:

- Define parameter sweeps with multiple theta values
- Configure multiple gradient estimators for comparison
- Run experiments with statistical repetitions
- Collect and organize results for analysis

The framework is designed to facilitate reproducible research and benchmarking
of different gradient estimation approaches on discrete optimization problems.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from mellowgate.api.estimators import (
    ReinforceState,
    finite_difference_gradient,
    gumbel_softmax_gradient,
    reinforce_gradient,
)
from mellowgate.api.functions import DiscreteProblem
from mellowgate.api.results import ResultsContainer
from mellowgate.logging import logger

ArrayType = jnp.ndarray


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
        ...     theta_values=jnp.linspace(-2, 2, 10),
        ...     num_repetitions=100,
        ...     estimator_configs={
        ...         "fd": {"cfg": FiniteDifferenceConfig(step_size=1e-3)}
        ...     }
        ... )
    """

    theta_values: ArrayType
    num_repetitions: int = 200
    estimator_configs: Optional[Dict[str, Dict[str, Any]]] = None


def _compute_sweep_statistics(
    gradient_samples: jnp.ndarray, num_repetitions: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute statistics for gradient samples with JIT compilation.

    Performance optimizations:
    - JIT compilation for statistical computations
    - Vectorized mean and std operations
    - Efficient overhead for repeated statistical calculations

    Args:
        gradient_samples: Shape (num_repetitions, num_theta_values)
        num_repetitions: Number of repetitions for static compilation

    Returns:
        tuple: (sample_mean, sample_std) each with shape (num_theta_values,)
    """
    if gradient_samples.shape[0] == 0:
        # Handle zero repetitions case
        theta_shape = gradient_samples.shape[1:]
        sample_mean = jnp.full(theta_shape, jnp.nan)
        sample_std = jnp.full(theta_shape, jnp.nan)
    else:
        # Compute statistics across repetitions (axis=0)
        sample_mean = gradient_samples.mean(axis=0)  # Mean across repetitions
        sample_std = gradient_samples.std(axis=0, ddof=1)  # Std across repetitions

    return sample_mean, sample_std


def run_parameter_sweep(
    discrete_problem: DiscreteProblem, sweep_config: Sweep
) -> Dict[str, ResultsContainer]:
    """Execute a parameter sweep experiment with multiple gradient estimators.

    Performance optimizations:
    - Precomputed invariants shared across estimators to avoid recomputation
    - Vectorized operations process all theta values simultaneously
    - Consistent static arguments for efficient JIT compilation
    - Batched statistical computations with JIT compilation

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
        "Starting parameter sweep over estimators: %s",
        list(sweep_config.estimator_configs.keys()),
    )

    results_containers = {}

    # Compute invariants once - shared across all estimators
    # These are used by all estimators to avoid recomputation
    discrete_distribution = discrete_problem.compute_function_values_deterministic(
        sweep_config.theta_values
    )

    # Compute expectation values once for all estimators
    expectation_values = discrete_problem.compute_expected_value(
        sweep_config.theta_values
    )

    # Generate a single random key for consistent sampling across estimators
    base_key = jax.random.PRNGKey(42)  # Use fixed seed for reproducibility

    # Compute sampled branch indices once for consistent sampling
    sampled_branch_indices = discrete_problem.sample_branch(
        sweep_config.theta_values,
        num_samples=sweep_config.num_repetitions,
        key=base_key,
    )

    # Store computed sampled points for reuse across estimators
    sampled_points = {"sampled_branch_indices": sampled_branch_indices}

    for estimator_name, estimator_spec in sweep_config.estimator_configs.items():
        logger.info(f"Running estimator: {estimator_name}")

        estimator_config = estimator_spec["cfg"]
        estimator_state = estimator_spec.get("state", None)

        # Start timing for the entire batch
        start_time = time.time()

        # Run multiple repetitions using vectorized gradient functions
        gradient_samples = []

        for rep in range(sweep_config.num_repetitions):
            # All estimators use JIT compilation for efficient computation
            if estimator_name == "fd":
                # Process all theta values at once with JIT compilation
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
                # Process all theta values at once with JIT compilation
                rep_gradients = reinforce_gradient(
                    discrete_problem,
                    sweep_config.theta_values,
                    estimator_config,
                    estimator_state,
                )
            elif estimator_name == "gs":
                # Process all theta values at once with JIT compilation
                rep_gradients = gumbel_softmax_gradient(
                    discrete_problem, sweep_config.theta_values, estimator_config
                )
            else:
                logger.error(f"Unknown estimator: {estimator_name}")
                raise ValueError(f"Unknown estimator: {estimator_name}")

            gradient_samples.append(rep_gradients)

        # Convert to JAX array and compute statistics using JIT-compiled function
        if len(gradient_samples) > 0:
            gradient_samples = jnp.array(gradient_samples)
            sample_mean, sample_std = _compute_sweep_statistics(
                gradient_samples, sweep_config.num_repetitions
            )
        else:
            # Handle zero repetitions case
            theta_shape = sweep_config.theta_values.shape
            sample_mean = jnp.full(theta_shape, jnp.nan)
            sample_std = jnp.full(theta_shape, jnp.nan)

        # Record timing for the entire batch
        elapsed_time = time.time() - start_time

        # Compute timing per theta value for backwards compatibility
        if len(sweep_config.theta_values) > 0:
            times_array = jnp.full(
                len(sweep_config.theta_values),
                elapsed_time / len(sweep_config.theta_values),
            )
        else:
            times_array = jnp.array([])

        # Log total time for the current estimator
        logger.info(
            f"Finished estimator: {estimator_name}, " f"Total Time: {elapsed_time:.2f}s"
        )

        # Store results for this estimator using computed invariants
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
            expectation_values=expectation_values,  # Reuse computed values
            discrete_distributions=discrete_distribution,  # Reuse computed values
            sampled_points=sampled_points,  # Reuse computed sampling
        )

    logger.info("Parameter sweep complete.")
    return results_containers
