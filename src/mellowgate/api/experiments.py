"""Experimental framework for running gradient estimation sweeps.

This module provides tools for conducting systematic experiments with different
gradient estimation methods across parameter ranges, collecting statistics,
and timing information for performance analysis.
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from mellowgate.api.functions import DiscreteProblem
from mellowgate.api.estimators import (
    finite_difference_gradient,
    reinforce_gradient,
    ReinforceState,
    gumbel_softmax_gradient,
)
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

    # Backward compatibility aliases
    @property
    def thetas(self) -> ArrayType:
        """Backward compatibility alias for theta_values."""
        return self.theta_values

    @property
    def repeats(self) -> int:
        """Backward compatibility alias for num_repetitions."""
        return self.num_repetitions

    @property
    def estimators(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Backward compatibility alias for estimator_configs."""
        return self.estimator_configs

    # Constructor functions for backward compatibility


def _patch_sweep_init() -> None:
    """Patch Sweep.__init__ to accept old parameter names."""
    original_init = Sweep.__init__

    def new_init(
        self,
        theta_values=None,
        num_repetitions=200,
        estimator_configs=None,
        thetas=None,
        repeats=None,
        estimators=None,
    ):
        # Support both old and new parameter names
        theta_vals = theta_values if theta_values is not None else thetas
        num_reps = num_repetitions if repeats is None else repeats
        estimator_cfgs = (
            estimator_configs if estimator_configs is not None else estimators
        )

        if theta_vals is None:
            raise ValueError("Must provide either 'theta_values' or 'thetas' parameter")

        original_init(self, theta_vals, num_reps, estimator_cfgs)

    Sweep.__init__ = new_init


# Apply the patch
_patch_sweep_init()


def run_parameter_sweep(
    discrete_problem: DiscreteProblem, sweep_config: Sweep
) -> Dict[str, Dict[str, ArrayType]]:
    """Execute a parameter sweep experiment with multiple gradient estimators.

    Runs the specified gradient estimators across all theta values in the sweep,
    collecting timing, mean, and standard deviation statistics for each
    estimator and parameter value.

    Args:
        discrete_problem: The discrete optimization problem to analyze.
        sweep_config: Configuration defining the experimental parameters.

    Returns:
        Dict containing results for each estimator. Structure:
        {
            "estimator_name": {
                "theta": array of theta values,
                "mean": array of gradient estimate means,
                "std": array of gradient estimate standard deviations,
                "time": array of computation times per theta
            }
        }

    Raises:
        ValueError: If an unknown estimator name is specified.
        TypeError: If REINFORCE estimator is used without proper state object.

    Examples:
        >>> results = run_parameter_sweep(problem, sweep)
        >>> fd_means = results["fd"]["mean"]
        >>> fd_stds = results["fd"]["std"]
    """
    if sweep_config.estimator_configs is None:
        logger.error("Sweep estimator_configs is None. Nothing to run.")
        return {}

    logger.info(
        "Starting sweep over estimators: %s",
        list(sweep_config.estimator_configs.keys()),
    )

    results = {}

    for estimator_name, estimator_spec in sweep_config.estimator_configs.items():
        logger.info(f"Running estimator: {estimator_name}")

        estimator_config = estimator_spec["cfg"]
        estimator_state = estimator_spec.get("state", None)

        means_list, stds_list, times_list = [], [], []

        for theta_value in sweep_config.theta_values:
            logger.info(f"  Theta: {theta_value}")

            gradient_samples = []
            start_time = time.time()

            for repetition_index in range(sweep_config.num_repetitions):
                # Log progress periodically
                if repetition_index % max(1, sweep_config.num_repetitions // 5) == 0:
                    logger.debug(
                        f"    Repetition {repetition_index+1}/"
                        f"{sweep_config.num_repetitions}"
                    )

                # Compute gradient estimate based on estimator type
                if estimator_name == "fd":
                    gradient_estimate = finite_difference_gradient(
                        discrete_problem, float(theta_value), estimator_config
                    )
                elif estimator_name == "reinforce":
                    if not isinstance(estimator_state, ReinforceState):
                        logger.error(
                            "State for 'reinforce' must be a ReinforceState instance."
                        )
                        raise TypeError(
                            "State for 'reinforce' must be a ReinforceState instance."
                        )
                    gradient_estimate = reinforce_gradient(
                        discrete_problem,
                        float(theta_value),
                        estimator_config,
                        estimator_state,
                    )
                elif estimator_name == "gs":
                    gradient_estimate = gumbel_softmax_gradient(
                        discrete_problem, float(theta_value), estimator_config
                    )
                else:
                    logger.error(f"Unknown estimator: {estimator_name}")
                    raise ValueError(f"Unknown estimator: {estimator_name}")

                gradient_samples.append(gradient_estimate)

            # Record timing and compute statistics
            elapsed_time = time.time() - start_time
            times_list.append(elapsed_time)

            gradient_samples_array = np.array(gradient_samples, dtype=float)
            sample_mean = gradient_samples_array.mean()
            sample_std = gradient_samples_array.std(ddof=1)

            means_list.append(sample_mean)
            stds_list.append(sample_std)

            logger.info(
                f"    Mean: {sample_mean:.4g}, Std: {sample_std:.4g}, "
                f"Time: {elapsed_time:.2f}s"
            )

        # Store results for this estimator
        results[estimator_name] = {
            "theta": sweep_config.theta_values.copy(),
            "mean": np.array(means_list),
            "std": np.array(stds_list),
            "time": np.array(times_list),
        }
        logger.info(f"Finished estimator: {estimator_name}")

    logger.info("Sweep complete.")
    return results
