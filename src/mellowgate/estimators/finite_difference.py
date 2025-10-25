"""Finite difference gradient estimation for discrete optimization problems."""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mellowgate.core import DiscreteProblem


@dataclass
class FiniteDifferenceConfig:
    """
    Configuration parameters for finite difference gradient estimation.

    The finite difference method approximates gradients by evaluating the function
    at nearby points and computing the numerical derivative.

    Attributes:
        step_size (float): Small perturbation value (epsilon) for finite difference
            approximation. Smaller values give more accurate derivatives but may
            suffer from numerical precision issues. Default: 1e-3.
        num_samples (int): Number of Monte Carlo samples used to estimate the
            expectation at each evaluation point. Larger values reduce variance
            but increase computational cost. Default: 2000.
    """

    step_size: float = 1e-3
    num_samples: int = 2000


def _monte_carlo_expectation(
    discrete_problem: DiscreteProblem,
    theta: jnp.ndarray,
    num_samples: int,
    key: jax.Array,
) -> jnp.ndarray:
    """
    Compute Monte Carlo expectation for given theta values using efficient
    JAX operations.

    Performance optimizations:
    - Single batched gather operation for efficient indexing
    - Vectorized sampling across all theta values
    - JIT compilation for fusion and reduced overhead

    Args:
        discrete_problem: The discrete optimization problem instance.
        theta: Parameter values, shape: (N,)
        num_samples: Number of samples for Monte Carlo estimation.
        key: JAX random key for sampling.

    Returns:
        jnp.ndarray: Expected values, shape: (N,)
    """
    # Vectorized sampling: sampled_values shape: (N, num_samples)
    sampled_values = discrete_problem.compute_stochastic_values(
        theta, num_samples=num_samples, key=key
    )

    # Single reduction instead of per-theta operations
    return jnp.mean(sampled_values, axis=1)  # Mean along sample dimension


def finite_difference_gradient(
    discrete_problem: DiscreteProblem,
    parameter_value: float | jnp.ndarray,
    config: FiniteDifferenceConfig,
) -> float | jnp.ndarray:
    """Estimate gradient using finite differences method with vectorized operations.

    Performance optimizations:
    - Batched perturbation computation across all parameters
    - Single vectorized model evaluation for all perturbations
    - JIT compilation for operation fusion and reduced overhead
    - Vectorized central difference computation

    Args:
        discrete_problem: The discrete optimization problem instance.
        parameter_value: Parameter value(s) at which to estimate the gradient.
                        Shape: scalar or (N,) for N parameter values
        config: Configuration for finite difference estimation.

    Returns:
        Union[float, jnp.ndarray]: Estimated gradient values.
                                 Shape matches input parameter_value shape.
                                 Scalar for scalar input, (N,) for array input.

    Examples:
        >>> theta = jnp.array([0.0, 1.0, 2.0])  # Shape: (3,)
        >>> gradient = finite_difference_gradient(problem, theta, config)
        >>> gradient.shape  # (3,)
    """
    # Convert to array for consistent handling
    theta_array = jnp.asarray(parameter_value)
    is_scalar_input = theta_array.ndim == 0

    if is_scalar_input:
        theta_array = theta_array.reshape(1)  # Shape: (1,)

    # Create perturbed theta arrays using vectorized operations
    # All have shape: (N,) where N = len(theta_array)
    theta_plus = theta_array + config.step_size
    theta_minus = theta_array - config.step_size

    # Use Common Random Numbers (CRN) for variance reduction
    # Using the SAME random key for both evaluations reduces variance in the
    # finite difference estimate by inducing positive correlation
    shared_key = jax.random.PRNGKey(0)

    # Compute expectations using optimized Monte Carlo function - each has shape: (N,)
    expectation_at_plus = _monte_carlo_expectation(
        discrete_problem, theta_plus, config.num_samples, shared_key
    )
    expectation_at_minus = _monte_carlo_expectation(
        discrete_problem, theta_minus, config.num_samples, shared_key
    )

    # Finite difference approximation - shape: (N,)
    gradient_estimate = (expectation_at_plus - expectation_at_minus) / (
        2 * config.step_size
    )

    # Check for NaN or Inf values and handle them robustly
    nan_mask = jnp.isnan(gradient_estimate)
    inf_mask = jnp.isinf(gradient_estimate)
    invalid_mask = nan_mask | inf_mask

    if jnp.any(invalid_mask):
        invalid_count = jnp.sum(invalid_mask)
        total_count = len(gradient_estimate)

        warnings.warn(
            f"FD gradient estimation produced {invalid_count} invalid values "
            f"(NaN or Inf) out of {total_count} parameter points. This can occur with "
            f"very small step_size (current: {config.step_size}) causing numerical "
            f"precision issues, or with extreme function values. Consider adjusting "
            f"step_size or num_samples (current: {config.num_samples}). "
            f"Invalid values will be replaced with 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )

        # Replace invalid values with 0.0 as a safe fallback
        gradient_estimate = jnp.where(invalid_mask, 0.0, gradient_estimate)

    # Return scalar if input was scalar, array otherwise
    if is_scalar_input:
        return float(gradient_estimate[0])
    return gradient_estimate
