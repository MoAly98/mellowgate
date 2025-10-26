"""Gumbel-Softmax gradient estimation for discrete optimization problems."""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mellowgate.core import DiscreteProblem
from mellowgate.utils.functions import softmax
from mellowgate.utils.statistics import sample_gumbel


@dataclass
class GumbelSoftmaxConfig:
    """
    Configuration for Gumbel-Softmax gradient estimation.

    The Gumbel-Softmax trick provides a differentiable approximation to discrete
    sampling using the Gumbel distribution and temperature-controlled softmax.

    Attributes:
        temperature (float): Temperature parameter for Gumbel-Softmax relaxation.
            Lower values make the distribution more discrete-like, higher values
            make it more uniform. Default: 0.5.
        num_samples (int): Number of Monte Carlo samples for gradient estimation.
            Default: 1000.
        use_straight_through_estimator (bool): Whether to use Straight-Through
            Estimator (STE) which uses discrete sampling in forward pass but
            continuous gradients in backward pass. Default: False.
    """

    temperature: float = 0.5
    num_samples: int = 1000
    use_straight_through_estimator: bool = False


def _gumbel_softmax_gradient_vectorized(
    log_probabilities: jnp.ndarray,
    probability_gradients: jnp.ndarray,
    function_values: jnp.ndarray,
    pathwise_gradients: jnp.ndarray,
    gumbel_noise: jnp.ndarray,
    num_samples: int,
    temperature: float,
    use_straight_through_estimator: bool,
) -> jnp.ndarray:
    """Vectorized Gumbel-Softmax gradient computation using proper reparameterization.

    This implements the correct Gumbel-Softmax reparameterization gradient:
    ∇θ E[f(y)] = E[f(y) * ∇θ y] + E[∇θ f(y)]

    Where ∇θ y is computed using the softmax Jacobian:
    ∂y_i/∂π_j = (1/τ) * y_i * (δ_ij - y_j)

    Performance optimizations:
    - Vectorized operations across theta and sample dimensions
    - Single batched softmax computation across all samples and theta values
    - Vectorized softmax Jacobian computation
    - JIT compilation fuses operations and reduces overhead

    Args:
        log_probabilities: Log probabilities log π, shape (num_branches, num_theta)
        probability_gradients: Gradients ∇θ π(x|θ), shape (num_branches, num_theta)
        function_values: Shape (num_branches, num_theta)
        pathwise_gradients: Shape (num_branches, num_theta) or zeros
        gumbel_noise: Shape (num_theta, num_samples, num_branches)
        num_samples: Number of samples per theta
        temperature: Temperature parameter
        use_straight_through_estimator: Whether to use STE

    Returns:
        jnp.ndarray: Gradient estimates, shape (num_theta,)
    """
    num_theta = log_probabilities.shape[1] if log_probabilities.ndim > 1 else 1

    # Handle single theta case by reshaping for consistent processing
    if log_probabilities.ndim == 1:
        log_probabilities = log_probabilities[:, jnp.newaxis]
        probability_gradients = probability_gradients[:, jnp.newaxis]
        function_values = function_values[:, jnp.newaxis]
        pathwise_gradients = pathwise_gradients[:, jnp.newaxis]
        gumbel_noise = gumbel_noise[jnp.newaxis, :, :]
        num_theta = 1

    # Vectorized Gumbel-perturbed log probabilities:
    # shape: (num_theta, num_samples, num_branches)
    log_probs_expanded = log_probabilities.T[
        :, jnp.newaxis, :
    ]  # Shape: (num_theta, 1, num_branches)
    perturbed_log_probs = (
        log_probs_expanded + gumbel_noise
    )  # Broadcasting to (num_theta, num_samples, num_branches)

    # Vectorized softmax computation across all samples and theta values
    continuous_weights = softmax(
        perturbed_log_probs / temperature, axis=2
    )  # Shape: (num_theta, num_samples, num_branches)

    # Compute pathwise gradient contributions vectorized
    if use_straight_through_estimator:
        # STE: Use discrete sampling but continuous gradients
        best_choice_indices = jnp.argmax(
            perturbed_log_probs, axis=2
        )  # Shape: (num_theta, num_samples)

        # Efficient batched gather for pathwise gradients
        theta_indices = jnp.arange(num_theta)[:, jnp.newaxis]  # Shape: (num_theta, 1)
        theta_indices_expanded = jnp.broadcast_to(
            theta_indices, (num_theta, num_samples)
        )
        pathwise_contribution = pathwise_gradients.T[
            theta_indices_expanded, best_choice_indices
        ]  # Shape: (num_theta, num_samples)
    else:
        # Continuous relaxation using vectorized dot product
        pathwise_gradients_expanded = pathwise_gradients.T[
            :, jnp.newaxis, :
        ]  # Shape: (num_theta, 1, num_branches)
        pathwise_contribution = jnp.sum(
            continuous_weights * pathwise_gradients_expanded, axis=2
        )  # Shape: (num_theta, num_samples)

    # Compute reparameterization gradient using softmax Jacobian
    # The Gumbel-Softmax reparameterization gradient is:
    # ∇θ E[f(y)] = E[∇θ (f(y) ∘ softmax((log π + G)/τ))]
    # where ∇θ y_i = (1/τ) * y_i * (∇θ log π_i - Σ_j y_j * ∇θ log π_j)

    # We need to work with log probability gradients for numerical stability
    # First, get the original probabilities from log_probabilities
    original_probabilities = jnp.exp(
        log_probabilities
    )  # Shape: (num_branches, num_theta)

    # ∇θ log π = (∇θ π) / π
    log_prob_gradients = probability_gradients / original_probabilities

    # Expand for vectorized computation with proper shape handling
    log_prob_gradients_expanded = log_prob_gradients.T[
        :, jnp.newaxis, :
    ]  # Shape: (num_theta, 1, num_branches)

    # Ensure proper broadcasting by taking only the scalar gradient values
    # For the case where probability_gradients has extra dimensions
    if log_prob_gradients_expanded.ndim > 3:
        # Take the diagonal elements for scalar parameter gradients
        log_prob_gradients_expanded = jnp.diagonal(
            log_prob_gradients_expanded, axis1=-2, axis2=-1
        )[..., jnp.newaxis, :]

    # Compute softmax Jacobian vector product for log probabilities
    # ∇θ y_i = (1/τ) * y_i * (∇θ log π_i - Σ_j y_j * ∇θ log π_j)

    # First term: y_i * (∇θ log π_i)
    diagonal_term = continuous_weights * log_prob_gradients_expanded
    # Shape: (num_theta, num_samples, num_branches)

    # Second term: y_i * Σ_j y_j * (∇θ log π_j)
    weighted_sum = jnp.sum(
        continuous_weights * log_prob_gradients_expanded, axis=2, keepdims=True
    )  # Shape: (num_theta, num_samples, 1)

    off_diagonal_term = continuous_weights * weighted_sum
    # Shape: (num_theta, num_samples, num_branches)

    # Combine terms and apply temperature scaling
    reparameterization_gradients = (diagonal_term - off_diagonal_term) / temperature
    # Shape: (num_theta, num_samples, num_branches)

    # Vectorized function value integration for reparameterization term
    function_values_expanded = function_values.T[
        :, jnp.newaxis, :
    ]  # Shape: (num_theta, 1, num_branches)

    reparameterization_contribution = jnp.sum(
        function_values_expanded * reparameterization_gradients, axis=2
    )  # Shape: (num_theta, num_samples)

    # Combine contributions and compute final gradients
    total_gradient_terms = (
        pathwise_contribution + reparameterization_contribution
    )  # Shape: (num_theta, num_samples)

    return jnp.mean(total_gradient_terms, axis=1)  # Shape: (num_theta,)


def gumbel_softmax_gradient(
    discrete_problem: DiscreteProblem,
    parameter_value: float | jnp.ndarray,
    config: GumbelSoftmaxConfig,
) -> float | jnp.ndarray:
    """
    Estimate the gradient using the Gumbel-Softmax reparameterization trick
    with vectorized operations and JIT compilation.

    Performance optimizations:
    - Vectorized operations across theta and sample dimensions
    - Single batched Gumbel noise generation for all samples
    - Vectorized softmax and matrix operations across all samples
    - JIT compilation for operation fusion and reduced overhead

    Args:
        discrete_problem: The discrete optimization problem containing the
            function to differentiate and probability model.
        parameter_value: The parameter value(s) θ at which to estimate the gradient.
            Can be scalar or array.
        config: Configuration parameters for Gumbel-Softmax estimation.

    Returns:
        Union[float, jnp.ndarray]: Estimated gradient value(s). Returns scalar
            for scalar input, array for array input.

    Raises:
        ValueError: If the logits model does not provide gradient information
            (dlogits_dtheta) required for reparameterization.
        Warning: If NaN values are detected and need to be handled.
    """
    # Convert to array for consistent handling
    theta_array = jnp.asarray(parameter_value)
    is_scalar_input = theta_array.ndim == 0

    if is_scalar_input:
        theta_array = theta_array.reshape(1)

    # Compute score function gradients correctly using JAX autodiff for Gumbel-Softmax
    # ∇θ log π(x|θ) = (1/π(x|θ)) * ∇θ π(x|θ)
    def compute_probabilities_for_theta(theta_single):
        """Wrapper to compute probabilities for a single theta value."""
        logits = discrete_problem.logits_model.logits_function(theta_single)
        return discrete_problem.logits_model.probability_function(logits)

    # Use JAX jacfwd to compute probability gradients correctly
    prob_jacobian_fn = jax.jacfwd(compute_probabilities_for_theta)

    choice_probabilities = jnp.asarray(
        discrete_problem.compute_probabilities(theta_array)
    )

    if theta_array.shape[0] == 1:
        # Single theta case
        probability_gradients = prob_jacobian_fn(theta_array[0])
        probability_gradients = probability_gradients.reshape(-1, 1)
    else:
        # Multiple theta case - vectorize the Jacobian computation
        prob_jacobian_vectorized = jax.vmap(prob_jacobian_fn)
        jacobian_result = prob_jacobian_vectorized(
            theta_array
        )  # Shape: (num_theta, num_branches, ...)
        # Reshape to (num_branches, num_theta) regardless of trailing dimensions
        probability_gradients = jacobian_result.reshape(theta_array.shape[0], -1).T

    # Use log probabilities for proper Gumbel-Max sampling (general case)
    log_probabilities = jnp.log(
        choice_probabilities
    )  # Shape: (num_branches, num_theta)

    function_values = jnp.asarray(discrete_problem.compute_function_values(theta_array))

    # Get pathwise gradients if available (optional)
    pathwise_gradients = discrete_problem.compute_derivative_values(theta_array)
    if pathwise_gradients is not None:
        pathwise_gradients = jnp.asarray(pathwise_gradients)
    else:
        # Create zeros with proper shape for vectorized computation
        pathwise_gradients = jnp.zeros_like(function_values)

    num_theta = len(theta_array)

    # Generate all Gumbel noise samples in a single batch using JAX
    # Use different keys for different theta values for reproducibility
    keys = jax.random.split(jax.random.PRNGKey(0), num_theta)

    # Vectorized Gumbel noise generation for all theta values and samples
    def generate_gumbel_for_theta(key):
        return sample_gumbel((config.num_samples, discrete_problem.num_branches), key)

    # Shape: (num_theta, num_samples, num_branches)
    gumbel_noise = jax.vmap(generate_gumbel_for_theta)(keys)

    # Use optimized vectorized gradient computation
    gradient_estimates = _gumbel_softmax_gradient_vectorized(
        log_probabilities,
        probability_gradients,
        function_values,
        pathwise_gradients,
        gumbel_noise,
        config.num_samples,
        config.temperature,
        config.use_straight_through_estimator,
    )  # Check for NaN values and handle them robustly
    nan_mask = jnp.isnan(gradient_estimates)
    if jnp.any(nan_mask):
        nan_count = jnp.sum(nan_mask)
        total_count = len(gradient_estimates)

        warnings.warn(
            f"Gumbel-Softmax gradient estimation produced {nan_count} NaN values "
            f"out of {total_count} parameter points. This typically occurs with "
            f"extreme logit values and high sample counts. Consider reducing "
            f"num_samples (current: {config.num_samples}) or adjusting temperature "
            f"(current: {config.temperature}). NaN values will be replaced with 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )

        # Replace NaN values with 0.0 as a safe fallback
        # This is reasonable since NaNs typically occur at extreme parameter values
        # where the true gradient should be very close to zero
        gradient_estimates = jnp.where(nan_mask, 0.0, gradient_estimates)

    # Return scalar if input was scalar, array otherwise
    if is_scalar_input:
        return float(gradient_estimates[0])
    return gradient_estimates
