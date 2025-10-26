"""REINFORCE gradient estimation for discrete optimization problems."""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mellowgate.core import DiscreteProblem


@dataclass
class ReinforceConfig:
    """
    Configuration for REINFORCE gradient estimation.

    REINFORCE is a policy gradient method that uses the score function to estimate
    gradients through sampling and reward weighting.

    Attributes:
        num_samples (int): Number of Monte Carlo samples to use for gradient
            estimation. More samples reduce variance but increase computational cost.
            Default: 2000.
        use_baseline (bool): Whether to use a baseline for variance reduction.
            Baselines can significantly reduce gradient variance without introducing
            bias. Default: True.
        baseline_momentum (float): Momentum coefficient for exponential moving
            average baseline update. Only used when use_baseline=True.
            Range: [0, 1]. Default: 0.9.
    """

    num_samples: int = 2000
    use_baseline: bool = True
    baseline_momentum: float = 0.9


class ReinforceState:
    """
    Maintains state for REINFORCE gradient estimation across multiple calls.

    This class tracks the running baseline used for variance reduction in the
    REINFORCE algorithm. The baseline is updated using an exponential moving
    average of observed rewards.

    Attributes:
        baseline (float): Current baseline value (running average of rewards).
        initialized (bool): Whether the baseline has been initialized with
            at least one observation.
    """

    def __init__(self) -> None:
        """Initialize REINFORCE state with empty baseline."""
        self.baseline: float = 0.0
        self.initialized: bool = False

    def update_baseline(self, new_reward: float, momentum: float) -> None:
        """
        Update the baseline using exponential moving average.

        Args:
            new_reward: New reward observation to incorporate.
            momentum: Momentum coefficient for the moving average.
        """
        if self.initialized:
            self.baseline = momentum * self.baseline + (1 - momentum) * new_reward
        else:
            self.baseline = new_reward
            self.initialized = True


def _reinforce_gradient_vectorized(
    choice_probabilities: jnp.ndarray,
    function_values: jnp.ndarray,
    pathwise_gradients: jnp.ndarray,
    score_function_gradients: jnp.ndarray,
    sampled_choice_indices: jnp.ndarray,
    baseline_values: jnp.ndarray,
    num_samples: int,
) -> jnp.ndarray:
    """Vectorized REINFORCE gradient computation with JIT compilation.

    Performance optimizations:
    - Single batched gather using jnp.take_along_axis for efficient indexing
    - Vectorized operations across theta/sample dimensions
    - JIT compilation fuses operations and reduces overhead
    - Vectorized score function computation across all samples simultaneously

    Args:
        choice_probabilities: Shape (num_branches, num_theta)
        function_values: Shape (num_branches, num_theta)
        pathwise_gradients: Shape (num_branches, num_theta) or zeros
        score_function_gradients: Shape (num_branches, num_theta)
        sampled_choice_indices: Shape (num_theta, num_samples)
        baseline_values: Shape (num_theta,)
        num_samples: Number of samples per theta

    Returns:
        jnp.ndarray: Gradient estimates, shape (num_theta,)
    """
    num_theta = choice_probabilities.shape[1] if choice_probabilities.ndim > 1 else 1

    # Handle single theta case by reshaping for consistent processing
    if choice_probabilities.ndim == 1:
        choice_probabilities = choice_probabilities[:, jnp.newaxis]
        function_values = function_values[:, jnp.newaxis]
        pathwise_gradients = pathwise_gradients[:, jnp.newaxis]
        score_function_gradients = score_function_gradients[:, jnp.newaxis]
        sampled_choice_indices = sampled_choice_indices[jnp.newaxis, :]
        baseline_values = baseline_values[jnp.newaxis]
        num_theta = 1

    # Compute score function center term: E[∇θ log π(x|θ)] = Σ π(x) * ∇θ log π(x|θ)
    score_function_center = jnp.sum(
        choice_probabilities * score_function_gradients, axis=0
    )  # Shape: (num_theta,)

    # Prepare indices for advanced indexing
    theta_indices = jnp.arange(num_theta)[:, jnp.newaxis]  # Shape: (num_theta, 1)
    theta_indices_expanded = jnp.broadcast_to(
        theta_indices, (num_theta, num_samples)
    )  # Shape: (num_theta, num_samples)

    # Single batched gather for function values
    sampled_function_vals = function_values[
        sampled_choice_indices, theta_indices_expanded
    ]  # Shape: (num_theta, num_samples)

    # Single batched gather for score function gradients
    sampled_score_grads = score_function_gradients[
        sampled_choice_indices, theta_indices_expanded
    ]  # Shape: (num_theta, num_samples)

    # Single batched gather for pathwise gradients
    sampled_pathwise_grads = pathwise_gradients[
        sampled_choice_indices, theta_indices_expanded
    ]  # Shape: (num_theta, num_samples)

    # Vectorized score function computation across all samples
    score_center_expanded = score_function_center[
        :, jnp.newaxis
    ]  # Shape: (num_theta, 1)
    baseline_expanded = baseline_values[:, jnp.newaxis]  # Shape: (num_theta, 1)

    # Compute reward differences and score function terms vectorized
    reward_differences = (
        sampled_function_vals - baseline_expanded
    )  # Shape: (num_theta, num_samples)
    score_function_terms = reward_differences * (
        sampled_score_grads - score_center_expanded
    )  # Shape: (num_theta, num_samples)

    # Combine pathwise and score function contributions
    total_gradient_terms = (
        sampled_pathwise_grads + score_function_terms
    )  # Shape: (num_theta, num_samples)

    # Final reduction: empirical mean as gradient estimate
    return jnp.mean(total_gradient_terms, axis=1)  # Shape: (num_theta,)


def reinforce_gradient(
    discrete_problem: DiscreteProblem,
    parameter_value: float | jnp.ndarray,
    config: ReinforceConfig,
    state: ReinforceState,
) -> float | jnp.ndarray:
    """
    Estimate the gradient using the REINFORCE algorithm with vectorized operations.

    REINFORCE uses the policy gradient theorem to estimate gradients by sampling
    from the current policy and weighting by rewards. The gradient estimator is:
    ∇θ E[f(θ)] ≈ E[f(x) * ∇θ log π(x|θ)] + E[∇θ f(x)]

    The first term is the score function (REINFORCE) and the second is the
    pathwise derivative when available. Supports both scalar and array inputs
    for efficient batch processing.

    Args:
        discrete_problem: The discrete optimization problem containing the
            function to differentiate and probability model.
        parameter_value: The parameter value(s) θ at which to estimate the gradient.
            Can be scalar or array.
        config: Configuration parameters for REINFORCE estimation.
        state: State object to maintain baseline across calls.

    Returns:
        Union[float, jnp.ndarray]: Estimated gradient value(s). Returns scalar
            for scalar input, array for array input.

    Raises:
        ValueError: If the logits model does not provide gradient information
            (dlogits_dtheta) required for the score function.

    Notes:
        - Uses baseline for variance reduction if enabled in config
        - Combines pathwise gradients (when available) with score function
        - Updates baseline state for future calls
        - All operations are vectorized for computational efficiency
    """
    # Convert to array for consistent handling
    theta_array = jnp.asarray(parameter_value)
    is_scalar_input = theta_array.ndim == 0

    if is_scalar_input:
        theta_array = theta_array.reshape(1)

    # Precompute all required arrays once - hoisting invariants
    choice_probabilities = jnp.asarray(
        discrete_problem.compute_probabilities(theta_array)
    )
    function_values = jnp.asarray(discrete_problem.compute_function_values(theta_array))

    # Get pathwise gradients if available (optional)
    pathwise_gradients = discrete_problem.compute_derivative_values(theta_array)
    if pathwise_gradients is not None:
        pathwise_gradients = jnp.asarray(pathwise_gradients)
    else:
        # Create zeros with proper shape for vectorized computation
        pathwise_gradients = jnp.zeros_like(function_values)

    # Compute score function gradients correctly using JAX autodiff
    # ∇θ log π(x|θ) = (1/π(x|θ)) * ∇θ π(x|θ)
    # where ∇θ π(x|θ) is computed using chain rule through logits
    def compute_probabilities_for_theta(theta_single):
        """Wrapper to compute probabilities for a single theta value."""
        logits = discrete_problem.logits_model.logits_function(theta_single)
        return discrete_problem.logits_model.probability_function(logits)

    # Use JAX jacfwd to compute probability gradients correctly
    prob_jacobian_fn = jax.jacfwd(compute_probabilities_for_theta)

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

    # Convert to score function gradients: ∇θ log π(x|θ) = (∇θ π(x|θ)) / π(x|θ)
    score_function_gradients = probability_gradients / choice_probabilities

    # Generate all random samples at once using batched key generation
    key = jax.random.PRNGKey(0)  # Use deterministic key for reproducibility
    sampled_choice_indices = discrete_problem.sample_branch(
        theta_array, num_samples=config.num_samples, key=key
    )

    # Handle baseline computation efficiently
    baseline_values = jnp.zeros(len(theta_array))
    if config.use_baseline:
        # Vectorized baseline computation - use efficient indexing
        theta_indices = jnp.arange(len(theta_array))[:, jnp.newaxis]
        theta_indices_expanded = jnp.broadcast_to(
            theta_indices, sampled_choice_indices.shape
        )
        sampled_rewards = function_values[
            sampled_choice_indices, theta_indices_expanded
        ]

        current_mean_rewards = jnp.mean(sampled_rewards, axis=-1)  # Mean over samples

        # Use existing baseline or initialize with current mean
        if state.initialized:
            baseline_values = jnp.full_like(current_mean_rewards, state.baseline)
        else:
            baseline_values = current_mean_rewards

        # Update baseline for future use (use mean of current rewards)
        overall_mean_reward = float(jnp.mean(current_mean_rewards))
        state.update_baseline(overall_mean_reward, config.baseline_momentum)

    # Use optimized vectorized gradient computation
    gradient_estimates = _reinforce_gradient_vectorized(
        choice_probabilities,
        function_values,
        pathwise_gradients,
        score_function_gradients,
        sampled_choice_indices,
        baseline_values,
        config.num_samples,
    )

    # Check for NaN or Inf values and handle them robustly
    nan_mask = jnp.isnan(gradient_estimates)
    inf_mask = jnp.isinf(gradient_estimates)
    invalid_mask = nan_mask | inf_mask

    if jnp.any(invalid_mask):
        invalid_count = jnp.sum(invalid_mask)
        total_count = len(gradient_estimates)

        warnings.warn(
            f"REINFORCE gradient estimation produced {invalid_count} invalid values "
            f"(NaN or Inf) out of {total_count} parameter points. This can occur with "
            f"extreme probability values or poorly conditioned sampling. "
            f"Consider adjusting num_samples (current: {config.num_samples}) or "
            f"using baseline (current: {config.use_baseline}). "
            f"Invalid values will be replaced with 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )

        # Replace invalid values with 0.0 as a safe fallback
        gradient_estimates = jnp.where(invalid_mask, 0.0, gradient_estimates)

    # Return scalar if input was scalar, array otherwise
    if is_scalar_input:
        return float(gradient_estimates[0])
    return gradient_estimates
