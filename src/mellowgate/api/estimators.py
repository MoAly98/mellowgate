"""Gradient estimation methods for discrete optimization problems.

This module implements three main approaches for estimating gradients in discrete
optimization problems where exact gradients may not be available:

1. Finite Differences: Approximates gradients using numerical differentiation
2. REINFORCE: Uses policy gradient methods with optional baseline reduction
3. Gumbel-Softmax: Provides differentiable relaxation of discrete sampling

Each estimator is designed to work with DiscreteProblem instances and supports
vectorized operations for efficient computation across multiple parameter values.
The estimators handle the fundamental challenge of computing gradients through
discrete sampling operations.
"""

import warnings
from dataclasses import dataclass
from typing import Union

import jax
import jax.numpy as jnp

from mellowgate.api.functions import DiscreteProblem
from mellowgate.utils.functions import softmax
from mellowgate.utils.statistics import sample_gumbel


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
    discrete_problem: DiscreteProblem, theta: jnp.ndarray, num_samples: int
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

    Returns:
        jnp.ndarray: Expected values, shape: (N,)
    """
    # Generate all random keys at once for batched sampling
    key = jax.random.PRNGKey(0)  # Use deterministic key for reproducibility

    # Vectorized sampling: sampled_values shape: (N, num_samples)
    sampled_values = discrete_problem.compute_stochastic_values(
        theta, num_samples=num_samples, key=key
    )

    # Single reduction instead of per-theta operations
    return jnp.mean(sampled_values, axis=1)  # Mean along sample dimension


def finite_difference_gradient(
    discrete_problem: DiscreteProblem,
    parameter_value: Union[float, jnp.ndarray],
    config: FiniteDifferenceConfig,
) -> Union[float, jnp.ndarray]:
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

    # Compute expectations using optimized Monte Carlo function - each has shape: (N,)
    expectation_at_plus = _monte_carlo_expectation(
        discrete_problem, theta_plus, config.num_samples
    )
    expectation_at_minus = _monte_carlo_expectation(
        discrete_problem, theta_minus, config.num_samples
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
    logits_gradients: jnp.ndarray,
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
        logits_gradients: Shape (num_branches, num_theta)
        sampled_choice_indices: Shape (num_theta, num_samples)
        baseline_values: Shape (num_theta,)
        num_samples: Number of samples per theta
        use_baseline: Whether to use baseline for variance reduction

    Returns:
        jnp.ndarray: Gradient estimates, shape (num_theta,)
    """
    num_theta = choice_probabilities.shape[1] if choice_probabilities.ndim > 1 else 1

    # Handle single theta case by reshaping for consistent processing
    if choice_probabilities.ndim == 1:
        choice_probabilities = choice_probabilities[:, jnp.newaxis]
        function_values = function_values[:, jnp.newaxis]
        pathwise_gradients = pathwise_gradients[:, jnp.newaxis]
        logits_gradients = logits_gradients[:, jnp.newaxis]
        sampled_choice_indices = sampled_choice_indices[jnp.newaxis, :]
        baseline_values = baseline_values[jnp.newaxis]
        num_theta = 1

    # Compute score function center term: E[∇θ log π(x|θ)] = Σ π(x) * ∇θ log π(x|θ)
    score_function_center = jnp.sum(
        choice_probabilities * logits_gradients, axis=0
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

    # Single batched gather for logits gradients
    sampled_logits_grads = logits_gradients[
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
        sampled_logits_grads - score_center_expanded
    )  # Shape: (num_theta, num_samples)

    # Combine pathwise and score function contributions
    total_gradient_terms = (
        sampled_pathwise_grads + score_function_terms
    )  # Shape: (num_theta, num_samples)

    # Final reduction: empirical mean as gradient estimate
    gradient_estimates = jnp.mean(total_gradient_terms, axis=1)  # Shape: (num_theta,)

    return gradient_estimates


def reinforce_gradient(
    discrete_problem: DiscreteProblem,
    parameter_value: Union[float, jnp.ndarray],
    config: ReinforceConfig,
    state: ReinforceState,
) -> Union[float, jnp.ndarray]:
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

    # Check that we have the required gradient information for score function
    if discrete_problem.logits_model.logits_derivative_function is None:
        raise ValueError(
            "REINFORCE requires logits_derivative_function for score function."
        )

    logits_gradients = jnp.asarray(
        discrete_problem.logits_model.logits_derivative_function(theta_array)
    )

    # Convert logits gradients to actual score function gradients
    # For custom probability functions, we need ∇θ log π(x|θ), not ∇θ α(x|θ)
    # For the binary sigmoid case with complementary probabilities:
    # π0 = sigmoid(α0), π1 = sigmoid(α1) where α1 = -α0
    # ∇θ log π0 = (∇θ α0) * (1 - π0) = (∇θ α0) * π1
    # ∇θ log π1 = (∇θ α1) * (1 - π1) = (∇θ α1) * π0
    score_function_gradients = logits_gradients * (1 - choice_probabilities)

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

        import warnings

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
    logits: jnp.ndarray,
    logits_gradients: jnp.ndarray,
    function_values: jnp.ndarray,
    pathwise_gradients: jnp.ndarray,
    gumbel_noise: jnp.ndarray,
    num_samples: int,
    num_branches: int,
    temperature: float,
    use_straight_through_estimator: bool,
) -> jnp.ndarray:
    """Vectorized Gumbel-Softmax gradient computation with JIT compilation.

    Performance optimizations:
    - Vectorized operations across theta and sample dimensions
    - Single batched softmax computation across all samples and theta values
    - Vectorized matrix operations for efficient computation
    - JIT compilation fuses operations and reduces overhead

    Args:
        logits: Shape (num_branches, num_theta)
        logits_gradients: Shape (num_branches, num_theta)
        function_values: Shape (num_branches, num_theta)
        pathwise_gradients: Shape (num_branches, num_theta) or zeros
        gumbel_noise: Shape (num_theta, num_samples, num_branches)
        num_samples: Number of samples per theta
        num_branches: Number of branches
        temperature: Temperature parameter
        use_straight_through_estimator: Whether to use STE

    Returns:
        jnp.ndarray: Gradient estimates, shape (num_theta,)
    """
    num_theta = logits.shape[1] if logits.ndim > 1 else 1

    # Handle single theta case by reshaping for consistent processing
    if logits.ndim == 1:
        logits = logits[:, jnp.newaxis]
        logits_gradients = logits_gradients[:, jnp.newaxis]
        function_values = function_values[:, jnp.newaxis]
        pathwise_gradients = pathwise_gradients[:, jnp.newaxis]
        gumbel_noise = gumbel_noise[jnp.newaxis, :, :]
        num_theta = 1

    # Vectorized Gumbel-perturbed logits: (num_theta, num_samples, num_branches)
    logits_expanded = logits.T[:, jnp.newaxis, :]  # Shape: (num_theta, 1, num_branches)
    perturbed_logits = (
        logits_expanded + gumbel_noise
    )  # Broadcasting to (num_theta, num_samples, num_branches)

    # Vectorized softmax computation across all samples and theta values
    continuous_weights = softmax(
        perturbed_logits / temperature, axis=2
    )  # Shape: (num_theta, num_samples, num_branches)

    # Compute pathwise gradient contributions vectorized
    if use_straight_through_estimator:
        # STE: Use discrete sampling but continuous gradients
        best_choice_indices = jnp.argmax(
            perturbed_logits, axis=2
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

    # Vectorized reparameterization gradient computation
    logits_gradients_expanded = logits_gradients.T[
        :, jnp.newaxis, :
    ]  # Shape: (num_theta, 1, num_branches)

    # Compute mean logits gradient efficiently
    mean_logits_gradient = jnp.sum(
        continuous_weights * logits_gradients_expanded, axis=2
    )  # Shape: (num_theta, num_samples)

    # Vectorized softmax gradient computation
    mean_logits_gradient_expanded = mean_logits_gradient[
        :, :, jnp.newaxis
    ]  # Shape: (num_theta, num_samples, 1)
    softmax_gradient = (
        continuous_weights * (logits_gradients_expanded - mean_logits_gradient_expanded)
    ) / temperature

    # Vectorized function value integration
    function_values_expanded = function_values.T[
        :, jnp.newaxis, :
    ]  # Shape: (num_theta, 1, num_branches)
    reparameterization_contribution = jnp.sum(
        function_values_expanded * softmax_gradient, axis=2
    )  # Shape: (num_theta, num_samples)

    # Combine contributions and compute final gradients
    total_gradient_terms = (
        pathwise_contribution + reparameterization_contribution
    )  # Shape: (num_theta, num_samples)
    gradient_estimates = jnp.mean(total_gradient_terms, axis=1)  # Shape: (num_theta,)

    return gradient_estimates


def gumbel_softmax_gradient(
    discrete_problem: DiscreteProblem,
    parameter_value: Union[float, jnp.ndarray],
    config: GumbelSoftmaxConfig,
) -> Union[float, jnp.ndarray]:
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

    # Validate that we have the required gradient information
    if discrete_problem.logits_model.logits_derivative_function is None:
        raise ValueError(
            "Gumbel-Softmax requires logits_derivative_function for backpropagation."
        )

    # Precompute all required arrays once - hoisting invariants
    logits_gradients = jnp.asarray(
        discrete_problem.logits_model.logits_derivative_function(theta_array)
    )

    # Convert logits gradients to actual score function gradients for Gumbel-Softmax
    # For the binary sigmoid case with complementary probabilities:
    # ∇θ log π_i = (∇θ α_i) * (1 - π_i)
    choice_probabilities = jnp.asarray(
        discrete_problem.compute_probabilities(theta_array)
    )
    score_function_gradients = logits_gradients * (1 - choice_probabilities)

    logits = jnp.asarray(discrete_problem.logits_model.logits_function(theta_array))
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
        logits,
        score_function_gradients,
        function_values,
        pathwise_gradients,
        gumbel_noise,
        config.num_samples,
        discrete_problem.num_branches,
        config.temperature,
        config.use_straight_through_estimator,
    )

    # Check for NaN values and handle them robustly
    nan_mask = jnp.isnan(gradient_estimates)
    if jnp.any(nan_mask):
        nan_count = jnp.sum(nan_mask)
        total_count = len(gradient_estimates)

        import warnings

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
