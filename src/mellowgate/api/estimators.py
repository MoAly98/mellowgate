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

from dataclasses import dataclass
from typing import Union

import numpy as np

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


def finite_difference_gradient(
    discrete_problem: DiscreteProblem,
    parameter_value: Union[float, np.ndarray],
    config: FiniteDifferenceConfig,
) -> Union[float, np.ndarray]:
    """Estimate gradient using finite differences method.

    Approximates the gradient by evaluating the function at theta + step_size
    and theta - step_size, then computing the numerical derivative.

    Args:
        discrete_problem: The discrete optimization problem instance.
        parameter_value: Parameter value(s) at which to estimate the gradient.
                        Shape: scalar or (N,) for N parameter values
        config: Configuration for finite difference estimation.

    Returns:
        Union[float, np.ndarray]: Estimated gradient values.
                                 Shape matches input parameter_value shape.
                                 Scalar for scalar input, (N,) for array input.

    Examples:
        >>> theta = np.array([0.0, 1.0, 2.0])  # Shape: (3,)
        >>> gradient = finite_difference_gradient(problem, theta, config)
        >>> gradient.shape  # (3,)
    """
    # Convert to array for consistent handling
    theta_array = np.asarray(parameter_value)
    is_scalar_input = theta_array.ndim == 0

    if is_scalar_input:
        theta_array = theta_array.reshape(1)  # Shape: (1,)

    def _monte_carlo_expectation(theta: np.ndarray) -> np.ndarray:
        """Compute Monte Carlo expectation for given theta values.

        Args:
            theta: Parameter values, shape: (N,)

        Returns:
            np.ndarray: Expected values, shape: (N,)
        """
        # sampled_values shape: (N, num_samples)
        sampled_values = discrete_problem.compute_stochastic_values(
            theta, num_samples=config.num_samples
        )
        # Return shape: (N,) - mean along sample dimension
        return np.mean(sampled_values, axis=1)

    # Create perturbed theta arrays
    # All have shape: (N,) where N = len(theta_array)
    theta_plus = theta_array + config.step_size
    theta_minus = theta_array - config.step_size

    # Compute expectations - each has shape: (N,)
    expectation_at_plus = _monte_carlo_expectation(theta_plus)
    expectation_at_minus = _monte_carlo_expectation(theta_minus)

    # Finite difference approximation - shape: (N,)
    gradient_estimate = (expectation_at_plus - expectation_at_minus) / (
        2 * config.step_size
    )

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


def reinforce_gradient(
    discrete_problem: DiscreteProblem,
    parameter_value: Union[float, np.ndarray],
    config: ReinforceConfig,
    state: ReinforceState,
) -> Union[float, np.ndarray]:
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
        Union[float, np.ndarray]: Estimated gradient value(s). Returns scalar
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
    theta_array = np.asarray(parameter_value)
    is_scalar_input = theta_array.ndim == 0

    if is_scalar_input:
        theta_array = theta_array.reshape(1)

    # Ensure all arrays are properly typed as numpy arrays
    choice_probabilities = np.asarray(
        discrete_problem.compute_probabilities(theta_array)
    )
    function_values = np.asarray(discrete_problem.compute_function_values(theta_array))

    # Get pathwise gradients if available (optional)
    pathwise_gradients = discrete_problem.compute_derivative_values(theta_array)
    if pathwise_gradients is not None:
        pathwise_gradients = np.asarray(pathwise_gradients)

    # Check that we have the required gradient information for score function
    if discrete_problem.logits_model.logits_derivative_function is None:
        raise ValueError(
            "REINFORCE requires logits_derivative_function for score function."
        )

    logits_gradients = np.asarray(
        discrete_problem.logits_model.logits_derivative_function(theta_array)
    )

    # Compute score function center term: E[∇θ log π(x|θ)] = Σ π(x) * ∇θ a(x)
    # where a(x) are the logits and π(x) = softmax(a(x))
    # For arrays: choice_probabilities shape: (num_branches, num_theta)
    #            logits_gradients shape: (num_branches, num_theta)
    if choice_probabilities.ndim == 1:
        # Single theta case
        score_function_center = np.dot(choice_probabilities, logits_gradients)
    else:
        # Multiple theta case - element-wise products then sum over branches
        score_function_center = np.sum(choice_probabilities * logits_gradients, axis=0)

    # Sample discrete choices according to current policy
    sampled_choice_indices = discrete_problem.sample_branch(
        theta_array, num_samples=config.num_samples
    )
    # sampled_choice_indices shape: (num_theta, num_samples) or
    # (num_samples,) for single theta

    # Handle baseline computation and updates
    baseline_values = np.zeros(len(theta_array))
    if config.use_baseline:
        # For vectorized case, we need to handle baseline per theta value
        # For now, use the same baseline for all theta values (could be enhanced)
        if function_values.ndim == 1:
            # Single theta case
            sampled_rewards = function_values[sampled_choice_indices]
        else:
            # Multiple theta case
            sampled_rewards = np.array(
                [
                    function_values[sampled_choice_indices[i], i]
                    for i in range(len(theta_array))
                ]
            )

        current_mean_rewards = np.mean(sampled_rewards, axis=-1)  # Mean over samples

        # Use existing baseline or initialize with current mean
        if state.initialized:
            baseline_values.fill(state.baseline)
        else:
            baseline_values = current_mean_rewards.copy()

        # Update baseline for future use (use mean of current rewards)
        overall_mean_reward = np.mean(current_mean_rewards)
        state.update_baseline(overall_mean_reward, config.baseline_momentum)

    # Compute pathwise gradient contribution (if available)
    pathwise_contribution = np.zeros((len(theta_array), config.num_samples))
    if pathwise_gradients is not None:
        if pathwise_gradients.ndim == 1:
            # Single theta case
            pathwise_contribution[0] = pathwise_gradients[sampled_choice_indices]
        else:
            # Multiple theta case
            for i in range(len(theta_array)):
                pathwise_contribution[i] = pathwise_gradients[
                    sampled_choice_indices[i], i
                ]

    # Compute score function contribution
    # Score function: (f(x) - baseline) * (∇θ log π(x|θ))
    # where ∇θ log π(x|θ) = ∇θ a(x) - Σ π(y) * ∇θ a(y)

    gradient_estimates = np.zeros(len(theta_array))

    for i in range(len(theta_array)):
        if function_values.ndim == 1:
            # Single theta case
            sampled_function_vals = function_values[sampled_choice_indices]
            sampled_logits_grads = logits_gradients[sampled_choice_indices]
            score_center = score_function_center
        else:
            # Multiple theta case
            sampled_function_vals = function_values[sampled_choice_indices[i], i]
            sampled_logits_grads = logits_gradients[sampled_choice_indices[i], i]
            score_center = score_function_center[i]

        # Compute score function terms: (reward - baseline) * score
        reward_differences = sampled_function_vals - baseline_values[i]
        score_function_terms = reward_differences * (
            sampled_logits_grads - score_center
        )

        # Combine pathwise and score function contributions
        total_gradient_terms = pathwise_contribution[i] + score_function_terms

        # Return empirical mean as gradient estimate
        gradient_estimates[i] = np.mean(total_gradient_terms)

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


def gumbel_softmax_gradient(
    discrete_problem: DiscreteProblem,
    parameter_value: Union[float, np.ndarray],
    config: GumbelSoftmaxConfig,
) -> Union[float, np.ndarray]:
    """
    Estimate the gradient using the Gumbel-Softmax reparameterization trick
    with vectorized operations.

    Fully vectorized implementation for improved performance that supports
    both scalar and array inputs for efficient batch processing.

    Args:
        discrete_problem: The discrete optimization problem containing the
            function to differentiate and probability model.
        parameter_value: The parameter value(s) θ at which to estimate the gradient.
            Can be scalar or array.
        config: Configuration parameters for Gumbel-Softmax estimation.

    Returns:
        Union[float, np.ndarray]: Estimated gradient value(s). Returns scalar
            for scalar input, array for array input.

    Raises:
        ValueError: If the logits model does not provide gradient information
            (dlogits_dtheta) required for reparameterization.
    """
    # Convert to array for consistent handling
    theta_array = np.asarray(parameter_value)
    is_scalar_input = theta_array.ndim == 0

    if is_scalar_input:
        theta_array = theta_array.reshape(1)

    # Validate that we have the required gradient information
    if discrete_problem.logits_model.logits_derivative_function is None:
        raise ValueError(
            "Gumbel-Softmax requires logits_derivative_function for backpropagation."
        )

    # Get all required arrays with proper type conversion
    logits_gradients = np.asarray(
        discrete_problem.logits_model.logits_derivative_function(theta_array)
    )
    logits = np.asarray(discrete_problem.logits_model.logits_function(theta_array))
    function_values = np.asarray(discrete_problem.compute_function_values(theta_array))

    # Get pathwise gradients if available (optional)
    pathwise_gradients = discrete_problem.compute_derivative_values(theta_array)
    if pathwise_gradients is not None:
        pathwise_gradients = np.asarray(pathwise_gradients)

    num_theta = len(theta_array)
    gradient_estimates = np.zeros(num_theta)

    # Process each theta value
    for i in range(num_theta):
        # Extract values for current theta
        if logits.ndim == 1 and num_theta == 1:
            # Single theta case
            current_logits = logits
            current_logits_gradients = logits_gradients
            current_function_values = (
                function_values[:, 0] if function_values.ndim > 1 else function_values
            )
            current_pathwise_gradients = (
                pathwise_gradients[:, 0]
                if pathwise_gradients is not None and pathwise_gradients.ndim > 1
                else pathwise_gradients
            )
        else:
            # Multiple theta case
            current_logits = logits[:, i]
            current_logits_gradients = logits_gradients[:, i]
            current_function_values = function_values[:, i]
            current_pathwise_gradients = (
                pathwise_gradients[:, i] if pathwise_gradients is not None else None
            )

        # Generate all Gumbel noise samples in a single batch
        gumbel_noise = sample_gumbel(
            (config.num_samples, discrete_problem.num_branches),
            np.random.default_rng(0),
        )

        # Compute Gumbel-perturbed logits for all samples
        perturbed_logits = (
            current_logits + gumbel_noise
        )  # Shape: (num_samples, num_branches)

        # Compute softmax weights for all samples
        continuous_weights = softmax(
            perturbed_logits / config.temperature, axis=1
        )  # Shape: (num_samples, num_branches)

        # Compute pathwise gradient contributions (if available)
        pathwise_contribution = np.zeros(config.num_samples)
        if current_pathwise_gradients is not None:
            if config.use_straight_through_estimator:
                # STE: Use discrete sampling but continuous gradients
                best_choice_indices = np.argmax(
                    perturbed_logits, axis=1
                )  # Shape: (num_samples,)
                pathwise_contribution = current_pathwise_gradients[best_choice_indices]
            else:
                # Continuous relaxation using softmax
                pathwise_contribution = np.dot(
                    continuous_weights, current_pathwise_gradients
                )

        # Compute reparameterization gradient contributions
        mean_logits_gradient = np.dot(
            continuous_weights, current_logits_gradients
        )  # Shape: (num_samples,)

        softmax_gradient = (
            continuous_weights
            * (current_logits_gradients - mean_logits_gradient[:, np.newaxis])
        ) / config.temperature  # Shape: (num_samples, num_branches)

        reparameterization_contribution = np.sum(
            current_function_values * softmax_gradient, axis=1
        )  # Shape: (num_samples,)

        # Combine pathwise and reparameterization contributions
        total_gradient_terms = pathwise_contribution + reparameterization_contribution

        # Return empirical mean as final gradient estimate for this theta
        gradient_estimates[i] = np.mean(total_gradient_terms)

    # Return scalar if input was scalar, array otherwise
    if is_scalar_input:
        return float(gradient_estimates[0])
    return gradient_estimates
