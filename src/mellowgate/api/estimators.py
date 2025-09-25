"""
Gradient estimators for discrete optimization problems.

This module provides various gradient estimation methods for discrete decision problems,
including finite differences, REINFORCE, and Gumbel-Softmax approaches. These methods
enable differentiable optimization of discrete choices through stochastic relaxations.
"""

from dataclasses import dataclass

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
    parameter_value: float,
    config: FiniteDifferenceConfig,
) -> float:
    """
    Estimate the gradient using finite differences.

    This method approximates the gradient by evaluating the function at
    parameter_value + step_size and parameter_value - step_size, then computing
    the central difference: (f(θ+ε) - f(θ-ε)) / (2ε).

    Args:
        discrete_problem (DiscreteProblem): The discrete optimization problem
            containing the function to differentiate and probability model.
        parameter_value (float): The parameter value θ at which to estimate
            the gradient.
        config (FiniteDifferenceConfig): Configuration containing step size
            and number of Monte Carlo samples.

    Returns:
        float: Estimated gradient value at the given parameter.

    Notes:
        The method uses Monte Carlo sampling to estimate expectations at each
        evaluation point, making it stochastic. The accuracy depends on both
        the step size and number of samples.
    """

    def _monte_carlo_expectation(theta: float) -> float:
        """
        Estimate E[f(θ)] using Monte Carlo sampling.

        Args:
            theta: Parameter value for evaluation.

        Returns:
            Monte Carlo estimate of the expectation.
        """
        sampled_values = discrete_problem.compute_stochastic_values(
            theta, num_samples=config.num_samples
        )
        return float(np.mean(sampled_values))

    # Evaluate expectations at perturbed parameter values
    expectation_at_plus = _monte_carlo_expectation(parameter_value + config.step_size)
    expectation_at_minus = _monte_carlo_expectation(parameter_value - config.step_size)

    # Compute central difference approximation
    gradient_estimate = (expectation_at_plus - expectation_at_minus) / (
        2 * config.step_size
    )

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
    parameter_value: float,
    config: ReinforceConfig,
    state: ReinforceState,
) -> float:
    """
    Estimate the gradient using the REINFORCE algorithm.

    REINFORCE uses the policy gradient theorem to estimate gradients by sampling
    from the current policy and weighting by rewards. The gradient estimator is:
    ∇θ E[f(θ)] ≈ E[f(x) * ∇θ log π(x|θ)] + E[∇θ f(x)]

    The first term is the score function (REINFORCE) and the second is the
    pathwise derivative when available.

    Args:
        discrete_problem: The discrete optimization problem containing the
            function to differentiate and probability model.
        parameter_value: The parameter value θ at which to estimate the gradient.
        config: Configuration parameters for REINFORCE estimation.
        state: State object to maintain baseline across calls.

    Returns:
        Estimated gradient value.

    Raises:
        ValueError: If the logits model does not provide gradient information
            (dlogits_dtheta) required for the score function.

    Notes:
        - Uses baseline for variance reduction if enabled in config
        - Combines pathwise gradients (when available) with score function
        - Updates baseline state for future calls
    """
    # Ensure all arrays are properly typed as numpy arrays
    choice_probabilities = np.asarray(
        discrete_problem.compute_probabilities(parameter_value)
    )
    function_values = np.asarray(
        discrete_problem.compute_function_values(parameter_value)
    )

    # Get pathwise gradients if available (optional)
    pathwise_gradients = discrete_problem.compute_derivative_values(parameter_value)
    if pathwise_gradients is not None:
        pathwise_gradients = np.asarray(pathwise_gradients)

    # Check that we have the required gradient information for score function
    if discrete_problem.logits_model.logits_derivative_function is None:
        raise ValueError(
            "REINFORCE requires logits_derivative_function for score function."
        )

    logits_gradients = np.asarray(
        discrete_problem.logits_model.logits_derivative_function(parameter_value)
    )

    # Compute score function center term: E[∇θ log π(x|θ)] = Σ π(x) * ∇θ a(x)
    # where a(x) are the logits and π(x) = softmax(a(x))
    score_function_center = float(np.dot(choice_probabilities, logits_gradients))

    # Sample discrete choices according to current policy
    random_generator = discrete_problem.random_generator
    sampled_choice_indices = random_generator.choice(
        discrete_problem.num_branches, size=config.num_samples, p=choice_probabilities
    )

    # Compute or update baseline for variance reduction
    baseline_value = 0.0
    if config.use_baseline:
        # Get rewards for sampled choices
        sampled_rewards = function_values[sampled_choice_indices]
        current_mean_reward = float(np.mean(sampled_rewards))

        # Use existing baseline or initialize with current mean
        if state.initialized:
            baseline_value = state.baseline
        else:
            baseline_value = current_mean_reward

        # Update baseline for future use
        state.update_baseline(current_mean_reward, config.baseline_momentum)

    # Compute pathwise gradient contribution (if available)
    pathwise_contribution = 0.0
    if pathwise_gradients is not None:
        pathwise_contribution = pathwise_gradients[sampled_choice_indices]

    # Compute score function contribution
    # Score function: (f(x) - baseline) * (∇θ log π(x|θ))
    # where ∇θ log π(x|θ) = ∇θ a(x) - Σ π(y) * ∇θ a(y)
    sampled_function_values = function_values[sampled_choice_indices]
    sampled_logits_gradients = logits_gradients[sampled_choice_indices]

    # Compute score function terms: (reward - baseline) * score
    reward_differences = sampled_function_values - baseline_value
    score_function_terms = reward_differences * (
        sampled_logits_gradients - score_function_center
    )

    # Combine pathwise and score function contributions
    total_gradient_terms = pathwise_contribution + score_function_terms

    # Return empirical mean as gradient estimate
    return float(np.mean(total_gradient_terms))


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
    parameter_value: float,
    config: GumbelSoftmaxConfig,
) -> float:
    """
    Estimate the gradient using the Gumbel-Softmax reparameterization trick.

    The Gumbel-Softmax method provides a differentiable relaxation of discrete
    sampling by adding Gumbel noise to logits and applying temperature-scaled
    softmax. This enables backpropagation through discrete decisions.

    The gradient estimator combines:
    1. Pathwise gradients (when available): ∇θ f(x)
    2. Reparameterization gradients: f(x) * ∇θ π(x|θ)

    Args:
        discrete_problem: The discrete optimization problem containing the
            function to differentiate and probability model.
        parameter_value: The parameter value θ at which to estimate the gradient.
        config: Configuration parameters for Gumbel-Softmax estimation.

    Returns:
        Estimated gradient value.

    Raises:
        ValueError: If the logits model does not provide gradient information
            (dlogits_dtheta) required for reparameterization.

    Notes:
        - When use_straight_through_estimator=True, uses discrete sampling
          in forward pass but continuous gradients in backward pass
        - Temperature controls the trade-off between discrete and continuous
          behavior
        - Requires differentiable logits model for reparameterization
    """
    # Validate that we have the required gradient information
    if discrete_problem.logits_model.logits_derivative_function is None:
        raise ValueError(
            "Gumbel-Softmax requires logits_derivative_function for backpropagation."
        )

    # Get all required arrays with proper type conversion
    logits_gradients = np.asarray(
        discrete_problem.logits_model.logits_derivative_function(parameter_value)
    )
    logits = np.asarray(discrete_problem.logits_model.logits_function(parameter_value))
    function_values = np.asarray(
        discrete_problem.compute_function_values(parameter_value)
    )

    # Get pathwise gradients if available (optional)
    pathwise_gradients = discrete_problem.compute_derivative_values(parameter_value)
    if pathwise_gradients is not None:
        pathwise_gradients = np.asarray(pathwise_gradients)

    # Collect gradient estimates across multiple samples
    gradient_estimates = []

    for sample_idx in range(config.num_samples):
        # Sample Gumbel noise for reparameterization
        gumbel_noise = np.asarray(
            sample_gumbel(
                discrete_problem.num_branches, discrete_problem.random_generator
            )
        )

        # Compute Gumbel-perturbed logits
        perturbed_logits = logits + gumbel_noise  # type: ignore

        # Compute pathwise gradient contribution
        pathwise_contribution = 0.0
        if config.use_straight_through_estimator:
            # STE: Use discrete sampling but continuous gradients
            best_choice_index = int(np.argmax(perturbed_logits))
            if pathwise_gradients is not None:
                pathwise_contribution = pathwise_gradients[best_choice_index]
        else:
            # Continuous relaxation using softmax
            continuous_weights = softmax(perturbed_logits / config.temperature)
            if pathwise_gradients is not None:
                pathwise_contribution = float(
                    np.dot(continuous_weights, pathwise_gradients)
                )

        # Compute reparameterization gradient contribution
        # This term captures how the discrete choice probabilities change with θ
        continuous_weights = softmax(perturbed_logits / config.temperature)

        # Compute weighted average of logits gradients
        mean_logits_gradient = float(np.dot(continuous_weights, logits_gradients))

        # Compute softmax gradient:
        # ∇θ softmax = softmax * (∇θ logits - mean_∇θ_logits) / τ
        softmax_gradient = (
            continuous_weights
            * (logits_gradients - mean_logits_gradient)  # type: ignore
        ) / config.temperature

        # Weight by function values to get reparameterization term
        reparameterization_contribution = float(
            np.sum(function_values * softmax_gradient)
        )

        # Combine both contributions for this sample
        total_gradient = pathwise_contribution + reparameterization_contribution
        gradient_estimates.append(total_gradient)

    # Return empirical mean as final gradient estimate
    return float(np.mean(gradient_estimates))
