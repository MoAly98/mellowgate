"""Core mathematical functions and problem definitions for discrete optimization.

This module defines the fundamental building blocks for discrete optimization problems,
including branch functions, logits models, and the complete discrete problem formulation
with exact gradient computation capabilities.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from mellowgate.utils.functions import softmax


@dataclass
class Bound:
    """Represents a bound (lower or upper) for a parameter.

    Attributes:
        value: The numerical value of the bound.
        inclusive: Whether the bound is inclusive (True) or exclusive (False).
    """

    value: float
    inclusive: bool = True


@dataclass
class Branch:
    """Represents a single branch in a discrete optimization problem.

    A branch consists of a function and optionally its derivative, representing
    one possible choice or path in the discrete decision space.

    Attributes:
        function: A callable that takes a parameter theta and returns a scalar value.
        derivative_function: Optional callable that returns the derivative of the
                           function with respect to theta. Required for exact
                           gradient computation.
        threshold: Optional tuple defining the range of theta values where this
                   branch is active. Each element in the tuple can be None:
                   - (None, upper): No lower threshold, active for theta < upper.
                   - (lower, None): No upper threshold, active for theta >= lower.
                   - (None, None): Always active.

    Examples:
        >>> import numpy as np
        >>> # Branch with trigonometric function
        >>> cos_branch = Branch(
        ...     function=lambda theta: float(np.cos(theta)),
        ...     derivative_function=lambda theta: float(-np.sin(theta))
        ... )
    """

    function: Callable[[float], float]
    derivative_function: Optional[Callable[[float], float]] = None
    threshold: Optional[tuple[Optional[Bound], Optional[Bound]]] = (None, None)


@dataclass
class LogitsModel:
    """Represents the logits model for discrete probability distributions.

    Attributes:
        logits_function: A callable that takes theta and returns an array of logits.
        logits_derivative_function: Optional callable for logits derivatives.
        probability_function: Optional callable to compute probabilities from logits.
                              Defaults to softmax.
    """

    logits_function: Callable[[float], np.ndarray]  # returns shape (K,)
    logits_derivative_function: Optional[Callable[[float], np.ndarray]] = None
    probability_function: Callable[[np.ndarray], np.ndarray] = softmax


@dataclass
class DiscreteProblem:
    """Complete discrete optimization problem definition.

    This class encapsulates a discrete optimization problem with multiple branches,
    each having associated functions and a logits model that determines the
    probability distribution over branches. Supports exact gradient computation
    when derivatives are available.

    Attributes:
        branches: List of Branch objects representing the discrete choices.
        logits_model: LogitsModel defining the probability distribution.
        sampling_function: Callable or random generator for sampling branches.

    Properties:
        num_branches: Number of branches in the problem.

    Examples:
        >>> import numpy as np
        >>> branches = [
        ...     Branch(lambda x: x**2, lambda x: 2*x),
        ...     Branch(lambda x: x**3, lambda x: 3*x**2)
        ... ]
        >>> logits_model = LogitsModel(lambda x: np.array([x, -x]))
        >>> problem = DiscreteProblem(branches, logits_model)
    """

    branches: List[Branch]
    logits_model: LogitsModel
    sampling_function: Callable[[np.ndarray], int] | np.random.Generator = (
        np.random.default_rng(0)
    )

    @property
    def num_branches(self) -> int:
        """Get the number of branches in the problem.

        Returns:
            int: Number of branches (discrete choices) in the problem.
        """
        return len(self.branches)

    def generate_threshold_conditions(
        self,
        theta: np.ndarray,
        threshold: Optional[tuple[Optional[Bound], Optional[Bound]]],
    ) -> np.ndarray:
        """Generate conditions for np.piecewise based on an array of theta values
        and a threshold.

        Args:
            theta: Array of parameter values to check.
            threshold: The threshold to check against. Can be:
                - (lower, upper): Active for lower < or <= theta < or <= upper.
                - (None, upper): No lower threshold, active for theta < or <= upper.
                - (lower, None): No upper threshold, active for theta > or >= lower.
                - (None, None): Always active.

        Returns:
            np.ndarray: Boolean array indicating whether each theta satisfies the
            threshold.

        Raises:
            ValueError: If theta is not an array.
        """
        if not isinstance(theta, np.ndarray):
            raise ValueError("Theta must be a numpy array.")
        if threshold is None or threshold == (None, None):
            return np.ones_like(theta, dtype=bool)
        lower, upper = threshold
        conditions = np.ones_like(theta, dtype=bool)
        if lower is not None:
            conditions &= (
                theta >= lower.value if lower.inclusive else theta > lower.value
            )
        if upper is not None:
            conditions &= (
                theta < upper.value if upper.inclusive else theta <= upper.value
            )
        return conditions

    def compute_probabilities(self, theta: float) -> np.ndarray:
        """Compute the probability distribution over branches for given theta.

        This method computes the logits for the given theta, transforms them into
        probabilities using the specified probability function (default: softmax),
        and ensures that the sum of probabilities is 1.

        Args:
            theta: The parameter value for which to compute probabilities.

        Returns:
            np.ndarray: Probability distribution over branches.

        Raises:
            ValueError: If the sum of probabilities is not approximately 1.
        """
        logits = self.logits_model.logits_function(theta)
        probabilities = self.logits_model.probability_function(logits)

        # Ensure the sum of probabilities is approximately 1
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError(
                f"Probabilities do not sum to 1 for theta={theta}. "
                f"Sum is {probabilities.sum():.4f}."
            )

        return probabilities

    def compute_function_values_deterministic(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate all branch functions at the given theta deterministically.

        This method evaluates functions considering thresholds. If a branch's
        threshold excludes the given theta, its function value will be set to NaN.

        Args:
            theta: Array of parameter values at which to evaluate functions.

        Returns:
            numpy.ndarray: Array of function values with shape (num_branches,).

        Raises:
            ValueError: If theta is not a numpy array.
        """
        if not isinstance(theta, np.ndarray):
            raise ValueError("Theta must be a numpy array.")
        conditions = [
            self.generate_threshold_conditions(theta, branch.threshold)
            for branch in self.branches
        ]
        functions = [
            lambda t, func=branch.function: func(t) for branch in self.branches
        ]
        return np.piecewise(theta, conditions, functions)

    def compute_function_values(self, theta: float) -> np.ndarray:
        """Evaluate all branch functions at the given theta.

        This method evaluates all functions without considering thresholds.
        The branch selection based on sampled indices happens later.

        Args:
            theta: The parameter value at which to evaluate functions.

        Returns:
            numpy.ndarray: Array of function values with shape (num_branches,).
        """
        return np.array([branch.function(theta) for branch in self.branches])

    def compute_derivative_values(self, theta: float) -> Optional[np.ndarray]:
        """Evaluate all branch function derivatives at the given theta.

        Args:
            theta: The parameter value at which to evaluate derivatives.

        Returns:
            numpy.ndarray: Array of derivative values with shape (num_branches,)
                          if all branches have derivatives defined.
            None: If any branch is missing its derivative function.
        """
        if any(branch.derivative_function is None for branch in self.branches):
            return None

        return np.array(
            [
                branch.derivative_function(theta)  # type: ignore
                for branch in self.branches
                if branch.derivative_function is not None
            ]
        )

    def compute_expected_value(self, theta: float) -> float:
        """Compute the expected value of the discrete problem at theta.

        This is the probability-weighted sum of all branch function values.

        Args:
            theta: The parameter value at which to evaluate the expectation.

        Returns:
            float: Expected value of the discrete problem.

        Formula:
            E[f] = sum_k p_k(theta) * f_k(theta)
        """
        probabilities = self.compute_probabilities(theta)
        function_values = self.compute_function_values(theta)
        return float(np.sum(probabilities * function_values))

    def compute_exact_gradient(self, theta: float) -> Optional[float]:
        """Compute the exact gradient of the expected value with respect to theta.

        This method computes the exact gradient using the policy gradient theorem
        and requires both logits derivatives and function derivatives to be available.

        Args:
            theta: The parameter value at which to evaluate the gradient.

        Returns:
            float: Exact gradient value if all derivatives are available.
            None: If any required derivative is missing.

        Formula:
            dE/dtheta = sum_k [p_k * df_k/dtheta + f_k * dp_k/dtheta]
            where dp_k/dtheta is computed using the chain rule through logits.

        Notes:
            Uses the policy gradient theorem for discrete distributions.
            Requires both function derivatives and logits derivatives.
        """
        # Check if logits derivatives are available
        if self.logits_model.logits_derivative_function is None:
            return None

        # Check if function derivatives are available
        function_derivatives = self.compute_derivative_values(theta)
        if function_derivatives is None:
            return None

        # Compute required quantities
        probabilities = self.compute_probabilities(theta)
        function_values = self.compute_function_values(theta)
        logits_gradients = self.logits_model.logits_derivative_function(theta)

        # Compute probability gradients using chain rule through softmax
        mean_logits_gradient = np.sum(probabilities * logits_gradients)
        probability_gradients = probabilities * (
            logits_gradients - mean_logits_gradient
        )

        # Apply policy gradient theorem
        function_term = np.sum(probabilities * function_derivatives)
        probability_term = np.sum(function_values * probability_gradients)

        return float(function_term + probability_term)

    def sample_branch(self, theta: float, num_samples: int = 1000) -> np.ndarray:
        """Sample branch indices based on the probability distribution.

        Args:
            theta: The parameter value at which to sample.
            num_samples: Number of samples for stochastic sampling.

        Returns:
            np.ndarray: Indices of the sampled branches.

        Notes:
            Uses the provided sampling function if available, otherwise defaults
            to sampling based on probabilities using the numpy generator.
        """
        probabilities = self.compute_probabilities(theta)

        match self.sampling_function:
            case func if callable(func):
                # Case 1: sampling_function is a Callable
                samples = np.fromiter(
                    (func(probabilities) for _ in range(num_samples)), dtype=int
                )
                return samples
            case generator if isinstance(generator, np.random.Generator):
                # Case 2: sampling_function is a np.random.Generator
                return generator.choice(
                    self.num_branches, size=num_samples, p=probabilities
                )
            case _:
                raise ValueError(
                    "sampling_function must be either a \
                        Callable or a np.random.Generator."
                )

    def compute_stochastic_values(
        self, theta: float, num_samples: int = 1000
    ) -> np.ndarray:
        """Compute stochastic values of the discrete problem at theta.

        This method samples branches based on the probability distribution and
        evaluates their function values.

        Args:
            theta: The parameter value at which to evaluate.
            num_samples: Number of samples for stochastic sampling.

        Returns:
            np.ndarray: Stochastic values of the discrete problem.
        """
        # Sample branch indices using the custom sampling function
        sampled_branches = self.sample_branch(theta, num_samples=num_samples)

        # Precompute function values for all branches
        function_values_at_choices = self.compute_function_values(theta)

        # Index the precomputed function values using sampled branches
        sampled_function_values = function_values_at_choices[sampled_branches]

        return sampled_function_values
