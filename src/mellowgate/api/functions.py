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
class Branch:
    """Represents a single branch in a discrete optimization problem.

    A branch consists of a function and optionally its derivative, representing
    one possible choice or path in the discrete decision space.

    Attributes:
        function: A callable that takes a parameter theta and returns a scalar value.
        derivative_function: Optional callable that returns the derivative of the
                           function with respect to theta. Required for exact
                           gradient computation.

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


@dataclass
class LogitsModel:
    """Represents the logits model for discrete probability distributions.

    The logits model defines how the logits (log-odds) for each branch are computed
    as a function of the parameter theta. This is used to create probability
    distributions over the discrete choices.

    Attributes:
        logits_function: A callable that takes theta and returns an array of logits
                        with shape (num_branches,).
        logits_derivative_function: Optional callable that returns the derivative
                                  of logits with respect to theta. Required for
                                  exact gradient computation.

    Examples:
        >>> import numpy as np
        >>> # Linear logits model
        >>> alpha = np.array([1.0, -0.5, 0.0])
        >>> logits_model = LogitsModel(
        ...     logits_function=lambda theta: alpha * theta,
        ...     logits_derivative_function=lambda theta: alpha
        ... )
    """

    logits_function: Callable[[float], np.ndarray]  # returns shape (K,)
    logits_derivative_function: Optional[Callable[[float], np.ndarray]] = None


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
        random_generator: NumPy random generator for reproducible sampling.

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
    random_generator: np.random.Generator = np.random.default_rng(0)  # type: ignore

    @property
    def num_branches(self) -> int:
        """Get the number of branches in the problem.

        Returns:
            int: Number of branches (discrete choices) in the problem.
        """
        return len(self.branches)

    def compute_probabilities(self, theta: float) -> np.ndarray:
        """Compute the probability distribution over branches for given theta.

        Uses the softmax function to convert logits into a valid probability
        distribution that sums to 1.

        Args:
            theta: The parameter value at which to evaluate probabilities.

        Returns:
            numpy.ndarray: Probability distribution over branches with shape
                          (num_branches,). All values are in [0, 1] and sum to 1.
        """
        logits = self.logits_model.logits_function(theta)
        return softmax(logits)

    def compute_function_values(self, theta: float) -> np.ndarray:
        """Evaluate all branch functions at the given theta.

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
