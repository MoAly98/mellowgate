"""Core mathematical functions and problem definitions for discrete optimization.

This module defines the fundamental building blocks for discrete optimization problems,
including branch functions, logits models, and the complete discrete problem
formulation with exact gradient computation capabilities. All operations are fully
vectorized to efficiently handle arrays of theta values without loops, enabling
high-performance computations for large parameter spaces.

Key Features:
- Vectorized branch functions supporting array-based theta operations
- Efficient probability computation using vectorized softmax
- Exact gradient computation with policy gradient theorem
- Optimized sampling and stochastic value computation
- Full NumPy array broadcasting support throughout
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np

from mellowgate.utils.functions import softmax


def _default_probability_function(logits: np.ndarray) -> np.ndarray:
    """Default probability function that applies softmax along branch dimension."""
    if logits.ndim == 1:
        return softmax(logits)
    else:
        # For 2D logits (num_branches, num_theta), apply softmax along branch
        # dimension (axis=0)
        return softmax(logits, axis=0)


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
    one possible choice or path in the discrete decision space. Functions are
    vectorized to handle arrays of theta values efficiently.

    Attributes:
        function: A callable that takes a theta array and returns function values.
                  For single theta: returns scalar or 1D array.
                  For multiple theta: returns array with shape matching theta.
        derivative_function: Optional callable that returns the derivative of the
                           function with respect to theta. Required for exact
                           gradient computation. Same shape behavior as function.
        threshold: Optional tuple defining the range of theta values where this
                   branch is active. Each element in the tuple can be None:
                   - (None, upper): No lower threshold, active for theta < upper.
                   - (lower, None): No upper threshold, active for theta >= lower.
                   - (None, None): Always active.

    Examples:
        >>> import numpy as np
        >>> # Vectorized branch with trigonometric function
        >>> cos_branch = Branch(
        ...     function=lambda theta: np.cos(theta),
        ...     derivative_function=lambda theta: -np.sin(theta)
        ... )
    """

    function: Callable[[np.ndarray], np.ndarray]
    derivative_function: Optional[Callable[[np.ndarray], np.ndarray]] = None
    threshold: Optional[tuple[Optional[Bound], Optional[Bound]]] = (None, None)


@dataclass
class LogitsModel:
    """Represents the logits model for discrete probability distributions.

    The logits model computes probability distributions over branches using
    vectorized operations. Supports both single theta and arrays of theta values.

    Attributes:
        logits_function: A callable that takes theta array and returns logits.
                        For single theta: returns shape (num_branches,).
                        For multiple theta: returns shape (num_branches, num_theta).
        logits_derivative_function: Optional callable for logits derivatives.
                                   Same shape behavior as logits_function.
        probability_function: Optional callable to compute probabilities from logits.
                              Defaults to vectorized softmax with appropriate axis.

    Examples:
        >>> import numpy as np
        >>> # Vectorized logits model
        >>> logits_model = LogitsModel(
        ...     logits_function=lambda theta: np.array([theta, -theta]),
        ...     logits_derivative_function=lambda theta: np.array([
        ...         np.ones_like(theta), -np.ones_like(theta)
        ...     ])
        ... )
    """

    logits_function: Callable[[np.ndarray], np.ndarray]  # returns shape (K,) or (K, N)
    logits_derivative_function: Optional[Callable[[np.ndarray], np.ndarray]] = None
    probability_function: Callable[[np.ndarray], np.ndarray] = (
        _default_probability_function
    )


@dataclass
class DiscreteProblem:
    """Complete discrete optimization problem definition with vectorized operations.

    This class encapsulates a discrete optimization problem with multiple branches,
    each having associated functions and a logits model that determines the
    probability distribution over branches. Supports exact gradient computation
    when derivatives are available. All operations are fully vectorized for
    efficient computation with arrays of theta values without loops.

    Attributes:
        branches: List of Branch objects representing the discrete choices.
        logits_model: LogitsModel defining the probability distribution.
        sampling_function: Callable or random generator for sampling branches.
                          Supports vectorized operations for efficiency.

    Properties:
        num_branches: Number of branches in the problem.

    Examples:
        >>> import numpy as np
        >>> # Vectorized branches
        >>> branches = [
        ...     Branch(lambda th: th**2, lambda th: 2*th),
        ...     Branch(lambda th: th**3, lambda th: 3*th**2)
        ... ]
        >>> # Vectorized logits model
        >>> logits_model = LogitsModel(lambda th: np.array([th, -th]))
        >>> problem = DiscreteProblem(branches, logits_model)
        >>>
        >>> # Efficient computation for multiple theta values
        >>> theta_array = np.array([0.0, 1.0, 2.0])
        >>> probs = problem.compute_probabilities(theta_array)  # Shape: (2, 3)
        >>> expected = problem.compute_expected_value(theta_array)  # Shape: (3,)
    """

    branches: List[Branch]
    logits_model: LogitsModel
    sampling_function: Union[Callable[[np.ndarray], int], np.random.Generator] = (
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

    def compute_probabilities(self, theta: np.ndarray) -> np.ndarray:
        """Compute the probability distribution over branches for given theta values.

        This method computes the logits for the given theta array, transforms them
        into probabilities using the specified probability function (default:
        vectorized softmax), and ensures that the sum of probabilities is 1 for
        each theta value.

        Args:
            theta: Array of parameter values at which to compute probabilities.
                   Can be 1D array for multiple theta values.

        Returns:
            np.ndarray: Probability distribution over branches.
                       Shape (num_branches,) for single theta value.
                       Shape (num_branches, num_theta) for multiple theta values.

        Raises:
            ValueError: If the sum of probabilities is not approximately 1 for
                       any theta.

        Examples:
            >>> theta = np.array([0.0, 1.0, 2.0])
            >>> probs = problem.compute_probabilities(theta)
            >>> probs.shape  # (num_branches, 3)
        """
        logits = self.logits_model.logits_function(theta)
        probabilities = self.logits_model.probability_function(logits)

        # Ensure the sum of probabilities is approximately 1
        # For 2D arrays (num_branches, num_theta), check sum along branch
        # dimension (axis=0)
        if probabilities.ndim == 1:
            prob_sums = probabilities.sum()
            if not np.isclose(prob_sums, 1.0):
                raise ValueError(
                    f"Probabilities do not sum to 1 for theta={theta}. "
                    f"Sum is {prob_sums:.4f}."
                )
        else:
            prob_sums = probabilities.sum(axis=0)
            if not np.allclose(prob_sums, 1.0):
                raise ValueError(
                    f"Probabilities do not sum to 1 for theta={theta}. "
                    f"Sums are {prob_sums}."
                )

        return probabilities

    def compute_function_values_deterministic(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate all branch functions at the given theta values deterministically.

        This method evaluates functions considering thresholds. If a branch's
        threshold excludes the given theta, its function value will be set to NaN.
        Uses vectorized operations for efficient computation across multiple
        theta values.

        Args:
            theta: Array of parameter values at which to evaluate functions.

        Returns:
            numpy.ndarray: Array of function values.
                          Shape matches input theta for threshold-aware evaluation.

        Raises:
            ValueError: If theta is not a numpy array.

        Examples:
            >>> theta = np.array([0.0, 1.0, 2.0])
            >>> values = problem.compute_function_values_deterministic(theta)
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

    def compute_function_values(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate all branch functions at the given theta values using
        vectorized operations.

        This method evaluates all functions without considering thresholds.
        The branch selection based on sampled indices happens later.
        Handles both scalar and array inputs efficiently.

        Args:
            theta: The parameter value(s) at which to evaluate functions.
                   Can be scalar, 1D array, or higher dimensional.

        Returns:
            np.ndarray: Array of function values.
                       Shape (num_branches,) for single theta value.
                       Shape (num_branches, num_theta) for array of theta values.

        Examples:
            >>> theta = np.array([0.0, 1.0, 2.0])
            >>> values = problem.compute_function_values(theta)
            >>> values.shape  # (num_branches, 3)
        """
        # Ensure theta is an array for consistent handling
        theta_array = np.asarray(theta)

        results = []
        for branch in self.branches:
            result = branch.function(theta_array)
            # Handle both scalar and array outputs from branch functions
            if np.isscalar(result):
                result = np.array([result])
            elif hasattr(result, "squeeze"):
                result = result.squeeze()
                if result.ndim == 0:  # Convert 0d array to 1d
                    result = np.array([result])
            results.append(result)

        return np.array(results)

    def compute_derivative_values(self, theta: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate all branch function derivatives at the given theta values
        using vectorized operations.

        Args:
            theta: The parameter value(s) at which to evaluate derivatives.
                   Can be scalar, 1D array, or higher dimensional.

        Returns:
            numpy.ndarray: Array of derivative values.
                          Shape (num_branches,) for single theta value.
                          Shape (num_branches, num_theta) for array of theta values.
                          Returns None if any branch is missing its derivative function.

        Examples:
            >>> theta = np.array([0.0, 1.0, 2.0])
            >>> derivs = problem.compute_derivative_values(theta)
            >>> derivs.shape if derivs is not None  # (num_branches, 3)
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

    def compute_expected_value(self, theta: np.ndarray) -> np.ndarray:
        """Compute the expected value of the discrete problem using vectorized
        operations.

        This is the probability-weighted sum of all branch function values,
        computed efficiently for arrays of theta values.

        Args:
            theta: The parameter value(s) at which to evaluate the expectation.
                   Can be scalar, 1D array, or higher dimensional.

        Returns:
            np.ndarray: Expected value(s) of the discrete problem.
                       Shape matches input theta shape.
                       Scalar for single theta, array for multiple theta values.

        Formula:
            E[f] = sum_k p_k(theta) * f_k(theta)

        Examples:
            >>> theta = np.array([0.0, 1.0, 2.0])
            >>> expected = problem.compute_expected_value(theta)
            >>> expected.shape  # (3,)
        """
        probabilities = self.compute_probabilities(theta)
        function_values = self.compute_function_values(theta)
        return np.sum(probabilities * function_values, axis=0)

    def compute_exact_gradient(self, theta: np.ndarray) -> Optional[np.ndarray]:
        """Compute the exact gradient of the expected value with respect to theta
        using vectorized operations.

        This method computes the exact gradient using the policy gradient theorem
        and requires both logits derivatives and function derivatives to be
        available.
        All computations are vectorized for efficiency with arrays of theta values.

        Args:
            theta: The parameter value(s) at which to evaluate the gradient.
                   Can be scalar, 1D array, or higher dimensional.

        Returns:
            np.ndarray: Exact gradient values.
                       Shape matches input theta shape.
                       Returns None if any required derivative is missing.

        Formula:
            dE/dtheta = sum_k [p_k * df_k/dtheta + f_k * dp_k/dtheta]
            where dp_k/dtheta is computed using the chain rule through logits.

        Notes:
            Uses the policy gradient theorem for discrete distributions.
            Requires both function derivatives and logits derivatives.
            All operations are vectorized for computational efficiency.

        Examples:
            >>> theta = np.array([0.0, 1.0, 2.0])
            >>> grad = problem.compute_exact_gradient(theta)
            >>> grad.shape if grad is not None  # (3,)
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
        # For 2D arrays, compute mean along branch dimension (axis=0) for each theta
        mean_logits_gradient = np.sum(
            probabilities * logits_gradients, axis=0, keepdims=True
        )
        probability_gradients = probabilities * (
            logits_gradients - mean_logits_gradient
        )

        # Apply policy gradient theorem
        # For 2D arrays, sum along branch dimension (axis=0)
        function_term = np.sum(probabilities * function_derivatives, axis=0)
        probability_term = np.sum(function_values * probability_gradients, axis=0)

        return function_term + probability_term

    def sample_branch(self, theta: np.ndarray, num_samples: int = 1000) -> np.ndarray:
        """Sample branch indices based on the probability distribution using
        vectorized operations.

        Efficiently samples branches for each theta value in the input array,
        supporting both callable sampling functions and numpy random generators.

        Args:
            theta: Array of parameter values at which to sample.
                   Can be 1D array for multiple theta values.
            num_samples: Number of samples for each theta value.

        Returns:
            np.ndarray: Indices of the sampled branches.
                       Shape (len(theta), num_samples) for multiple theta values.
                       Shape (num_samples,) for single theta value.

        Notes:
            Uses the provided sampling function if available, otherwise defaults
            to vectorized sampling based on probabilities using the numpy generator.
            Optimized for computational efficiency with large theta arrays.

        Examples:
            >>> theta = np.array([0.0, 1.0, 2.0])
            >>> samples = problem.sample_branch(theta, num_samples=100)
            >>> samples.shape  # (3, 100)
        """
        probabilities = self.compute_probabilities(theta)
        # Shape: (num_branches, len(theta))

        match self.sampling_function:
            case func if callable(func):
                # Case 1: sampling_function is a Callable
                # Need to handle the callable case - for now, use loop but
                # optimize later
                if probabilities.ndim == 1:
                    # Single theta case
                    samples = np.fromiter(
                        (func(probabilities) for _ in range(num_samples)), dtype=int
                    )
                else:
                    # Multiple theta case
                    samples = np.array(
                        [
                            np.fromiter(
                                (func(probabilities[:, i]) for _ in range(num_samples)),
                                dtype=int,
                            )
                            for i in range(theta.shape[0])
                        ]
                    )
                return samples
            case generator if isinstance(generator, np.random.Generator):
                # Case 2: sampling_function is a np.random.Generator - can
                # vectorize this
                if probabilities.ndim == 1:
                    # Single theta case
                    return generator.choice(
                        self.num_branches, size=num_samples, p=probabilities
                    )
                else:
                    # Multiple theta case - vectorized sampling
                    samples = np.array(
                        [
                            generator.choice(
                                self.num_branches,
                                size=num_samples,
                                p=probabilities[:, i],
                            )
                            for i in range(theta.shape[0])
                        ]
                    )
                    return samples
            case _:
                raise ValueError(
                    "sampling_function must be either a Callable or a "
                    "np.random.Generator."
                )

    def compute_stochastic_values(
        self, theta: Union[float, np.ndarray], num_samples: int = 1000
    ) -> np.ndarray:
        """Compute stochastic values of the discrete problem using vectorized sampling.

        This method samples branches based on the probability distribution and
        evaluates their function values efficiently using vectorized operations.
        Handles both scalar and array theta inputs seamlessly.

        Args:
            theta: Parameter value(s) at which to evaluate.
                   Can be scalar, 1D array, or higher dimensional.
            num_samples: Number of samples for each theta value.

        Returns:
            np.ndarray: Stochastic values of the discrete problem.
                       Shape (1, num_samples) for scalar theta input.
                       Shape (len(theta), num_samples) for array theta input.

        Examples:
            >>> theta = np.array([0.0, 1.0, 2.0])
            >>> stoch_vals = problem.compute_stochastic_values(
            ...     theta, num_samples=100
            ... )
            >>> stoch_vals.shape  # (3, 100)

            >>> scalar_theta = 1.5
            >>> stoch_vals = problem.compute_stochastic_values(
            ...     scalar_theta, num_samples=100
            ... )
            >>> stoch_vals.shape  # (1, 100)
        """
        # Convert to array for consistent handling
        theta_array = np.asarray(theta)
        if theta_array.ndim == 0:  # Handle scalar input
            theta_array = theta_array.reshape(1)

        # Sample branch indices using the custom sampling function
        sampled_branches = self.sample_branch(theta_array, num_samples=num_samples)

        # Precompute function values for all branches
        function_values_at_choices = self.compute_function_values(theta_array)

        # Handle both single and multiple theta cases
        if function_values_at_choices.ndim == 1:
            # Single theta case: function_values is (num_branches,), samples is
            # (num_samples,)
            sampled_function_values = function_values_at_choices[sampled_branches]
        else:
            # Multiple theta case: function_values is (num_branches, num_theta),
            # samples is (num_theta, num_samples)
            if sampled_branches.ndim == 1:
                # Single theta case but with 2D function values
                sampled_function_values = function_values_at_choices[
                    sampled_branches, 0
                ]
            else:
                sampled_function_values = np.array(
                    [
                        function_values_at_choices[sampled_branches[i], i]
                        for i in range(theta_array.shape[0])
                    ]
                )

        return sampled_function_values
