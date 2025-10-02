"""Core mathematical functions and problem definitions for discrete optimization.

This module provides the fundamental building blocks for defining and solving
discrete optimization problems where decisions involve choosing between multiple
branches or paths. The module implements vectorized operations throughout to
efficiently handle both single parameter values and arrays of parameters.

The main components are:
- Branch: Represents a single choice/path with its associated function
- LogitsModel: Defines probability distributions over branches using logits
- DiscreteProblem: Complete problem formulation with sampling and gradient computation

All operations support NumPy array broadcasting and are optimized for performance
with large parameter spaces.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import jax
import jax.numpy as jnp
from sigmatch import SignatureMatcher, SignatureMismatchError

from mellowgate.utils.functions import softmax


def _default_probability_function(logits: jnp.ndarray) -> jnp.ndarray:
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
        >>> import jax.numpy as jnp
        >>> # Vectorized branch with trigonometric function
        >>> cos_branch = Branch(
        ...     function=lambda theta: jnp.cos(theta),
        ...     derivative_function=lambda theta: -jnp.sin(theta)
        ... )
    """

    function: Callable[[jnp.ndarray], jnp.ndarray]
    derivative_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
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
        >>> import jax.numpy as jnp
        >>> # Vectorized logits model
        >>> logits_model = LogitsModel(
        ...     logits_function=lambda theta: jnp.array([theta, -theta]),
        ...     logits_derivative_function=lambda theta: jnp.array([
        ...         jnp.ones_like(theta), -jnp.ones_like(theta)
        ...     ])
        ... )
    """

    logits_function: Callable[
        [jnp.ndarray], jnp.ndarray
    ]  # returns shape (K,) or (K, N)
    logits_derivative_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    probability_function: Callable[[jnp.ndarray], jnp.ndarray] = (
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
        sampling_function:  Optional callable for custom sampling. Must implement with
                            signature:
                            (probabilities: jnp.ndarray, key: jax.Array) -> jnp.ndarray
                            Will be integrated into the vectorized sampling pipeline.

    Properties:
        num_branches: Number of branches in the problem.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax
        >>> # Vectorized branches
        >>> branches = [
        ...     Branch(lambda th: th**2, lambda th: 2*th),
        ...     Branch(lambda th: th**3, lambda th: 3*th**2)
        ... ]
        >>> # Vectorized logits model
        >>> logits_model = LogitsModel(lambda th: jnp.array([th, -th]))
        >>>
        >>> # Custom sampling function (automatically validated)
        >>> def custom_sampler(probs: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        ...     return jax.random.choice(key, len(probs), shape=(), p=probs)
        >>>
        >>> problem = DiscreteProblem(branches, logits_model, custom_sampler)
        >>>
        >>> # Efficient computation for multiple theta values
        >>> theta_array = jnp.array([0.0, 1.0, 2.0])
        >>> probs = problem.compute_probabilities(theta_array)  # Shape: (2, 3)
        >>> expected = problem.compute_expected_value(theta_array)  # Shape: (3,)
    """

    branches: List[Branch]
    logits_model: LogitsModel
    sampling_function: Optional[Callable[[jnp.ndarray, jax.Array], int]] = None

    def __post_init__(self):
        """Validate the sampling function signature after initialization."""
        if self.sampling_function is not None:
            self._validate_sampling_function()

    def _validate_sampling_function(self) -> None:
        """Validate that the sampling function has the correct signature.

        Raises:
            SignatureMismatchError: If the function signature doesn't match the
                                    expected signature.
            ValueError: If the function doesn't behave correctly with test inputs.
        """
        if self.sampling_function is None:
            return

        try:
            SignatureMatcher(".", ".").match(
                self.sampling_function, raise_exception=True
            )
        except SignatureMismatchError as e:
            raise e

        # Test the function with sample inputs to ensure it works correctly
        try:
            test_probs = jnp.array([0.3, 0.7])
            test_key = jax.random.PRNGKey(42)
            result = self.sampling_function(test_probs, test_key)

            # Validate the result
            if not isinstance(result, jnp.ndarray):
                raise ValueError(
                    f"Sampling function must return a jnp.ndarray, got {type(result)}"
                )

            if result.shape != ():
                raise ValueError(
                    "Sampling function must return a scalar array (shape ()), "
                    f"got shape {result.shape}.\n"
                    "Hint: Use jax.random.choice(key, len(probs), "
                    "shape=(), p=probs)"
                )

            if not (0 <= result < len(test_probs)):
                raise ValueError(
                    f"Sampling function returned invalid index {result}, "
                    f"must be in range [0, {len(test_probs)})"
                )

        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            else:
                raise ValueError(
                    f"Sampling function failed validation test: {str(e)}\n"
                    f"Make sure your function works with test inputs: "
                    f"probabilities=jnp.array([0.3, 0.7]), key=jax.random.PRNGKey(42)"
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
        theta: jnp.ndarray,
        threshold: Optional[tuple[Optional[Bound], Optional[Bound]]],
    ) -> jnp.ndarray:
        """Generate conditions for jnp.piecewise based on an array of theta values
        and a threshold.

        Args:
            theta: Array of parameter values to check.
            threshold: The threshold to check against. Can be:
                - (lower, upper): Active for lower < or <= theta < or <= upper.
                - (None, upper): No lower threshold, active for theta < or <= upper.
                - (lower, None): No upper threshold, active for theta > or >= lower.
                - (None, None): Always active.

        Returns:
            jnp.ndarray: Boolean array indicating whether each theta satisfies the
            threshold.

        Raises:
            ValueError: If theta is not an array.
        """
        if not isinstance(theta, jnp.ndarray):
            raise ValueError("Theta must be a numpy array.")
        if threshold is None or threshold == (None, None):
            return jnp.ones_like(theta, dtype=bool)
        lower, upper = threshold
        conditions = jnp.ones_like(theta, dtype=bool)
        if lower is not None:
            conditions &= (
                theta >= lower.value if lower.inclusive else theta > lower.value
            )
        if upper is not None:
            conditions &= (
                theta < upper.value if upper.inclusive else theta <= upper.value
            )
        return conditions

    def compute_probabilities(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the probability distribution over branches for given theta values.

        This method computes the logits for the given theta array, transforms them
        into probabilities using the specified probability function (default:
        vectorized softmax), and ensures that the sum of probabilities is 1 for
        each theta value.

        Args:
            theta: Array of parameter values at which to compute probabilities.
                   Shape: (N,) for N theta values

        Returns:
            jnp.ndarray: Probability distribution over branches.
                       Shape: (num_branches, N) for N theta values
                       Each column sums to 1.0

        Raises:
            ValueError: If the sum of probabilities is not approximately 1 for
                       any theta.

        Examples:
            >>> theta = jnp.array([0.0, 1.0, 2.0])  # Shape: (3,)
            >>> probs = problem.compute_probabilities(theta)
            >>> probs.shape  # (num_branches, 3)
        """
        # logits shape: (num_branches, N) where N = len(theta)
        logits = self.logits_model.logits_function(theta)

        # probabilities shape: (num_branches, N)
        probabilities = self.logits_model.probability_function(logits)

        # Validate probability sums along branch dimension (axis=0)
        if probabilities.ndim == 1:
            # Shape: (num_branches,) - single theta case
            prob_sums = probabilities.sum()
            if not jnp.isclose(prob_sums, 1.0):
                raise ValueError(
                    f"Probabilities do not sum to 1 for theta={theta}. "
                    f"Sum is {prob_sums:.4f}."
                )
        else:
            # Shape: (num_branches, N) - multiple theta case
            # prob_sums shape: (N,) - one sum per theta value
            prob_sums = probabilities.sum(axis=0)
            if not jnp.allclose(prob_sums, 1.0):
                raise ValueError(
                    f"Probabilities do not sum to 1 for theta={theta}. "
                    f"Sums are {prob_sums}."
                )

        return probabilities

    def compute_function_values_deterministic(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Evaluate all branch functions at the given theta values deterministically.

        This method evaluates functions considering thresholds. If a branch's
        threshold excludes the given theta, its function value will be set to NaN.
        Uses vectorized operations for efficient computation across multiple
        theta values.

        Args:
            theta: Array of parameter values at which to evaluate functions.

        Returns:
            jax.numpy.ndarray: Array of function values.
                          Shape matches input theta for threshold-aware evaluation.

        Raises:
            ValueError: If theta is not a numpy array.

        Examples:
            >>> theta = jnp.array([0.0, 1.0, 2.0])
            >>> values = problem.compute_function_values_deterministic(theta)
        """
        if not isinstance(theta, jnp.ndarray):
            raise ValueError("Theta must be a numpy array.")

        # Handle empty theta arrays
        if theta.size == 0:
            return jnp.empty((self.num_branches, 0))

        # Compute function values for each branch at all theta values
        result = jnp.zeros((self.num_branches, theta.size))

        for i, branch in enumerate(self.branches):
            conditions = self.generate_threshold_conditions(theta, branch.threshold)
            # Where conditions are met, compute function values; otherwise NaN
            values = jnp.full_like(theta, jnp.nan, dtype=float)
            values = values.at[conditions].set(branch.function(theta[conditions]))
            result = result.at[i, :].set(values)

        return result

    def compute_function_values(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Evaluate all branch functions at the given theta values using
        vectorized operations.

        This method evaluates all functions without considering thresholds.
        The branch selection based on sampled indices happens later.
        Handles both scalar and array inputs efficiently.

        Args:
            theta: The parameter value(s) at which to evaluate functions.
                   Shape: (N,) for N theta values

        Returns:
            jnp.ndarray: Array of function values.
                       Shape: (num_branches, N) for N theta values
                       Each row corresponds to one branch function

        Examples:
            >>> theta = jnp.array([0.0, 1.0, 2.0])  # Shape: (3,)
            >>> values = problem.compute_function_values(theta)
            >>> values.shape  # (num_branches, 3)
        """
        # Ensure theta is an array for consistent handling
        theta_array = jnp.asarray(theta)  # Shape: (N,)

        results = []
        for branch in self.branches:
            # Each branch.function(theta_array) returns shape: (N,)
            result = branch.function(theta_array)

            # Handle both scalar and array outputs from branch functions
            if jnp.isscalar(result):
                result = jnp.array([result])
            elif hasattr(result, "squeeze"):
                result = result.squeeze()
                if result.ndim == 0:  # Convert 0d array to 1d
                    result = jnp.array([result])
            results.append(result)

        # Stack results to get shape: (num_branches, N)
        return jnp.array(results)

    def compute_derivative_values(self, theta: jnp.ndarray) -> Optional[jnp.ndarray]:
        """Evaluate all branch function derivatives at the given theta values
        using vectorized operations.

        Args:
            theta: The parameter value(s) at which to evaluate derivatives.
                   Can be scalar, 1D array, or higher dimensional.

        Returns:
            jax.numpy.ndarray: Array of derivative values.
                          Shape (num_branches,) for single theta value.
                          Shape (num_branches, num_theta) for array of theta values.
                          Returns None if any branch is missing its derivative function.

        Examples:
            >>> theta = jnp.array([0.0, 1.0, 2.0])
            >>> derivs = problem.compute_derivative_values(theta)
            >>> derivs.shape if derivs is not None  # (num_branches, 3)
        """
        if any(branch.derivative_function is None for branch in self.branches):
            return None

        return jnp.array(
            [
                branch.derivative_function(theta)  # type: ignore
                for branch in self.branches
                if branch.derivative_function is not None
            ]
        )

    def compute_expected_value(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the expected value of the discrete problem using vectorized
        operations.

        This is the probability-weighted sum of all branch function values,
        computed efficiently for arrays of theta values.

        Args:
            theta: The parameter value(s) at which to evaluate the expectation.
                   Can be scalar, 1D array, or higher dimensional.

        Returns:
            jnp.ndarray: Expected value(s) of the discrete problem.
                       Shape matches input theta shape.
                       Scalar for single theta, array for multiple theta values.

        Formula:
            E[f] = sum_k p_k(theta) * f_k(theta)

        Examples:
            >>> theta = jnp.array([0.0, 1.0, 2.0])
            >>> expected = problem.compute_expected_value(theta)
            >>> expected.shape  # (3,)
        """
        probabilities = self.compute_probabilities(theta)
        function_values = self.compute_function_values(theta)
        return jnp.sum(probabilities * function_values, axis=0)

    def compute_exact_gradient(
        self, theta: Union[float, jnp.ndarray]
    ) -> Optional[Union[float, jnp.ndarray]]:
        """Compute the exact gradient of the expected value with respect to theta
        using vectorized operations.

        This method computes the exact gradient using the policy gradient theorem
        and requires both logits derivatives and function derivatives to be
        available.
        All computations are vectorized for efficiency with arrays of theta values.
        Supports both scalar and array inputs for maximum flexibility.

        Args:
            theta: The parameter value(s) at which to evaluate the gradient.
                   Can be scalar, 1D array, or higher dimensional.

        Returns:
            Union[float, jnp.ndarray]: Exact gradient values.
                                     Returns scalar for scalar input,
                                     array for array input.
                                     Returns None if any required derivative is missing.

        Formula:
            dE/dtheta = sum_k [p_k * df_k/dtheta + f_k * dp_k/dtheta]
            where dp_k/dtheta is computed using the chain rule through logits.

        Notes:
            Uses the policy gradient theorem for discrete distributions.
            Requires both function derivatives and logits derivatives.
            All operations are vectorized for computational efficiency.

        Examples:
            >>> theta = jnp.array([0.0, 1.0, 2.0])
            >>> grad = problem.compute_exact_gradient(theta)
            >>> grad.shape if grad is not None  # (3,)
            >>>
            >>> scalar_theta = 1.0
            >>> scalar_grad = problem.compute_exact_gradient(scalar_theta)
            >>> type(scalar_grad)  # float
        """
        # Convert to array for consistent handling
        theta_array = jnp.asarray(theta)
        is_scalar_input = theta_array.ndim == 0

        if is_scalar_input:
            theta_array = theta_array.reshape(1)

        # Check if logits derivatives are available
        if self.logits_model.logits_derivative_function is None:
            return None

        # Check if function derivatives are available
        function_derivatives = self.compute_derivative_values(theta_array)
        if function_derivatives is None:
            return None

        # Compute required quantities
        probabilities = self.compute_probabilities(theta_array)
        function_values = self.compute_function_values(theta_array)

        # Compute probability gradients using automatic differentiation
        # This is more general and handles any probability function correctly
        def compute_probabilities_for_theta(theta_single):
            """Wrapper to compute probabilities for a single theta value."""
            logits = self.logits_model.logits_function(theta_single)
            return self.logits_model.probability_function(logits)

        # Use JAX jacfwd to compute the Jacobian (gradient of vector-valued function)
        prob_jacobian_fn = jax.jacfwd(compute_probabilities_for_theta)

        if theta_array.shape[0] == 1:
            # Single theta case
            probability_gradients = prob_jacobian_fn(theta_array[0])
            probability_gradients = probability_gradients.reshape(-1, 1)
        else:
            # Multiple theta case - vectorize the Jacobian computation
            prob_jacobian_vectorized = jax.vmap(prob_jacobian_fn)
            probability_gradients = prob_jacobian_vectorized(theta_array).T

        # Apply policy gradient theorem
        # For 2D arrays, sum along branch dimension (axis=0)
        function_term = jnp.sum(probabilities * function_derivatives, axis=0)
        probability_term = jnp.sum(function_values * probability_gradients, axis=0)

        gradient_result = function_term + probability_term

        # Return scalar if input was scalar, array otherwise
        if is_scalar_input:
            return float(gradient_result[0])
        return gradient_result

    def sample_branch(
        self,
        theta: jnp.ndarray,
        num_samples: int = 1000,
        key: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """Sample branch indices based on the probability distribution using
        vectorized operations.

        Performance optimizations:
        - Batched key splitting for efficient key generation
        - Vectorized sampling across all theta values using vmap
        - JIT compilation for operation fusion and reduced overhead
        - Custom sampling functions integrated into vectorized pipeline

        Efficiently samples branches for each theta value in the input array.
        Uses either a custom sampling function or JAX's categorical sampling,
        both integrated into the same vectorized computation pipeline.

        Args:
            theta: Array of parameter values at which to sample.
                   Shape: (N,) for N theta values
            num_samples: Number of samples for each theta value.
            key: JAX random key for sampling. If None, defaults to PRNGKey(0).

        Returns:
            jnp.ndarray: Indices of the sampled branches.
                       Shape: (N, num_samples) for N theta values
                       Each row contains samples for one theta value
                       Values are integers in range [0, num_branches)

        Notes:
            Both custom sampling functions and JAX categorical sampling use the same
            vectorized pipeline for optimal performance. Custom functions should
            expect (probabilities, key) and return a single sample index.

        Examples:
            >>> theta = jnp.array([0.0, 1.0, 2.0])  # Shape: (3,)
            >>> key = jax.random.PRNGKey(42)
            >>> samples = problem.sample_branch(theta, num_samples=100, key=key)
            >>> samples.shape  # (3, 100)
        """
        # probabilities shape: (num_branches, N) where N = len(theta)
        probabilities = self.compute_probabilities(theta)

        if key is None:
            key = jax.random.PRNGKey(0)  # Default key for reproducibility

        # Define the sampling function to use (custom or default)
        if self.sampling_function is not None:
            # Use user-provided sampling function in vectorized pipeline
            user_sampling_fn = self.sampling_function  # Type guard for mypy

            def sampling_fn(probs, key_sample):
                return user_sampling_fn(probs, key_sample)

        else:
            # Default JAX categorical sampling function
            def sampling_fn(probs, key_sample):
                logits = jnp.log(probs + 1e-8)  # Add epsilon for numerical stability
                return jax.random.categorical(key_sample, logits)

        if probabilities.ndim == 1:
            # Single theta case: probabilities shape (num_branches,)
            # Generate keys for each sample
            sample_keys = jax.random.split(key, num_samples)

            # Vectorized sampling using vmap over samples
            samples = jax.vmap(lambda k: sampling_fn(probabilities, k))(sample_keys)
            return samples
        else:
            # Multiple theta case: probabilities shape (num_branches, N)
            # Batched key generation for efficient sampling
            keys = jax.random.split(key, theta.shape[0])

            # Vectorized sampling function for one theta value
            def sample_for_theta(key_i, probs_i):
                # Generate keys for each sample for this theta
                sample_keys = jax.random.split(key_i, num_samples)
                # Vectorized sampling across samples using vmap
                return jax.vmap(lambda k: sampling_fn(probs_i, k))(sample_keys)

            # Single vmap call for vectorized sampling across all theta values
            samples = jax.vmap(sample_for_theta)(keys, probabilities.T)
            return samples

    def compute_stochastic_values(
        self,
        theta: Union[float, jnp.ndarray],
        num_samples: int = 1000,
        key: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """Compute stochastic values of the discrete problem using vectorized sampling.

        Performance optimizations:
        - Single batched gather using jnp.take_along_axis for efficient indexing
        - Vectorized operations across samples
        - JIT compilation for operation fusion and reduced overhead
        - Precomputes function values once and reuses efficiently

        This method samples branches based on the probability distribution and
        evaluates their function values efficiently using vectorized operations.
        Handles both scalar and array theta inputs seamlessly.

        Args:
            theta: Parameter value(s) at which to evaluate.
                   Can be scalar, 1D array, or higher dimensional.
            num_samples: Number of samples for each theta value.
            key: JAX random key for sampling. If None, defaults to PRNGKey(0).

        Returns:
            jnp.ndarray: Stochastic values of the discrete problem.
                       Shape (1, num_samples) for scalar theta input.
                       Shape (len(theta), num_samples) for array theta input.

        Examples:
            >>> theta = jnp.array([0.0, 1.0, 2.0])
            >>> key = jax.random.PRNGKey(42)
            >>> stoch_vals = problem.compute_stochastic_values(
            ...     theta, num_samples=100, key=key
            ... )
            >>> stoch_vals.shape  # (3, 100)

            >>> scalar_theta = 1.5
            >>> stoch_vals = problem.compute_stochastic_values(
            ...     scalar_theta, num_samples=100, key=key
            ... )
            >>> stoch_vals.shape  # (1, 100)
        """
        # Convert to array for consistent handling
        theta_array = jnp.asarray(theta)
        if theta_array.ndim == 0:  # Handle scalar input
            theta_array = theta_array.reshape(1)

        # Handle empty array input
        if theta_array.size == 0:
            return jnp.empty((0, num_samples))

        # Precompute function values for all branches once - efficient computation
        function_values_at_choices = self.compute_function_values(theta_array)

        # Sample branch indices using vectorized sampling
        sampled_branches = self.sample_branch(
            theta_array, num_samples=num_samples, key=key
        )

        # Handle both single and multiple theta cases with efficient batched gather
        if function_values_at_choices.ndim == 1:
            # Single theta case: function_values is (num_branches,),
            # samples is (num_samples,)
            sampled_function_values = jnp.take(
                function_values_at_choices, sampled_branches, axis=0
            )
        else:
            # Multiple theta case: function_values is (num_branches, num_theta)
            # samples is (num_theta, num_samples)
            if sampled_branches.ndim == 1:
                # Single theta case but with 2D function values
                sampled_function_values = jnp.take(
                    function_values_at_choices[:, 0], sampled_branches, axis=0
                )
            else:
                # Efficient batched gather using take_along_axis for advanced indexing
                # Prepare indices for advanced indexing
                theta_indices = jnp.arange(theta_array.shape[0])[
                    :, jnp.newaxis
                ]  # Shape: (num_theta, 1)
                theta_indices_expanded = jnp.broadcast_to(
                    theta_indices, sampled_branches.shape
                )  # Shape: (num_theta, num_samples)

                # Single batched gather operation for efficient indexing
                sampled_function_values = function_values_at_choices[
                    sampled_branches, theta_indices_expanded
                ]

        return sampled_function_values
