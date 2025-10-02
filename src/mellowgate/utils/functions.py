"""Mathematical utility functions for discrete optimization.

This module provides core mathematical functions used throughout the mellowgate
library, particularly for probability computations and numerical operations.
The functions are implemented with numerical stability in mind and support
vectorized operations for efficient computation.

Key functions include softmax for converting logits to probabilities, with
proper handling of edge cases and numerical overflow prevention.
"""

from typing import Union

import jax.numpy as jnp

ArrayLike = Union[jnp.ndarray, list, tuple, float, int]


def softmax(logits: ArrayLike, axis: int = -1) -> jnp.ndarray:
    """Compute the softmax of the input logits using JAX.

    The softmax function converts a vector of real numbers into a probability
    distribution. This implementation uses JAX's numerically stable softmax
    with proper handling of edge cases.

    Args:
        logits: Input array-like object containing the logits values.
                Can be a JAX array, list, or tuple.
        axis: Axis along which to apply the softmax. Default is -1 (last axis).
              For 2D logits with shape (num_branches, num_theta), use axis=0.

    Returns:
        jax.numpy.ndarray: Probability distribution with the same shape as input.
                          All values are in [0, 1] and sum to 1 along the
                          specified axis.

    Examples:
        >>> import jax.numpy as jnp
        >>> # 1D case
        >>> logits = jnp.array([1.0, 2.0, 3.0])
        >>> probabilities = softmax(logits)
        >>> print(probabilities)
        [0.09003057 0.24472847 0.66524096]
        >>> print(jnp.sum(probabilities))
        1.0

        >>> # 2D case (num_branches=2, num_theta=3)
        >>> logits_2d = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        >>> probabilities_2d = softmax(logits_2d, axis=0)
        >>> print(probabilities_2d.sum(axis=0))  # Should be [1.0, 1.0, 1.0]

    Notes:
        Uses JAX's numerically stable softmax implementation with proper
        handling of scalar and empty array edge cases.
    """
    import jax.nn

    logits_array = jnp.asarray(logits)

    # Handle empty arrays
    if logits_array.size == 0:
        return logits_array.astype(jnp.float64)

    # Handle scalar case (0-dimensional array)
    if logits_array.ndim == 0:
        # For a scalar, softmax should return 1.0 (probability of single event)
        return jnp.array(1.0, dtype=jnp.float64)

    # For all other cases, use JAX's softmax
    return jax.nn.softmax(logits_array, axis=axis)
