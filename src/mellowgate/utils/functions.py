"""Utility functions for mathematical operations in mellowgate.

This module provides mathematical utility functions commonly used throughout
the mellowgate library, particularly for discrete optimization problems.
"""

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]


def softmax(logits: ArrayLike, axis: int = -1) -> np.ndarray:
    """Compute the softmax of the input logits.

    The softmax function converts a vector of real numbers into a probability
    distribution. This implementation uses the numerically stable version that
    subtracts the maximum value to prevent overflow.

    Args:
        logits: Input array-like object containing the logits values.
                Can be a numpy array, list, or tuple. For 2D arrays with shape
                (num_branches, num_theta), normalizes along the branch dimension.
        axis: Axis along which to apply the softmax. Default is -1 (last axis).
              For 2D logits with shape (num_branches, num_theta), use axis=0.

    Returns:
        numpy.ndarray: Probability distribution with the same shape as input.
                      All values are in [0, 1] and sum to 1 along the specified axis.

    Examples:
        >>> import numpy as np
        >>> # 1D case
        >>> logits = np.array([1.0, 2.0, 3.0])
        >>> probabilities = softmax(logits)
        >>> print(probabilities)
        [0.09003057 0.24472847 0.66524096]
        >>> print(np.sum(probabilities))
        1.0

        >>> # 2D case (num_branches=2, num_theta=3)
        >>> logits_2d = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        >>> probabilities_2d = softmax(logits_2d, axis=0)
        >>> print(probabilities_2d.sum(axis=0))  # Should be [1.0, 1.0, 1.0]

    Notes:
        Uses the numerically stable softmax formulation:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        Applied along the specified axis for multi-dimensional arrays.
    """
    logits_array = np.asarray(logits)

    # Handle empty arrays
    if logits_array.size == 0:
        return logits_array.astype(float)

    # Handle both 1D and multi-dimensional cases
    if logits_array.ndim == 1:
        # Original 1D behavior
        normalized_logits = logits_array - np.max(logits_array)
        exponentials = np.exp(normalized_logits)
        return exponentials / np.sum(exponentials)
    else:
        # Multi-dimensional case with specified axis
        # Subtract max for numerical stability along the specified axis
        max_logits = np.max(logits_array, axis=axis, keepdims=True)
        normalized_logits = logits_array - max_logits
        exponentials = np.exp(normalized_logits)
        sum_exponentials = np.sum(exponentials, axis=axis, keepdims=True)
        return exponentials / sum_exponentials
