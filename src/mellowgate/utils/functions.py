"""Utility functions for mathematical operations in mellowgate.

This module provides mathematical utility functions commonly used throughout
the mellowgate library, particularly for discrete optimization problems.
"""

import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, list, tuple]


def softmax(logits: ArrayLike) -> np.ndarray:
    """Compute the softmax of the input logits.

    The softmax function converts a vector of real numbers into a probability
    distribution. This implementation uses the numerically stable version that
    subtracts the maximum value to prevent overflow.

    Args:
        logits: Input array-like object containing the logits values.
                Can be a numpy array, list, or tuple.

    Returns:
        numpy.ndarray: Probability distribution with the same shape as input.
                      All values are in [0, 1] and sum to 1.

    Examples:
        >>> import numpy as np
        >>> logits = np.array([1.0, 2.0, 3.0])
        >>> probabilities = softmax(logits)
        >>> print(probabilities)
        [0.09003057 0.24472847 0.66524096]
        >>> print(np.sum(probabilities))
        1.0

    Notes:
        Uses the numerically stable softmax formulation:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    logits_array = np.asarray(logits)
    # Subtract max for numerical stability
    normalized_logits = logits_array - np.max(logits_array)
    exponentials = np.exp(normalized_logits)
    return exponentials / np.sum(exponentials)
