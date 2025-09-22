"""Statistical utility functions for stochastic gradient estimation.

This module provides statistical sampling functions used in gradient estimation
algorithms, particularly for generating random variates from various distributions.
"""

from typing import Tuple, Union

import numpy as np

ShapeType = Union[int, Tuple[int, ...]]


def sample_gumbel(
    shape: ShapeType, random_generator: np.random.Generator
) -> np.ndarray:
    """Sample from the Gumbel distribution using the inverse transform method.

    The Gumbel distribution is used in the Gumbel-Softmax trick for creating
    differentiable samples from discrete distributions. This implementation
    uses the standard Gumbel distribution with location=0 and scale=1.

    Args:
        shape: The shape of the output array. Can be an integer for 1D arrays
               or a tuple of integers for multi-dimensional arrays.
        random_generator: A numpy random generator instance for reproducible
                         random number generation.

    Returns:
        numpy.ndarray: Array of Gumbel-distributed random samples with the
                      specified shape.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> samples = sample_gumbel((3, 2), rng)
        >>> print(samples.shape)
        (3, 2)

    Notes:
        Uses the inverse transform method: G = -log(-log(U)) where U ~ Uniform(0,1).
        This is the standard Gumbel distribution with CDF F(x) = exp(-exp(-x)).
    """
    # Sample from uniform distribution
    uniform_samples = random_generator.uniform(0.0, 1.0, shape)
    # Apply inverse transform for Gumbel distribution
    return -np.log(-np.log(uniform_samples))
