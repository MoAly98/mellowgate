"""Statistical utilities for analyzing experimental results.

This module provides functions for computing statistical measures and metrics
commonly used in analyzing the performance of gradient estimation methods.
The functions support both single estimates and collections of estimates,
enabling comprehensive analysis of experimental results.

The utilities focus on variance estimation, bias analysis, and other metrics
relevant to evaluating the quality of gradient estimators.
"""

from typing import Tuple, Union

import jax
import jax.numpy as jnp

ShapeType = Union[int, Tuple[int, ...]]


def sample_gumbel(shape: ShapeType, key: jax.Array) -> jnp.ndarray:
    """Sample from the Gumbel distribution using the inverse transform method.

    The Gumbel distribution is used in the Gumbel-Softmax trick for creating
    differentiable samples from discrete distributions. This implementation
    uses the standard Gumbel distribution with location=0 and scale=1.

    Args:
        shape: The shape of the output array. Can be an integer for 1D arrays
               or a tuple of integers for multi-dimensional arrays.
        key: A JAX random key for reproducible random number generation.

    Returns:
        jax.numpy.ndarray: Array of Gumbel-distributed random samples with the
                          specified shape.

    Examples:
        >>> import jax
        >>> key = jax.random.PRNGKey(42)
        >>> samples = sample_gumbel((3, 2), key)
        >>> print(samples.shape)
        (3, 2)

    Notes:
        Uses the inverse transform method: G = -log(-log(U)) where U ~ Uniform(0,1).
        This is the standard Gumbel distribution with CDF F(x) = exp(-exp(-x)).
    """
    # Ensure shape is a tuple for JAX
    if isinstance(shape, int):
        shape = (shape,)

    # Sample from uniform distribution using JAX
    uniform_samples = jax.random.uniform(key, shape, minval=0.0, maxval=1.0)
    # Apply inverse transform for Gumbel distribution
    # Add small epsilon for numerical stability
    return -jnp.log(-jnp.log(uniform_samples + 1e-8))


def sigmoid_2d(logits: jnp.ndarray) -> jnp.ndarray:
    """Custom sigmoid function for 2D arrays.

    Args:
        logits: A 2D array of logits with shape (num_branches, num_samples).

    Returns:
        jax.numpy.ndarray: Sigmoid probabilities with the same shape as logits.

    Examples:
        >>> logits = jnp.array([[1, 2], [3, 4]])
        >>> probabilities = sigmoid_2d(logits)
        >>> print(probabilities)
        [[0.73105858 0.88079708]
         [0.95257413 0.98201379]]
    """
    exp_logits = jnp.exp(-logits)
    return 1 / (1 + exp_logits)
