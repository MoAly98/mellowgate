"""Logits model for discrete probability distributions."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from mellowgate.utils.functions import softmax


def _default_probability_function(logits: jnp.ndarray) -> jnp.ndarray:
    """Default probability function that applies softmax along branch dimension."""
    if logits.ndim == 1:
        return softmax(logits)
    # For 2D logits (num_branches, num_theta), apply softmax along branch
    # dimension (axis=0)
    return softmax(logits, axis=0)


@dataclass
class LogitsModel:
    """Represents the logits model for discrete probability distributions.

    The logits model computes probability distributions over branches using
    vectorized operations. Supports both single theta and arrays of theta values.
    Gradient computation is handled automatically through JAX's automatic
    differentiation system.

    Attributes:
        logits_function: A callable that takes theta array and returns logits.
                        For single theta: returns shape (num_branches,).
                        For multiple theta: returns shape (num_branches, num_theta).
        probability_function: Optional callable to compute probabilities from logits.
                              Defaults to vectorized softmax with appropriate axis.

    Examples:
        >>> import jax.numpy as jnp
        >>> # Vectorized logits model
        >>> logits_model = LogitsModel(
        ...     logits_function=lambda theta: jnp.array([theta, -theta])
        ... )
    """

    logits_function: Callable[
        [jnp.ndarray], jnp.ndarray
    ]  # returns shape (K,) or (K, N)
    probability_function: Callable[[jnp.ndarray], jnp.ndarray] = (
        _default_probability_function
    )
