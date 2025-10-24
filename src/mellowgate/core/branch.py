"""Branch and Bound definitions for discrete optimization problems."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp


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
    derivative_function: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    threshold: tuple[Bound | None, Bound | None] | None = (None, None)
