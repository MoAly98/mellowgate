"""Configuration settings for mellowgate.

This module handles global configuration settings for the mellowgate library,
including numerical precision settings for JAX.
"""

import jax


# Configure JAX for high precision numerical computations
def configure_precision(enable_x64: bool = True) -> None:
    """Configure JAX numerical precision.

    Args:
        enable_x64: If True, enables float64/int64 precision. If False, uses
                   float32/int32 (JAX default).

    Note:
        This should be called before any JAX computations to ensure consistent
        precision throughout the library. By default, mellowgate enables x64
        precision for improved numerical accuracy in gradient computations.
    """
    jax.config.update("jax_enable_x64", enable_x64)


# Enable high precision by default when mellowgate is imported
configure_precision(True)
