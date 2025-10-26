"""Configuration settings for mellowgate.

This module handles global configuration settings for the mellowgate library,
including numerical precision settings, platform configuration, and logging
settings for JAX.
"""

import logging

import jax


def configure_platforms(platforms: str = "cpu,gpu") -> None:
    """Configure JAX platforms to use.

    Args:
        platforms: Comma-separated list of platforms to enable.
                  Options: 'cpu', 'gpu', 'tpu' or combinations like 'cpu,gpu'.
                  Default: 'cpu' to avoid TPU initialization warnings on most systems.

    Note:
        This must be called before importing JAX to take effect.
        Setting to 'cpu' eliminates TPU/GPU initialization warnings.
    """
    jax.config.update("jax_platforms", platforms)


def configure_logging(level: int = logging.WARNING) -> None:
    """Configure JAX logging to suppress unnecessary warnings.

    Args:
        level: Logging level for JAX components. Default: WARNING to suppress
               INFO messages about platform initialization.

    Note:
        This helps reduce noise from JAX backend initialization messages.
    """
    # Suppress JAX XLA bridge warnings (like TPU initialization failures)
    logging.getLogger("jax._src.xla_bridge").setLevel(level)
    # Suppress other JAX warnings that may be noisy
    logging.getLogger("jax._src.lib.xla_bridge").setLevel(level)


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


def configure_mellowgate(
    enable_x64: bool = True, platforms: str = "cpu", log_level: int = logging.WARNING
) -> None:
    """Configure all mellowgate JAX settings in one call.

    Args:
        enable_x64: Enable float64 precision for numerical accuracy.
        platforms: JAX platforms to enable ('cpu', 'gpu', 'tpu', or combinations).
        log_level: Logging level for JAX messages.

    Note:
        This is a convenience function that applies all common JAX configurations
        for mellowgate usage.
    """
    configure_platforms(platforms)
    configure_logging(log_level)
    configure_precision(enable_x64)


# Apply default configuration when mellowgate is imported
configure_mellowgate()
