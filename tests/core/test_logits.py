"""Tests for mellowgate.core.logits module."""

import jax.numpy as jnp

from mellowgate.core import LogitsModel


class TestLogitsModel:
    """Test the LogitsModel dataclass."""

    def test_logits_model_creation(self):
        """Test creating a LogitsModel."""

        def logits_func(th):
            return jnp.array([th, -th])

        model = LogitsModel(logits_function=logits_func)
        assert model.logits_function is logits_func

    def test_logits_model_with_custom_probability_function(self):
        """Test LogitsModel with custom probability function."""

        def logits_func(th):
            return jnp.array([th, -th])

        def custom_prob_func(logits):
            # Custom: just normalize without exp
            return logits / jnp.sum(logits, axis=0, keepdims=True)

        model = LogitsModel(
            logits_function=logits_func, probability_function=custom_prob_func
        )
        assert model.logits_function is logits_func
        assert model.probability_function is custom_prob_func
