"""Simplified unit tests for utils.functions module.

This module provides essential tests for mathematical utility functions.
"""

import jax.numpy as jnp

from mellowgate.utils.functions import softmax


class TestSoftmax:
    """Test softmax function for converting logits to probabilities."""

    def test_softmax_1d_basic(self):
        """Test basic 1D softmax functionality."""
        logits = jnp.array([1.0, 2.0, 3.0])
        probabilities = softmax(logits)

        # Check that probabilities sum to 1
        assert jnp.isclose(jnp.sum(probabilities), 1.0)
        # Check that all probabilities are positive
        assert jnp.all(probabilities > 0)
        # Check that highest logit gets highest probability
        assert jnp.argmax(probabilities) == jnp.argmax(logits)

    def test_softmax_2d_axis0(self):
        """Test 2D softmax along axis 0 (normalizing across branches)."""
        logits = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        probabilities = softmax(logits, axis=0)

        # Check that probabilities sum to 1 along axis 0
        column_sums = jnp.sum(probabilities, axis=0)
        assert jnp.allclose(column_sums, 1.0)

    def test_softmax_numerical_stability(self):
        """Test numerical stability with large values."""
        logits = jnp.array([1000.0, 1001.0, 1002.0])
        probabilities = softmax(logits)

        # Should not overflow or produce NaN/inf
        assert jnp.all(jnp.isfinite(probabilities))
        assert jnp.isclose(jnp.sum(probabilities), 1.0)

    def test_softmax_empty_array(self):
        """Test softmax with empty array."""
        logits = jnp.array([])
        probabilities = softmax(logits)

        # Should return empty array of same shape
        assert probabilities.shape == (0,)
        assert jnp.issubdtype(probabilities.dtype, jnp.floating)

    def test_softmax_scalar(self):
        """Test softmax with scalar input."""
        logits = 1.0
        probabilities = softmax(logits)

        # For a single value, probability should be 1.0
        assert probabilities == 1.0
        assert probabilities.shape == ()

    def test_softmax_2d_axis1(self):
        """Test 2D softmax along axis 1 (default axis)."""
        logits = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        probabilities = softmax(logits, axis=1)

        # Check that probabilities sum to 1 along axis 1
        row_sums = jnp.sum(probabilities, axis=1)
        assert jnp.allclose(row_sums, 1.0)

    def test_softmax_list_input(self):
        """Test softmax with list input."""
        logits = [1.0, 2.0, 3.0]
        probabilities = softmax(logits)

        # Should work the same as array input
        assert jnp.isclose(jnp.sum(probabilities), 1.0)
        assert jnp.all(probabilities > 0)

    def test_softmax_single_element(self):
        """Test softmax with single-element array."""
        logits = jnp.array([5.0])
        probabilities = softmax(logits)

        # Single element should have probability 1.0
        assert jnp.allclose(probabilities, jnp.array([1.0]))
        assert probabilities.shape == (1,)

    def test_softmax_known_values_1d(self):
        """Test softmax with known analytical values."""
        # Test case: logits = [0, ln(2), ln(3)]
        # Expected: softmax = [1/6, 2/6, 3/6] = [1/6, 1/3, 1/2]
        logits = jnp.array([0.0, jnp.log(2.0), jnp.log(3.0)])
        probabilities = softmax(logits)
        expected = jnp.array([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])

        assert jnp.allclose(probabilities, expected, rtol=1e-6)

    def test_softmax_known_values_equal_logits(self):
        """Test softmax with equal logits - should give uniform distribution."""
        logits = jnp.array([1.0, 1.0, 1.0, 1.0])
        probabilities = softmax(logits)
        expected = jnp.array([0.25, 0.25, 0.25, 0.25])

        assert jnp.allclose(probabilities, expected, rtol=1e-6)

    def test_softmax_known_values_2d_axis0(self):
        """Test 2D softmax along axis 0 with known values."""
        # Each column should normalize independently
        # Column 1: [0, ln(2)] -> [1/3, 2/3]
        # Column 2: [ln(3), 0] -> [3/4, 1/4]
        logits = jnp.array([[0.0, jnp.log(3.0)], [jnp.log(2.0), 0.0]])
        probabilities = softmax(logits, axis=0)
        expected = jnp.array([[1.0 / 3.0, 3.0 / 4.0], [2.0 / 3.0, 1.0 / 4.0]])

        assert jnp.allclose(probabilities, expected, rtol=1e-6)

    def test_softmax_known_values_2d_axis1(self):
        """Test 2D softmax along axis 1 with known values."""
        # Each row should normalize independently
        # Row 1: [0, ln(2)] -> [1/3, 2/3]
        # Row 2: [ln(4), 0] -> [4/5, 1/5]
        logits = jnp.array([[0.0, jnp.log(2.0)], [jnp.log(4.0), 0.0]])
        probabilities = softmax(logits, axis=1)
        expected = jnp.array([[1.0 / 3.0, 2.0 / 3.0], [4.0 / 5.0, 1.0 / 5.0]])

        assert jnp.allclose(probabilities, expected, rtol=1e-6)

    def test_softmax_temperature_scaling(self):
        """Test softmax with temperature scaling (dividing logits)."""
        logits = jnp.array([1.0, 2.0, 3.0])

        # High temperature (divide by 10) should make distribution more uniform
        high_temp_probs = softmax(logits / 10.0)

        # Low temperature (multiply by 10) should make distribution more peaked
        low_temp_probs = softmax(logits * 10.0)

        # Check that high temperature is more uniform (lower max prob)
        assert jnp.max(high_temp_probs) < jnp.max(low_temp_probs)

        # Check that low temperature is more peaked (higher max prob)
        assert jnp.max(low_temp_probs) > 0.99  # Should be very close to 1
