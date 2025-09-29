"""Simplified unit tests for utils.functions module.

This module provides essential tests for mathematical utility functions.
"""

import numpy as np

from mellowgate.utils.functions import softmax


class TestSoftmax:
    """Test softmax function for converting logits to probabilities."""

    def test_softmax_1d_basic(self):
        """Test basic 1D softmax functionality."""
        logits = np.array([1.0, 2.0, 3.0])
        probabilities = softmax(logits)

        # Check that probabilities sum to 1
        assert np.isclose(np.sum(probabilities), 1.0)
        # Check that all probabilities are positive
        assert np.all(probabilities > 0)
        # Check that highest logit gets highest probability
        assert np.argmax(probabilities) == np.argmax(logits)

    def test_softmax_2d_axis0(self):
        """Test 2D softmax along axis 0 (normalizing across branches)."""
        logits = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        probabilities = softmax(logits, axis=0)

        # Check that probabilities sum to 1 along axis 0
        column_sums = np.sum(probabilities, axis=0)
        assert np.allclose(column_sums, 1.0)

    def test_softmax_numerical_stability(self):
        """Test numerical stability with large values."""
        logits = np.array([1000.0, 1001.0, 1002.0])
        probabilities = softmax(logits)

        # Should not overflow or produce NaN/inf
        assert np.all(np.isfinite(probabilities))
        assert np.isclose(np.sum(probabilities), 1.0)

    def test_softmax_empty_array(self):
        """Test softmax with empty array."""
        logits = np.array([])
        probabilities = softmax(logits)

        # Should return empty array of same shape
        assert probabilities.shape == (0,)
        assert probabilities.dtype == float
