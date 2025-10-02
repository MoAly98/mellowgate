"""Tests for mellowgate.utils.functions module."""

import jax.numpy as jnp

from mellowgate.utils.functions import softmax


class TestSoftmax:
    """Test the softmax utility function."""

    def test_softmax_basic_1d(self):
        """Test basic softmax computation with 1D array."""
        logits = jnp.array([1.0, 2.0, 3.0])
        probs = softmax(logits)

        # Check that probabilities sum to 1
        assert jnp.allclose(jnp.sum(probs), 1.0)

        # Check that all probabilities are positive
        assert jnp.all(probs > 0)

        # Check shape
        assert probs.shape == logits.shape

        # Check that largest logit gives largest probability
        assert jnp.argmax(probs) == jnp.argmax(logits)

    def test_softmax_2d_axis_0(self):
        """Test softmax computation with 2D array along axis 0."""
        logits = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        probs = softmax(logits, axis=0)

        # Check that probabilities sum to 1 along axis 0
        assert jnp.allclose(jnp.sum(probs, axis=0), 1.0)

        # Check shape
        assert probs.shape == logits.shape

        # Check that all probabilities are positive
        assert jnp.all(probs > 0)

    def test_softmax_2d_axis_1(self):
        """Test softmax computation with 2D array along axis 1."""
        logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        probs = softmax(logits, axis=1)

        # Check that probabilities sum to 1 along axis 1
        assert jnp.allclose(jnp.sum(probs, axis=1), 1.0)

        # Check shape
        assert probs.shape == logits.shape

        # Check that all probabilities are positive
        assert jnp.all(probs > 0)

    def test_softmax_default_axis(self):
        """Test softmax with default axis (-1)."""
        logits = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        probs = softmax(logits)  # Default axis=-1

        # Should be equivalent to axis=1 for 2D array
        probs_axis1 = softmax(logits, axis=1)
        assert jnp.allclose(probs, probs_axis1)

    def test_softmax_zero_logits(self):
        """Test softmax with zero logits."""
        logits = jnp.array([0.0, 0.0, 0.0])
        probs = softmax(logits)

        # Should give uniform distribution
        expected = jnp.array([1 / 3, 1 / 3, 1 / 3])
        assert jnp.allclose(probs, expected)

    def test_softmax_large_logits(self):
        """Test softmax with large logits (numerical stability)."""
        logits = jnp.array([100.0, 101.0, 102.0])
        probs = softmax(logits)

        # Should not overflow
        assert jnp.all(jnp.isfinite(probs))
        assert jnp.allclose(jnp.sum(probs), 1.0)

        # Largest logit should dominate
        assert jnp.argmax(probs) == 2

    def test_softmax_negative_logits(self):
        """Test softmax with negative logits."""
        logits = jnp.array([-10.0, -5.0, -1.0])
        probs = softmax(logits)

        # Should work correctly
        assert jnp.allclose(jnp.sum(probs), 1.0)
        assert jnp.all(probs > 0)

        # Less negative should have higher probability
        assert jnp.argmax(probs) == 2

    def test_softmax_single_element(self):
        """Test softmax with single element."""
        logits = jnp.array([5.0])
        probs = softmax(logits)

        # Should give probability 1
        assert jnp.allclose(probs, jnp.array([1.0]))

    def test_softmax_scalar_input(self):
        """Test softmax with scalar input (0-dimensional array)."""
        # Test with scalar array (ndim == 0)
        scalar_logits = jnp.array(5.0)
        probs = softmax(scalar_logits)

        # Should return 1.0 with float64 dtype
        assert jnp.allclose(probs, 1.0)
        assert probs.dtype == jnp.float64
        assert probs.shape == ()  # Scalar shape

        # Test with different scalar values
        scalar_logits_2 = jnp.array(-10.0)
        probs_2 = softmax(scalar_logits_2)
        assert jnp.allclose(probs_2, 1.0)
        assert probs_2.dtype == jnp.float64

    def test_softmax_extreme_values(self):
        """Test softmax with extreme values."""
        # Very large difference
        logits = jnp.array([-1000.0, 0.0])
        probs = softmax(logits)

        # First element should be essentially 0, second should be 1
        assert probs[0] < 1e-10
        assert jnp.allclose(probs[1], 1.0, atol=1e-10)

    def test_softmax_broadcast_compatibility(self):
        """Test softmax with broadcasting scenarios."""
        # Test with different shaped arrays
        logits_1d = jnp.array([1.0, 2.0])
        logits_2d = jnp.array([[1.0, 2.0]])

        probs_1d = softmax(logits_1d)
        probs_2d = softmax(logits_2d, axis=1)

        # Results should be equivalent
        assert jnp.allclose(probs_1d, probs_2d.squeeze())


class TestSoftmaxNumericalStability:
    """Test numerical stability of softmax implementation."""

    def test_softmax_overflow_protection(self):
        """Test that softmax handles potential overflow."""
        # These values would cause exp() to overflow without proper scaling
        logits = jnp.array([1000.0, 1001.0, 1002.0])
        probs = softmax(logits)

        # Should not contain NaN or Inf
        assert jnp.all(jnp.isfinite(probs))
        assert jnp.allclose(jnp.sum(probs), 1.0)

    def test_softmax_underflow_protection(self):
        """Test that softmax handles potential underflow."""
        # Very negative values
        logits = jnp.array([-1000.0, -999.0, -998.0])
        probs = softmax(logits)

        # Should not contain NaN or Inf
        assert jnp.all(jnp.isfinite(probs))
        assert jnp.allclose(jnp.sum(probs), 1.0)

        # Largest (least negative) should dominate
        assert jnp.argmax(probs) == 2

    def test_softmax_mixed_extreme_values(self):
        """Test softmax with mixed extreme values."""
        logits = jnp.array([-1000.0, 0.0, 1000.0])
        probs = softmax(logits)

        # Middle and last should have significant probability
        # First should be negligible
        assert probs[0] < 1e-10
        assert probs[2] > 0.99  # Last element should dominate

    def test_softmax_precision_consistency(self):
        """Test that softmax gives consistent results."""
        logits = jnp.array([1.0, 2.0, 3.0])

        # Compute softmax multiple times
        probs1 = softmax(logits)
        probs2 = softmax(logits)

        # Should be identical
        assert jnp.allclose(probs1, probs2, rtol=1e-15)


class TestSoftmaxEdgeCases:
    """Test edge cases for softmax function."""

    def test_softmax_empty_array(self):
        """Test softmax with empty array."""
        logits = jnp.array([])
        probs = softmax(logits)

        # Should return empty array
        assert probs.shape == (0,)

    def test_softmax_3d_array(self):
        """Test softmax with 3D array."""
        logits = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        probs = softmax(logits, axis=-1)

        # Check shape preservation
        assert probs.shape == logits.shape

        # Check that last axis sums to 1
        assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0)

    def test_softmax_axis_validation(self):
        """Test softmax with different axis values."""
        logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Test axis=0
        probs_0 = softmax(logits, axis=0)
        assert jnp.allclose(jnp.sum(probs_0, axis=0), 1.0)

        # Test axis=1
        probs_1 = softmax(logits, axis=1)
        assert jnp.allclose(jnp.sum(probs_1, axis=1), 1.0)

        # Test axis=-1 (equivalent to axis=1 for 2D)
        probs_neg1 = softmax(logits, axis=-1)
        assert jnp.allclose(probs_1, probs_neg1)

    def test_softmax_dtype_preservation(self):
        """Test that softmax preserves input dtype when possible."""
        # Test with float32
        logits_f32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        probs_f32 = softmax(logits_f32)
        assert probs_f32.dtype == jnp.float32

        # Test with float64
        logits_f64 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        probs_f64 = softmax(logits_f64)
        assert probs_f64.dtype == jnp.float64


class TestSoftmaxMathematicalProperties:
    """Test mathematical properties of softmax."""

    def test_softmax_translation_invariance(self):
        """Test that softmax is invariant to translation."""
        logits = jnp.array([1.0, 2.0, 3.0])
        constant = 10.0

        probs1 = softmax(logits)
        probs2 = softmax(logits + constant)

        # Should be identical
        assert jnp.allclose(probs1, probs2, rtol=1e-15)

    def test_softmax_monotonicity(self):
        """Test that softmax preserves order."""
        logits = jnp.array([1.0, 3.0, 2.0])
        probs = softmax(logits)

        # Order of probabilities should match order of logits
        assert probs[1] > probs[2] > probs[0]  # 3.0 > 2.0 > 1.0

    def test_softmax_temperature_scaling(self):
        """Test softmax behavior with temperature scaling."""
        logits = jnp.array([1.0, 2.0, 3.0])

        # High temperature (more uniform)
        probs_high_temp = softmax(logits / 10.0)

        # Low temperature (more peaked)
        probs_low_temp = softmax(logits * 10.0)

        # High temperature should be more uniform
        entropy_high = -jnp.sum(probs_high_temp * jnp.log(probs_high_temp))
        entropy_low = -jnp.sum(probs_low_temp * jnp.log(probs_low_temp))

        assert entropy_high > entropy_low

    def test_softmax_gradient_properties(self):
        """Test that softmax gradients have expected properties."""
        import jax

        def softmax_at_index(logits, index):
            """Get softmax probability at specific index."""
            return softmax(logits)[index]

        logits = jnp.array([1.0, 2.0, 3.0])

        # Compute gradient of softmax[1] w.r.t. logits
        grad_fn = jax.grad(softmax_at_index, argnums=0)
        gradient = grad_fn(logits, 1)

        # Gradient should sum to 0 (constraint that probabilities sum to 1)
        assert jnp.allclose(jnp.sum(gradient), 0.0, atol=1e-15)
