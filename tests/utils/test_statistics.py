"""Tests for mellowgate.utils.statistics module."""

import jax
import jax.numpy as jnp

from mellowgate.utils.statistics import sample_gumbel, sigmoid_2d


class TestSampleGumbel:
    """Test the sample_gumbel utility function."""

    def test_sample_gumbel_basic_shape(self):
        """Test basic Gumbel sampling with specified shape."""
        shape = (10, 5)
        key = jax.random.PRNGKey(42)

        samples = sample_gumbel(shape, key)

        # Check shape
        assert samples.shape == shape

        # Check that samples are finite
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_gumbel_1d_shape(self):
        """Test Gumbel sampling with 1D shape."""
        shape = (100,)
        key = jax.random.PRNGKey(123)

        samples = sample_gumbel(shape, key)

        assert samples.shape == shape
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_gumbel_scalar_shape(self):
        """Test Gumbel sampling with scalar shape."""
        shape = ()
        key = jax.random.PRNGKey(456)

        samples = sample_gumbel(shape, key)

        assert samples.shape == shape
        assert jnp.isfinite(samples)

    def test_sample_gumbel_large_shape(self):
        """Test Gumbel sampling with large shape."""
        shape = (1000, 100)
        key = jax.random.PRNGKey(789)

        samples = sample_gumbel(shape, key)

        assert samples.shape == shape
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_gumbel_reproducibility(self):
        """Test that Gumbel sampling is reproducible with same key."""
        shape = (50, 10)
        key = jax.random.PRNGKey(42)

        samples1 = sample_gumbel(shape, key)
        samples2 = sample_gumbel(shape, key)

        # Should be identical with same key
        assert jnp.allclose(samples1, samples2, rtol=1e-15)

    def test_sample_gumbel_different_keys(self):
        """Test that different keys give different samples."""
        shape = (100,)
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        samples1 = sample_gumbel(shape, key1)
        samples2 = sample_gumbel(shape, key2)

        # Should be different with different keys
        assert not jnp.allclose(samples1, samples2, rtol=1e-10)

    def test_sample_gumbel_statistical_properties(self):
        """Test statistical properties of Gumbel samples."""
        shape = (10000,)
        key = jax.random.PRNGKey(42)

        samples = sample_gumbel(shape, key)

        # Gumbel distribution has mean ≈ γ (Euler's constant) ≈ 0.5772
        # and variance ≈ π²/6 ≈ 1.6449
        sample_mean = jnp.mean(samples)
        sample_var = jnp.var(samples)

        # Check approximate statistical properties (with some tolerance)
        expected_mean = 0.5772  # Euler's gamma constant
        expected_var = jnp.pi**2 / 6

        # Allow for statistical fluctuation
        assert jnp.abs(sample_mean - expected_mean) < 0.1
        assert jnp.abs(sample_var - expected_var) < 0.2

    def test_sample_gumbel_distribution_shape(self):
        """Test that samples follow expected Gumbel distribution shape."""
        shape = (10000,)
        key = jax.random.PRNGKey(42)

        samples = sample_gumbel(shape, key)

        # Gumbel distribution is right-skewed
        # Most values should be between -2 and 5
        assert jnp.mean(samples > -2) > 0.9
        assert jnp.mean(samples < 5) > 0.9

        # Should have some extreme positive values (right tail)
        assert jnp.max(samples) > 3.0

    def test_sample_gumbel_multidimensional(self):
        """Test Gumbel sampling with multidimensional shapes."""
        shape = (5, 10, 3)
        key = jax.random.PRNGKey(42)

        samples = sample_gumbel(shape, key)

        assert samples.shape == shape
        assert jnp.all(jnp.isfinite(samples))

        # Each slice should have different values
        assert not jnp.allclose(samples[0], samples[1], rtol=1e-10)

    def test_sample_gumbel_edge_case_zero_size(self):
        """Test Gumbel sampling with zero-size dimension."""
        shape = (0, 5)
        key = jax.random.PRNGKey(42)

        samples = sample_gumbel(shape, key)

        assert samples.shape == shape
        # Should be empty array
        assert samples.size == 0

    def test_sample_gumbel_jit_compatibility(self):
        """Test that sample_gumbel works with JIT compilation."""

        def jit_sample_gumbel(shape, key):
            return sample_gumbel(shape, key)

        # JIT compile with static shape argument
        jit_compiled = jax.jit(jit_sample_gumbel, static_argnums=0)

        shape = (100, 50)
        key = jax.random.PRNGKey(42)

        # Should work without error
        samples = jit_compiled(shape, key)

        assert samples.shape == shape
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_gumbel_vmap_compatibility(self):
        """Test that sample_gumbel works with vmap."""

        def sample_with_key(key):
            return sample_gumbel((10,), key)

        # Create multiple keys
        keys = jax.random.split(jax.random.PRNGKey(42), 5)

        # Use vmap to sample with multiple keys
        vmapped_sample = jax.vmap(sample_with_key)
        samples = vmapped_sample(keys)

        assert samples.shape == (5, 10)
        assert jnp.all(jnp.isfinite(samples))

        # Each row should be different
        for i in range(5):
            for j in range(i + 1, 5):
                assert not jnp.allclose(samples[i], samples[j], rtol=1e-10)


class TestSampleGumbelMathematical:
    """Test mathematical properties of Gumbel sampling."""

    def test_gumbel_cdf_properties(self):
        """Test that samples follow Gumbel CDF properties."""
        n_samples = 10000
        key = jax.random.PRNGKey(42)

        samples = sample_gumbel((n_samples,), key)

        # Test CDF at a few points
        # Gumbel CDF: F(x) = exp(-exp(-x))
        test_points = jnp.array([0.0, 1.0, 2.0])

        for x in test_points:
            # Empirical CDF
            empirical_cdf = jnp.mean(samples <= x)

            # Theoretical CDF
            theoretical_cdf = jnp.exp(-jnp.exp(-x))

            # Should be close (within statistical error)
            assert jnp.abs(empirical_cdf - theoretical_cdf) < 0.05

    def test_gumbel_quantiles(self):
        """Test that sample quantiles match theoretical quantiles."""
        n_samples = 10000
        key = jax.random.PRNGKey(42)

        samples = sample_gumbel((n_samples,), key)

        # Test a few quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        for q in quantiles:
            # Empirical quantile
            empirical_q = jnp.quantile(samples, q)

            # Theoretical quantile: F^(-1)(q) = -ln(-ln(q))
            theoretical_q = -jnp.log(-jnp.log(q))

            # Should be close
            assert jnp.abs(empirical_q - theoretical_q) < 0.2

    def test_gumbel_max_stability(self):
        """Test max-stability property of Gumbel distribution."""
        # This is a more advanced test of the max-stability property
        n_samples = 1000
        n_blocks = 10
        key = jax.random.PRNGKey(42)

        # Generate samples in blocks
        keys = jax.random.split(key, n_blocks)
        block_samples = jax.vmap(lambda k: sample_gumbel((n_samples,), k))(keys)

        # Take maximum of each block
        block_maxima = jnp.max(block_samples, axis=1)

        # The maxima should still follow a Gumbel distribution
        # (approximately, after appropriate normalization)
        # This is a simplified test - in practice, normalization constants
        # would need to be applied

        # At least check that we get reasonable values
        assert jnp.all(jnp.isfinite(block_maxima))
        assert jnp.all(block_maxima > 0)  # Should have some large positive values


class TestSampleGumbelEdgeCases:
    """Test edge cases for Gumbel sampling."""

    def test_sample_gumbel_extreme_shapes(self):
        """Test with extreme shape configurations."""
        key = jax.random.PRNGKey(42)

        # Very large single dimension
        shape = (1000000,)
        samples = sample_gumbel(shape, key)
        assert samples.shape == shape

        # Many small dimensions
        shape = (2, 2, 2, 2, 2, 2)
        samples = sample_gumbel(shape, key)
        assert samples.shape == shape

    def test_sample_gumbel_dtype_consistency(self):
        """Test that sample_gumbel returns consistent dtype."""
        shape = (100,)
        key = jax.random.PRNGKey(42)

        samples = sample_gumbel(shape, key)

        # Should return float type (typically float32 in JAX)
        assert jnp.issubdtype(samples.dtype, jnp.floating)

    def test_sample_gumbel_numerical_stability(self):
        """Test numerical stability with edge cases."""
        shape = (1000,)
        key = jax.random.PRNGKey(42)

        samples = sample_gumbel(shape, key)

        # Should not contain NaN or Inf
        assert jnp.all(jnp.isfinite(samples))

        # Should not be all identical (would indicate a bug)
        assert jnp.var(samples) > 0.1

    def test_sample_gumbel_deterministic_with_key(self):
        """Test that results are deterministic given the same key."""
        shape = (1000,)
        key = jax.random.PRNGKey(12345)

        # Multiple calls with same key should give identical results
        samples1 = sample_gumbel(shape, key)
        samples2 = sample_gumbel(shape, key)
        samples3 = sample_gumbel(shape, key)

        assert jnp.array_equal(samples1, samples2)
        assert jnp.array_equal(samples2, samples3)

    def test_sample_gumbel_different_shape_same_key(self):
        """Test sampling different shapes with same key."""
        key = jax.random.PRNGKey(42)

        samples1 = sample_gumbel((100,), key)
        samples2 = sample_gumbel((50, 2), key)
        samples3 = sample_gumbel((10, 10), key)

        # Should work for all shapes
        assert samples1.shape == (100,)
        assert samples2.shape == (50, 2)
        assert samples3.shape == (10, 10)

        # All should be finite
        assert jnp.all(jnp.isfinite(samples1))
        assert jnp.all(jnp.isfinite(samples2))
        assert jnp.all(jnp.isfinite(samples3))

    def test_sample_gumbel_integer_shape(self):
        """Test Gumbel sampling with integer shape (tests shape conversion)."""
        # This tests the line: if isinstance(shape, int): shape = (shape,)
        shape = 50  # Pass int instead of tuple
        key = jax.random.PRNGKey(42)

        samples = sample_gumbel(shape, key)

        # Should convert to (50,) internally
        assert samples.shape == (50,)
        assert jnp.all(jnp.isfinite(samples))


class TestSigmoid2D:
    """Test the sigmoid_2d utility function."""

    def test_sigmoid_2d_basic(self):
        """Test basic sigmoid operation on 2D array."""
        logits = jnp.array([[0.0, 1.0], [-1.0, 2.0]])

        probs = sigmoid_2d(logits)

        # Check shape is preserved
        assert probs.shape == logits.shape

        # Check values are in [0, 1] range
        assert jnp.all(probs >= 0.0)
        assert jnp.all(probs <= 1.0)

        # Check known values (sigmoid(0) = 0.5)
        assert jnp.isclose(probs[0, 0], 0.5, atol=1e-6)

    def test_sigmoid_2d_extreme_values(self):
        """Test sigmoid with extreme logit values."""
        # Test very positive and very negative values
        logits = jnp.array([[-100.0, 100.0], [-10.0, 10.0]])

        probs = sigmoid_2d(logits)

        # Should not produce NaN or Inf
        assert jnp.all(jnp.isfinite(probs))

        # Very negative logits should be close to 0
        assert probs[0, 0] < 0.01
        assert probs[1, 0] < 0.01

        # Very positive logits should be close to 1
        assert probs[0, 1] > 0.99
        assert probs[1, 1] > 0.99

    def test_sigmoid_2d_mathematical_properties(self):
        """Test mathematical properties of sigmoid function."""
        logits = jnp.array([[1.0, -1.0], [0.5, -0.5]])

        probs = sigmoid_2d(logits)

        # sigmoid(-x) = 1 - sigmoid(x)
        assert jnp.isclose(probs[0, 1], 1 - probs[0, 0], atol=1e-6)
        assert jnp.isclose(probs[1, 1], 1 - probs[1, 0], atol=1e-6)
