import numpy as np

Array = np.ndarray

def sample_gumbel(shape, rng) -> Array:
    u = rng.uniform(0.0, 1.0, shape)
    return -np.log(-np.log(u))

