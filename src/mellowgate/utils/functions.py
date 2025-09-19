import numpy as np
Array = np.ndarray

def softmax(x: Array) -> Array:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()
