from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np
from mellowgate.utils.functions import softmax

Array = np.ndarray

@dataclass
class Branch:
    f: Callable[[float], float]
    df: Optional[Callable[[float], float]] = None   # optional

@dataclass
class LogitsModel:
    logits: Callable[[float], Array]                 # returns sh
    dlogits_dtheta: Optional[Callable[[float], Array]] = None  # optionalape (K,)

@dataclass
class DiscreteProblem:
    branches: List[Branch]       # length K
    logits_model: LogitsModel
    rng: np.random.Generator = np.random.default_rng(0) # type: ignore

    @property
    def K(self) -> int:
        return len(self.branches)

    def pi(self, theta: float) -> Array:
        return softmax(self.logits_model.logits(theta))

    def f_vals(self, theta: float) -> Array:
        return np.array([b.f(theta) for b in self.branches])

    def df_vals(self, theta: float) -> Optional[Array]:
        if any(b.df is None for b in self.branches):
            return None
        else:
            return np.array([b.df(theta) for b in self.branches if b.df is not None])

    # exact expectation and exact gradient (if all ingredients available)
    def L(self, theta: float) -> float:
        p = self.pi(theta)
        return float(np.sum(p * self.f_vals(theta))) # type: ignore

    def true_grad(self, theta: float) -> Optional[float]:
        # needs dlogits_dtheta and df_vals
        if self.logits_model.dlogits_dtheta is None:
            return None
        df = self.df_vals(theta)
        if df is None:
            return None
        p = self.pi(theta)
        f = self.f_vals(theta)
        da = self.logits_model.dlogits_dtheta(theta)  # (K,)
        # d pi / d theta = J_softmax(a(theta)) * da
        # but for 1D theta and vector a we can use: dpi_i = pi_i * (da_i - sum_j pi_j*da_j)
        mean_da = np.sum(p * da) # type: ignore
        dpi = p * (da - mean_da)
        return float(np.sum(p * df) + np.sum(f * dpi)) # type: ignore