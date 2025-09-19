import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
from mellowgate.utils.statistics import sample_gumbel
from mellowgate.api.functions import DiscreteProblem
from mellowgate.utils.functions import softmax

Array = np.ndarray

@dataclass
class FDConfig:
    eps: float = 1e-3
    M: int = 2000

def fd_gradient(prob: DiscreteProblem, theta: float, cfg: FDConfig) -> float:
    rng = prob.rng
    def mc_expectation(th):
        p = prob.pi(th)
        ks = rng.choice(prob.K, size=cfg.M, p=p)
        return float(np.mean(prob.f_vals(th)[ks]))
    Lp = mc_expectation(theta + cfg.eps)
    Lm = mc_expectation(theta - cfg.eps)
    return (Lp - Lm) / (2 * cfg.eps)

@dataclass
class ReinforceConfig:
    M: int = 2000
    use_baseline: bool = True
    baseline_momentum: float = 0.9  # running avg if no oracle

class ReinforceState:
    def __init__(self):
        self.baseline = 0.0
        self.initialized = False

def reinforce_gradient(prob: DiscreteProblem, theta: float,
                       cfg: ReinforceConfig, state: ReinforceState) -> float:
    p = prob.pi(theta)
    f = prob.f_vals(theta)
    df = prob.df_vals(theta)  # may be None
    a = prob.logits_model.logits(theta)
    if prob.logits_model.dlogits_dtheta is None:
        raise ValueError("REINFORCE needs dlogits_dtheta for score function.")
    da = prob.logits_model.dlogits_dtheta(theta)

    # score for sample k: d log pi_k / d theta = da_k - sum_j p_j da_j
    score_center = float(np.sum(p * da))

    rng = prob.rng
    ks = rng.choice(prob.K, size=cfg.M, p=p)

    # baseline
    if cfg.use_baseline:
        # running average of reward
        rewards = f[ks]
        b = np.mean(rewards) if not state.initialized else state.baseline
        if state.initialized:
            state.baseline = cfg.baseline_momentum * state.baseline + (1-cfg.baseline_momentum) * float(np.mean(rewards))
        else:
            state.baseline = float(np.mean(rewards)); state.initialized = True
    else:
        b = 0.0

    # pathwise part (if df available) + score term
    path = df[ks] if df is not None else 0.0
    score = (f[ks] - b) * (da[ks] - score_center)
    return float(np.mean(path + score))

@dataclass
class GSConfig:
    tau: float = 0.5
    M: int = 1000
    use_ste: bool = False  # True => hard forward, soft backward

def gs_gradient(prob: DiscreteProblem, theta: float, cfg: GSConfig) -> float:
    if prob.logits_model.dlogits_dtheta is None:
        raise ValueError("Gumbel-Softmax needs dlogits_dtheta for backprop.")
    da = prob.logits_model.dlogits_dtheta(theta)  # (K,)
    a  = prob.logits_model.logits(theta)          # (K,)
    f  = prob.f_vals(theta)
    df = prob.df_vals(theta)  # optional

    grads = []
    for _ in range(cfg.M):
        g = sample_gumbel(prob.K, prob.rng)
        if cfg.use_ste:
            k = int(np.argmax(a + g))
            path = df[k] if df is not None else 0.0
        else:
            path = float(np.sum(softmax((a + g) / cfg.tau) * (df if df is not None else 0.0))) # type: ignore

        s = softmax((a + g) / cfg.tau) # type: ignore
        mean_da = float(np.sum(s * da)) # type: ignore
        ds = (s * (da - mean_da)) / cfg.tau
        rep = float(np.sum(f * ds))
        grads.append(path + rep)
    return float(np.mean(grads))