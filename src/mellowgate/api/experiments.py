import time
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import numpy as np
from mellowgate.api.functions import DiscreteProblem
from mellowgate.api.estimators import fd_gradient, FDConfig, reinforce_gradient, ReinforceConfig, ReinforceState, gs_gradient, GSConfig
from mellowgate.logging import logger
Array = np.ndarray

@dataclass
class Sweep:
    thetas: Array                  # 1D array of theta values
    repeats: int = 200             # repetitions per theta for mean/std
    estimators: Optional[Dict[str, Dict[str, Any]]] = None
    # example:
    # {"fd": {"cfg": FDConfig()},
    #  "reinforce": {"cfg": ReinforceConfig(), "state": ReinforceState()},
    #  "gs": {"cfg": GSConfig(tau=0.5)}}

def run_sweep(prob: DiscreteProblem, sweep: Sweep) -> Dict[str, Dict[str, Array]]:
    if sweep.estimators is None:
        logger.error("Sweep.estimators is None. Nothing to run.")
        return {}
    logger.info("Starting sweep over estimators: %s", list(sweep.estimators.keys()))
    out = {}
    for name, spec in sweep.estimators.items():
        logger.info(f"Running estimator: {name}")
        cfg = spec["cfg"]
        state = spec.get("state", None)
        means, stds, times = [], [], []
        for th in sweep.thetas:
            logger.info(f"  Theta: {th}")
            samples = []
            t0 = time.time()
            for i in range(sweep.repeats):
                if i % max(1, sweep.repeats // 5) == 0:
                    logger.debug(f"    Repeat {i+1}/{sweep.repeats}")
                if name == "fd":
                    samples.append(fd_gradient(prob, float(th), cfg))
                elif name == "reinforce":
                    if not isinstance(state, ReinforceState):
                        logger.error("State for 'reinforce' must be a ReinforceState instance.")
                        raise TypeError("State for 'reinforce' must be a ReinforceState instance.")
                    samples.append(reinforce_gradient(prob, float(th), cfg, state))
                elif name == "gs":
                    samples.append(gs_gradient(prob, float(th), cfg))
                else:
                    logger.error(f"Unknown estimator: {name}")
                    raise ValueError("Unknown estimator: " + name)
            times.append(time.time() - t0)
            samples = np.array(samples, dtype=float)
            means.append(samples.mean())
            stds.append(samples.std(ddof=1))
            logger.info(f"    Mean: {means[-1]:.4g}, Std: {stds[-1]:.4g}, Time: {times[-1]:.2f}s")
        out[name] = {
            "theta": sweep.thetas.copy(),
            "mean": np.array(means),
            "std":  np.array(stds),
            "time": np.array(times),
        }
        logger.info(f"Finished estimator: {name}")
    logger.info("Sweep complete.")
    return out