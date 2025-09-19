# example_toy.py
import numpy as np
from mellowgate.api.functions import (Branch, LogitsModel, DiscreteProblem)
from mellowgate.api.estimators import (FDConfig, ReinforceConfig, ReinforceState, GSConfig)
from mellowgate.plots.metrics import plot_means_vs_true, plot_bias_var_mse, plot_time
from mellowgate.api.experiments import Sweep, run_sweep
from mellowgate.utils.outputs import OutputManager
from pathlib import Path
import warnings

# Initialise output manager for this script
output_manager = OutputManager(base_dir="outputs")

# branches
branches = [
    Branch(f=lambda th: float(np.cos(th)),  df=lambda th: float(-np.sin(th))),
    Branch(f=lambda th: float(np.sin(th)),  df=lambda th: float(np.cos(th))),
    Branch(f=lambda th: float(np.tanh(th)), df=lambda th: float(1 - np.tanh(th)**2)),
]
alpha = np.array([-1.0, 0.0, 1.0])
logits_model = LogitsModel(
    logits=lambda th: alpha * th,
    dlogits_dtheta=lambda th: alpha
)

prob = DiscreteProblem(branches=branches, logits_model=logits_model)

thetas = np.linspace(-2.5, 2.5, 21)
sweep = Sweep(
    thetas=thetas,
    repeats=1000,
    estimators={
        "fd": {"cfg": FDConfig(eps=1e-3, M=20000)},
        "reinforce": {"cfg": ReinforceConfig(M=20000, use_baseline=True),
                      "state": ReinforceState()},
        "gs": {"cfg": GSConfig(tau=0.5, M=800, use_ste=True)},
    }
)

results = run_sweep(prob, sweep)

# plotting (uses exact gradient if available)
plot_means_vs_true(results, prob.true_grad, output_manager=output_manager)
plot_bias_var_mse(results, prob.true_grad, output_manager=output_manager)
plot_time(results, output_manager=output_manager)
