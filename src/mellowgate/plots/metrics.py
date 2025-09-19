import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, Optional
from mellowgate.logging import logger

def plot_means_vs_true(
    results: Dict[str, Dict[str, np.ndarray]],
    true_grad_fn: Callable[[float], Optional[float]],
    output_manager,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
    subdir: Optional[str] = None,
):
    title = title or "Gradient means vs truth"
    outfile = outfile or "means_vs_true.pdf"
    subdir = subdir or "metrics"
    plt.figure(figsize=(7,4))
    for name, R in results.items():
        th, mu, sd = R["theta"], R["mean"], R["std"]
        plt.plot(th, mu, marker="o", label=name)
        plt.fill_between(th, mu-sd, mu+sd, alpha=0.15) # type: ignore
    tg = np.array([true_grad_fn(float(t)) for t in th])
    if None in tg:
        logger.warning("Some true gradient values are missing. Skipping true gradient plot.")
    else:
        plt.semilogy(th, tg, "r--", linewidth=2, label="true grad")
    plt.xlabel("theta"); plt.ylabel("gradient")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    outpath = output_manager.get_path(subdir, filename=outfile)
    logger.info(f"Saving plot to {outpath}")
    plt.savefig(outpath)
    plt.close()

def plot_bias_var_mse(
    results: Dict[str, Dict[str, np.ndarray]],
    true_grad_fn: Callable[[float], Optional[float]],
    output_manager,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
    subdir: Optional[str] = None,
):
    title = title or "Bias / std / MSE"
    outfile = outfile or "bias_std_mse.pdf"
    subdir = subdir or "metrics"

    plt.figure(figsize=(7,4))
    for name, R in results.items():
        th, mu, sd = R["theta"], R["mean"], R["std"]
        tg = np.array([true_grad_fn(float(t)) for t in th])
        if None in tg:
            logger.warning(f"Some true gradient values are missing for {name}. Skipping bias/variance/MSE plot.")
            continue
        bias = np.abs(mu - tg) # type: ignore
        mse  = bias**2 + sd**2
        plt.semilogy(th, bias, label=name+" bias")
        plt.semilogy(th, sd,   "--", label=name+" std")
        plt.semilogy(th, mse,  ":", label=name+" mse")
    plt.xlabel("theta"); plt.ylabel("magnitude")
    plt.title(title)
    plt.legend(frameon=False, ncol=3)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    outpath = output_manager.get_path(subdir, filename=outfile)
    logger.info(f"Saving plot to {outpath}")
    plt.savefig(outpath)
    plt.close()

def plot_time(
    results: Dict[str, Dict[str, np.ndarray]],
    output_manager,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
    subdir: Optional[str] = None,
):
    title = title or "Time per theta"
    outfile = outfile or "time_per_theta.pdf"
    subdir = subdir or "metrics"
    plt.figure(figsize=(7,4))
    for name, R in results.items():
        plt.semilogy(R["theta"], R["time"], marker="o", label=name)
    plt.xlabel("theta"); plt.ylabel("seconds per theta (repeats aggregated)")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    outpath = output_manager.get_path(subdir, filename=outfile)
    logger.info(f"Saving plot to {outpath}")
    plt.savefig(outpath)
    plt.close()