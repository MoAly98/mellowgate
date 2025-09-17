#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Differentiable Discrete Decisions via Stochastic Processes

A demonstration of how to make discrete decisions differentiable using stochastic
relaxations and various gradient estimation techniques. Uses a simple sine/cosine
selection problem as an illustrative example.

Includes:
  • Gradient estimators:
      1) Analytical gradient of relaxed expectation
      2) Finite-difference Monte Carlo on E[f]
      3) REINFORCE (no baseline)
      4) REINFORCE (mean baseline)
      5) Gumbel-Softmax Straight-Through (GS-STE)
  • Visualizations:
      - Piecewise function definition
      - Stochastic decision probability
      - Sampling visualization across parameter space
      - Expected value curves
      - Computation graph
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import matplotlib.patches

# Type alias for numpy random generator
RandomGenerator = np.random.Generator

# Optional animation support
try:
    import matplotlib.animation as animation
    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False
    print("Warning: matplotlib.animation not available. Animations will be skipped.")

# -----------------------------
# Utilities
# -----------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    return np.log(p) - np.log1p(-p)

# -----------------------------
# Problem definition
# -----------------------------

def f_discrete(theta: float, k: int) -> float:
    """
    Piecewise function with discrete decision.

    Args:
        theta: Parameter value
        k: Binary decision variable (0 or 1)

    Returns:
        sin(theta) if k==1, cos(theta) if k==0
    """
    return math.sin(theta) if k == 1 else math.cos(theta)

def df_dtheta_pathwise(theta: float, k: int) -> float:
    """
    Pathwise derivative of f with respect to theta.

    Args:
        theta: Parameter value
        k: Binary decision variable (0 or 1)

    Returns:
        cos(theta) if k==1, -sin(theta) if k==0
    """
    return math.cos(theta) if k == 1 else -math.sin(theta)

def f_relaxed(theta: float) -> float:
    """
    Relaxed expectation: E[f] = p(theta) * sin(theta) + (1-p(theta)) * cos(theta).

    This is the differentiable version of the discrete function, where the
    discrete decision is replaced by its expectation under a Bernoulli distribution.

    Args:
        theta: Parameter value

    Returns:
        Expected value of f under stochastic decision
    """
    p = sigmoid(theta)
    return p * math.sin(theta) + (1.0 - p) * math.cos(theta)

def df_relaxed(theta: float) -> float:
    """
    Analytical gradient of the relaxed expectation with respect to theta.

    This demonstrates how the stochastic relaxation makes the originally
    non-differentiable function differentiable everywhere.

    Args:
        theta: Parameter value

    Returns:
        Analytical gradient of E[f] with respect to theta
    """
    p = sigmoid(theta)
    dp = p * (1.0 - p)  # derivative of sigmoid
    # Chain rule: d/dθ [p*sin + (1-p)*cos] = dp*(sin-cos) + p*cos - (1-p)*(-sin)
    return dp * (math.sin(theta) - math.cos(theta)) + p * math.cos(theta) - (1.0 - p) * math.sin(theta)

# -----------------------------
# Gradient estimators
# -----------------------------

def grad_finite_difference(theta: float, n_samples: int = 50_000, delta: float = 1e-3, rng: Optional[RandomGenerator] = None) -> float:
    """Finite-difference Monte Carlo on E[f] using θ and θ+δ."""
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None  # Help type checker
    p0 = sigmoid(theta)
    p1 = sigmoid(theta + delta)
    k0 = rng.random(n_samples) < p0
    k1 = rng.random(n_samples) < p1
    f0 = np.where(k0, np.sin(theta), np.cos(theta)).mean()
    f1 = np.where(k1, np.sin(theta + delta), np.cos(theta + delta)).mean()
    return (f1 - f0) / delta

def grad_reinforce(theta: float, n_samples: int = 50_000, baseline: Optional[float] = None, rng: Optional[RandomGenerator] = None) -> float:
    """REINFORCE estimator with optional baseline (unbiased)."""
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None  # Help type checker
    p = sigmoid(theta)
    ks = (rng.random(n_samples) < p).astype(np.int32)
    path_terms = np.where(ks == 1, np.cos(theta), -np.sin(theta))
    f_vals = np.where(ks == 1, np.sin(theta), np.cos(theta))
    b = 0.0 if baseline is None else baseline
    score_terms = (f_vals - b) * (ks - p)  # since d/dθ log pθ(k) = (k - p)
    return (path_terms + score_terms).mean()

def grad_reinforce_mean_baseline(theta: float, n_samples: int = 50_000, rng: Optional[RandomGenerator] = None) -> float:
    """REINFORCE with mean baseline b(θ) ≈ E[f] estimated from same samples."""
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None  # Help type checker
    p = sigmoid(theta)
    ks = (rng.random(n_samples) < p).astype(np.int32)
    f_vals = np.where(ks == 1, np.sin(theta), np.cos(theta))
    baseline = float(f_vals.mean())
    path_terms = np.where(ks == 1, np.cos(theta), -np.sin(theta))
    score_terms = (f_vals - baseline) * (ks - p)
    return (path_terms + score_terms).mean()

def sample_gumbel(shape: Union[int, Tuple[int, ...]], rng: RandomGenerator) -> np.ndarray:
    u = rng.random(shape)
    return -np.log(-np.log(u))

def grad_gs_ste(theta: float, n_samples: int = 50_000, tau: float = 0.5, rng: Optional[RandomGenerator] = None) -> float:
    """
    Gumbel-Softmax Straight-Through estimator for Bernoulli.
    Forward: hard sample z ∈ {0,1}; Backward: surrogate gradient via relaxed s.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = sigmoid(theta)
    log_p = np.log(p)
    log_q = np.log(1.0 - p)
    g1 = sample_gumbel(n_samples, rng)
    g0 = sample_gumbel(n_samples, rng)

    # Hard decision in forward
    z = (log_p + g1 > log_q + g0).astype(np.float64)

    # Soft relaxation for backward
    diff_g = np.subtract(g1, g0)  # type: ignore
    s = 1.0 / (1.0 + np.exp(-((logit(p) + diff_g) / tau)))
    ds_dtheta = (s * (1.0 - s)) / tau  # because d/dθ logit(p) = 1 for p=σ(θ)

    # f = z*sinθ + (1 - z)*cosθ
    path_term = z * math.cos(theta) - (1.0 - z) * math.sin(theta)
    ste_term = (math.sin(theta) - math.cos(theta)) * ds_dtheta
    return float((path_term + ste_term).mean())

# -----------------------------
# Grid evaluation & gradient demo plotting
# -----------------------------

@dataclass
class EstimatorSpec:
    name: str
    fn: Callable[[float], float]

def evaluate_grid(thetas: np.ndarray, estimators: Dict[str, Callable[[float], float]]) -> Dict[str, np.ndarray]:
    return {k: np.array([fn(float(t)) for t in thetas]) for k, fn in estimators.items()}

def demo_gradients(outdir: Path = Path("."), seed: int = 1234, n_samples: int = 50_000, delta: float = 1e-3, tau: float = 0.5) -> Dict[str, Path]:
    """
    Demonstrate different gradient estimation techniques for the stochastic decision problem.

    Args:
        outdir: Directory to save plots
        seed: Random seed for reproducibility
        n_samples: Number of samples for Monte Carlo estimators
        delta: Step size for finite difference
        tau: Temperature for Gumbel-Softmax

    Returns:
        Dictionary mapping plot names to file paths
    """
    rng = np.random.default_rng(seed)
    thetas = np.linspace(-4.0, 4.0, 121)

    print(f"  Computing gradients with {n_samples:,} samples...")

    estimators = {
        "Analytical (Relaxed)": lambda th: df_relaxed(th),
        "Finite Difference MC": lambda th: grad_finite_difference(th, n_samples=n_samples, delta=delta, rng=rng),
        "REINFORCE (no baseline)": lambda th: grad_reinforce(th, n_samples=n_samples, baseline=None, rng=rng),
        "REINFORCE (mean baseline)": lambda th: grad_reinforce_mean_baseline(th, n_samples=n_samples, rng=rng),
        f"Gumbel-Softmax STE (τ={tau})": lambda th: grad_gs_ste(th, n_samples=n_samples, tau=tau, rng=rng),
    }

    results = evaluate_grid(thetas, estimators)

    # Plot 1: Gradient estimates comparison
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    styles = ['-', '--', '-.', ':', '-']

    for i, (name, vals) in enumerate(results.items()):
        plt.plot(thetas, vals, label=name, color=colors[i], linestyle=styles[i], linewidth=2)

    plt.xlabel(r"Parameter $\theta$", fontsize=12)
    plt.ylabel(r"Gradient estimate $\frac{d}{d\theta}\,\mathbb{E}[f(\theta, k)]$", fontsize=12)
    plt.title("Gradient Estimation Techniques Comparison", fontsize=14)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out1 = outdir / "gradient_estimators_comparison.png"
    plt.savefig(out1, dpi=200, bbox_inches='tight')
    plt.show()

    # Plot 2: Function values comparison
    plt.figure(figsize=(12, 7))
    f_relaxed_vals = np.array([f_relaxed(float(t)) for t in thetas])
    f_deterministic = np.where(thetas < 0.0, np.cos(thetas), np.sin(thetas))  # type: ignore

    plt.plot(thetas, f_relaxed_vals, label="Relaxed expectation E[f] (differentiable)",
             linewidth=3, color='blue')
    plt.plot(thetas, f_deterministic, label="Deterministic function (non-differentiable at θ=0)",  # type: ignore
             linewidth=3, color='red', linestyle='--')

    # Highlight the discontinuity
    plt.axvline(0, color='red', linestyle=':', alpha=0.7, label="Discontinuity")

    plt.xlabel(r"Parameter $\theta$", fontsize=12)
    plt.ylabel(r"Function value $f(\theta)$", fontsize=12)
    plt.title("Discrete vs Relaxed Function Comparison", fontsize=14)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out2 = outdir / "function_comparison.png"
    plt.savefig(out2, dpi=200, bbox_inches='tight')
    plt.show()

    return {"gradient_estimators": out1, "function_comparison": out2}

# -----------------------------
# Visualization functions
# -----------------------------

def plot_piecewise_function(savepath: Union[Path, str]) -> Path:
    """Visualize the piecewise sine/cosine function."""
    thetas = np.linspace(-4*np.pi/3, 4*np.pi/3, 1000)
    f = np.where(thetas < 0, np.cos(thetas), np.sin(thetas))  # type: ignore

    plt.figure(figsize=(10, 6))
    plt.plot(thetas, f, linewidth=2, label=r"$f(\theta) = \cos\theta$ if $\theta < 0$, $\sin\theta$ if $\theta \geq 0$")  # type: ignore
    plt.axvline(0, color='red', linestyle="--", alpha=0.7, label="Decision boundary")
    plt.title("Piecewise Function: Non-differentiable at θ=0", fontsize=14)
    plt.xlabel(r"$\theta$", fontsize=12)
    plt.ylabel(r"$f(\theta)$", fontsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    savepath = Path(savepath)
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()
    return savepath

def plot_decision_probability(savepath: Union[Path, str]) -> Path:
    """Visualize the stochastic decision probability function."""
    thetas = np.linspace(-6, 6, 600)
    p = sigmoid(thetas)

    plt.figure(figsize=(10, 6))
    plt.plot(thetas, p, linewidth=2, label=r"$p(\theta) = \sigma(\theta) = \frac{1}{1 + e^{-\theta}}$")
    plt.axhline(0.5, color='red', linestyle="--", alpha=0.7, label="Decision threshold")
    plt.axvline(0, color='gray', linestyle=":", alpha=0.7)
    plt.title("Stochastic Decision: Probability of Choosing sin(θ)", fontsize=14)
    plt.xlabel(r"$\theta$", fontsize=12)
    plt.ylabel(r"$p(\theta)$ = P(choose sin)", fontsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    savepath = Path(savepath)
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()
    return savepath

def plot_stochastic_sampling(savepath: Union[Path, str], n: int = 2000, seed: int = 0) -> Path:
    """Visualize samples from the stochastic decision process."""
    rng = np.random.default_rng(seed)
    thetas = np.linspace(-6, 6, n)
    p = sigmoid(thetas)
    k = (rng.random(n) < p).astype(int)

    plt.figure(figsize=(10, 6))
    plt.plot(thetas, p, linewidth=2, color='blue', label=r"$p(\theta)$ = P(choose sin)")

    # Add some jitter to y-coordinates for better visualization
    k_jittered = k + 0.02 * rng.standard_normal(n)
    plt.scatter(thetas, k_jittered, s=8, alpha=0.6, color='red',  # type: ignore
               label=r"Samples: $k \sim \text{Bernoulli}(p(\theta))$")

    plt.title("Stochastic Sampling Across Parameter Space", fontsize=14)
    plt.xlabel(r"$\theta$", fontsize=12)
    plt.ylabel(r"$p(\theta)$ and binary samples $k$", fontsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    savepath = Path(savepath)
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()
    return savepath

def plot_expectation_curves(savepath: Union[Path, str]) -> Path:
    """Compare the discrete function with its stochastic expectation."""
    thetas = np.linspace(-4*np.pi/3, 4*np.pi/3, 1000)
    p = sigmoid(thetas)

    # Discrete (non-differentiable) version
    f_discrete = np.where(thetas < 0, np.cos(thetas), np.sin(thetas))  # type: ignore

    # Stochastic expectation (differentiable)
    f_expectation = p * np.sin(thetas) + (1.0 - p) * np.cos(thetas)  # type: ignore

    plt.figure(figsize=(10, 6))
    plt.plot(thetas, np.sin(thetas), '--', alpha=0.7, label=r"$\sin\theta$")  # type: ignore
    plt.plot(thetas, np.cos(thetas), '--', alpha=0.7, label=r"$\cos\theta$")  # type: ignore
    plt.plot(thetas, f_discrete, linewidth=2, color='red',  # type: ignore
             label=r"Discrete: $f(\theta, k_{det})$ (non-differentiable)")
    plt.plot(thetas, f_expectation, linewidth=2, color='blue',  # type: ignore
             label=r"Stochastic: $\mathbb{E}[f(\theta, k)] = p(\theta)\sin\theta + (1-p(\theta))\cos\theta$")

    plt.axvline(0, color='gray', linestyle=":", alpha=0.7)
    plt.title("Discrete vs Stochastic (Differentiable) Objective", fontsize=14)
    plt.xlabel(r"$\theta$", fontsize=12)
    plt.ylabel("Function value", fontsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    savepath = Path(savepath)
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()
    return savepath

def draw_computation_graph(savepath: Union[Path, str]) -> Path:
    """Draw the computation graph for the stochastic decision process."""
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Node positions
    positions = {
        "theta": (0.1, 0.5),
        "sin": (0.3, 0.75),
        "cos": (0.3, 0.25),
        "p": (0.3, 0.5),
        "k": (0.5, 0.5),
        "select": (0.7, 0.5),
        "f": (0.85, 0.5),
    }

    def draw_node(name, text, color='lightblue'):
        x, y = positions[name]
        circle = matplotlib.patches.Circle((x, y), 0.04, fill=True, color=color, ec='black')
        ax.add_patch(circle)
        ax.text(x, y, text, ha="center", va="center", fontsize=10, weight='bold')

    def draw_arrow(frm, to, label=None, offset: float = 0.0):
        x1, y1 = positions[frm]
        x2, y2 = positions[to]

        # Calculate arrow positions accounting for node radius
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        dx_norm, dy_norm = dx/length, dy/length

        start_x = x1 + 0.04 * dx_norm
        start_y = y1 + 0.04 * dy_norm + offset
        end_x = x2 - 0.04 * dx_norm
        end_y = y2 - 0.04 * dy_norm + offset

        ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle="->", lw=1.5, color='darkblue'))

        if label:
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2 + 0.03
            ax.text(mid_x, mid_y, label, ha="center", va="bottom", fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    # Draw nodes
    draw_node("theta", r"$\theta$", 'lightgreen')
    draw_node("sin", r"$\sin\theta$", 'lightcoral')
    draw_node("cos", r"$\cos\theta$", 'lightcoral')
    draw_node("p", r"$p(\theta)$", 'lightyellow')
    draw_node("k", r"$k$", 'lightgray')
    draw_node("select", "select", 'lightsteelblue')
    draw_node("f", r"$f$", 'lightpink')

    # Draw arrows
    draw_arrow("theta", "sin")
    draw_arrow("theta", "cos")
    draw_arrow("theta", "p")
    draw_arrow("p", "k", r"$k \sim \text{Ber}(p)$")
    draw_arrow("sin", "select", "if k=1", offset=0.02)
    draw_arrow("cos", "select", "if k=0", offset=-0.02)
    draw_arrow("k", "select")
    draw_arrow("select", "f")

    plt.title("Computation Graph: Stochastic Decision Process f(θ,k)", fontsize=14, pad=20)

    # Add legend
    legend_text = [
        "θ: parameter to optimize",
        "p(θ): decision probability",
        "k: stochastic binary decision",
        "f: final output"
    ]
    ax.text(0.02, 0.95, '\n'.join(legend_text), transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))

    plt.tight_layout()
    savepath = Path(savepath)
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()
    return savepath

def create_all_visualizations(outdir: Path = Path(".")) -> Dict[str, Path]:
    """Create all demonstration visualizations and return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    print("Creating visualizations...")
    paths = {
        "piecewise_function": plot_piecewise_function(outdir / "piecewise_function.png"),
        "decision_probability": plot_decision_probability(outdir / "decision_probability.png"),
        "stochastic_sampling": plot_stochastic_sampling(outdir / "stochastic_sampling.png"),
        "expectation_curves": plot_expectation_curves(outdir / "expectation_curves.png"),
        "computation_graph": draw_computation_graph(outdir / "computation_graph.png"),
    }
    print(f"Visualizations saved to: {outdir}")
    return paths

# -----------------------------
# Animation functions (optional)
# -----------------------------

def animate_stochastic_decision(savepath: Union[Path, str], duration: float = 5.0, fps: int = 30) -> Optional[Path]:
    """
    Create an animation showing how the stochastic decision evolves with θ.

    Args:
        savepath: Where to save the animation
        duration: Animation duration in seconds
        fps: Frames per second

    Returns:
        Path to saved animation, or None if animation not available
    """
    if not HAS_ANIMATION:
        print("Skipping animation: matplotlib.animation not available")
        return None

    savepath = Path(savepath)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Setup static elements
    theta_range = np.linspace(-6, 6, 300)
    p_curve = sigmoid(theta_range)

    ax1.plot(theta_range, p_curve, 'b-', linewidth=2, alpha=0.7, label=r'$p(\theta) = \sigma(\theta)$')
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.7)
    ax1.set_ylabel(r'$p(\theta)$')
    ax1.set_title('Stochastic Decision Probability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Moving elements
    theta_point, = ax1.plot([], [], 'ro', markersize=10, label=r'Current $\theta$')
    prob_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Function values
    sin_line, = ax2.plot(theta_range, np.sin(theta_range), '--', alpha=0.7, label=r'$\sin\theta$')  # type: ignore
    cos_line, = ax2.plot(theta_range, np.cos(theta_range), '--', alpha=0.7, label=r'$\cos\theta$')  # type: ignore
    expectation_line, = ax2.plot([], [], 'b-', linewidth=2, label=r'$\mathbb{E}[f]$')
    current_point, = ax2.plot([], [], 'ro', markersize=10)

    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel('Function value')
    ax2.set_title('Expected Function Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-1.1, 1.1)

    n_frames = int(duration * fps)
    theta_values = np.linspace(-6, 6, n_frames)

    def animate(frame):
        theta = theta_values[frame]
        p = sigmoid(theta)

        # Update probability point
        theta_point.set_data([theta], [p])
        prob_text.set_text(f'θ = {theta:.2f}\np(θ) = {p:.3f}')

        # Update expectation curve up to current point
        current_thetas = theta_values[:frame+1]
        current_expectations = [sigmoid(t) * np.sin(t) + (1 - sigmoid(t)) * np.cos(t)
                               for t in current_thetas]
        expectation_line.set_data(current_thetas, current_expectations)

        # Update current point
        current_expectation = sigmoid(theta) * np.sin(theta) + (1 - sigmoid(theta)) * np.cos(theta)
        current_point.set_data([theta], [current_expectation])

        return theta_point, prob_text, expectation_line, current_point

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps,
                                 blit=True, repeat=True)

    try:
        anim.save(str(savepath), writer='pillow', fps=fps)
        plt.show()
        print(f"Animation saved to: {savepath}")
        return savepath
    except Exception as e:
        print(f"Failed to save animation: {e}")
        plt.show()
        return None

def create_gradient_flow_animation(savepath: Union[Path, str], duration: float = 8.0, fps: int = 20) -> Optional[Path]:
    """
    Animate the gradient flow for different estimators.

    Args:
        savepath: Where to save the animation
        duration: Animation duration in seconds
        fps: Frames per second

    Returns:
        Path to saved animation, or None if animation not available
    """
    if not HAS_ANIMATION:
        print("Skipping gradient animation: matplotlib.animation not available")
        return None

    savepath = Path(savepath)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    theta_range = np.linspace(-4, 4, 200)
    analytical_grads = [df_relaxed(t) for t in theta_range]

    # Plot analytical gradient as reference
    ax1.plot(theta_range, analytical_grads, 'b-', linewidth=2, alpha=0.7,
             label='Analytical gradient')
    ax1.set_ylabel('Gradient')
    ax1.set_title('Gradient Estimates vs Analytical')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Animated elements
    reinforce_line, = ax1.plot([], [], 'r-', linewidth=2, alpha=0.8, label='REINFORCE')
    gs_ste_line, = ax1.plot([], [], 'g-', linewidth=2, alpha=0.8, label='GS-STE')

    # Function evolution
    function_line, = ax2.plot([], [], 'b-', linewidth=2, label='E[f]')
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel('Function value')
    ax2.set_title('Function Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-1.1, 1.1)

    n_frames = int(duration * fps)

    def animate(frame):
        # Simulate noisy gradient estimates
        noise_scale = 0.1
        theta_subset = theta_range[:frame*2+10]  # Growing window

        if len(theta_subset) > 10:
            # Add noise to simulate Monte Carlo estimates
            reinforce_noise = np.random.normal(0, noise_scale, len(theta_subset))
            gs_ste_noise = np.random.normal(0, noise_scale*0.7, len(theta_subset))

            analytical_subset = [df_relaxed(t) for t in theta_subset]
            reinforce_est = np.array(analytical_subset) + reinforce_noise
            gs_ste_est = np.array(analytical_subset) + gs_ste_noise

            reinforce_line.set_data(theta_subset, reinforce_est)
            gs_ste_line.set_data(theta_subset, gs_ste_est)

            # Update function
            f_vals = [f_relaxed(t) for t in theta_subset]
            function_line.set_data(theta_subset, f_vals)

        return reinforce_line, gs_ste_line, function_line

    try:
        anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps,
                                     blit=True, repeat=True)
        anim.save(str(savepath), writer='pillow', fps=fps)
        plt.show()
        print(f"Gradient animation saved to: {savepath}")
        return savepath
    except Exception as e:
        print(f"Failed to save gradient animation: {e}")
        plt.show()
        return None

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class DemoConfig:
    """Configuration for the demonstration."""
    seed: int = 1234
    n_samples: int = 50_000
    finite_diff_delta: float = 1e-3
    gumbel_tau: float = 0.5
    create_animations: bool = True
    animation_duration: float = 5.0
    animation_fps: int = 20

    def __post_init__(self):
        """Validate configuration."""
        if self.n_samples < 1000:
            raise ValueError("n_samples should be at least 1000 for reliable estimates")
        if self.finite_diff_delta <= 0:
            raise ValueError("finite_diff_delta must be positive")
        if self.gumbel_tau <= 0:
            raise ValueError("gumbel_tau must be positive")

# -----------------------------
# Main demonstration
# -----------------------------

def main(config: Optional[DemoConfig] = None):
    """
    Run the complete demonstration of differentiable discrete decisions.

    Args:
        config: Configuration object. If None, uses default configuration.
    """
    if config is None:
        config = DemoConfig()

    outdir = Path("./differentiable_decisions_demo")
    outdir.mkdir(exist_ok=True)

    print("🎯 Differentiable Discrete Decisions Demonstration")
    print("=" * 55)
    print(f"Configuration:")
    print(f"  • Samples: {config.n_samples:,}")
    print(f"  • Random seed: {config.seed}")
    print(f"  • Animations: {'Enabled' if config.create_animations and HAS_ANIMATION else 'Disabled'}")
    print(f"  • Output directory: {outdir}")

    # 1) Generate gradient comparison plots
    print("\n📊 Generating gradient estimator comparisons...")
    gradient_paths = demo_gradients(
        outdir=outdir,
        seed=config.seed,
        n_samples=config.n_samples,
        delta=config.finite_diff_delta,
        tau=config.gumbel_tau
    )

    # 2) Create conceptual visualizations
    print("\n🎨 Creating conceptual visualizations...")
    visual_paths = create_all_visualizations(outdir=outdir)

    all_paths = {**gradient_paths, **visual_paths}

    # 3) Create animations if requested and available
    if config.create_animations and HAS_ANIMATION:
        print("\n🎬 Creating animations...")
        try:
            anim1 = animate_stochastic_decision(
                outdir / "stochastic_decision_animation.gif",
                duration=config.animation_duration,
                fps=config.animation_fps
            )
            if anim1:
                all_paths["stochastic_animation"] = anim1

            anim2 = create_gradient_flow_animation(
                outdir / "gradient_flow_animation.gif",
                duration=config.animation_duration * 1.6,
                fps=config.animation_fps
            )
            if anim2:
                all_paths["gradient_animation"] = anim2
        except Exception as e:
            print(f"Animation creation failed: {e}")

    print(f"\n✅ All demonstrations complete!")
    print(f"\n📁 Generated files in {outdir}:")
    for name, path in all_paths.items():
        print(f"  • {name}: {path.name}")

    # Print summary of concepts demonstrated
    print(f"\n🧠 Concepts demonstrated:")
    print("  • Non-differentiable discrete decisions")
    print("  • Stochastic relaxation via Bernoulli distribution")
    print("  • Multiple gradient estimation techniques:")
    print("    - Analytical gradients of relaxed expectation")
    print("    - Finite difference Monte Carlo")
    print("    - REINFORCE (with and without baselines)")
    print("    - Gumbel-Softmax Straight-Through Estimator")
    print("  • Computation graphs for stochastic processes")

    return all_paths

# -----------------------------
# Convenience functions
# -----------------------------

def quick_demo(n_samples: int = 10_000, create_animations: bool = False) -> Dict[str, Path]:
    """
    Run a quick demonstration with fewer samples for faster execution.

    Args:
        n_samples: Number of Monte Carlo samples (default: 10,000)
        create_animations: Whether to create animations (default: False)

    Returns:
        Dictionary of generated file paths
    """
    config = DemoConfig(
        n_samples=n_samples,
        create_animations=create_animations,
        animation_duration=3.0,
        animation_fps=15
    )
    return main(config)

def full_demo() -> Dict[str, Path]:
    """
    Run the full demonstration with high-quality settings.

    Returns:
        Dictionary of generated file paths
    """
    config = DemoConfig(
        n_samples=100_000,
        create_animations=True,
        animation_duration=8.0,
        animation_fps=30
    )
    return main(config)

def compare_estimators_at_point(theta: float = 0.5, n_samples: int = 10_000, n_runs: int = 100) -> None:
    """
    Compare gradient estimators at a specific point with multiple runs to show variance.

    Args:
        theta: Point at which to evaluate gradients
        n_samples: Samples per estimator per run
        n_runs: Number of independent runs
    """
    print(f"\n🔍 Comparing estimators at θ = {theta}")
    print(f"Running {n_runs} independent trials with {n_samples:,} samples each...")

    analytical = df_relaxed(theta)

    estimators = {
        "Finite Diff": lambda rng: grad_finite_difference(theta, n_samples=n_samples, rng=rng),
        "REINFORCE": lambda rng: grad_reinforce(theta, n_samples=n_samples, rng=rng),
        "REINFORCE+baseline": lambda rng: grad_reinforce_mean_baseline(theta, n_samples=n_samples, rng=rng),
        "GS-STE": lambda rng: grad_gs_ste(theta, n_samples=n_samples, rng=rng),
    }

    results = {}
    for name, estimator in estimators.items():
        estimates = []
        for run in range(n_runs):
            rng = np.random.default_rng(run)
            estimates.append(estimator(rng))
        results[name] = np.array(estimates)

    print(f"\nAnalytical gradient: {analytical:.4f}")
    print("\nEstimator comparison:")
    print("Method              | Mean      | Std       | Bias      | RMSE")
    print("-" * 65)

    for name, estimates in results.items():
        mean_est = estimates.mean()
        std_est = estimates.std()
        bias = mean_est - analytical
        rmse = np.sqrt(((estimates - analytical) ** 2).mean())
        print(f"{name:18} | {mean_est:8.4f} | {std_est:8.4f} | {bias:8.4f} | {rmse:8.4f}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_demo()
        elif sys.argv[1] == "full":
            full_demo()
        elif sys.argv[1] == "compare":
            theta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
            compare_estimators_at_point(theta)
        else:
            print("Usage: python sine_cos_gradients.py [quick|full|compare [theta]]")
    else:
        # Default: run with standard configuration
        main()
