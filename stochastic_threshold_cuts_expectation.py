#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic Threshold Cuts with Expectation Values and Gumbel-Max Trick

This script demonstrates the proper application of differentiable discrete decisions
to threshold cuts, using the same principles as the sine/cosine example:
• Expectation values for differentiable loss functions
• Gumbel-Max trick for unbiased sampling
• Proper gradient estimation techniques (REINFORCE, Gumbel-Softmax STE)

The key insight: replace hard threshold cuts with stochastic selections,
then optimize using expectation values to make everything differentiable.

Author: Enhanced for differentiable discrete decisions tutorial
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from typing import Optional, Tuple, Dict, List, Union, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Type alias for random generator
RandomGenerator = np.random.Generator

# Set random seed for reproducibility
np.random.seed(42)

# -----------------------------
# Core differentiable functions
# -----------------------------

def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # type: ignore

def logit(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Inverse sigmoid (logit) function."""
    p_clipped = np.clip(p, 1e-7, 1 - 1e-7)  # type: ignore
    return np.log(p_clipped / (1 - p_clipped))

def sample_gumbel(shape: Union[int, Tuple[int, ...]], rng: RandomGenerator) -> np.ndarray:
    """
    Sample from standard Gumbel distribution using inverse CDF method.

    Mathematical Background:
    =======================
    The Gumbel distribution is essential for the Gumbel-Max trick:
    - PDF: f(x) = exp(-x - exp(-x))
    - CDF: F(x) = exp(-exp(-x))
    - Inverse CDF: F^(-1)(u) = -log(-log(u))

    Key Property (Gumbel-Max Trick):
    If g₁, g₂ ~ Gumbel(0,1) and we have logits l₁, l₂, then:
    argmax_i(l_i + g_i) is distributed according to Categorical(softmax(l))

    For Bernoulli case: sample k = 1 if logit(p) + g₁ > 0 + g₀
    This gives exactly k ~ Bernoulli(p), but in a differentiable way!
    """
    if isinstance(shape, int):
        shape = (shape,)
    u = rng.uniform(0, 1, shape)
    return -np.log(-np.log(u))

# -----------------------------
# Data generation
# -----------------------------

def generate_physics_like_data(n_samples: int = 10000, signal_fraction: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic physics-like data with signal and background."""
    n_signal = int(n_samples * signal_fraction)
    n_background = n_samples - n_signal

    # Background: lower values, broader distribution
    background_data = np.random.normal(loc=50, scale=15, size=n_background)
    background_labels = np.zeros(n_background)

    # Signal: higher values, narrower distribution
    signal_data = np.random.normal(loc=80, scale=10, size=n_signal)
    signal_labels = np.ones(n_signal)

    # Combine and shuffle
    all_data = np.concatenate([background_data, signal_data])
    all_labels = np.concatenate([background_labels, signal_labels])

    indices = np.random.permutation(len(all_data))
    return all_data[indices], all_labels[indices]

# -----------------------------
# Cut function analogy to sine/cosine example
# -----------------------------

def f_cut_discrete(data_point: float, threshold: float, k: int) -> float:
    """
    Pure Heaviside step function for threshold cuts.

    This is the canonical discrete decision function that creates the
    non-differentiability problem we're trying to solve:

    H(x - threshold) = {1 if x >= threshold, 0 if x < threshold}

    The discrete decision k should match this Heaviside:
    - k=1 when data_point >= threshold
    - k=0 when data_point < threshold

    Args:
        data_point: Single data point value
        threshold: Threshold parameter
        k: Binary decision (should be Heaviside(data_point - threshold))

    Returns:
        1.0 if accepted (k=1), 0.0 if rejected (k=0)
        This is the pure discrete step function that's non-differentiable!
    """
    # Pure Heaviside step function
    return float(k)  # Simply return the binary decision

def f_cut_relaxed(data: np.ndarray, threshold: float, sharpness: float = 1.0) -> float:
    """
    Relaxed expectation version - analogous to E[f] in sine/cosine example.

    This is the stochastic relaxation of the Heaviside function:
    E[H(x - threshold)] = p(x) * 1 + (1-p(x)) * 0 = p(x) = σ(β(x - threshold))

    The expectation of the Heaviside step function becomes a sigmoid!
    This is the fundamental insight: discrete decisions become continuous
    expectations under stochastic relaxation.

    Args:
        data: Array of data points
        threshold: Threshold parameter
        sharpness: Controls steepness of sigmoid decision (β parameter)

    Returns:
        Expected value of Heaviside decisions: E[H] = mean(σ(β(x - threshold)))
    """
    # Selection probabilities for each data point
    p = sigmoid(sharpness * (data - threshold))

    # E[Heaviside] = E[k] = p * 1 + (1-p) * 0 = p
    # So the expectation is just the mean selection probability
    return float(np.mean(p))  # type: ignore

def df_cut_relaxed(data: np.ndarray, threshold: float, sharpness: float = 1.0) -> float:
    """
    Analytical gradient of the relaxed Heaviside function.

    For E[H] = E[k] = mean(σ(β(x - threshold))), the gradient is:
    d/dθ E[H] = mean(d/dθ σ(β(x - θ))) = mean(-β * σ * (1-σ))

    This demonstrates how stochastic relaxation makes the originally
    non-differentiable Heaviside differentiable everywhere.
    """
    # Selection probabilities
    p = sigmoid(sharpness * (data - threshold))

    # Derivative of sigmoid: dp/dθ = -sharpness * p * (1-p)
    dp_dtheta = -sharpness * p * (1 - p)

    # Since E[H] = mean(p), the gradient is mean(dp/dθ)
    return float(np.mean(dp_dtheta))

# -----------------------------
# Gumbel-Max trick implementation
# -----------------------------

def bernoulli_gumbel_max(data: np.ndarray, threshold: float, sharpness: float = 1.0,
                        rng: Optional[RandomGenerator] = None) -> np.ndarray:
    """
    Sample Bernoulli decisions using the Gumbel-Max trick.

    Mathematical Formulation:
    ========================

    1) Compute selection probability: p = σ(sharpness * (data - threshold))
    2) Convert to logits: logit(p) = log(p/(1-p))
    3) Sample Gumbel noise: g₁, g₀ ~ Gumbel(0,1)
    4) Decision rule: k = 1 if logit(p) + g₁ > 0 + g₀

    This is exactly equivalent to k ~ Bernoulli(p) but enables
    the straight-through estimator for gradient computation.

    Args:
        data: Input data points
        threshold: Threshold parameter
        sharpness: Controls steepness of decision boundary
        rng: Random number generator

    Returns:
        Binary decisions sampled via Gumbel-Max trick
    """
    if rng is None:
        rng = np.random.default_rng()

    # Selection probabilities
    p = sigmoid(sharpness * (data - threshold))

    # Convert to logits (with numerical stability)
    p_clipped = np.clip(p, 1e-7, 1 - 1e-7)  # type: ignore
    logit_p = np.log(p_clipped / (1 - p_clipped))

    # Sample Gumbel noise for both options
    g1 = sample_gumbel(data.shape, rng)  # Noise for "select" option
    g0 = sample_gumbel(data.shape, rng)  # Noise for "reject" option

    # Gumbel-Max decision: select if logit(p) + g1 > 0 + g0
    decisions = (logit_p + g1 > g0).astype(float)

    return decisions

def bernoulli_gumbel_max_soft(data: np.ndarray, threshold: float, tau: float = 1.0,
                             sharpness: float = 1.0, rng: Optional[RandomGenerator] = None) -> np.ndarray:
    """
    Soft version of Gumbel-Max for continuous relaxation.

    Used in Gumbel-Softmax Straight-Through Estimator.
    Returns continuous values in (0,1) instead of discrete {0,1}.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Selection probabilities
    p = sigmoid(sharpness * (data - threshold))
    p_clipped = np.clip(p, 1e-7, 1 - 1e-7)  # type: ignore
    logit_p = np.log(p_clipped / (1 - p_clipped))

    # Sample Gumbel noise
    g1 = sample_gumbel(data.shape, rng)
    g0 = sample_gumbel(data.shape, rng)

    # Soft decisions using temperature tau
    soft_decisions = sigmoid((logit_p + g1 - g0) / tau)

    return np.asarray(soft_decisions)

# -----------------------------
# Efficiency and performance metrics
# -----------------------------

def compute_efficiency(selected: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    """
    Compute signal efficiency and background rejection metrics.

    Args:
        selected: Binary array indicating selected events (1=selected, 0=rejected)
        true_labels: True labels (1=signal, 0=background)

    Returns:
        Dictionary with efficiency metrics
    """
    signal_mask = true_labels == 1
    background_mask = true_labels == 0

    # Signal efficiency: fraction of signal events selected
    signal_eff = np.sum(selected[signal_mask]) / np.sum(signal_mask) if np.sum(signal_mask) > 0 else 0

    # Background efficiency (we want this low): fraction of background selected
    background_eff = np.sum(selected[background_mask]) / np.sum(background_mask) if np.sum(background_mask) > 0 else 0

    # Background rejection: 1 - background efficiency
    background_rej = 1 - background_eff

    # Purity: fraction of selected events that are signal
    purity = np.sum(selected[signal_mask]) / np.sum(selected) if np.sum(selected) > 0 else 0  # type: ignore

    return {
        'signal_efficiency': signal_eff,
        'background_efficiency': background_eff,
        'background_rejection': background_rej,
        'purity': purity,
        'n_selected': int(np.sum(selected))  # type: ignore
    }

def compute_soft_efficiency(soft_probs: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    """
    Compute expected efficiency metrics using soft probabilities.

    This gives the expected performance under the stochastic relaxation.
    """
    signal_mask = true_labels == 1
    background_mask = true_labels == 0

    # Expected signal efficiency
    signal_eff = np.mean(soft_probs[signal_mask]) if np.sum(signal_mask) > 0 else 0

    # Expected background efficiency
    background_eff = np.mean(soft_probs[background_mask]) if np.sum(background_mask) > 0 else 0

    # Expected background rejection
    background_rej = 1 - background_eff

    # Expected number of selected events
    n_selected_expected = np.sum(soft_probs)  # type: ignore

    # Expected purity (bit more complex for soft case)
    signal_contribution = np.sum(soft_probs[signal_mask])  # type: ignore
    purity = signal_contribution / n_selected_expected if n_selected_expected > 0 else 0

    return {
        'signal_efficiency': signal_eff,
        'background_efficiency': background_eff,
        'background_rejection': background_rej,
        'purity': purity,
        'n_selected': n_selected_expected
    }

# -----------------------------
# Gradient estimation methods
# -----------------------------

def grad_finite_difference_cuts(threshold: float, data: np.ndarray, sharpness: float = 1.0,
                               n_samples: int = 5000, delta: float = 1e-3,
                               rng: Optional[RandomGenerator] = None) -> float:
    """
    Finite-difference gradient estimator for threshold cuts.

    Analogous to grad_finite_difference in sine/cosine example.
    """
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None  # Type checker hint

    # Sample subset for efficiency
    if len(data) > n_samples:
        indices = rng.choice(len(data), n_samples, replace=False)
        data_sample = data[indices]
    else:
        data_sample = data

    # Estimate E[f(θ)] and E[f(θ+δ)] using Gumbel-Max sampling
    k0 = bernoulli_gumbel_max(data_sample, threshold, sharpness, rng)
    k1 = bernoulli_gumbel_max(data_sample, threshold + delta, sharpness, rng)

    f0 = np.mean([f_cut_discrete(x, threshold, int(k))
                  for x, k in zip(data_sample, k0)])
    f1 = np.mean([f_cut_discrete(x, threshold + delta, int(k))
                  for x, k in zip(data_sample, k1)])

    return (f1 - f0) / delta

def grad_reinforce_cuts(threshold: float, data: np.ndarray, sharpness: float = 1.0,
                       n_samples: int = 5000, baseline: Optional[float] = None,
                       rng: Optional[RandomGenerator] = None) -> float:
    """
    REINFORCE gradient estimator for threshold cuts.

    Analogous to grad_reinforce in sine/cosine example.
    Uses the score function to estimate gradients.
    """
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None

    # Sample subset
    if len(data) > n_samples:
        indices = rng.choice(len(data), n_samples, replace=False)
        data_sample = data[indices]
    else:
        data_sample = data

    # Selection probabilities
    p = sigmoid(sharpness * (data_sample - threshold))

    # Sample decisions using Gumbel-Max
    decisions = bernoulli_gumbel_max(data_sample, threshold, sharpness, rng)

    # Compute function values (Heaviside decisions)
    f_values = np.array([f_cut_discrete(x, threshold, int(k))
                        for x, k in zip(data_sample, decisions)])

    # For Heaviside, there are no pathwise gradients w.r.t. threshold
    # The function is f(x,k) = k, which doesn't depend on threshold directly

    # REINFORCE score function: d/dθ log p(k|θ) = -sharpness * (k - p)
    if baseline is None:
        baseline = 0.0
    score_terms = (f_values - baseline) * (-sharpness * (decisions - p))  # type: ignore

    return float(np.mean(score_terms))

def grad_reinforce_mean_baseline_cuts(threshold: float, data: np.ndarray, sharpness: float = 1.0,
                                     n_samples: int = 5000, rng: Optional[RandomGenerator] = None) -> float:
    """
    REINFORCE with mean baseline for threshold cuts.

    Analogous to grad_reinforce_mean_baseline in sine/cosine example.
    """
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None

    # Sample subset
    if len(data) > n_samples:
        indices = rng.choice(len(data), n_samples, replace=False)
        data_sample = data[indices]
    else:
        data_sample = data

    # Selection probabilities
    p = sigmoid(sharpness * (data_sample - threshold))

    # Sample decisions
    decisions = bernoulli_gumbel_max(data_sample, threshold, sharpness, rng)

    # Function values (Heaviside)
    f_values = np.array([f_cut_discrete(x, threshold, int(k))
                        for x, k in zip(data_sample, decisions)])

    # Use mean as baseline
    baseline = float(np.mean(f_values))  # type: ignore

    # For Heaviside function f(x,k) = k, no pathwise gradients w.r.t. threshold
    # Only score function gradients remain

    # Score function terms with baseline
    score_terms = (f_values - baseline) * (-sharpness * (decisions - p))  # type: ignore

    return float(np.mean(score_terms))

def grad_gumbel_softmax_ste_cuts(threshold: float, data: np.ndarray, sharpness: float = 1.0,
                                tau: float = 0.5, n_samples: int = 5000,
                                rng: Optional[RandomGenerator] = None) -> float:
    """
    Gumbel-Softmax Straight-Through Estimator for threshold cuts.

    Analogous to grad_gs_ste in sine/cosine example.
    Forward pass: hard decisions; Backward pass: soft gradients.
    """
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None

    # Sample subset
    if len(data) > n_samples:
        indices = rng.choice(len(data), n_samples, replace=False)
        data_sample = data[indices]
    else:
        data_sample = data

    # Hard decisions (forward pass)
    hard_decisions = bernoulli_gumbel_max(data_sample, threshold, sharpness, rng)

    # Soft decisions (for gradients)
    soft_decisions = bernoulli_gumbel_max_soft(data_sample, threshold, tau, sharpness, rng)

    # Function evaluation using hard decisions (Heaviside)
    f_values = np.array([f_cut_discrete(x, threshold, int(k))
                        for x, k in zip(data_sample, hard_decisions)])

    # For Heaviside f(x,k) = k, no direct pathwise gradients w.r.t. threshold

    # Straight-through gradients using soft decisions
    # The STE approximates d/dθ E[f(k)] ≈ E[f'(soft_k) * d(soft_k)/dθ]
    # For Heaviside: f'(k) = 1, so we need d(soft_k)/dθ

    # Soft decision gradient: d(soft_k)/dθ where soft_k = σ((logit(p) + g₁ - g₀)/τ)
    p = sigmoid(sharpness * (data_sample - threshold))

    # Chain rule: d(soft_k)/dθ = d(soft_k)/d(logit_p) * d(logit_p)/dp * dp/dθ
    # For the Gumbel-Softmax relaxation, this simplifies to:
    ds_dtheta = -sharpness * soft_decisions * (1 - soft_decisions) / tau

    # STE gradient: since f(k) = k, and we use soft decisions for backward pass
    ste_gradient = np.mean(ds_dtheta)

    return float(ste_gradient)

# -----------------------------
# Comparison and visualization
# -----------------------------

def compare_cut_methods():
    """Compare different cut approaches: hard, expectation, and sampling with efficiency analysis."""
    print("🎯 Comparing Cut Methods: Hard vs Expectation vs Sampling")
    print("=" * 65)

    # Generate data
    data, labels = generate_physics_like_data(n_samples=2000, signal_fraction=0.3)
    threshold = 65.0
    sharpness = 2.0

    print(f"Data: {len(data)} events, threshold = {threshold}, sharpness = {sharpness}")
    print(f"Signal events: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")  # type: ignore
    print(f"Background events: {np.sum(1-labels)} ({(1-np.mean(labels))*100:.1f}%)")  # type: ignore

    # Method 1: Hard cut (non-differentiable)
    hard_decisions = (data >= threshold).astype(float)
    hard_reward = np.mean([f_cut_discrete(x, threshold, int(k))
                          for x, k in zip(data, hard_decisions)])

    # Method 2: Expectation (differentiable)
    expectation_reward = f_cut_relaxed(data, threshold, sharpness)

    # Soft probabilities for efficiency analysis
    soft_probs = sigmoid(sharpness * (data - threshold))
    assert isinstance(soft_probs, np.ndarray)  # Type hint for checker

    # Method 3: Stochastic sampling (unbiased but high variance)
    rng = np.random.default_rng(42)
    stoch_decisions = bernoulli_gumbel_max(data, threshold, sharpness, rng)
    stoch_reward = np.mean([f_cut_discrete(x, threshold, int(k))
                           for x, k in zip(data, stoch_decisions)])

    print(f"\nReward comparison:")
    print(f"  • Hard cut (non-differentiable):     {hard_reward:.6f}")
    print(f"  • Expectation (differentiable):      {expectation_reward:.6f}")
    print(f"  • Stochastic sampling (1 sample):    {stoch_reward:.6f}")

    # Compute efficiency metrics
    hard_metrics = compute_efficiency(hard_decisions, labels)
    soft_metrics = compute_soft_efficiency(soft_probs, labels)
    stoch_metrics = compute_efficiency(stoch_decisions, labels)

    print(f"\n📊 Efficiency Analysis:")
    print("-" * 80)
    print("Method           | Signal Eff | Bkg Rej | Purity | N Selected | Reward")
    print("-" * 80)
    print(f"Hard Cut         | {hard_metrics['signal_efficiency']:.3f}      | {hard_metrics['background_rejection']:.3f}   | {hard_metrics['purity']:.3f}  | {hard_metrics['n_selected']:4d}       | {hard_reward:.6f}")
    print(f"Soft (Expected)  | {soft_metrics['signal_efficiency']:.3f}      | {soft_metrics['background_rejection']:.3f}   | {soft_metrics['purity']:.3f}  | {soft_metrics['n_selected']:7.1f}    | {expectation_reward:.6f}")
    print(f"Stochastic       | {stoch_metrics['signal_efficiency']:.3f}      | {stoch_metrics['background_rejection']:.3f}   | {stoch_metrics['purity']:.3f}  | {stoch_metrics['n_selected']:4d}       | {stoch_reward:.6f}")

    # Show convergence of stochastic sampling to expectation
    n_trials = 100
    stoch_rewards = []
    stoch_efficiencies = []

    for trial in range(n_trials):
        rng_trial = np.random.default_rng(trial)
        stoch_dec = bernoulli_gumbel_max(data, threshold, sharpness, rng_trial)
        reward = np.mean([f_cut_discrete(x, threshold, int(k))
                         for x, k in zip(data, stoch_dec)])
        stoch_rewards.append(reward)

        # Track efficiency
        trial_metrics = compute_efficiency(stoch_dec, labels)
        stoch_efficiencies.append(trial_metrics['signal_efficiency'])

    print(f"\nStochastic sampling convergence ({n_trials} trials):")
    print(f"  • Mean reward:                        {np.mean(stoch_rewards):.6f}")
    print(f"  • Std reward:                         {np.std(stoch_rewards):.6f}")
    print(f"  • Expectation target:                 {expectation_reward:.6f}")
    print(f"  • Bias:                               {np.mean(stoch_rewards) - expectation_reward:.6f}")
    print(f"  • Mean signal efficiency:             {np.mean(stoch_efficiencies):.6f}")
    print(f"  • Expected signal efficiency:         {soft_metrics['signal_efficiency']:.6f}")

    # Create plots directory
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"\nCreated directory: {plots_dir}")

    # Enhanced visualization - individual plots saved to files

    # Plot 1: Data distributions and cuts
    plt.figure(figsize=(10, 6))
    signal_data = data[labels == 1]
    background_data = data[labels == 0]

    plt.hist(background_data, bins=30, alpha=0.7, label='Background', color='red', density=True)
    plt.hist(signal_data, bins=30, alpha=0.7, label='Signal', color='blue', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')

    # Show selection probabilities
    x_range = np.linspace(data.min(), data.max(), 100)
    p_range = sigmoid(sharpness * (x_range - threshold))
    plt.plot(x_range, p_range * plt.gca().get_ylim()[1], 'green', linewidth=2,
             label='Selection Probability')

    plt.xlabel('Data Value')
    plt.ylabel('Density')
    plt.title('Data Distribution and Soft Cut')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_data_distribution_and_soft_cut.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Hard cut result
    plt.figure(figsize=(10, 6))
    selected_hard = data[hard_decisions == 1]
    rejected_hard = data[hard_decisions == 0]

    plt.hist(rejected_hard, bins=30, alpha=0.7, label='Rejected', color='red', density=True)
    plt.hist(selected_hard, bins=30, alpha=0.7, label='Selected', color='green', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Data Value')
    plt.ylabel('Density')
    plt.title(f'Hard Cut Result\n({len(selected_hard)} selected)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_hard_cut_result.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Single stochastic cut result
    plt.figure(figsize=(10, 6))
    selected_stoch = data[stoch_decisions == 1]
    rejected_stoch = data[stoch_decisions == 0]

    plt.hist(rejected_stoch, bins=30, alpha=0.7, label='Rejected', color='red', density=True)
    plt.hist(selected_stoch, bins=30, alpha=0.7, label='Selected', color='green', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Data Value')
    plt.ylabel('Density')
    plt.title(f'Stochastic Cut Result\n({len(selected_stoch)} selected)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_stochastic_cut_result.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3b: Multiple stochastic cut samples (from original stochastic_threshold_cuts.py style)
    plt.figure(figsize=(10, 6))
    colors = ['green', 'orange', 'purple']
    n_stochastic_samples = 3

    for i in range(min(n_stochastic_samples, len(colors))):
        rng_sample = np.random.default_rng(i + 42)  # Different seed for each sample
        stoch_sample_decisions = bernoulli_gumbel_max(data, threshold, sharpness, rng_sample)
        selected_data_stoch = data[stoch_sample_decisions == 1]

        plt.hist(selected_data_stoch, bins=30, alpha=0.5,
                label=f'Stochastic Sample {i+1} ({len(selected_data_stoch)} events)',
                color=colors[i], density=True)

    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Data Value')
    plt.ylabel('Density')
    plt.title('Multiple Stochastic Cut Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03b_multiple_stochastic_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: ROC-like curve
    plt.figure(figsize=(10, 6))
    # Sweep threshold to create ROC curve
    threshold_sweep = np.linspace(data.min(), data.max(), 100)
    signal_effs = []
    background_effs = []

    for thresh in threshold_sweep:
        probs = sigmoid(sharpness * (data - thresh))
        soft_met = compute_soft_efficiency(probs, labels)  # type: ignore
        signal_effs.append(soft_met['signal_efficiency'])
        background_effs.append(soft_met['background_efficiency'])

    plt.plot(background_effs, signal_effs, 'g-', linewidth=2, label='Soft Cut ROC')

    # Mark current operating point
    plt.scatter(soft_metrics['background_efficiency'], soft_metrics['signal_efficiency'],
               color='red', s=100, zorder=5, label=f'Current Operating Point')

    plt.xlabel('Background Efficiency')
    plt.ylabel('Signal Efficiency')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 5: Reward convergence
    plt.figure(figsize=(10, 6))
    plt.hist(stoch_rewards, bins=20, alpha=0.7, density=True, label='Stochastic Samples')
    plt.axvline(expectation_reward, color='red', linestyle='-', linewidth=2, label='Expectation')
    plt.axvline(hard_reward, color='green', linestyle='--', linewidth=2, label='Hard Cut')
    plt.xlabel('Reward Value')
    plt.ylabel('Density')
    plt.title('Reward Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '05_reward_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 6: Cumulative average convergence
    plt.figure(figsize=(10, 6))
    cumulative_avg = np.cumsum(stoch_rewards) / np.arange(1, len(stoch_rewards) + 1)
    plt.plot(cumulative_avg, 'b-', linewidth=2, label='Cumulative Average')
    plt.axhline(expectation_reward, color='red', linestyle='-', linewidth=2, label='Expectation')
    plt.xlabel('Trial Number')
    plt.ylabel('Cumulative Average Reward')
    plt.title('Convergence to Expectation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '06_cumulative_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 7: Efficiency comparison bar chart
    plt.figure(figsize=(10, 6))
    methods = ['Hard', 'Soft\n(Expected)', 'Stochastic\n(Sample)']
    signal_effs_comp = [hard_metrics['signal_efficiency'],
                       soft_metrics['signal_efficiency'],
                       stoch_metrics['signal_efficiency']]
    background_rejs_comp = [hard_metrics['background_rejection'],
                           soft_metrics['background_rejection'],
                           stoch_metrics['background_rejection']]

    x_pos = np.arange(len(methods))
    width = 0.35

    plt.bar(x_pos - width/2, signal_effs_comp, width, label='Signal Efficiency', alpha=0.7, color='blue')
    plt.bar(x_pos + width/2, background_rejs_comp, width, label='Background Rejection', alpha=0.7, color='red')

    plt.xlabel('Method')
    plt.ylabel('Efficiency')
    plt.title('Efficiency Comparison')
    plt.xticks(x_pos, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '07_efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 8: Efficiency variance over trials
    plt.figure(figsize=(10, 6))
    plt.plot(stoch_efficiencies, 'b-', alpha=0.7, linewidth=1)
    plt.axhline(np.mean(stoch_efficiencies), color='blue', linestyle='-', linewidth=2, label='Mean')
    plt.axhline(soft_metrics['signal_efficiency'], color='red', linestyle='--', linewidth=2, label='Expected')
    plt.xlabel('Trial Number')
    plt.ylabel('Signal Efficiency')
    plt.title('Signal Efficiency Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '08_efficiency_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 9: Selection probability vs data value
    plt.figure(figsize=(10, 6))
    sorted_indices = np.argsort(data)  # type: ignore
    sorted_data = data[sorted_indices]
    sorted_probs = soft_probs[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Plot probabilities
    plt.plot(sorted_data, sorted_probs, 'g-', linewidth=2, label='Selection Probability')

    # Overlay true labels
    signal_indices = sorted_labels == 1
    background_indices = sorted_labels == 0
    plt.scatter(sorted_data[signal_indices], np.ones(np.sum(signal_indices))*1.05,
               c='blue', s=10, alpha=0.5, label='Signal Events')
    plt.scatter(sorted_data[background_indices], np.ones(np.sum(background_indices))*-0.05,
               c='red', s=10, alpha=0.5, label='Background Events')

    plt.axvline(threshold, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Data Value')
    plt.ylabel('Selection Probability')
    plt.title('Selection Function vs True Labels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '09_selection_probability_vs_labels.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 10: Purity vs threshold
    plt.figure(figsize=(10, 6))
    purities = []
    for thresh in threshold_sweep:
        probs = sigmoid(sharpness * (data - thresh))
        soft_met = compute_soft_efficiency(probs, labels)  # type: ignore
        purities.append(soft_met['purity'])

    plt.plot(threshold_sweep, purities, 'purple', linewidth=2, label='Expected Purity')
    plt.axvline(threshold, color='black', linestyle='--', alpha=0.7, label='Current Threshold')
    plt.axhline(soft_metrics['purity'], color='red', linestyle='-', alpha=0.7, label='Current Purity')
    plt.xlabel('Threshold')
    plt.ylabel('Purity')
    plt.title('Purity vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '10_purity_vs_threshold.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 11: N selected vs threshold
    plt.figure(figsize=(10, 6))
    n_selected_vals = []
    for thresh in threshold_sweep:
        probs = sigmoid(sharpness * (data - thresh))
        soft_met = compute_soft_efficiency(probs, labels)  # type: ignore
        n_selected_vals.append(soft_met['n_selected'])

    plt.plot(threshold_sweep, n_selected_vals, 'orange', linewidth=2, label='Expected N Selected')
    plt.axvline(threshold, color='black', linestyle='--', alpha=0.7, label='Current Threshold')
    plt.axhline(soft_metrics['n_selected'], color='red', linestyle='-', alpha=0.7, label='Current N Selected')
    plt.xlabel('Threshold')
    plt.ylabel('Number of Selected Events')
    plt.title('Selection Rate vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '11_selection_rate_vs_threshold.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 12: Summary metrics
    plt.figure(figsize=(10, 6))
    metric_names = ['Signal\nEff', 'Bkg\nRej', 'Purity']
    hard_vals = [hard_metrics['signal_efficiency'], hard_metrics['background_rejection'], hard_metrics['purity']]
    soft_vals = [soft_metrics['signal_efficiency'], soft_metrics['background_rejection'], soft_metrics['purity']]
    stoch_vals = [stoch_metrics['signal_efficiency'], stoch_metrics['background_rejection'], stoch_metrics['purity']]

    x_pos = np.arange(len(metric_names))
    width = 0.25

    plt.bar(x_pos - width, hard_vals, width, label='Hard', alpha=0.7, color='green')
    plt.bar(x_pos, soft_vals, width, label='Soft', alpha=0.7, color='blue')
    plt.bar(x_pos + width, stoch_vals, width, label='Stoch', alpha=0.7, color='orange')

    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Performance Summary')
    plt.xticks(x_pos, metric_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '12_performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ All plots saved to '{plots_dir}' directory:")
    for i, plot_name in enumerate([
        '01_data_distribution_and_soft_cut.png',
        '02_hard_cut_result.png',
        '03_stochastic_cut_result.png',
        '03b_multiple_stochastic_samples.png',
        '04_roc_curve.png',
        '05_reward_convergence.png',
        '06_cumulative_convergence.png',
        '07_efficiency_comparison.png',
        '08_efficiency_variance.png',
        '09_selection_probability_vs_labels.png',
        '10_purity_vs_threshold.png',
        '11_selection_rate_vs_threshold.png',
        '12_performance_summary.png'
    ], 1):
        print(f"  {i:2d}. {plot_name}")

    return data, labels, threshold

def compare_gradient_methods(plots_dir: str = "plots"):
    """Compare different gradient estimation methods for threshold cuts."""
    print("\n🎯 Comparing Gradient Estimation Methods")
    print("=" * 55)

    # Import os if not already imported
    import os

    # Generate data
    data, _ = generate_physics_like_data(n_samples=1500, signal_fraction=0.3)

    # Range of thresholds to test
    thresholds = np.linspace(45, 85, 25)
    sharpness = 2.0

    print(f"Computing gradients for {len(thresholds)} threshold values...")
    print(f"Using {len(data)} data points with sharpness = {sharpness}")

    # Analytical gradients (ground truth)
    analytical_grads = [df_cut_relaxed(data, t, sharpness) for t in thresholds]

    # Monte Carlo gradient estimators
    rng = np.random.default_rng(42)
    finite_diff_grads = [grad_finite_difference_cuts(t, data, sharpness, 1000, rng=rng)
                        for t in thresholds]

    rng = np.random.default_rng(42)  # Reset for fair comparison
    reinforce_grads = [grad_reinforce_cuts(t, data, sharpness, 1000, rng=rng)
                      for t in thresholds]

    rng = np.random.default_rng(42)
    reinforce_baseline_grads = [grad_reinforce_mean_baseline_cuts(t, data, sharpness, 1000, rng=rng)
                               for t in thresholds]

    rng = np.random.default_rng(42)
    gumbel_grads = [grad_gumbel_softmax_ste_cuts(t, data, sharpness, 0.5, 1000, rng=rng)
                   for t in thresholds]

    # Create gradient plots directory
    grad_plots_dir = os.path.join(plots_dir, "gradient_methods")
    if not os.path.exists(grad_plots_dir):
        os.makedirs(grad_plots_dir)

    # Plot 1: Gradient comparison
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, analytical_grads, 'k-', linewidth=3, label='Analytical (Truth)')
    plt.plot(thresholds, finite_diff_grads, 'r--', linewidth=2, label='Finite Difference')
    plt.plot(thresholds, reinforce_grads, 'g:', linewidth=2, label='REINFORCE')
    plt.plot(thresholds, reinforce_baseline_grads, 'b-.', linewidth=2, label='REINFORCE + Baseline')
    plt.plot(thresholds, gumbel_grads, 'm-', linewidth=2, alpha=0.7, label='Gumbel-Softmax STE')
    plt.xlabel('Threshold')
    plt.ylabel('Gradient')
    plt.title('Gradient Estimation Methods Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(grad_plots_dir, 'gradient_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Error analysis
    plt.figure(figsize=(10, 6))
    finite_diff_errors = np.abs(np.array(finite_diff_grads) - np.array(analytical_grads))  # type: ignore
    reinforce_errors = np.abs(np.array(reinforce_grads) - np.array(analytical_grads))  # type: ignore
    reinforce_baseline_errors = np.abs(np.array(reinforce_baseline_grads) - np.array(analytical_grads))  # type: ignore
    gumbel_errors = np.abs(np.array(gumbel_grads) - np.array(analytical_grads))  # type: ignore

    plt.plot(thresholds, finite_diff_errors, 'r--', label='Finite Difference')
    plt.plot(thresholds, reinforce_errors, 'g:', label='REINFORCE')
    plt.plot(thresholds, reinforce_baseline_errors, 'b-.', label='REINFORCE + Baseline')
    plt.plot(thresholds, gumbel_errors, 'm-', alpha=0.7, label='Gumbel-Softmax STE')
    plt.xlabel('Threshold')
    plt.ylabel('Absolute Error')
    plt.title('Gradient Estimation Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(grad_plots_dir, 'gradient_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Function values
    plt.figure(figsize=(10, 6))
    function_vals = [f_cut_relaxed(data, t, sharpness) for t in thresholds]
    plt.plot(thresholds, function_vals, 'g-', linewidth=2, label='Relaxed Function E[f]')

    # Mark where gradient is zero
    zero_crossings = []
    for i in range(len(analytical_grads)-1):
        if analytical_grads[i] * analytical_grads[i+1] < 0:
            zero_crossings.append(thresholds[i])

    for zc in zero_crossings:
        plt.axvline(zc, color='red', linestyle='--', alpha=0.7, label='Gradient = 0')

    plt.xlabel('Threshold')
    plt.ylabel('Function Value')
    plt.title('Relaxed Function and Critical Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(grad_plots_dir, 'function_values_and_critical_points.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Statistics summary
    plt.figure(figsize=(10, 6))
    methods = ['Finite\nDiff', 'REINFORCE', 'REINFORCE\n+Baseline', 'Gumbel\nSoftmax']
    mean_errors = [
        np.mean(finite_diff_errors),
        np.mean(reinforce_errors),
        np.mean(reinforce_baseline_errors),
        np.mean(gumbel_errors)
    ]
    std_errors = [
        np.std(finite_diff_errors),
        np.std(reinforce_errors),
        np.std(reinforce_baseline_errors),
        np.std(gumbel_errors)
    ]

    x_pos = np.arange(len(methods))
    plt.bar(x_pos, mean_errors, yerr=std_errors, capsize=5,
            color=['red', 'green', 'blue', 'magenta'], alpha=0.7)
    plt.xticks(x_pos, methods)
    plt.ylabel('Mean Absolute Error')
    plt.title('Method Comparison Summary')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(grad_plots_dir, 'method_comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Gradient method plots saved to '{grad_plots_dir}' directory:")
    for plot_name in [
        'gradient_comparison.png',
        'gradient_errors.png',
        'function_values_and_critical_points.png',
        'method_comparison_summary.png'
    ]:
        print(f"  - {plot_name}")

    # Print statistics
    print("\n📊 Gradient Estimation Statistics:")
    print("-" * 60)
    print("Method              | Mean Error | Std Error  | Max Error")
    print("-" * 60)
    for method, errors in zip(methods, [finite_diff_errors, reinforce_errors,
                                       reinforce_baseline_errors, gumbel_errors]):
        print(f"{method:18} | {np.mean(errors):9.6f} | {np.std(errors):9.6f} | {np.max(errors):9.6f}")

def demonstrate_gumbel_max_trick():
    """Demonstrate the Gumbel-Max trick in detail."""
    print("\n🎯 Demonstrating Gumbel-Max Trick")
    print("=" * 45)

    # Simple example with known probability
    data_point = 70.0
    threshold = 65.0
    sharpness = 2.0

    true_prob = sigmoid(sharpness * (data_point - threshold))
    print(f"Data point: {data_point}, Threshold: {threshold}")
    print(f"True selection probability: {true_prob:.4f}")

    # Sample many times using Gumbel-Max
    n_trials = 10000
    rng = np.random.default_rng(42)

    decisions = []
    for _ in range(n_trials):
        decision = bernoulli_gumbel_max(np.array([data_point]), threshold, sharpness, rng)[0]
        decisions.append(decision)

    empirical_prob = np.mean(decisions)
    print(f"Empirical probability (Gumbel-Max): {empirical_prob:.4f}")
    print(f"Error: {abs(empirical_prob - true_prob):.6f}")

    # Compare with direct Bernoulli sampling
    rng = np.random.default_rng(42)
    direct_decisions = rng.random(n_trials) < true_prob
    direct_prob = np.mean(direct_decisions)
    print(f"Empirical probability (direct Bernoulli): {direct_prob:.4f}")
    print(f"Error: {abs(direct_prob - true_prob):.6f}")

    print(f"\n✅ Gumbel-Max trick produces equivalent results to direct sampling!")
    print(f"🔑 But Gumbel-Max enables straight-through gradient estimation")

def main():
    """Run all demonstrations."""
    print("🚀 Stochastic Threshold Cuts with Expectation Values")
    print("=" * 70)
    print("This script demonstrates proper application of differentiable discrete")
    print("decisions to threshold cuts, using the same principles as the sine/cosine example:")
    print("• Expectation values make discrete decisions differentiable")
    print("• Gumbel-Max trick enables unbiased sampling")
    print("• Multiple gradient estimation techniques (REINFORCE, Gumbel-Softmax STE)")
    print("• Choice of method depends on specific use case and requirements")

    # Run demonstrations
    demonstrate_gumbel_max_trick()
    data, labels, threshold = compare_cut_methods()
    compare_gradient_methods("plots")  # Pass the plots directory

if __name__ == "__main__":
    main()
