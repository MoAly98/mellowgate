#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic Threshold Cuts: From Hard Cuts to Differentiable Selection

This script demonstrates how to replace hard threshold cuts with stochastic,
differentiable alternatives using real data. This is particularly relevant
for particle physics analysis where cuts are commonly used for event selection.

The example shows:
• Hard cut: traditional binary selection based on threshold
• Soft cut: probabilistic selection using sigmoid function
• Stochastic cut: random sampling based on soft probabilities
• Gradient estimation for optimizing the threshold parameter

Author: Generated for differentiable discrete decisions tutorial
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Type alias for random generator
RandomGenerator = np.random.Generator

# Set random seed for reproducibility
np.random.seed(42)

# -----------------------------
# Data generation utilities
# -----------------------------

def generate_physics_like_data(n_samples: int = 10000, signal_fraction: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data resembling particle physics measurements.

    Creates a dataset with:
    - Background: Normal distribution (mean=50, std=15)
    - Signal: Normal distribution (mean=80, std=10)

    This simulates a common scenario where we want to separate signal
    from background using a discriminating variable.

    Args:
        n_samples: Total number of data points
        signal_fraction: Fraction of events that are signal

    Returns:
        Tuple of (data, labels) where labels are 1 for signal, 0 for background
    """
    n_signal = int(n_samples * signal_fraction)
    n_background = n_samples - n_signal

    # Generate background events (lower values, broader distribution)
    background_data = np.random.normal(loc=50, scale=15, size=n_background)
    background_labels = np.zeros(n_background)

    # Generate signal events (higher values, narrower distribution)
    signal_data = np.random.normal(loc=80, scale=10, size=n_signal)
    signal_labels = np.ones(n_signal)

    # Combine and shuffle
    all_data = np.concatenate([background_data, signal_data])
    all_labels = np.concatenate([background_labels, signal_labels])

    # Shuffle to mix signal and background
    indices = np.random.permutation(len(all_data))
    return all_data[indices], all_labels[indices]

def generate_multimodal_data(n_samples: int = 8000) -> np.ndarray:
    """
    Generate more complex multimodal data for demonstration.

    Creates a mixture of three Gaussian distributions to show
    how threshold cuts affect different populations.
    """
    # Three populations with different characteristics
    pop1 = np.random.normal(loc=30, scale=8, size=n_samples//3)    # Low peak
    pop2 = np.random.normal(loc=60, scale=12, size=n_samples//3)   # Middle peak
    pop3 = np.random.normal(loc=90, scale=6, size=n_samples//3)    # High peak

    return np.concatenate([pop1, pop2, pop3])

# -----------------------------
# Cut functions
# -----------------------------

def sigmoid(x: float) -> float:
    """Sigmoid function with numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def hard_cut(data: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply hard threshold cut to data.

    Args:
        data: Input data values
        threshold: Threshold value

    Returns:
        Binary array: 1 if data >= threshold, 0 otherwise
    """
    return (data >= threshold).astype(float)

def soft_cut_probability(data: np.ndarray, threshold: float, sharpness: float = 1.0) -> np.ndarray:
    """
    Compute soft cut probabilities using sigmoid function.

    Args:
        data: Input data values
        threshold: Threshold parameter (inflection point)
        sharpness: Controls steepness of sigmoid (higher = sharper transition)

    Returns:
        Array of probabilities between 0 and 1
    """
    return 1.0 / (1.0 + np.exp(-sharpness * (data - threshold)))

def stochastic_cut(data: np.ndarray, threshold: float, sharpness: float = 1.0,
                  rng: Optional[RandomGenerator] = None) -> np.ndarray:
    """
    Apply stochastic cut based on soft probabilities.

    Args:
        data: Input data values
        threshold: Threshold parameter
        sharpness: Controls steepness of transition
        rng: Random number generator

    Returns:
        Binary array sampled from soft probabilities
    """
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None  # Help type checker

    probabilities = soft_cut_probability(data, threshold, sharpness)
    return (rng.random(len(data)) < probabilities).astype(float)

# -----------------------------
# Efficiency and loss functions
# -----------------------------

def compute_efficiency(selected: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    """
    Compute signal efficiency and background rejection.

    Args:
        selected: Binary array indicating selected events
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

def loss_function(threshold: float, data: np.ndarray, true_labels: np.ndarray,
                 sharpness: float = 1.0, signal_weight: float = 1.0) -> float:
    """
    Loss function for optimizing threshold parameter.

    Combines signal efficiency (maximize) with background rejection (maximize).
    This is differentiable when using soft cuts.

    Args:
        threshold: Threshold parameter to optimize
        data: Input data
        true_labels: True signal/background labels
        sharpness: Sigmoid sharpness parameter
        signal_weight: Relative weight for signal vs background

    Returns:
        Loss value (lower is better)
    """
    # Compute soft selection probabilities
    probs = soft_cut_probability(data, threshold, sharpness)

    # Expected signal efficiency (fraction of signal selected)
    signal_mask = true_labels == 1
    background_mask = true_labels == 0

    signal_eff = np.mean(probs[signal_mask]) if np.sum(signal_mask) > 0 else 0
    background_eff = np.mean(probs[background_mask]) if np.sum(background_mask) > 0 else 0

    # Loss: we want high signal efficiency and low background efficiency
    # This is a simple example - in practice you might use more sophisticated metrics
    loss = -signal_weight * signal_eff + (1 - signal_weight) * background_eff

    return loss

# -----------------------------
# Gradient estimation
# -----------------------------

def finite_difference_gradient(threshold: float, data: np.ndarray, true_labels: np.ndarray,
                             sharpness: float = 1.0, signal_weight: float = 1.0,
                             delta: float = 0.1) -> float:
    """Finite difference gradient of the loss function."""
    loss_plus = loss_function(threshold + delta, data, true_labels, sharpness, signal_weight)
    loss_minus = loss_function(threshold - delta, data, true_labels, sharpness, signal_weight)
    return (loss_plus - loss_minus) / (2 * delta)

def analytical_gradient(threshold: float, data: np.ndarray, true_labels: np.ndarray,
                       sharpness: float = 1.0, signal_weight: float = 1.0) -> float:
    """
    Analytical gradient of the loss function.

    This demonstrates the power of differentiable cuts - we can compute
    exact gradients for optimization.
    """
    # Compute probabilities and their derivatives
    z = sharpness * (data - threshold)
    probs = 1.0 / (1.0 + np.exp(-z))

    # Derivative of sigmoid: p(1-p) * sharpness
    dprobs_dtheta = -sharpness * probs * (1 - probs)

    # Gradient of loss components
    signal_mask = true_labels == 1
    background_mask = true_labels == 0

    # d/dθ E[p | signal]
    signal_grad = np.mean(dprobs_dtheta[signal_mask]) if np.sum(signal_mask) > 0 else 0

    # d/dθ E[p | background]
    background_grad = np.mean(dprobs_dtheta[background_mask]) if np.sum(background_mask) > 0 else 0

    # Total gradient
    grad = -signal_weight * signal_grad + (1 - signal_weight) * background_grad

    return grad

# -----------------------------
# Visualization functions
# -----------------------------

def plot_data_and_cuts(data: np.ndarray, true_labels: Optional[np.ndarray] = None,
                      threshold: float = 60.0, sharpness: float = 1.0,
                      n_stochastic_samples: int = 3) -> None:
    """
    Visualize the data and different types of cuts.

    Args:
        data: Input data
        true_labels: Optional true signal/background labels
        threshold: Threshold value
        sharpness: Sigmoid sharpness
        n_stochastic_samples: Number of stochastic cut examples to show
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Data range for plotting
    x_min, x_max = data.min() - 5, data.max() + 5
    x_range = np.linspace(x_min, x_max, 1000)

    # Plot 1: Original data histogram
    ax1 = axes[0, 0]
    if true_labels is not None:
        # Separate signal and background
        signal_data = data[true_labels == 1]
        background_data = data[true_labels == 0]

        ax1.hist(background_data, bins=50, alpha=0.7, label='Background', color='red', density=True)
        ax1.hist(signal_data, bins=50, alpha=0.7, label='Signal', color='blue', density=True)
        ax1.legend()
    else:
        ax1.hist(data, bins=50, alpha=0.7, color='gray', density=True)

    ax1.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
    ax1.set_title('Original Data Distribution')
    ax1.set_xlabel('Variable Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cut functions
    ax2 = axes[0, 1]

    # Hard cut
    hard_cut_vals = (x_range >= threshold).astype(float)
    ax2.plot(x_range, hard_cut_vals, 'r-', linewidth=3, label='Hard Cut', alpha=0.8)

    # Soft cut probability
    soft_probs = soft_cut_probability(x_range, threshold, sharpness)
    ax2.plot(x_range, soft_probs, 'b-', linewidth=3, label=f'Soft Cut (β={sharpness})', alpha=0.8)

    ax2.axvline(threshold, color='black', linestyle='--', alpha=0.7)
    ax2.set_title('Cut Functions Comparison')
    ax2.set_xlabel('Variable Value')
    ax2.set_ylabel('Selection Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Plot 3: Hard cut applied to data
    ax3 = axes[1, 0]
    hard_selected = hard_cut(data, threshold)
    selected_data = data[hard_selected == 1]
    rejected_data = data[hard_selected == 0]

    ax3.hist(rejected_data, bins=50, alpha=0.7, label='Rejected', color='red', density=True)
    ax3.hist(selected_data, bins=50, alpha=0.7, label='Selected', color='green', density=True)
    ax3.axvline(threshold, color='black', linestyle='--', linewidth=2)
    ax3.set_title(f'Hard Cut Result (Selected: {len(selected_data)}/{len(data)})')
    ax3.set_xlabel('Variable Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Multiple stochastic cut samples
    ax4 = axes[1, 1]
    colors = ['green', 'orange', 'purple']

    for i in range(min(n_stochastic_samples, len(colors))):
        rng = np.random.default_rng(i + 42)  # Different seed for each sample
        stoch_selected = stochastic_cut(data, threshold, sharpness, rng)
        selected_data_stoch = data[stoch_selected == 1]

        ax4.hist(selected_data_stoch, bins=30, alpha=0.5,
                label=f'Stochastic Sample {i+1} ({len(selected_data_stoch)} events)',
                color=colors[i], density=True)

    ax4.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax4.set_title('Stochastic Cut Samples')
    ax4.set_xlabel('Variable Value')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_cut_optimization(data: np.ndarray, true_labels: np.ndarray,
                         sharpness: float = 1.0, signal_weight: float = 0.7) -> tuple[float, float]:
    """
    Visualize the optimization of threshold parameter.

    Shows how loss function varies with threshold and demonstrates
    gradient-based optimization.
    """
    # Range of thresholds to test
    thresholds = np.linspace(data.min(), data.max(), 100)

    # Compute loss for each threshold
    losses = [loss_function(t, data, true_labels, sharpness, signal_weight) for t in thresholds]

    # Find optimal threshold
    optimal_idx = np.argmin(losses)
    optimal_threshold = thresholds[optimal_idx]

    # Compute gradients at various points
    grad_thresholds = thresholds[::10]  # Subsample for clarity
    analytical_grads = [analytical_gradient(t, data, true_labels, sharpness, signal_weight)
                       for t in grad_thresholds]
    finite_diff_grads = [finite_difference_gradient(t, data, true_labels, sharpness, signal_weight)
                        for t in grad_thresholds]

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Loss function
    ax1.plot(thresholds, losses, 'b-', linewidth=2, label='Loss Function')
    ax1.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Optimal Threshold = {optimal_threshold:.1f}')
    ax1.scatter(optimal_threshold, losses[optimal_idx], color='red', s=100, zorder=5)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Function vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient comparison
    ax2.plot(grad_thresholds, analytical_grads, 'bo-', label='Analytical Gradient', markersize=6)
    ax2.plot(grad_thresholds, finite_diff_grads, 'r^-', label='Finite Difference', markersize=6)
    ax2.axhline(0, color='black', linestyle=':', alpha=0.7)
    ax2.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Gradient')
    ax2.set_title('Gradient Estimates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: ROC-like curve (signal eff vs background eff)
    signal_effs = []
    background_effs = []

    for threshold in thresholds:
        probs = soft_cut_probability(data, threshold, sharpness)
        signal_mask = true_labels == 1
        background_mask = true_labels == 0

        sig_eff = np.mean(probs[signal_mask])
        bkg_eff = np.mean(probs[background_mask])

        signal_effs.append(sig_eff)
        background_effs.append(bkg_eff)

    ax3.plot(background_effs, signal_effs, 'g-', linewidth=2, label='Soft Cut')

    # Mark optimal point
    opt_sig_eff = signal_effs[optimal_idx]
    opt_bkg_eff = background_effs[optimal_idx]
    ax3.scatter(opt_bkg_eff, opt_sig_eff, color='red', s=100, zorder=5,
               label=f'Optimal (BGE={opt_bkg_eff:.3f}, SGE={opt_sig_eff:.3f})')

    ax3.set_xlabel('Background Efficiency')
    ax3.set_ylabel('Signal Efficiency')
    ax3.set_title('Signal vs Background Efficiency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # Plot 4: Efficiency metrics at optimal threshold
    metrics = compute_efficiency(
        soft_cut_probability(data, optimal_threshold, sharpness) > 0.5,
        true_labels
    )

    metric_names = ['Signal\nEfficiency', 'Background\nRejection', 'Purity']
    metric_values = [metrics['signal_efficiency'], metrics['background_rejection'], metrics['purity']]

    bars = ax4.bar(metric_names, metric_values, color=['blue', 'red', 'green'], alpha=0.7)
    ax4.set_ylabel('Value')
    ax4.set_title(f'Performance Metrics at Optimal Threshold\n({metrics["n_selected"]} events selected)')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    return optimal_threshold, losses[optimal_idx]

def compare_sharpness_effects(data: np.ndarray, threshold: float = 60.0) -> None:
    """
    Show how the sharpness parameter affects the soft cut behavior.
    """
    sharpness_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    x_range = np.linspace(data.min() - 5, data.max() + 5, 1000)

    plt.figure(figsize=(12, 8))

    # Plot soft cut probabilities for different sharpness values
    plt.subplot(2, 2, 1)
    for sharpness in sharpness_values:
        probs = soft_cut_probability(x_range, threshold, sharpness)
        plt.plot(x_range, probs, linewidth=2, label=f'β = {sharpness}')

    plt.axvline(threshold, color='black', linestyle='--', alpha=0.7, label='Threshold')
    plt.xlabel('Variable Value')
    plt.ylabel('Selection Probability')
    plt.title('Soft Cut Probability vs Sharpness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot histograms of selected events for different sharpness values
    plt.subplot(2, 2, 2)
    rng = np.random.default_rng(42)

    for i, sharpness in enumerate([0.5, 1.0, 2.0]):
        selected = stochastic_cut(data, threshold, sharpness, rng)
        selected_data = data[selected == 1]
        plt.hist(selected_data, bins=30, alpha=0.6, label=f'β = {sharpness} ({len(selected_data)} events)',
                density=True)

    plt.axvline(threshold, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Variable Value')
    plt.ylabel('Density')
    plt.title('Selected Events vs Sharpness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot gradient magnitude vs sharpness
    plt.subplot(2, 2, 3)
    if len(data) > 1000:  # Only if we have labels
        sample_data = data[:1000]  # Use subset for speed
        sample_labels = np.random.binomial(1, 0.3, 1000)  # Mock labels

        gradients = []
        for sharpness in np.linspace(0.1, 5.0, 50):
            grad = abs(analytical_gradient(threshold, sample_data, sample_labels, sharpness))
            gradients.append(grad)

        plt.plot(np.linspace(0.1, 5.0, 50), gradients, 'b-', linewidth=2)
        plt.xlabel('Sharpness (β)')
        plt.ylabel('|Gradient|')
        plt.title('Gradient Magnitude vs Sharpness')
        plt.grid(True, alpha=0.3)

    # Plot transition width vs sharpness
    plt.subplot(2, 2, 4)
    transition_widths = []
    for sharpness in sharpness_values:
        # Transition width: range where probability goes from 0.1 to 0.9
        prob_01 = threshold - np.log(9) / sharpness  # Where p = 0.1
        prob_09 = threshold + np.log(9) / sharpness  # Where p = 0.9
        width = prob_09 - prob_01
        transition_widths.append(width)

    plt.plot(sharpness_values, transition_widths, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Sharpness (β)')
    plt.ylabel('Transition Width (0.1 → 0.9)')
    plt.title('Cut Transition Width vs Sharpness')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

# -----------------------------
# Main demonstration functions
# -----------------------------

def demo_basic_cuts():
    """Demonstrate basic hard vs soft vs stochastic cuts."""
    print("🎯 Demonstrating Basic Threshold Cuts")
    print("=" * 50)

    # Generate synthetic physics-like data
    data, labels = generate_physics_like_data(n_samples=5000, signal_fraction=0.3)

    print(f"Generated {len(data)} events:")
    print(f"  • Signal events: {np.sum(labels)}")  # type: ignore
    print(f"  • Background events: {np.sum(1 - labels)}")
    print(f"  • Data range: [{data.min():.1f}, {data.max():.1f}]")

    # Choose threshold
    threshold = 65.0
    sharpness = 1.0

    print(f"\nApplying cuts with threshold = {threshold}")

    # Apply different types of cuts
    hard_selected = hard_cut(data, threshold)
    soft_probs = soft_cut_probability(data, threshold, sharpness)
    stoch_selected = stochastic_cut(data, threshold, sharpness, np.random.default_rng(42))

    # Compute and display metrics
    hard_metrics = compute_efficiency(hard_selected, labels)
    soft_metrics = compute_efficiency((soft_probs > 0.5).astype(float), labels)
    stoch_metrics = compute_efficiency(stoch_selected, labels)

    print("\n📊 Cut Performance Comparison:")
    print("-" * 60)
    print("Cut Type        | Signal Eff | Bkg Rej | Purity | N Selected")
    print("-" * 60)
    print(f"Hard Cut        | {hard_metrics['signal_efficiency']:.3f}      | {hard_metrics['background_rejection']:.3f}   | {hard_metrics['purity']:.3f}  | {hard_metrics['n_selected']}")
    print(f"Soft Cut (p>0.5)| {soft_metrics['signal_efficiency']:.3f}      | {soft_metrics['background_rejection']:.3f}   | {soft_metrics['purity']:.3f}  | {soft_metrics['n_selected']}")
    print(f"Stochastic Cut  | {stoch_metrics['signal_efficiency']:.3f}      | {stoch_metrics['background_rejection']:.3f}   | {stoch_metrics['purity']:.3f}  | {stoch_metrics['n_selected']}")

    # Visualize results
    plot_data_and_cuts(data, labels, threshold, sharpness)

    return data, labels, threshold

def demo_cut_optimization():
    """Demonstrate gradient-based optimization of threshold parameter."""
    print("\n🎯 Demonstrating Cut Optimization")
    print("=" * 50)

    # Generate data
    data, labels = generate_physics_like_data(n_samples=3000, signal_fraction=0.25)

    sharpness = 2.0  # Sharper sigmoid for better optimization
    signal_weight = 0.8  # Prioritize signal efficiency

    print(f"Optimizing threshold with:")
    print(f"  • Sharpness parameter β = {sharpness}")
    print(f"  • Signal weight = {signal_weight}")
    print(f"  • Background weight = {1 - signal_weight}")

    # Find optimal threshold
    optimal_threshold, optimal_loss = plot_cut_optimization(data, labels, sharpness, signal_weight)

    print(f"\n🎯 Optimization Results:")
    print(f"  • Optimal threshold: {optimal_threshold:.2f}")
    print(f"  • Optimal loss: {optimal_loss:.4f}")

    # Compare with naive threshold choice
    naive_threshold = np.median(data)
    naive_loss = loss_function(naive_threshold, data, labels, sharpness, signal_weight)

    print(f"  • Naive threshold (median): {naive_threshold:.2f}")
    print(f"  • Naive loss: {naive_loss:.4f}")
    print(f"  • Improvement: {((naive_loss - optimal_loss) / naive_loss * 100):.1f}%")

    return optimal_threshold

def demo_sharpness_effects():
    """Demonstrate the effect of sharpness parameter."""
    print("\n🎯 Demonstrating Sharpness Parameter Effects")
    print("=" * 50)

    # Generate multimodal data for better demonstration
    data = generate_multimodal_data(n_samples=6000)
    threshold = np.percentile(data, 60)  # 60th percentile as threshold

    print(f"Generated multimodal data with {len(data)} events")
    print(f"Using threshold at 60th percentile: {threshold:.1f}")

    # Show effects of different sharpness values
    compare_sharpness_effects(data, threshold)

    print("\n💡 Key insights about sharpness parameter β:")
    print("  • Higher β → sharper transition (closer to hard cut)")
    print("  • Lower β → smoother transition (more gradual selection)")
    print("  • Higher β → larger gradients (faster optimization)")
    print("  • Lower β → more exploration, less exploitation")

def main():
    """Run all demonstrations."""
    print("🚀 Stochastic Threshold Cuts Demonstration")
    print("=" * 60)
    print("This script shows how to replace hard threshold cuts with")
    print("differentiable stochastic alternatives for optimization.")
    print()

    # Run demonstrations
    data, labels, threshold = demo_basic_cuts()
    optimal_threshold = demo_cut_optimization()
    demo_sharpness_effects()

if __name__ == "__main__":
    main()
