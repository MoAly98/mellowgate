import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from mellowgate.api.estimators import (
    GumbelSoftmaxConfig,
    ReinforceConfig,
    ReinforceState,
    gumbel_softmax_gradient,
    reinforce_gradient,
)
from mellowgate.api.functions import Branch, DiscreteProblem, LogitsModel
from mellowgate.utils.outputs import OutputManager

# Initialize output manager
output_manager = OutputManager(base_directory="outputs")


# Define the hard cut function
def hard_cut(data, threshold):
    return data[data > (threshold)]


# Define the sigmoid relaxation function
def sigmoid_cut(data, threshold, k=10):
    weights = 1 / (1 + np.exp(-k * (data - threshold)))
    return data, weights


# Define mellowgate stochastic relaxation
def pass_branch(data):
    """Branch for passing the data point."""
    return data


def fail_branch(data):
    """Branch for failing the data point."""
    return np.zeros_like(data)


branches = [
    Branch(function=fail_branch, derivative_function=lambda x: np.zeros_like(x)),
    Branch(function=pass_branch, derivative_function=lambda x: np.ones_like(x)),
]

temperature = 0.1  # Define a temperature parameter for scaling logits
# Generate synthetic data
data = np.linspace(-3, 3, 100000)
alpha = np.array([-1.0, 1.0])
threshold = 0.0

# Update logits model to use a Heaviside function for branch selection
logits_model = LogitsModel(
    logits_function=lambda th: np.array(
        [-(th - threshold) / 0.1, (th - threshold) / 0.1]
    ),
    logits_derivative_function=lambda th: np.array(
        [np.zeros_like(th), np.zeros_like(th)]
    ),  # Derivative is zero for Heaviside
    probability_function=lambda logits: 1.0 / ((1 + np.exp(-logits))),
)

problem = DiscreteProblem(branches=branches, logits_model=logits_model)

# Apply hard cut
hard_cut_result = hard_cut(data, threshold)

# Apply sigmoid relaxation
sigmoid_result, sigmoid_weights = sigmoid_cut(data, threshold)

# Sample branch indices using Heaviside-based logits
num_samples = 1  # Number of samples per data point
sampled_branch_indices = problem.sample_branch(data, num_samples=num_samples)

# Filter data based on sampled indices
stochastic_result = np.array(
    [
        data[i] if branch_idx == 1 else None  # Keep data if pass branch is sampled
        for i, branch_idx in enumerate(sampled_branch_indices.flatten())
    ]
)

# Remove None values
stochastic_result = stochastic_result[stochastic_result != np.array(None)]

# Calculate total number of events left after the "cut" in all three cases
num_events_hard_cut = len(hard_cut_result)
num_events_sigmoid_cut = np.sum(sigmoid_weights)
num_events_stochastic_cut = len(stochastic_result)

print(f"Total events after hard cut: {num_events_hard_cut}")
print(f"Total events after sigmoid cut: {num_events_sigmoid_cut}")
print(f"Total events after stochastic cut: {num_events_stochastic_cut}")

# Set all plot styles in rcParams
plt.rcParams.update(
    {
        "axes.linewidth": 1.5,  # Thicker axes
        "xtick.direction": "in",  # Inside ticks
        "ytick.direction": "in",  # Inside ticks
        "xtick.major.size": 6,  # Major tick size
        "ytick.major.size": 6,  # Major tick size
        "xtick.minor.size": 4,  # Minor tick size
        "ytick.minor.size": 4,  # Minor tick size
        "xtick.minor.width": 0.8,  # Minor tick width
        "ytick.minor.width": 0.8,  # Minor tick width
        "xtick.major.width": 1.5,  # Major tick width
        "ytick.major.width": 1.5,  # Major tick width
        "font.size": 12,  # Font size for labels
        "legend.fontsize": 10,  # Font size for legend
        "figure.figsize": (10, 6),  # Default figure size
        "axes.labelsize": 12,  # Label font size
        "axes.titlesize": 14,  # Title font size
        "axes.grid": False,  # Disable grid by default
        "xtick.minor.visible": True,  # Enable minor ticks on x-axis
        "ytick.minor.visible": True,  # Enable minor ticks on y-axis
        "xtick.top": True,  # Enable ticks on the top side
        "xtick.bottom": True,  # Enable ticks on the bottom side
        "ytick.left": True,  # Enable ticks on the left side
        "ytick.right": True,  # Enable ticks on the right side
    }
)

# Update the color scheme to a popular internet-favored palette
plt.rcParams.update(
    {
        "axes.prop_cycle": cycler(
            color=[
                "#E15759",  # Red
                "#F28E2B",  # Orange
                "#4E79A7",  # Blue
                "#76B7B2",  # Teal
                "#59A14F",  # Green
                "#EDC949",  # Yellow
                "#AF7AA1",  # Purple
                "#FF9DA7",  # Pink
                "#9C755F",  # Brown
                "#BAB0AC",  # Gray
            ]
        )
    }
)

# Plot results
plt.figure(figsize=(10, 6))

# Hard cut
hard_cut_counts, hard_cut_bins = np.histogram(hard_cut_result, bins=50)
plt.plot(hard_cut_bins[:-1], hard_cut_counts, label="Hard Cut", drawstyle="steps-post")

# Sigmoid relaxation
sigmoid_counts, sigmoid_bins = np.histogram(
    sigmoid_result, bins=50, weights=sigmoid_weights
)
plt.plot(
    sigmoid_bins[:-1],
    sigmoid_counts,
    label="Sigmoid Relaxation",
    drawstyle="steps-post",
)

# Stochastic relaxation
stochastic_counts, stochastic_bins = np.histogram(stochastic_result, bins=50)
temperature_label = f"Stochastic Relaxation (T={temperature})"
plt.plot(
    stochastic_bins[:-1],
    stochastic_counts,
    label=temperature_label,
    drawstyle="steps-post",
)

plt.xlabel("Data Values", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# Update legends to include totals and remove legend box
plt.legend(
    labels=[
        f"Hard Cut (Total: {num_events_hard_cut:,})",
        f"Sigmoid Relaxation (Total: {num_events_sigmoid_cut:,.2f})",
        f"Stochastic Relaxation (Total: {num_events_stochastic_cut:,})",
    ],
    fontsize=10,
    frameon=False,  # Remove legend box
)

plt.tight_layout()

# Save the plot using OutputManager
plot_path = output_manager.get_path("plots", "cut_relaxation_comparison.png")
plt.savefig(plot_path)

# Normalize the histogram counts for each method
plt.figure(figsize=(10, 6))

# Hard cut (normalized)
hard_cut_counts, hard_cut_bins = np.histogram(hard_cut_result, bins=50, density=True)
plt.plot(
    hard_cut_bins[:-1],
    hard_cut_counts,
    label="Hard Cut (Normalized)",
    drawstyle="steps-post",
)

# Sigmoid relaxation (normalized)
sigmoid_counts, sigmoid_bins = np.histogram(
    sigmoid_result, bins=50, weights=sigmoid_weights, density=True
)
plt.plot(
    sigmoid_bins[:-1],
    sigmoid_counts,
    label="Sigmoid Relaxation (Normalized)",
    drawstyle="steps-post",
)

# Stochastic relaxation (normalized)
stochastic_counts, stochastic_bins = np.histogram(
    stochastic_result, bins=50, density=True
)
plt.plot(
    stochastic_bins[:-1],
    stochastic_counts,
    label=f"Stochastic Relaxation (T={temperature}, Normalized)",
    drawstyle="steps-post",
)

plt.xlabel("Data Values", fontsize=12)
plt.ylabel("Normalized Frequency", fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()

# Save the normalized plot using OutputManager
normalized_plot_path = output_manager.get_path(
    "plots", "normalized_cut_relaxation_comparison.png"
)
plt.savefig(normalized_plot_path)

# Define configurations for Gumbel-Softmax and REINFORCE
config_gumbel = GumbelSoftmaxConfig(
    temperature=0.5, num_samples=500, use_straight_through_estimator=True
)
config_reinforce = ReinforceConfig(num_samples=500, use_baseline=True)
state_reinforce = ReinforceState()

# Compute gradients using Gumbel-Softmax and REINFORCE
gradient_gumbel = gumbel_softmax_gradient(problem, data, config=config_gumbel)
gradient_reinforce = reinforce_gradient(
    problem, data, config=config_reinforce, state=state_reinforce
)

print(f"Gradient using Gumbel-Softmax: {gradient_gumbel}")
print(f"Gradient using REINFORCE: {gradient_reinforce}")

# Plot gradients
plt.figure(figsize=(10, 6))

# Plot Gumbel-Softmax gradient
plt.plot(data, gradient_gumbel, label="Gumbel-Softmax Gradient", color="blue")

# Plot REINFORCE gradient
plt.plot(data, gradient_reinforce, label="REINFORCE Gradient", color="orange")

plt.xlabel("Data Values", fontsize=12)
plt.ylabel("Gradient", fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()

# Save the gradient plot using OutputManager
gradient_plot_path = output_manager.get_path("plots", "gradient_comparison.png")
plt.savefig(gradient_plot_path)
