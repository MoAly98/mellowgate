# Import necessary libraries
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Import mellowgate components
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

# Define font properties
# ----------------------------------------
gs_font = fm.FontProperties(fname="/System/Library/Fonts/Supplemental/GillSans.ttc")


# Define helper functions
# ----------------------------------------
# Correct the apply_gill_sans_font function to ensure the legend
# splits into 3 columns and increases font size
def apply_gill_sans_font(ax):
    """Apply Gill Sans font and font size to axis labels, tick labels, and legend."""
    ax.set_xlabel(
        ax.get_xlabel(), fontproperties=gs_font, fontsize=13
    )  # Adjust font size
    ax.set_ylabel(
        ax.get_ylabel(), fontproperties=gs_font, fontsize=13
    )  # Adjust font size
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(gs_font)
        label.set_fontsize(12)  # Adjust tick label font size

    # Update legend to use Gill Sans font, split into 3 columns, and increase font size
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(gs_font)
            text.set_fontsize(12)  # Increase legend font size
        legend.set_frame_on(True)  # Enable legend frame
        legend.get_frame().set_edgecolor("none")  # Remove box edge
        legend.get_frame().set_linewidth(0)  # Set frame line width to 0
        legend.set_bbox_to_anchor((0.0, 1.1))  # Position above the top axis
        legend.set_loc("upper left")  # Center the legend horizontally


# Apply the old style
plt.style.use("./50s.mplstyle")
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


# Define cut functions
# ----------------------------------------
def hard_cut(data, threshold):
    """Apply a hard cut to the data."""
    return data[data > threshold]


def sigmoid_cut(data, threshold, k=10):
    """Apply a sigmoid relaxation to the data."""
    weights = 1 / (1 + np.exp(-k * (data - threshold)))
    return data, weights


def pass_branch(data):
    """Branch for passing the data point."""
    return data


def fail_branch(data):
    """Branch for failing the data point."""
    return np.zeros_like(data)


# Set up mellowgate problem
# ----------------------------------------
branches = [
    Branch(function=fail_branch, derivative_function=lambda x: np.zeros_like(x)),
    Branch(function=pass_branch, derivative_function=lambda x: np.ones_like(x)),
]

threshold = 0.0
logits_model = LogitsModel(
    logits_function=lambda th: np.array(
        [-(th - threshold) / 0.1, (th - threshold) / 0.1]
    ),
    logits_derivative_function=lambda th: np.array(
        [np.zeros_like(th), np.zeros_like(th)]
    ),
    probability_function=lambda logits: 1.0 / (1 + np.exp(-logits)),
)

problem = DiscreteProblem(branches=branches, logits_model=logits_model)

# Generate synthetic data
# ----------------------------------------
data = np.linspace(-3, 3, 100000)

# Apply cuts
# ----------------------------------------
hard_cut_result = hard_cut(data, threshold)
sigmoid_result, sigmoid_weights = sigmoid_cut(data, threshold)

# Sample branch indices using mellowgate
num_samples = 1
sampled_branch_indices = problem.sample_branch(data, num_samples=num_samples)

# Filter data based on sampled indices
stochastic_result = np.array(
    [
        data[i] if branch_idx == 1 else None
        for i, branch_idx in enumerate(sampled_branch_indices.flatten())
    ]
)
stochastic_result = stochastic_result[stochastic_result != np.array(None)]

# Calculate total events after cuts
# ----------------------------------------
num_events_hard_cut = len(hard_cut_result)
num_events_sigmoid_cut = np.sum(sigmoid_weights)
num_events_stochastic_cut = len(stochastic_result)

# Print results
print(f"Total events after hard cut: {num_events_hard_cut}")
print(f"Total events after sigmoid cut: {num_events_sigmoid_cut}")
print(f"Total events after stochastic cut: {num_events_stochastic_cut}")

# Plot results
# ----------------------------------------
# First plot: Cut relaxation comparison
fig, ax = plt.subplots(figsize=(10, 6))
hard_cut_counts, hard_cut_bins = np.histogram(hard_cut_result, bins=50)
ax.plot(hard_cut_bins[:-1], hard_cut_counts, label="Hard Cut", drawstyle="steps-post")

sigmoid_counts, sigmoid_bins = np.histogram(
    sigmoid_result, bins=50, weights=sigmoid_weights
)
ax.plot(
    sigmoid_bins[:-1],
    sigmoid_counts,
    label="Sigmoid Relaxation",
    drawstyle="steps-post",
)

stochastic_counts, stochastic_bins = np.histogram(stochastic_result, bins=50)
ax.plot(
    stochastic_bins[:-1],
    stochastic_counts,
    label="Stochastic Relaxation (T=0.1)",
    drawstyle="steps-post",
)

ax.set_xlabel("Data Values")
ax.set_ylabel("Frequency")
ax.legend(fontsize=10, ncols=3)
ax.grid(True)
apply_gill_sans_font(ax)
plot_path = output_manager.get_path("plots", "cut_relaxation_comparison.png")
fig.savefig(plot_path)

# Second plot: Normalized comparison
fig, ax = plt.subplots(figsize=(10, 6))
hard_cut_counts, hard_cut_bins = np.histogram(hard_cut_result, bins=50, density=True)
ax.plot(hard_cut_bins[:-1], hard_cut_counts, label="Hard Cut ", drawstyle="steps-post")

sigmoid_counts, sigmoid_bins = np.histogram(
    sigmoid_result, bins=50, weights=sigmoid_weights, density=True
)
ax.plot(
    sigmoid_bins[:-1],
    sigmoid_counts,
    label="Sigmoid Relaxation (T=0.1) ",
    drawstyle="steps-post",
)

stochastic_counts, stochastic_bins = np.histogram(
    stochastic_result, bins=50, density=True
)
ax.plot(
    stochastic_bins[:-1],
    stochastic_counts,
    label="Stochastic Relaxation (T=0.1)",
    drawstyle="steps-post",
)

ax.set_xlabel("Data Values")
ax.set_ylabel("Normalized Frequency")
ax.legend(fontsize=10, ncols=3)
ax.grid(True)
apply_gill_sans_font(ax)
normalized_plot_path = output_manager.get_path(
    "plots", "normalized_cut_relaxation_comparison.png"
)
fig.savefig(normalized_plot_path)

# Third plot: Gradient comparison
# ----------------------------------------
config_gumbel = GumbelSoftmaxConfig(
    temperature=0.5, num_samples=500, use_straight_through_estimator=True
)
config_reinforce = ReinforceConfig(num_samples=500, use_baseline=True)
state_reinforce = ReinforceState()

gradient_gumbel = gumbel_softmax_gradient(problem, data, config=config_gumbel)
gradient_reinforce = reinforce_gradient(
    problem, data, config=config_reinforce, state=state_reinforce
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data, gradient_gumbel, label="Gumbel-Softmax Gradient", color="blue")
ax.plot(data, gradient_reinforce, label="REINFORCE Gradient", color="orange")

ax.set_xlabel("Data Values")
ax.set_ylabel("Gradient")
ax.legend(fontsize=10, ncols=3)
ax.grid(True)
apply_gill_sans_font(ax)
gradient_plot_path = output_manager.get_path("plots", "gradient_comparison.png")
fig.savefig(gradient_plot_path)
