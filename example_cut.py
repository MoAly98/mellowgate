# Import necessary libraries
import jax.numpy as jnp
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
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
def apply_gill_sans_font(ax, legloc=None, axcolor="black", ylabcolor="black"):
    """Apply Gill Sans font and font size to axis labels, tick labels, and legend."""
    ax.set_xlabel(
        ax.get_xlabel(), fontproperties=gs_font, fontsize=15
    )  # Adjust font size
    ax.set_ylabel(
        ax.get_ylabel(), fontproperties=gs_font, fontsize=15, color=ylabcolor
    )  # Adjust font size
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(gs_font)
        label.set_fontsize(15)  # Adjust tick label font size
        label.set_color(axcolor)  # Adjust tick label color

    # Update legend to use Gill Sans font, split into 3 columns, and increase font size
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(gs_font)
            text.set_fontsize(15)  # Increase legend font size
        legend.set_frame_on(True)  # Enable legend frame
        legend.get_frame().set_edgecolor("none")  # Remove box edge
        legend.get_frame().set_linewidth(0)  # Set frame line width to 0
        if legloc is None:
            legend.set_bbox_to_anchor((0.0, 1.1))  # Position above the top axis
        legend.set_loc("upper left")  # Center the legend horizontally


# Apply the old style
plt.style.use("./src/mellowgate/plots/50s.mplstyle")
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


def sigmoid_cut(data, threshold, temp=0.1):
    """Apply a sigmoid relaxation to the data."""
    weights = 1 / (1 + jnp.exp(-(data - threshold) / temp))
    return data, weights


def pass_branch(data):
    """Branch for passing the data point."""
    return jnp.ones_like(data)


def fail_branch(data):
    """Branch for failing the data point."""
    return jnp.zeros_like(data)


# Set up mellowgate problem
# ----------------------------------------
branches = [
    Branch(function=pass_branch, derivative_function=lambda x: jnp.zeros_like(x)),
    Branch(function=fail_branch, derivative_function=lambda x: jnp.zeros_like(x)),
]

threshold = 0.0
temp = 0.1
logits_model = LogitsModel(
    logits_function=lambda th: jnp.array(
        [(th - threshold) / temp, -(th - threshold) / temp]
    ),
    logits_derivative_function=lambda th: jnp.array(
        [jnp.ones_like(th) / temp, -jnp.ones_like(th) / temp]
    ),
    probability_function=lambda logits: 1.0 / (1 + jnp.exp(-logits)),
)

problem = DiscreteProblem(branches=branches, logits_model=logits_model)

# Generate synthetic data
# ----------------------------------------
data = jnp.linspace(-3, 3, 10000)

# Apply cuts
# ----------------------------------------
hard_cut_result = hard_cut(data, threshold)
sigmoid_result, sigmoid_weights = sigmoid_cut(data, threshold)

# Sample branch indices using mellowgate
num_samples = 1
sampled_branch_indices = problem.sample_branch(data, num_samples=num_samples)

# Filter data based on sampled indices (JAX-compatible)
pass_mask = sampled_branch_indices.flatten() == 0
stochastic_result = data[pass_mask]

# Calculate total events after cuts
# ----------------------------------------
num_events_hard_cut = len(hard_cut_result)
num_events_sigmoid_cut = jnp.sum(sigmoid_weights)
num_events_stochastic_cut = len(stochastic_result)

# Print results
print(f"Total events after hard cut: {num_events_hard_cut}")
print(f"Total events after sigmoid cut: {num_events_sigmoid_cut}")
print(f"Total events after stochastic cut: {num_events_stochastic_cut}")

# Plot results
# ----------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Top plot: Raw frequencies
bins = jnp.linspace(-3, 3, 51)
hard_cut_counts, hard_cut_bins = jnp.histogram(hard_cut_result, bins=bins)
ax1.plot(hard_cut_bins[:-1], hard_cut_counts, label="Hard Cut", drawstyle="steps-post")

sigmoid_counts, sigmoid_bins = jnp.histogram(
    sigmoid_result, bins=bins, weights=sigmoid_weights
)
ax1.plot(
    sigmoid_bins[:-1],
    sigmoid_counts,
    label="Sigmoid Relaxation",
    drawstyle="steps-post",
)

stochastic_counts, stochastic_bins = jnp.histogram(stochastic_result, bins=bins)
ax1.plot(
    stochastic_bins[:-1],
    stochastic_counts,
    label="Stochastic Relaxation",
    drawstyle="steps-post",
)

ax1.set_xlabel("Data Values")
ax1.set_ylabel("Raw Frequency")
ax1.legend(fontsize=10)
ax1.grid(True)
ax1.set_title("Raw Frequencies")
apply_gill_sans_font(ax1, legloc="upper left")

# Bottom plot: Expectation values (probability of passing)
# Compute expectation values for a subset of data for cleaner visualization
data_subset = jnp.linspace(-3, 3, 1000)
expected_values = problem.compute_expected_value(data_subset)

# Also compute the sigmoid probability for comparison
sigmoid_probs = 1 / (1 + jnp.exp(-(data_subset - threshold) / temp))

# Hard cut expectation (step function)
hard_cut_expectation = (data_subset > threshold).astype(float)

ax2.plot(
    data_subset,
    expected_values,
    label="Stochastic Expectation",
    color="purple",
    linewidth=2,
)
ax2.plot(
    data_subset,
    sigmoid_probs,
    label="Sigmoid Probability",
    color="orange",
    linewidth=2,
    linestyle="--",
)
ax2.plot(
    data_subset,
    hard_cut_expectation,
    label="Hard Cut Expectation",
    color="blue",
    linewidth=2,
    linestyle="-.",
)

ax2.set_xlabel("Data Values")
ax2.set_ylabel("Expectation Value (Probability of Passing)")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title("Expectation Values: Probability of Selecting Pass Branch")
ax2.set_ylim(-0.05, 1.05)
apply_gill_sans_font(ax2, legloc="upper left")

plt.tight_layout()
plot_path = output_manager.get_path("plots", "cut_relaxation_comparison.png")
fig.savefig(plot_path)

# Print some key statistics
print("Expected value statistics:")
print(f"  Mean expectation: {jnp.mean(expected_values):.4f}")
print(f"  Min expectation:  {jnp.min(expected_values):.4f}")
print(f"  Max expectation:  {jnp.max(expected_values):.4f}")
threshold_expectation = problem.compute_expected_value(jnp.array([threshold]))[0]
print(f"  Threshold crossing at data = {threshold}: E[f] = {threshold_expectation:.4f}")


# plot: Gradient comparison
# ----------------------------------------
# Note: Higher temperature (0.5 vs 0.1) reduces variance in Gumbel-Softmax gradients
# Low temperatures make the distribution very sharp, causing high variance
config_gumbel = GumbelSoftmaxConfig(
    temperature=0.5, num_samples=10000, use_straight_through_estimator=True
)
config_reinforce = ReinforceConfig(num_samples=1000, use_baseline=True)
state_reinforce = ReinforceState()

# Compute gradient estimates
gradient_gumbel = gumbel_softmax_gradient(problem, data, config=config_gumbel)
gradient_reinforce = reinforce_gradient(
    problem, data, config=config_reinforce, state=state_reinforce
)

# Compute exact gradient using MellowGate's built-in method
exact_gradient = problem.compute_exact_gradient(data)

fig, ax = plt.subplots(figsize=(12, 8))

if exact_gradient is not None:
    ax.plot(
        data,
        exact_gradient,
        label="Exact Gradient",
        color="red",
        linewidth=2,
        linestyle="-",
    )

ax.plot(data, gradient_gumbel, label="Gumbel-Softmax Gradient", color="blue", alpha=0.7)
ax.plot(data, gradient_reinforce, label="REINFORCE Gradient", color="orange", alpha=0.7)

ax.set_xlabel("Data Values")
ax.set_ylabel("Gradient")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_title("Gradient Comparison: Exact vs Estimated", fontsize=14)
apply_gill_sans_font(ax, legloc="upper left")
gradient_plot_path = output_manager.get_path("plots", "gradient_comparison.png")
fig.savefig(gradient_plot_path)

# Print some statistics for comparison
if exact_gradient is not None:
    print("Exact gradient statistics:")
    print(f"  Mean: {jnp.mean(exact_gradient):.6f}")
    print(f"  Std:  {jnp.std(exact_gradient):.6f}")
    print(f"  Max:  {jnp.max(exact_gradient):.6f}")

print("\nGumbel-Softmax gradient statistics:")
print(f"  Mean: {jnp.mean(gradient_gumbel):.6f}")
print(f"  Std:  {jnp.std(gradient_gumbel):.6f}")
if exact_gradient is not None:
    gumbel_mse = jnp.mean((gradient_gumbel - exact_gradient) ** 2)
    print(f"  MSE vs exact: {gumbel_mse:.8f}")

print("\nREINFORCE gradient statistics:")
print(f"  Mean: {jnp.mean(gradient_reinforce):.6f}")
print(f"  Std:  {jnp.std(gradient_reinforce):.6f}")
if exact_gradient is not None:
    print(f"  MSE vs exact: {jnp.mean((gradient_reinforce - exact_gradient)**2):.8f}")


# Temperature Analysis for Gumbel-Softmax
# ========================================
print("\n" + "=" * 60)
print("TEMPERATURE ANALYSIS FOR GUMBEL-SOFTMAX")
print("=" * 60)

# Test a subset of data for computational efficiency
data_test = jnp.linspace(-1, 1, 1000)
exact_gradient_test = problem.compute_exact_gradient(data_test)

if exact_gradient_test is None:
    print("Warning: Exact gradient not available. Skipping temperature analysis.")
else:
    # Temperature values to test
    temperatures = jnp.array([0.1, 0.3, 0.5, 1.0, 2.0])
    num_repetitions = 10  # Number of repetitions for variance calculation
    num_samples_temp = 200  # Samples per gradient estimate

    # Storage for results
    gradient_estimates = []
    bias_values = []
    variance_values = []
    mse_values = []

    num_temps = len(temperatures)
    test_msg = (
        f"Testing {num_temps} temperature values with "
        f"{num_repetitions} repetitions each..."
    )
    print(test_msg)

    # Run the temperature experiment
    for i, temp in enumerate(temperatures):
        print(f"Temperature {temp:.2f}... ", end="", flush=True)

        # Multiple repetitions for this temperature
        temp_gradients = []
        for rep in range(num_repetitions):
            config_temp = GumbelSoftmaxConfig(
                temperature=temp,
                num_samples=num_samples_temp,
                use_straight_through_estimator=True,
            )
            grad_est = gumbel_softmax_gradient(problem, data_test, config=config_temp)
            temp_gradients.append(grad_est)

        temp_gradients = jnp.array(temp_gradients)
        gradient_estimates.append(temp_gradients)

        # Calculate statistics
        mean_gradient = jnp.mean(temp_gradients, axis=0)
        bias_squared = jnp.mean((mean_gradient - exact_gradient_test) ** 2)

        # Calculate variance across repetitions at each data point, then average
        point_wise_variance = jnp.var(temp_gradients, axis=0)
        variance = jnp.mean(point_wise_variance)

        # Calculate MSE (combining bias and variance)
        all_squared_errors = []
        for rep_grad in temp_gradients:
            squared_error = (rep_grad - exact_gradient_test) ** 2
            all_squared_errors.append(squared_error)
        mse = jnp.mean(jnp.array(all_squared_errors))

        bias_values.append(float(bias_squared))
        variance_values.append(float(variance))
        mse_values.append(float(mse))

        # Debug info for first temperature
        if i == 0:
            print(f"\nDebug for T={temp}:")
            print(f"  Shape of temp_gradients: {temp_gradients.shape}")
            print(f"  Sample variance values: {point_wise_variance[:5]}")
            print(f"  Mean variance: {variance}")
            print(f"  Bias squared: {bias_squared}")
            print(f"  MSE: {mse}")
        else:
            print("✓")

    # Convert to arrays for plotting
    bias_values = jnp.array(bias_values)
    variance_values = jnp.array(variance_values)
    mse_values = jnp.array(mse_values)

    print("Analysis complete!")

    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

    # Plot 1: Sample gradient estimates for selected temperatures
    sample_temps = [0.1, 0.5, 1.0, 2.0]
    temp_indices = []
    for t in sample_temps:
        matches = jnp.where(temperatures == t)[0]
        if len(matches) > 0:
            temp_indices.append(int(matches[0]))

    sample_temps = [temperatures[i] for i in temp_indices]  # Use actual temperatures

    ax1.plot(
        data_test,
        exact_gradient_test,
        "k-",
        linewidth=3,
        label="Exact Gradient",
        alpha=0.8,
    )
    colors = ["red", "blue", "green", "orange"]
    for idx, (temp_idx, temp, color) in enumerate(
        zip(temp_indices, sample_temps, colors)
    ):
        # Plot a few sample estimates
        for rep in range(min(3, num_repetitions)):
            alpha_val = 0.3 if rep > 0 else 0.7
            label_val = f"T={temp}" if rep == 0 else None
            ax1.plot(
                data_test,
                gradient_estimates[temp_idx][rep],
                color=color,
                alpha=alpha_val,
                linewidth=1,
                label=label_val,
            )

    ax1.set_xlabel("Data Values")
    ax1.set_ylabel("Gradient")
    ax1.set_title(
        "Gradient Estimates vs Temperature\n(Multiple Samples per Temperature)"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    apply_gill_sans_font(ax1, legloc="upper left")

    # Plot 2: Bias vs Temperature
    ax2.semilogy(temperatures, bias_values, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Bias² (log scale)")
    ax2.set_title("Bias² vs Temperature")
    ax2.grid(True, alpha=0.3)
    apply_gill_sans_font(ax2)

    # Plot 3: Variance vs Temperature
    ax3.semilogy(temperatures, variance_values, "bo-", linewidth=2, markersize=8)
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel("Variance (log scale)")
    ax3.set_title("Variance vs Temperature")
    ax3.grid(True, alpha=0.3)
    apply_gill_sans_font(ax3)

    # Plot 4: MSE vs Temperature (bias-variance decomposition with dual y-axes)
    ax4_left = ax4
    ax4_right = ax4.twinx()

    # Left axis: MSE and Bias² (larger values)
    line1 = ax4_left.semilogy(
        temperatures, mse_values, "go-", linewidth=2, markersize=8, label="Total MSE"
    )
    line2 = ax4_left.semilogy(
        temperatures, bias_values, "r--", linewidth=2, alpha=0.7, label="Bias²"
    )
    ax4_left.set_xlabel("Temperature")
    ax4_left.set_ylabel("MSE & Bias² (log scale)", color="black")
    ax4_left.tick_params(axis="y", labelcolor="black")

    # Right axis: Variance (smaller values) with blue styling
    line3 = ax4_right.semilogy(
        temperatures,
        variance_values,
        "bo-",
        linewidth=2,
        markersize=8,
        alpha=0.7,
        label="Variance",
    )
    ax4_right.set_ylabel("Variance (log scale)", color="blue")
    ax4_right.tick_params(axis="y", labelcolor="blue")
    ax4_right.spines["right"].set_color("blue")
    ax4_right.spines["right"].set_linewidth(2)

    # Combine legends from both axes and position center right
    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax4_left.legend(lines, labels, loc="center right")

    ax4_left.set_title("Bias-Variance Decomposition (Dual Scale)")
    ax4_left.grid(True, alpha=0.3)

    # Apply consistent font styling to both axes
    apply_gill_sans_font(ax4_left, legloc="center right")
    apply_gill_sans_font(
        ax4_right, legloc="center right", axcolor="blue", ylabcolor="blue"
    )

    plt.tight_layout()
    temp_analysis_path = output_manager.get_path(
        "plots", "gumbel_temperature_analysis.png"
    )
    fig.savefig(temp_analysis_path, dpi=300)

    # Print numerical results with scientific notation for better readability
    print("\nTEMPERATURE ANALYSIS RESULTS:")
    col_temp = "Temp"
    col_bias = "Bias²"
    col_var = "Variance"
    col_mse = "MSE"
    col_ratio = "Ratio V/B²"
    header = f"{col_temp:<6} {col_bias:<12} {col_var:<15} {col_mse:<12} {col_ratio:<10}"
    print(header)
    print("-" * 70)
    for i, temp in enumerate(temperatures):
        ratio = (
            variance_values[i] / bias_values[i] if bias_values[i] > 0 else float("inf")
        )
        result_line = (
            f"{temp:<6.2f} {bias_values[i]:<12.6f} {variance_values[i]:<15.2e} "
            f"{mse_values[i]:<12.6f} {ratio:<10.2e}"
        )
        print(result_line)

    # Find optimal temperature (minimum MSE)
    optimal_idx = jnp.argmin(mse_values)
    optimal_temp = temperatures[optimal_idx]
    print(f"\nOptimal temperature (min MSE): {optimal_temp:.2f}")
    print(f"MSE at optimal temperature: {mse_values[optimal_idx]:.6f}")

    # Compare specific temperatures if they exist
    if len(temperatures) >= 2:
        temp0_info = (
            f"• T={temperatures[0]:.1f}: Bias²={bias_values[0]:.6f}, "
            f"Variance={variance_values[0]:.2e}"
        )
        print(temp0_info)
        if len(temperatures) >= 3:
            mid_idx = len(temperatures) // 2
            temp_mid_info = (
                f"• T={temperatures[mid_idx]:.1f}: Bias²={bias_values[mid_idx]:.6f}, "
                f"Variance={variance_values[mid_idx]:.2e}"
            )
            print(temp_mid_info)
