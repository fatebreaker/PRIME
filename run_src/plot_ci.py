import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generate or provide your data
# ---------------------------------------------------------
np.random.seed(123)
num_subjects = 10

before = np.random.normal(loc=150, scale=40, size=num_subjects)
after  = before - np.random.normal(loc=20, scale=10, size=num_subjects)

# Each subject can be assigned a unique color
colors = plt.cm.tab10(np.linspace(0, 1, num_subjects))

# ---------------------------------------------------------
# 2. Create a small function for x-jitter
# ---------------------------------------------------------
def jitter_values(base_x, n_points, spread=0.08):
    """
    Returns an array of length n_points with random offsets
    around base_x, in the range [-spread, spread].
    """
    return base_x + np.random.uniform(-spread, spread, size=n_points)

# Generate jittered x-coordinates for 'before' and 'after'
x_before = jitter_values(1, num_subjects, spread=0.1)
x_after  = jitter_values(2, num_subjects, spread=0.1)

# ---------------------------------------------------------
# 3. Create the figure and axes
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 7))

# ---------------------------------------------------------
# 4. Plot individual data points (with jittered x)
# ---------------------------------------------------------
for i in range(num_subjects):
    ax.scatter(x_before[i], before[i], color=colors[i], s=80, zorder=3)
    ax.scatter(x_after[i],  after[i],  color=colors[i], s=80, zorder=3)

# ---------------------------------------------------------
# 5. Compute and plot mean ± SD for each group
# ---------------------------------------------------------
mean_before = np.mean(before)
std_before  = np.std(before)
mean_after  = np.mean(after)
std_after   = np.std(after)

# Plot means (black squares) with error bars for ±1 SD
# (We'll keep them exactly at x=1 and x=2)
ax.errorbar(1, mean_before, yerr=std_before, fmt='s',
            color='black', capsize=5, markersize=8, zorder=4)
ax.errorbar(2, mean_after,  yerr=std_after,  fmt='s',
            color='black', capsize=5, markersize=8, zorder=4)

# ---------------------------------------------------------
# 6. Place significance text in the top-right corner
# ---------------------------------------------------------
ax.text(0.95, 0.95, 'p < 0.001',
        transform=ax.transAxes,   # coordinates relative to the axes
        ha='right', va='top',
        fontsize=10)

# ---------------------------------------------------------
# 7. Styling the axes
# ---------------------------------------------------------
ax.set_xticks([1, 2])
ax.set_xticklabels(['Before', 'After'])
ax.set_ylabel('Reading Time (Seconds)')
ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
