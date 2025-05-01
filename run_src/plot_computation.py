# Re-import necessary libraries after execution state reset
import matplotlib.pyplot as plt
import numpy as np

# Define data
methods = ["CoT", "MedRAG", "SC", "i-MedRAG", "Search-O1", "PRIME (Ours)"]
accuracy = [75.71, 78.77, 77.47, 80.35,81.17, 86.29]
steps = [395.94, 12376.45, 2718.81,4822.43, 7144.36, 1642.04]

# Define different markers for each method
markers = ["o", "s", "D", "^", "*", "v"]  # Different markers for each method

# Create scatter plot
plt.figure(figsize=(10, 5))

for i, method in enumerate(methods):
    plt.scatter(steps[i], accuracy[i], label=method, marker=markers[i], s=100)

# Set x-axis to display only powers of 2
x_min, x_max = min(steps), max(steps)
x_ticks = [2**i for i in range(int(np.floor(np.log2(x_min))), int(np.ceil(np.log2(x_max))) + 1)]
x_ticks = [256, 2048, 4096, 8192, 16384]
# x_ticks = [1, 2, 4, 8, 16, 32, 40]
# print(x_ticks)
plt.xticks(x_ticks)

# Labels and title
plt.xlabel("Number of Generated Tokens (log scale)")
plt.ylabel("Accuracy (%)")
plt.legend(title="Methods", loc="lower right")
plt.grid(True, linestyle="--", alpha=0.7)

# Save plot
plt.savefig("computational.png")

# Display the plot
# plt.show()
