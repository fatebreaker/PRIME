import matplotlib.pyplot as plt
import numpy as np
# Define data
methods =   ["CoT", "MedRAG", "i-MedRAG", "Search-O1", "PRIME (Ours)", "SC"]
accuracy1 = [62.76, 64.81, 71.21, 75.98, 71.47,  64.61]
accuracy2 = [75.71, 78.77, 80.35, 86.29, 81.17, 77.47]
steps =     [395.94, 1642.04, 2718.81, 4144.36, 4822.43, 12376.45]
# Define different markers for each method
markers = ["o", "s", "D", "^", "*", "v"]  # Different markers for each method
colors = ["b", "g", "r", "c", "m", "y"]  # Assign unique colors
# Create scatter plot
plt.figure(figsize=(8, 5))
for i, method in enumerate(methods):
    # plt.scatter(steps[i], accuracy1[i], label=method, color=colors[i], marker=markers[i], s=100)
    plt.scatter(steps[i], accuracy1[i], label=method, color=colors[i], marker=markers[i], s=100)
    plt.scatter(steps[i], accuracy2[i], color=colors[i], marker=markers[i], s=100)
plt.plot(steps, accuracy1, linestyle="--", color="gray", label="LLaMA 8B", alpha=0.7)
plt.plot(steps, accuracy2, linestyle="-", color="black", label="LLaMA 70B", alpha=0.7)
# Set x-axis to log scale
plt.xscale("log", base=2)
# Set x-axis ticks
x_ticks = [256, 2048, 4096, 8192, 16384]
plt.xticks(x_ticks)
plt.xlim(256, 14000)
plt.ylim(60, 90)
# Labels and title
plt.xlabel("Number of Generated Tokens (log scale)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.legend(title="Methods", loc="upper left", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.7)
# Save plot
plt.savefig("computational_loglinear.png")
# Display the plot
# plt.show()