import matplotlib.pyplot as plt

# X-axis steps
x = [1, 2, 3, 4, 5]
# Y-axis scores (peak at 2 = 87.2)
y = [85.3, 87.2, 86.0, 85.1, 84.7]

# Plotting
plt.plot(x, y, marker='o', linestyle='-', linewidth=2)
plt.xticks(x)
plt.xlabel("Number of debate iteration")
plt.ylabel("Accuracy")
# plt.title("Score Distribution (Peak at Step 2)")
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("debate_iterations.png")
