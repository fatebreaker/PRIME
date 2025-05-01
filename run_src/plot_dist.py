import matplotlib.pyplot as plt

# Data: (System 1, System 2)
data = [
    ("MedQA", [1074, 199]),
    ("MedMCQA", [3640, 543]),
    ("MMLU", [920, 169]),
    ("Musique", [230, 820]),
    ("HotpotQA", [368, 682]),
    ("2Wiki", [390, 660]),
]

# Color mapping: blue for System 1, red for System 2
colors = ['blue', 'red']

# Create figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# Plot each pie chart
for ax, (title, values) in zip(axes, data):
    ax.pie(values, labels=["System 1", "System 2"], colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)

plt.tight_layout()
# plt.show()
plt.savefig("distribution.png")