import matplotlib.pyplot as plt

# Categories and their counts
labels = [
    'Incorrect Plan',
    'Incorrect Search Queries',
    'Incorrect Retrieval Documents',
    'Incorrect Knowledge',
    'Incorrect Initial Hypothesis',
    'Incorrect Integrated Hypothesis',
    'Incorrect Conclusion'
]
sizes = [1, 1, 2, 1, 34, 5, 6]

# Colors (optional: you can customize them)
colors = plt.cm.Paired.colors[:len(labels)]

# Create pie chart
plt.figure(figsize=(12, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
# plt.title('Distribution of Incorrect Reasoning Types')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.tight_layout()
# plt.show()
plt.savefig("error.png")
