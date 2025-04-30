from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure output directory exists
img_dir = Path("../../images/demos")
img_dir.mkdir(parents=True, exist_ok=True)

# Custom style
plt.style.use("../../misc/custom_style.mplstyle")

# Classification report data
data = {
    "Class": ["DCM", "HCM", "MINF", "NOR", "RV"],
    "Precision": [1.0, 1.0, 0.6667, 1.0, 0.8],
    "Recall": [0.75, 0.75, 1.0, 0.75, 1.0],
    "F1-Score": [0.8571, 0.8571, 0.8, 0.8571, 0.8889],
}
df = pd.DataFrame(data)

# Plot settings
x = np.arange(len(df["Class"]))
width = 0.25
fig, ax = plt.subplots(figsize=(14, 10))

# Define color palette
colors = {
    "Precision": "#7FB3D5",
    "Recall": "#F7DC6F",
    "F1-Score": "#82E0AA",
}

# Plot bars
bars1 = ax.bar(
    x - width,
    df["Precision"],
    width,
    label='Precision',
    color=colors["Precision"],
    edgecolor='black',
    linewidth=0.8,
)
bars2 = ax.bar(
    x,
    df["Recall"],
    width,
    label='Recall',
    color=colors["Recall"],
    edgecolor='black',
    linewidth=0.8,
)
bars3 = ax.bar(
    x + width,
    df["F1-Score"],
    width,
    label='F1-Score',
    color=colors["F1-Score"],
    edgecolor='black',
    linewidth=0.8,
)

# Axes settings
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('')
ax.set_title(
    'Class-wise Precision, Recall and F1-Score (SVM K-Fold)',
    fontsize=14,
    weight='bold',
)
ax.set_xticks(x)
ax.set_xticklabels(df["Class"], fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10)

# Annotate bars with values
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10,
        )

# Save figure
fig.tight_layout()
plt.savefig(img_dir / "best_model.png", dpi=300)
plt.show()
