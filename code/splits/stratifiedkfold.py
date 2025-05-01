from pathlib import Path
from typing import cast

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("../../data/datasets/norm_acdc_radiomics.csv")

# Custom style
plt.style.use("../../misc/custom_style.mplstyle")

# Ensure output directory exists
img_dir = Path("../../images/splits")
img_dir.mkdir(parents=True, exist_ok=True)

# Distinct colors for classes
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
class_labels = np.unique(df["class"])
label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
legend_patches = [
    mpatches.Patch(color=colors[label_to_idx[label]], label=f"{label}")
    for label in class_labels
]

# Encode labels
le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])

# Save the label encoder
Path("../../results/models/").mkdir(parents=True, exist_ok=True)
joblib.dump(le, "../../results/models/label_encoder.pkl")

# Separate features and classes
X = df.drop(columns=["class"])
y = df["class"]

# 80% Train+Val, 20% Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Fix for Pyright
X_temp = cast(pd.DataFrame, X_temp)
X_test = cast(pd.DataFrame, X_test)
y_temp = cast(pd.Series, y_temp)
y_test = cast(pd.Series, y_test)

# Save datasets
Path("../../data/kfold/").mkdir(parents=True, exist_ok=True)

pd.DataFrame(X_temp).to_csv("../../data/kfold/X_temp_norm.csv", index=False)
pd.DataFrame(X_test).to_csv("../../data/kfold/X_test_norm.csv", index=False)
pd.DataFrame(y_temp).to_csv("../../data/kfold/y_temp_norm.csv", index=False)
pd.DataFrame(y_test).to_csv("../../data/kfold/y_test_norm.csv", index=False)

# Visualize StratifiedKFold splits for training set
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
n_samples = len(X_temp)
n_classes = len(np.unique(y_temp))

plt.figure()
box_height = 0.8

# Iterate over each fold to visualize the train/val assignment
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_temp, y_temp)):
    block_size = len(test_idx)
    block_start = fold_idx * block_size
    block_end = block_start + block_size

    # Initialize an array to determine plotting positions for each sample
    plot_positions = np.empty(n_samples, dtype=int)
    remaining_positions = [
        i for i in range(n_samples) if i < block_start or i >= block_end
    ]
    for i, idx in enumerate(test_idx):
        plot_positions[idx] = block_start + i
    for i, idx in enumerate(train_idx):
        plot_positions[idx] = remaining_positions[i]

    # Sort samples according to their plotting positions
    sorted_idx = np.argsort(plot_positions)

    # Plot rectangles representing each sample, marking validation samples with hatching
    for i, sample_idx in enumerate(sorted_idx):
        c = colors[y_temp.iloc[sample_idx]]
        is_test = sample_idx in test_idx
        rect = mpatches.Rectangle(
            (i, fold_idx - box_height / 2),
            1,
            box_height,
            facecolor=c,
            edgecolor="k",
            linewidth=0.5,
            hatch="///" if is_test else None,
        )
        plt.gca().add_patch(rect)

    # Compute class distributions separately for training and validation sets
    y_train_fold = y_temp.iloc[train_idx]
    y_val_fold = y_temp.iloc[test_idx]
    train_counts = y_train_fold.value_counts().sort_index()
    val_counts = y_val_fold.value_counts().sort_index()

    print(f"Fold {fold_idx + 1} class distribution:")

    # Print the number of samples per class for training and validation sets
    for cls in sorted(y_temp.unique()):
        train_n = train_counts.get(cls, 0)
        val_n = val_counts.get(cls, 0)
        print(f"  Class {cls}: train={train_n}, val={val_n}")

plt.title("Stratified K-Fold Splits")
plt.xlabel("Sample Index")
plt.ylabel("Fold Index")
plt.yticks(
    range(cv.get_n_splits()),
    [f"Fold {i+1}" for i in range(cv.get_n_splits())],
)
plt.legend(
    handles=legend_patches,
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
)
plt.xlim(0, n_samples)
plt.ylim(-0.5, cv.get_n_splits() - 0.5)
plt.tight_layout()
plt.savefig(img_dir / "kfold.png")
plt.show()
