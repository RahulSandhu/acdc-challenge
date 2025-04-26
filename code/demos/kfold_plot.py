from pathlib import Path
from typing import cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("../../data/raw/acdc_radiomics.csv")

# Distinct colors for classes
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
class_labels = np.unique(df["class"])
label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
legend_patches = [
    mpatches.Patch(color=colors[label_to_idx[label]], label=f'{label}')
    for label in class_labels
]

# Encode labels
le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])

# Separate features and classes
X = df.drop(columns=["class"])
y = df["class"]

# Apply StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_features, columns=X.columns)

# Normalized dataframe
norm_df = pd.concat([scaled_df, y.reset_index(drop=True)], axis=1)

# Separate features and classes
X = norm_df.drop(columns=["class"])
y = norm_df["class"]

# 80% Train+Val, 20% Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Fix for Pyright
X_test = cast(pd.DataFrame, X_test)
X_temp = cast(pd.DataFrame, X_temp)
y_test = cast(pd.Series, y_test)
y_temp = cast(pd.Series, y_temp)

# Visualize StratifiedKFold splits for training set
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
n_samples = len(X_temp)
n_classes = len(np.unique(y_temp))

# Ensure output directory exists
img_dir = Path("../../images/demos")
img_dir.mkdir(parents=True, exist_ok=True)

# Custom style
plt.style.use("../../misc/custom_style.mplstyle")

# Initialize the figure and box height for the fold blocks
plt.figure(figsize=(14, 5))
box_height = 0.8

# Iterate over each fold to visualize the train/val assignment
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_temp, y_temp)):
    # Determine size and location of the current validation block
    block_size = len(test_idx)
    block_start = fold_idx * block_size
    block_end = block_start + block_size

    plot_positions = np.empty(n_samples, dtype=int)
    remaining_positions = [
        i for i in range(n_samples) if i < block_start or i >= block_end
    ]

    # Assign test samples into their block positions
    for i, idx in enumerate(test_idx):
        plot_positions[idx] = block_start + i

    # Assign train samples into the remaining space
    for i, idx in enumerate(train_idx):
        plot_positions[idx] = remaining_positions[i]

    # Get sorted sample indices
    sorted_idx = np.argsort(plot_positions)

    # Draw rectangles per sample with color per class
    for i, sample_idx in enumerate(sorted_idx):
        # Choose color based on class label
        c = colors[y_temp.iloc[sample_idx]]

        # Use pattern to indicate validation samples
        is_test = sample_idx in test_idx
        rect = mpatches.Rectangle(
            (i, fold_idx - box_height / 2),
            1,
            box_height,
            facecolor=c,
            edgecolor='k',
            linewidth=0.5,
            hatch='///' if is_test else None,
        )
        plt.gca().add_patch(rect)

    # Get class label distributions in training and validation sets
    y_train_fold = y_temp.iloc[train_idx]
    y_val_fold = y_temp.iloc[test_idx]

    # Count samples per class in training and validation
    train_counts = y_train_fold.value_counts().sort_index()
    val_counts = y_val_fold.value_counts().sort_index()

    # Print class distribution for this fold
    print(f"Fold {fold_idx + 1} class distribution:")

    # Print number of samples per class in train and val
    for cls in sorted(y_temp.unique()):
        # Get sample counts
        train_n = train_counts.get(cls, 0)
        val_n = val_counts.get(cls, 0)
        print(f"  Class {cls}: train={train_n}, val={val_n}")

# Save figure
plt.title("StratifiedKFold Splits Visualization")
plt.xlabel("Sample Index")
plt.ylabel("Fold Index")
plt.yticks(
    range(cv.get_n_splits()), [f"Fold {i+1}" for i in range(cv.get_n_splits())]
)
plt.legend(
    handles=legend_patches,
    bbox_to_anchor=(1.01, 1),
    loc='upper left',
)
plt.xlim(0, n_samples)
plt.ylim(-0.5, cv.get_n_splits() - 0.5)
plt.tight_layout()
plt.savefig(img_dir / "kfold_plot.png")
plt.show()
