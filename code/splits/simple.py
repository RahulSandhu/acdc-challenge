from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
raw_df = pd.read_csv("../../data/datasets/raw_acdc_radiomics.csv")
norm_df = pd.read_csv("../../data/datasets/norm_acdc_radiomics.csv")

# Custom style
plt.style.use("../../config/custom_style.mplstyle")

# Ensure output directory exists
img_dir = Path("../../images/splits")
img_dir.mkdir(parents=True, exist_ok=True)

# Distinct colors for classes
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
class_labels = np.unique(norm_df["class"])
label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
legend_patches = [
    mpatches.Patch(color=colors[label_to_idx[label]], label=f"{label}")
    for label in class_labels
]

# Encode labels
le = LabelEncoder()
raw_df["class"] = le.fit_transform(raw_df["class"])
norm_df["class"] = le.fit_transform(norm_df["class"])

# Save the label encoder
Path("../../results/models/").mkdir(parents=True, exist_ok=True)
joblib.dump(le, "../../results/models/label_encoder.pkl")

# Separate features and classes
X_raw = raw_df.drop(columns=["class"])
y_raw = raw_df["class"]

X_norm = norm_df.drop(columns=["class"])
y_norm = norm_df["class"]

# 80% Train+Val, 20% Test
X_temp_raw, X_test_raw, y_temp_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

# 75% Train, 25% Val from the 80% (→ 60% train, 20% val)
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_temp_raw, y_temp_raw, test_size=0.25, stratify=y_temp_raw, random_state=42
)

# 80% Train+Val, 20% Test
X_temp_norm, X_test_norm, y_temp_norm, y_test_norm = train_test_split(
    X_norm, y_norm, test_size=0.2, stratify=y_norm, random_state=42
)

# 75% Train, 25% Val from the 80% (→ 60% train, 20% val)
X_train_norm, X_val_norm, y_train_norm, y_val_norm = train_test_split(
    X_temp_norm, y_temp_norm, test_size=0.25, stratify=y_temp_norm, random_state=42
)

# Save datasets
Path("../../data/simple/").mkdir(parents=True, exist_ok=True)

pd.DataFrame(X_temp_raw).to_csv("../../data/simple/X_temp_raw.csv", index=False)
pd.DataFrame(X_test_raw).to_csv("../../data/simple/X_test_raw.csv", index=False)
pd.DataFrame(y_temp_raw).to_csv("../../data/simple/y_temp_raw.csv", index=False)
pd.DataFrame(y_test_raw).to_csv("../../data/simple/y_test_raw.csv", index=False)
pd.DataFrame(X_train_raw).to_csv("../../data/simple/X_train_raw.csv", index=False)
pd.DataFrame(X_val_raw).to_csv("../../data/simple/X_val_raw.csv", index=False)
pd.DataFrame(y_train_raw).to_csv("../../data/simple/y_train_raw.csv", index=False)
pd.DataFrame(y_val_raw).to_csv("../../data/simple/y_val_raw.csv", index=False)

pd.DataFrame(X_temp_norm).to_csv("../../data/simple/X_temp_norm.csv", index=False)
pd.DataFrame(X_test_norm).to_csv("../../data/simple/X_test_norm.csv", index=False)
pd.DataFrame(y_temp_norm).to_csv("../../data/simple/y_temp_norm.csv", index=False)
pd.DataFrame(y_test_norm).to_csv("../../data/simple/y_test_norm.csv", index=False)
pd.DataFrame(X_train_norm).to_csv("../../data/simple/X_train_norm.csv", index=False)
pd.DataFrame(X_val_norm).to_csv("../../data/simple/X_val_norm.csv", index=False)
pd.DataFrame(y_train_norm).to_csv("../../data/simple/y_train_norm.csv", index=False)
pd.DataFrame(y_val_norm).to_csv("../../data/simple/y_val_norm.csv", index=False)

# Visualize simple train-test split
n_samples = len(X_norm)

plt.figure()
bar_height = 0.8

# Sort indices to group samples by class
sorted_idx = np.argsort(np.array(y_norm.values))

# Iterate over sorted samples to draw colored rectangles
for i, sample_idx in enumerate(sorted_idx):
    c = colors[y_norm.iloc[sample_idx]]
    is_test = sample_idx in X_test_norm.index
    rect = mpatches.Rectangle(
        (i, 0),
        1,
        bar_height,
        facecolor=c,
        edgecolor="k",
        linewidth=0.5,
        hatch="///" if is_test else None,
    )
    plt.gca().add_patch(rect)

plt.title("Simple Train-Test Split")
plt.xlabel("Sample Index")
plt.yticks([])
plt.legend(
    handles=legend_patches,
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
)
plt.xlim(0, n_samples)
plt.tight_layout()
plt.savefig(img_dir / "simple.png")
plt.show()
