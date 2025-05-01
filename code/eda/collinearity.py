from pathlib import Path
from typing import List, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from utils.generate_short_labels import generate_short_labels

# Load dataset
df = pd.read_csv("../../data/datasets/norm_acdc_radiomics.csv")

# Custom style
plt.style.use("../../misc/custom_style.mplstyle")

# Ensure output directory exists
img_dir = Path("../../images/eda")
img_dir.mkdir(parents=True, exist_ok=True)

# Drop target column from features
features_df = df.drop(columns=["class"])

# Binarize class labels into indicator matrix
lb = LabelBinarizer()
class_binarized = pd.DataFrame(lb.fit_transform(df["class"]), columns=lb.classes_)

# Collect top correlations for each class
rows = []
top_n = 5

for cls in class_binarized.columns:
    class_vector = class_binarized[cls]
    correlations = features_df.corrwith(class_vector).sort_values(
        key=abs, ascending=False
    )
    for feature, value in correlations.head(top_n).items():
        rows.append({"feature": feature, "correlation": value, "class": cls})

# Build DataFrame and add absolute correlation
corr_df = pd.DataFrame(rows)
corr_df["abs_corr"] = corr_df["correlation"].abs()

# Sort by class then strength
corr_df = corr_df.sort_values(by=["class", "abs_corr"], ascending=[True, False])

# Shorten feature names for plotting
short_labels = generate_short_labels(corr_df["feature"].tolist())

# Visualize top correlated features
plt.figure(figsize=(12, 10))
g = sns.barplot(
    data=corr_df,
    x="correlation",
    y="feature",
    hue="class",
    dodge=True,
)
g.set_yticks(g.get_yticks())
g.set_yticklabels(short_labels)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("Top 5 Correlated Features per Class")
plt.xlabel("Pearson Correlation")
plt.ylabel("Feature")
plt.legend()
plt.tight_layout()
plt.savefig(img_dir / "ovr_correlation.png")
plt.show()

# Convert class column to a NumPy array
class_array = df["class"].to_numpy()
selected_class = np.random.choice(class_array)

# Filter features from the selected class
class_df = df[df["class"] == selected_class]
feature_names = class_df.drop(columns=["height", "weight", "class"]).columns.to_numpy()

# Group features by their PyRadiomics prefix
feature_groups = {}
for feature in feature_names:
    group_name = feature.split("_")[1]
    if group_name not in feature_groups:
        feature_groups[group_name] = []
    feature_groups[group_name].append(feature)

# Set seed
np.random.seed(42)

# Randomly pick one group and features
selected_group = np.random.choice(list(feature_groups.keys()))
selected_features_np = np.random.choice(
    np.array(feature_groups[selected_group]), size=5, replace=False
)

# Fix for Pyright
selected_features = cast(List[str], selected_features_np.tolist())

# Build a DataFrame only with selected features
selected_feature_data = df.loc[df["class"] == selected_class, selected_features]

# Compute correlation matrix
corr_matrix = selected_feature_data.corr()

# Generate short labels for the selected features
short_labels = generate_short_labels(selected_features)

# Plot heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    cbar_kws={"shrink": 0.8},
)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels(short_labels)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(short_labels, rotation=45, ha="right")
plt.title(
    f"Multicollinearity Matrix for {selected_group.capitalize()} Features - Class {selected_class}"
)
plt.tight_layout()
plt.savefig(img_dir / "multicollinearity.png")
plt.show()
