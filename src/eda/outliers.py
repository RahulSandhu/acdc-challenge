from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils.generate_short_labels import generate_short_labels

# Load dataset
df = pd.read_csv("../../data/datasets/raw_acdc_radiomics.csv")

# Custom style
plt.style.use("../../config/custom_style.mplstyle")

# Ensure output directory exists
img_dir = Path("../../images/eda")
img_dir.mkdir(parents=True, exist_ok=True)

# Initialize outlier_data container
outlier_data = []

# Separate features and target
features_df = df.drop(columns=["class"])
classes = df["class"].unique()

# Loop through each class and compute outliers
for cls in classes:
    class_data = features_df[df["class"] == cls]
    for column in class_data.columns:
        Q1 = pd.Series(class_data[column]).quantile(0.25)
        Q3 = pd.Series(class_data[column]).quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((class_data[column] < lower) | (class_data[column] > upper)).sum()
        outlier_data.append({"feature": column, "class": cls, "outliers": outliers})

# Create DataFrame of outlier counts
outlier_df = pd.DataFrame(outlier_data)

# Identify top 10 features by outlier count
top_features = (
    outlier_df.groupby("feature")["outliers"]
    .sum()
    .reset_index()
    .sort_values("outliers", ascending=False)
    .head(10)["feature"]
    .tolist()
)

# Filter DataFrame to top features and set order
top_df = outlier_df[outlier_df["feature"].isin(top_features)].copy()
top_df["feature"] = pd.Categorical(
    top_df["feature"], categories=top_features, ordered=True
)
top_df = cast(pd.DataFrame, top_df)

# Generate short labels for plotting
short_labels = generate_short_labels(top_features)

# Compute per-class mean and std of outlier counts
class_stats = outlier_df.groupby("class")["outliers"].agg(["mean", "std"]).reset_index()

# Plot bar chart of outliers
plt.figure()
g = sns.barplot(data=top_df, x="feature", y="outliers", hue="class", legend=False)
g.set_xticks(range(len(short_labels)))
g.set_xticklabels(short_labels, rotation=45, ha="right")

# Add horizontal lines for each class's mean ± std
class_colors = sns.color_palette(n_colors=len(class_stats))
line_handles = []
line_labels = []

for (cls, mean, std), color in zip(class_stats.values, class_colors):
    line = plt.axhline(
        mean,
        color=color,
        linestyle="--",
        label=f"{cls} Mean = {mean:.2f} ± {std:.2f}",
        linewidth=2,
    )
    line_handles.append(line)
    line_labels.append(f"{cls} Mean = {mean:.2f} ± {std:.2f}")

# Save figure
plt.legend(handles=line_handles, labels=line_labels)
plt.title("Top 10 Features with Most Outliers by Class")
plt.xlabel("")
plt.ylabel("Number of Outliers")
plt.tight_layout()
plt.savefig(img_dir / "outliers.png")
plt.show()
