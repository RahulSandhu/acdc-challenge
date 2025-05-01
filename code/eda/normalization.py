from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("../../data/datasets/raw_acdc_radiomics.csv")

# Custom style
plt.style.use("../../misc/custom_style.mplstyle")

# Ensure output directory exists
img_dir = Path("../../images/eda")
img_dir.mkdir(parents=True, exist_ok=True)

# Set seed
np.random.seed(42)

# Randomly select a class and features
selected_class = np.random.choice(np.array(df["class"].unique()))
class_df = df[df["class"] == selected_class].drop(columns=["class"])
selected_features = np.random.choice(class_df.columns, size=9, replace=False)

# Plot histograms with KDE for selected features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

# Plot histogram with KDE for each feature
for ax, feature in zip(axes, selected_features):
    sns.histplot(x=class_df[feature], kde=True, bins=20, color="steelblue", ax=ax)
    label = feature.split("_", 2)[-1]
    ax.set_title(label, fontsize=10, pad=1)
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.suptitle("KDE Plots for Selected Features", fontsize=16)
fig.supxlabel("Feature Value", fontsize=12)
fig.supylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

# Plot Q-Q plots with Shapiro p-values
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

# Generate Q-Q plot and annotate with p-value
for ax, feature in zip(axes, selected_features):
    stats.probplot(class_df[feature], dist="norm", plot=ax)
    _, pval = stats.shapiro(class_df[feature])
    label = feature.split("_", 2)[-1]
    ax.set_title(f"{label}\np = {pval:.3f}", fontsize=9, pad=2)
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.suptitle("Q-Q Plots with Shapiro-Wilk Normality Test", fontsize=16)
fig.supxlabel("Theoretical Quantiles", fontsize=12)
fig.supylabel("Sample Quantiles", fontsize=12)
plt.tight_layout()
plt.savefig(img_dir / "qq.png")
plt.show()

# Separate features/class
features = df.drop(columns=["class"])
classes = df["class"]

# Apply StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
norm_df = pd.concat([scaled_df, classes.reset_index(drop=True)], axis=1)

# Save normalized DataFrame
Path("../../data/datasets/").mkdir(parents=True, exist_ok=True)
norm_df.to_csv("../../data/datasets/norm_acdc_radiomics.csv", index=False)
