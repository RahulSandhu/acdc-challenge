from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Load dataset
df = pd.read_csv("../../data/datasets/raw_acdc_radiomics.csv")

# Custom style
plt.style.use("../../config/custom_style.mplstyle")

# Ensure output directory exists
img_dir = Path("../../images/eda")
img_dir.mkdir(parents=True, exist_ok=True)

# Count columns by data type
dtype_summary = df.dtypes.value_counts()

# Plot distribution of data types
plt.figure()
dtype_summary.plot(kind="bar")
plt.title("Distribution of Data Types")
plt.xlabel("")
plt.ylabel("Number of Columns")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Compute missing value counts
nan_counts = df.isna().sum()
num_nan_columns = (nan_counts > 0).sum()
null_counts = df.isnull().sum()
num_null_columns = (null_counts > 0).sum()

print(f"Number of NaN columns: {num_nan_columns}")
print(f"Number of null columns: {num_null_columns}")
print(f"Total number of NaN values: {df.isna().sum().sum()}")

# Plot class distribution of patients
plt.figure()
df["class"].value_counts().plot(kind="bar")
plt.xticks(rotation=0)
plt.title("Distribution of Classes")
plt.xlabel("")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# Set up demographics plots
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

# Height distribution by class
sns.boxplot(
    data=df,
    x="class",
    y="height",
    ax=axes[0],
    showfliers=False,
    color="lightgray",
)
sns.stripplot(
    data=df,
    x="class",
    y="height",
    hue="class",
    ax=axes[0],
    dodge=True,
    alpha=0.7,
)
axes[0].set_title("Height by Class")
axes[0].set_xlabel("")
axes[0].set_ylabel("Height (cm)")

# Weight distribution by class
sns.boxplot(
    data=df,
    x="class",
    y="weight",
    ax=axes[1],
    showfliers=False,
    color="lightgray",
)
sns.stripplot(
    data=df,
    x="class",
    y="weight",
    hue="class",
    ax=axes[1],
    dodge=True,
    alpha=0.7,
)
axes[1].set_title("Weight by Class")
axes[1].set_xlabel("")
axes[1].set_ylabel("Weight (kg)")
plt.tight_layout()
plt.show()

# Compute BMI = weight (kg) / (height (m))^2
df_bmi = df.copy()
df_bmi["bmi"] = df_bmi["weight"] / ((df_bmi["height"] / 100) ** 2)

# WHO BMI category bands
bmi_bands = [
    (18.5, 24.9, "Normal (18.5–24.9)", "green"),
    (25.0, 29.9, "Overweight (25–29.9)", "yellow"),
    (30.0, 34.9, "Obese I (30–34.9)", "orange"),
    (35.0, 39.9, "Obese II (35–39.9)", "red"),
    (40.0, df_bmi["bmi"].max(), "Obese III (≥40)", "darkred"),
]

# Plot BMI distribution by class
plt.figure()
ax = sns.boxplot(
    data=df_bmi,
    x="class",
    y="bmi",
    showfliers=False,
    color="lightgray",
)
sns.stripplot(
    data=df_bmi,
    x="class",
    y="bmi",
    hue="class",
    dodge=True,
    alpha=0.7,
)

# Plot BMI bands
patches = []

# Draw a horizontal shaded band for the current BMI category
for low, high, label, color in bmi_bands:
    ax.axhspan(low, high, color=color, alpha=0.1)
    patches.append(Patch(facecolor=color, alpha=0.2, label=label))

plt.title("BMI by Class with WHO Obesity Categories")
plt.xlabel("")
plt.ylabel("BMI (kg/m²)")
ax.legend(handles=patches, loc="upper right")
plt.tight_layout()
plt.savefig(img_dir / "bmi_by_class.png")
plt.show()
