from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.preprocessing import LabelBinarizer
from utils.generate_short_labels import generate_short_labels


def compute_top_correlations(
    df: pd.DataFrame, target_col: str = "class", top_n: int = 5
) -> pd.DataFrame:
    """
    Compute the top N features most correlated with each class label.

    Inputs:
        - df (pd.DataFrame): Input DataFrame containing feature columns and a
          target column.
        - target_col (str): Name of the target column to correlate against.
          Defaults to 'class'.
        - top_n (int): Number of top features to select for each class.
          Defaults to 5.

    Outputs:
        - pd.DataFrame: A DataFrame with the top_n correlated features.
    """
    # Drop target column from analysis
    features_df = df.drop(columns=[target_col])

    # Binarize class labels into indicator matrix
    lb = LabelBinarizer()
    class_binarized = pd.DataFrame(
        lb.fit_transform(df[target_col]), columns=lb.classes_
    )

    # Collect top correlations for each class
    rows = []

    # Iterate over each class label
    for cls in class_binarized.columns:
        # Correlate features with current class vector
        class_vector = class_binarized[cls]
        correlations = features_df.corrwith(class_vector).sort_values(
            key=abs, ascending=False
        )

        # Take top_n features
        for feature, value in correlations.head(top_n).items():
            # Save feature, correlation, and class
            rows.append(
                {"feature": feature, "correlation": value, "class": cls}
            )

    # Build DataFrame and add absolute correlation
    corr_df = pd.DataFrame(rows)
    corr_df["abs_corr"] = corr_df["correlation"].abs()

    # Sort by class then by strength
    sort_df = corr_df.sort_values(
        by=["class", "abs_corr"], ascending=[True, False]
    )

    return sort_df


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/raw/acdc_radiomics.csv")

    # Count columns by data type
    dtype_summary = df.dtypes.value_counts()

    # Custom style
    plt.style.use("../../misc/custom_style.mplstyle")

    # Ensure output directory exists
    img_dir = Path("../../images/eda")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Plot distribution of data types
    plt.figure(figsize=(6, 4))
    dtype_summary.plot(kind="bar")
    plt.title("Distribution of Data Types")
    plt.xlabel("")
    plt.ylabel("Number of Columns")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(img_dir / "data_types.png")
    plt.show()

    # Compute missing value counts
    nan_counts = df.isna().sum()
    num_nan_columns = (nan_counts > 0).sum()
    null_counts = df.isnull().sum()
    num_null_columns = (null_counts > 0).sum()

    # Print value counts
    print(f"Number of NaN columns: {num_nan_columns}")
    print(f"Number of null columns: {num_null_columns}")
    print(f"Total number of NaN values: {df.isna().sum().sum()}")

    # Plot class distribution of patients
    plt.figure(figsize=(6, 4))
    df["class"].value_counts().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.title("Distribution of Classes")
    plt.xlabel("")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(img_dir / "class_count.png")
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
    plt.savefig(img_dir / "demographics.png")
    plt.show()

    # Compute BMI = weight (kg) / (height (m))^2
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    # WHO BMI category bands
    bmi_bands = [
        (18.5, 24.9, "Normal (18.5–24.9)", "green"),
        (25.0, 29.9, "Overweight (25–29.9)", "yellow"),
        (30.0, 34.9, "Obese I (30–34.9)", "orange"),
        (35.0, 39.9, "Obese II (35–39.9)", "red"),
        (40.0, df["bmi"].max(), "Obese III (≥40)", "darkred"),
    ]

    # Plot BMI distribution by class
    plt.figure(figsize=(9, 5))
    ax = sns.boxplot(
        data=df,
        x="class",
        y="bmi",
        showfliers=False,
        color="lightgray",
    )
    sns.stripplot(
        data=df,
        x="class",
        y="bmi",
        hue="class",
        dodge=True,
        alpha=0.7,
    )

    # Plot BMI bands
    patches = []

    # Loop through BMI band ranges, labels, and colors
    for low, high, label, color in bmi_bands:
        # Draw a horizontal shaded band for the current BMI category
        ax.axhspan(low, high, color=color, alpha=0.1)
        patches.append(Patch(facecolor=color, alpha=0.2, label=label))

    # Finalize plot with title, labels, legend, and save
    plt.title("BMI by Class with WHO Obesity Categories")
    plt.xlabel("")
    plt.ylabel("BMI (kg/m²)")
    ax.legend(handles=patches, loc="upper right")
    plt.tight_layout()
    plt.savefig(img_dir / "bmi_by_class.png")
    plt.show()

    # Compute and retrieve top correlations
    corr_df = compute_top_correlations(df)

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
    plt.savefig(img_dir / "correlation.png")
    plt.show()
