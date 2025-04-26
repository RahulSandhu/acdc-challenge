from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from utils.generate_short_labels import generate_short_labels


def compute_outlier_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IQR-based outlier counts for each feature within each class.

    Inputs:
        - df (pd.DataFrame): Input DataFrame containing feature columns and a
          'class' column indicating class membership.

    Outputs:
        - pd.DataFrame: A DataFrame with computed outliers.
    """
    # Initialize outlier_data container
    outlier_data = []

    # Separate features and target
    features_df = df.drop(columns=['class'])
    classes = df["class"].unique()

    # Loop through each class
    for cls in classes:
        # Select data for current class
        class_data = features_df[df["class"] == cls]

        # Loop through each feature
        for column in class_data.columns:
            # Calculate IQR bounds
            Q1 = pd.Series(class_data[column]).quantile(0.25)
            Q3 = pd.Series(class_data[column]).quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            # Count outliers in feature
            outliers = (
                (class_data[column] < lower) | (class_data[column] > upper)
            ).sum()
            outlier_data.append(
                {"feature": column, "class": cls, "outliers": outliers}
            )

    # Create DataFrame of outlier counts
    outlier_df = pd.DataFrame(outlier_data)

    return outlier_df


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/raw/acdc_radiomics.csv")

    # Compute outliers
    outlier_df = compute_outlier_df(df)

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
    class_stats = (
        outlier_df.groupby("class")["outliers"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Custom style
    plt.style.use("../../misc/custom_style.mplstyle")

    # Ensure output directory exists
    img_dir = Path("../../images/processing/")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Plot bar chart of outliers
    plt.figure(figsize=(12, 6))
    g = sns.barplot(
        data=top_df, x="feature", y="outliers", hue="class", legend=False
    )
    g.set_xticks(range(len(short_labels)))
    g.set_xticklabels(short_labels, rotation=45, ha="right")

    # Add horizontal lines for each class's mean ± std
    class_colors = sns.color_palette(n_colors=len(class_stats))
    line_handles = []
    line_labels = []

    # Loop through each class's mean and std to draw lines
    for (cls, mean, std), color in zip(class_stats.values, class_colors):
        # Draw a dashed horizontal line at the mean value for the class
        line = plt.axhline(
            mean,
            color=color,
            linestyle="--",
            label=f"{cls} Mean = {mean:.2f} ± {std:.2f}",
            linewidth=2,
        )
        line_handles.append(line)
        line_labels.append(f"{cls} Mean = {mean:.2f} ± {std:.2f}")

    # Add the legend with the collected lines and labels
    plt.legend(handles=line_handles, labels=line_labels)
    plt.title("Top 10 Features with Most Outliers by Class")
    plt.xlabel("Feature")
    plt.ylabel("Number of Outliers")
    plt.tight_layout()
    plt.savefig(img_dir / "outliers.png")
    plt.show()

    # Randomly select a class and features
    np.random.seed(42)
    selected_class = np.random.choice(np.array(df["class"].unique()))
    class_df = df[df["class"] == selected_class].drop(columns=['class'])
    selected_features = np.random.choice(
        class_df.columns, size=9, replace=False
    )

    # Plot histograms with KDE for selected features
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    # Loop through axes and features
    for ax, feature in zip(axes, selected_features):
        # Plot histogram with KDE for each feature
        sns.histplot(
            x=class_df[feature], kde=True, bins=20, color="steelblue", ax=ax
        )
        label = feature.split("_", 2)[-1]
        ax.set_title(label, fontsize=10, pad=1)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Save figure
    fig.supxlabel("Feature Value", fontsize=12)
    fig.supylabel("Frequency", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(img_dir / "kde.png")
    plt.show()

    # Plot Q-Q plots with Shapiro p-values
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    # Loop through axes and features
    for ax, feature in zip(axes, selected_features):
        # Generate Q-Q plot and annotate with p-value
        stats.probplot(class_df[feature], dist="norm", plot=ax)
        _, pval = stats.shapiro(class_df[feature])
        label = feature.split("_", 2)[-1]
        ax.set_title(f"{label}\np = {pval:.3f}", fontsize=9, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Save figure
    fig.supxlabel("Theoretical Quantiles", fontsize=12)
    fig.supylabel("Sample Quantiles", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(img_dir / "qq.png")
    plt.show()

    # Separate features/class
    features = df.drop(columns=['class'])
    classes = df['class']

    # Apply StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Normalized dataframe
    norm_df = pd.concat([scaled_df, classes.reset_index(drop=True)], axis=1)

    # Save normalized dataframe
    Path("../../data/processed/").mkdir(parents=True, exist_ok=True)
    norm_df.to_csv("../../data/processed/norm_acdc_radiomics.csv", index=False)
