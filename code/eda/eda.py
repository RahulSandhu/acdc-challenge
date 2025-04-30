from pathlib import Path
from typing import List, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from utils.generate_short_labels import generate_short_labels


def compute_outlier_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IQR-based outlier counts for each feature within each class.

    Inputs:
        - df (pd.DataFrame): Input DataFrame containing feature columns and a
          class column.

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
    # Custom style
    plt.style.use("../../misc/custom_style.mplstyle")

    # Load dataset
    df = pd.read_csv("../../data/datasets/raw_acdc_radiomics.csv")

    # Count columns by data type
    dtype_summary = df.dtypes.value_counts()

    # Ensure output directory exists
    img_dir = Path("../../images/eda")
    img_dir.mkdir(parents=True, exist_ok=True)

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

    # Print value counts
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

    # Loop through BMI band ranges, labels, and colors
    for low, high, label, color in bmi_bands:
        # Draw a horizontal shaded band for the current BMI category
        ax.axhspan(low, high, color=color, alpha=0.1)
        patches.append(Patch(facecolor=color, alpha=0.2, label=label))

    # Save figure
    plt.title("BMI by Class with WHO Obesity Categories")
    plt.xlabel("")
    plt.ylabel("BMI (kg/m²)")
    ax.legend(handles=patches, loc="upper right")
    plt.tight_layout()
    plt.savefig(img_dir / "bmi_by_class.png")
    plt.show()

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

    # Plot bar chart of outliers
    plt.figure()
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

    # Save figure
    plt.legend(handles=line_handles, labels=line_labels)
    plt.title("Top 10 Features with Most Outliers by Class")
    plt.xlabel("")
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
    fig.suptitle("Q-Q Plots with Shapiro-Wilk Normality Test", fontsize=16)
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
    Path("../../data/datasets/").mkdir(parents=True, exist_ok=True)
    norm_df.to_csv("../../data/datasets/norm_acdc_radiomics.csv", index=False)

    # Compute and retrieve top correlations
    corr_df = compute_top_correlations(norm_df)

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
    class_array = norm_df["class"].to_numpy()
    selected_class = np.random.choice(class_array)

    # Filter features from the selected class
    class_df = norm_df[norm_df["class"] == selected_class]
    feature_names = class_df.drop(
        columns=['height', 'weight', 'class']
    ).columns.to_numpy()

    # Group features by their PyRadiomics prefix
    feature_groups = {}
    for feature in feature_names:
        group_name = feature.split("_")[1]
        if group_name not in feature_groups:
            feature_groups[group_name] = []
        feature_groups[group_name].append(feature)

    # Randomly pick one group and features
    np.random.seed(42)
    selected_group = np.random.choice(list(feature_groups.keys()))
    selected_features_np = np.random.choice(
        np.array(feature_groups[selected_group]), size=5, replace=False
    )

    # Fix for Pyright
    selected_features = cast(List[str], selected_features_np.tolist())

    # Build a DataFrame only with selected features
    selected_feature_data = norm_df.loc[
        norm_df["class"] == selected_class, selected_features
    ]

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
    norm_df["class"] = le.fit_transform(df["class"])

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

    # Initialize the figure and box height for the fold blocks
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
                edgecolor='k',
                linewidth=0.5,
                hatch='///' if is_test else None,
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

    # Save figure
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
        loc='upper left',
    )
    plt.xlim(0, n_samples)
    plt.ylim(-0.5, cv.get_n_splits() - 0.5)
    plt.tight_layout()
    plt.savefig(img_dir / "kfold_plot.png")
    plt.show()
