from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from utils.CoefficientThresholdLasso import CoefficientThresholdLasso


def rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_temp: pd.DataFrame,
    y_temp: pd.Series,
    n_estimators: Sequence[int],
    max_depth: Sequence[Optional[int]],
    min_samples_leaf: Sequence[int],
    SEED: int = 42,
    pen: float = 0.5,
) -> Tuple[Dict[str, Optional[Pipeline]], pd.DataFrame, pd.DataFrame, str]:
    """
    Train and evaluate Random Forest (RF) models using two selection
    strategies: simple train/validation split and K-fold cross-validation, both
    incorporating CTL (Coefficient Threshold Lasso) and LDA (Linear
    Discriminant Analysis) as preprocessing steps.

    The function performs:
    1. A manual grid search on a train/validation split using CTL → LDA → RF,
    selecting the model based on a penalized validation accuracy that accounts
    for overfitting (train-validation gap).
    2. A GridSearchCV over a combined training+validation set using the same
    CTL → LDA → RF pipeline and penalized scoring.

    Inputs:
        - X_train (pd.DataFrame): Training features (already scaled).
        - y_train (pd.Series): Labels for the training set.
        - X_val (pd.DataFrame): Validation features (already scaled).
        - y_val (pd.Series): Labels for the validation set.
        - X_temp (pd.DataFrame): Combined training+validation features.
        - y_temp (pd.Series): Labels corresponding to X_temp.
        - n_estimators (Sequence[int]): Number of trees to try in the forest.
        - max_depth (Sequence[Optional[int]]): Maximum depth values to try (use
          None for unlimited depth).
        - min_samples_leaf (Sequence[int]): Minimum number of samples required
          to be at a leaf node.
        - SEED (int, optional): Random seed for reproducibility. Default is 42.
        - pen (float, optional): Penalty weight applied to the train-validation
          gap when scoring. Default is 0.5.

    Outputs:
        - Tuple containing:
            - Dict[str, Optional[Pipeline]]: Trained models under keys 'simple'
              and 'kfold'.
            - pd.DataFrame: DataFrame summarizing the manual grid search
              results.
            - pd.DataFrame: DataFrame summarizing the cross-validation search
              results.
            - str: Summary string reporting the selected hyperparameters and
              model performance.
    """
    # Initialize result container
    models: Dict[str, Optional[Pipeline]] = {}

    # Grid search pipeline: CTL → LDA → RF
    grid_pipeline_simple = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )
    grid_parameters_simple = [
        {
            "n_estimators": n,
            "max_depth": d,
            "min_samples_leaf": s,
        }
        for n in n_estimators
        for d in max_depth
        for s in min_samples_leaf
    ]

    # Fit on training data and transform both train and val
    X_train_lda = grid_pipeline_simple.fit_transform(X_train.values, y_train)
    X_val_lda = grid_pipeline_simple.transform(X_val.values)

    # Manual grid search with scoring collected in a dataframe
    df_simple = []

    # Manual grid search (simple train-val split)
    for params in grid_parameters_simple:
        # Fit RF with current parameters
        rf_mdl = RandomForestClassifier(**params, random_state=SEED)
        rf_mdl.fit(X_train_lda, y_train)

        # Evaluate and penalize overfitting
        train_score = accuracy_score(y_train, rf_mdl.predict(X_train_lda))
        val_score = accuracy_score(y_val, rf_mdl.predict(X_val_lda))
        gap = abs(train_score - val_score)
        penalized = val_score - pen * gap

        # Append metrics
        df_simple.append(
            {
                "params": params,
                "train_score": train_score,
                "val_score": val_score,
                "gap": gap,
                "score": penalized,
            }
        )
    df_simple = pd.DataFrame(df_simple)

    # Pick best model from penalized score
    best_idx_simple = df_simple["score"].idxmax()
    best_parameters_simple = df_simple.loc[best_idx_simple, "params"]
    performance_simple = {
        "train_acc": df_simple.loc[best_idx_simple, "train_score"],
        "val_acc": df_simple.loc[best_idx_simple, "val_score"],
        "gap": df_simple.loc[best_idx_simple, "gap"],
        "score": df_simple.loc[best_idx_simple, "score"],
    }

    # Refit best model on the full training set
    model_simple = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
            (
                "rf",
                RandomForestClassifier(**best_parameters_simple, random_state=SEED),
            ),
        ]
    )
    model_simple.fit(X_train, y_train)

    # Update models
    models["simple"] = model_simple

    # Report results
    summary = "=" * 50 + "\n"
    summary += "  Best Parameters (SIMPLE + CTL/LDA + RF)\n"
    summary += "=" * 50 + "\n"
    summary += f"  n_estimators     : {best_parameters_simple['n_estimators']}\n"
    summary += f"  max_depth        : {best_parameters_simple['max_depth']}\n"
    summary += f"  min_samples_leaf : {best_parameters_simple['min_samples_leaf']}\n"
    summary += "-" * 50 + "\n"
    summary += (
        f"Train Acc: {performance_simple['train_acc']:.4f}, "
        f"Val Acc: {performance_simple['val_acc']:.4f}, "
        f"Score: {performance_simple['score']:.4f}"
    )

    # Grid search pipeline: CTL → LDA → RF
    grid_pipeline_kfold = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
            ("rf", RandomForestClassifier(random_state=SEED)),
        ]
    )
    grid_parameters_kfold = {
        "rf__n_estimators": n_estimators,
        "rf__max_depth": max_depth,
        "rf__min_samples_leaf": min_samples_leaf,
    }

    # Define stratified K-Fold cross-validator
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Run stratified K-fold grid search
    grid_search_kfold = GridSearchCV(
        estimator=grid_pipeline_kfold,
        param_grid=grid_parameters_kfold,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
        refit=False,
    )
    grid_search_kfold.fit(X_temp, y_temp)

    # Penalize overfit models
    df_kfold = pd.DataFrame(grid_search_kfold.cv_results_)
    df_kfold["gap"] = abs(df_kfold["mean_train_score"] - df_kfold["mean_test_score"])
    df_kfold["score"] = df_kfold["mean_test_score"] - pen * df_kfold["gap"]

    # Pick best model from penalized score
    best_idx_kfold = df_kfold["score"].idxmax()
    best_parameters_kfold = df_kfold.loc[best_idx_kfold, "params"]
    performance_kfold = {
        "train_acc": df_kfold.loc[best_idx_kfold, "mean_train_score"],
        "val_acc": df_kfold.loc[best_idx_kfold, "mean_test_score"],
        "gap": df_kfold.loc[best_idx_kfold, "gap"],
        "score": df_kfold.loc[best_idx_kfold, "score"],
    }

    # Refit best model on the full training+val set
    best_parameters_kfold = {
        k.replace("rf__", ""): v for k, v in best_parameters_kfold.items()
    }
    model_kfold = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
            (
                "rf",
                RandomForestClassifier(random_state=SEED),
            ),
        ]
    )
    model_kfold.fit(X_temp, y_temp)

    # Update models
    models["kfold"] = model_kfold

    # Report results
    summary += "\n\n" + "=" * 50 + "\n"
    summary += "  Best Parameters (K-FOLD + CTL/LDA + RF)\n"
    summary += "=" * 50 + "\n"
    summary += f"  n_estimators     : {best_parameters_kfold['n_estimators']}\n"
    summary += f"  max_depth        : {best_parameters_kfold['max_depth']}\n"
    summary += f"  min_samples_leaf : {best_parameters_kfold['min_samples_leaf']}\n"
    summary += "-" * 50 + "\n"
    summary += (
        f"Train Acc: {performance_kfold['train_acc']:.4f}, "
        f"Val Acc: {performance_kfold['val_acc']:.4f}, "
        f"Score: {performance_kfold['score']:.4f}\n"
    )

    return models, df_simple, df_kfold, summary


if __name__ == "__main__":
    # Load dataset
    X_temp = pd.read_csv("../../data/kfold/X_temp_norm.csv").squeeze()
    X_train = pd.read_csv("../../data/simple/X_train_norm.csv").squeeze()
    X_val = pd.read_csv("../../data/simple/X_val_norm.csv").squeeze()
    y_temp = pd.read_csv("../../data/kfold/y_temp_norm.csv").squeeze()
    y_train = pd.read_csv("../../data/simple/y_train_norm.csv").squeeze()
    y_val = pd.read_csv("../../data/simple/y_val_norm.csv").squeeze()

    # Fix for Pyright
    X_temp = cast(pd.DataFrame, X_temp)
    X_train = cast(pd.DataFrame, X_train)
    X_val = cast(pd.DataFrame, X_val)
    y_temp = cast(pd.Series, y_temp)
    y_train = cast(pd.Series, y_train)
    y_val = cast(pd.Series, y_val)

    # Define RF hyperparameters
    n_estimators = [10, 25, 50, 75, 100, 150, 200, 300]
    max_depth = [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
    min_samples_leaf = [1, 2, 4, 8]

    # Run RF
    models, df_simple, df_kfold, summary = rf(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_temp=X_temp,
        y_temp=y_temp,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    # Ensure output directory exists
    Path("../../results/models/rf").mkdir(parents=True, exist_ok=True)
    img_dir = Path("../../images/models/")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    Path("../../results/models/rf/rf_summary.txt").write_text(summary)

    # Expand parameters for simple strategy
    df_simple_expanded = df_simple.copy()
    df_simple_expanded["n_estimators"] = df_simple_expanded["params"].apply(
        lambda d: d["n_estimators"]
    )
    df_simple_expanded["max_depth"] = df_simple_expanded["params"].apply(
        lambda d: d["max_depth"]
    )
    df_simple_expanded["min_samples_leaf"] = df_simple_expanded["params"].apply(
        lambda d: d["min_samples_leaf"]
    )

    # Expand parameters for kfold strategy
    df_kfold_expanded = df_kfold.copy()
    df_kfold_expanded["n_estimators"] = df_kfold_expanded["param_rf__n_estimators"]
    df_kfold_expanded["max_depth"] = df_kfold_expanded["param_rf__max_depth"]
    df_kfold_expanded["min_samples_leaf"] = df_kfold_expanded[
        "param_rf__min_samples_leaf"
    ]

    # Custom style
    plt.style.use("../../config/custom_style.mplstyle")

    # Hyperparameters evolution plot
    fig, axs = plt.subplots(3, 1)

    # Score vs n_estimators
    sns.lineplot(
        data=df_simple_expanded,
        x="n_estimators",
        y="score",
        ax=axs[0],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="n_estimators",
        y="score",
        ax=axs[0],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[0].set_title("n_estimators")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Mean Score")
    axs[0].legend(loc="upper right")

    # Score vs max_depth
    sns.lineplot(
        data=df_simple_expanded,
        x="max_depth",
        y="score",
        ax=axs[1],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="max_depth",
        y="score",
        ax=axs[1],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[1].set_title("max_depth")
    axs[1].set_xlabel("")
    axs[1].set_ylabel("Mean Score")
    axs[1].legend(loc="upper right")

    # Score vs min_samples_leaf
    sns.lineplot(
        data=df_simple_expanded,
        x="min_samples_leaf",
        y="score",
        ax=axs[2],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="min_samples_leaf",
        y="score",
        ax=axs[2],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[2].set_title("min_samples_leaf")
    axs[2].set_xlabel("")
    axs[2].set_ylabel("Mean Score")
    axs[2].legend(loc="upper right")

    # Save figure
    plt.tight_layout()
    plt.savefig(img_dir / "rf_hyperparameters_evolution.png")
    plt.show()
