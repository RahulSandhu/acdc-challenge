from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from utils.CoefficientThresholdLasso import CoefficientThresholdLasso


def knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_temp: pd.DataFrame,
    y_temp: pd.Series,
    knn_neighbors: Sequence[int],
    knn_weights: Sequence[str],
    knn_metrics: Sequence[str],
    pen: float = 0.5,
) -> Tuple[Dict[str, Optional[Pipeline]], pd.DataFrame, pd.DataFrame, str]:
    """
    Train and evaluate KNN models using two selection strategies: simple
    train/validation split and K-fold cross-validation, both incorporating CTL
    (Coefficient Threshold Lasso) and LDA (Linear Discriminant Analysis) as
    preprocessing steps.

    The function performs:
    1. A manual grid search on a train/validation split using CTL → LDA → KNN,
    selecting the model based on a penalized validation accuracy that accounts
    for overfitting (train-validation gap).
    2. A GridSearchCV over a combined training+validation set using the same
       CTL → LDA → KNN pipeline and penalized scoring.

    Inputs:
        - X_train (pd.DataFrame): Training features (already scaled).
        - y_train (pd.Series): Labels for the training set.
        - X_val (pd.DataFrame): Validation features (already scaled).
        - y_val (pd.Series): Labels for the validation set.
        - X_temp (pd.DataFrame): Combined training+validation features.
        - y_temp (pd.Series): Labels corresponding to X_temp.
        - knn_neighbors (Sequence[int]): Values of n_neighbors to try.
        - knn_weights (Sequence[str]): Weight strategies to try (e.g.,
          'uniform', 'distance').
        - knn_metrics (Sequence[str]): Distance metrics to try (e.g.,
          'euclidean', 'manhattan').
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

    # Grid search pipeline: CTL → LDA → KNN
    grid_pipeline_simple = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )
    grid_parameters_simple = [
        {"n_neighbors": n, "weights": w, "metric": m}
        for n in knn_neighbors
        for w in knn_weights
        for m in knn_metrics
    ]

    # Fit on training data and transform both train and val
    X_train_lda = grid_pipeline_simple.fit_transform(X_train.values, y_train)
    X_val_lda = grid_pipeline_simple.transform(X_val.values)

    # Manual grid search with scoring collected in a dataframe
    df_simple = []

    # Manual grid search (simple train-val split)
    for params in grid_parameters_simple:
        # Fit KNN with current parameters
        knn_mdl = KNeighborsClassifier(**params)
        knn_mdl.fit(X_train_lda, y_train)

        # Evaluate and penalize overfitting
        train_score = accuracy_score(y_train, knn_mdl.predict(X_train_lda))
        val_score = accuracy_score(y_val, knn_mdl.predict(X_val_lda))
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
            ("knn", KNeighborsClassifier(**best_parameters_simple)),
        ]
    )
    model_simple.fit(X_train, y_train)

    # Update models
    models["simple"] = model_simple

    # Report results
    summary = "=" * 50 + "\n"
    summary += "  Best Parameters (SIMPLE + CTL/LDA + KNN)\n"
    summary += "=" * 50 + "\n"
    summary += f"  n_neighbors : {best_parameters_simple['n_neighbors']}\n"
    summary += f"  weights     : {best_parameters_simple['weights']}\n"
    summary += f"  metric      : {best_parameters_simple['metric']}\n"
    summary += "-" * 50 + "\n"
    summary += (
        f"Train Acc: {performance_simple['train_acc']:.4f}, "
        f"Val Acc: {performance_simple['val_acc']:.4f}, "
        f"Score: {performance_simple['score']:.4f}"
    )

    # Grid search pipeline: CTL → LDA → KNN
    grid_pipeline_kfold = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
            ("knn", KNeighborsClassifier()),
        ]
    )
    grid_parameters_kfold = {
        "knn__n_neighbors": knn_neighbors,
        "knn__weights": knn_weights,
        "knn__metric": knn_metrics,
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
        k.replace("knn__", ""): v for k, v in best_parameters_kfold.items()
    }
    model_kfold = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
            ("knn", KNeighborsClassifier()),
        ]
    )
    model_kfold.fit(X_temp, y_temp)

    # Update models
    models["kfold"] = model_kfold

    # Report results
    summary += "\n\n" + "=" * 50 + "\n"
    summary += "  Best Parameters (K-FOLD + CTL/LDA + KNN)\n"
    summary += "=" * 50 + "\n"
    summary += f"  n_neighbors : {best_parameters_kfold['n_neighbors']}\n"
    summary += f"  weights     : {best_parameters_kfold['weights']}\n"
    summary += f"  metric      : {best_parameters_kfold['metric']}\n"
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

    # Define KNN hyperparameters
    knn_neighbors = list(range(1, 60, 2))
    knn_weights = ["uniform", "distance"]
    knn_metrics = [
        "euclidean",
        "manhattan",
        "chebyshev",
        "cosine",
        "hamming",
        "canberra",
        "correlation",
        "braycurtis",
        "minkowski",
    ]

    # Run KNN
    models, df_simple, df_kfold, summary = knn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_temp=X_temp,
        y_temp=y_temp,
        knn_neighbors=knn_neighbors,
        knn_weights=knn_weights,
        knn_metrics=knn_metrics,
    )

    # Ensure output directory exists
    Path("../../results/models/knn").mkdir(parents=True, exist_ok=True)
    img_dir = Path("../../images/models/")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    Path("../../results/models/knn/knn_summary.txt").write_text(summary)

    # Expand parameters for simple strategy
    df_simple_expanded = df_simple.copy()
    df_simple_expanded["n_neighbors"] = df_simple_expanded["params"].apply(
        lambda d: d["n_neighbors"]
    )
    df_simple_expanded["weights"] = df_simple_expanded["params"].apply(
        lambda d: d["weights"]
    )
    df_simple_expanded["metric"] = df_simple_expanded["params"].apply(
        lambda d: d["metric"]
    )

    # Expand parameters for kfold strategy
    df_kfold_expanded = df_kfold.copy()
    df_kfold_expanded["n_neighbors"] = df_kfold_expanded["param_knn__n_neighbors"]
    df_kfold_expanded["weights"] = df_kfold_expanded["param_knn__weights"]
    df_kfold_expanded["metric"] = df_kfold_expanded["param_knn__metric"]

    # Custom style
    plt.style.use("../../misc/custom_style.mplstyle")

    # Hyperparameters evolution plot
    fig, axs = plt.subplots(3, 1)

    # Score vs n_neighbors
    sns.lineplot(
        data=df_simple_expanded,
        x="n_neighbors",
        y="score",
        ax=axs[0],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="n_neighbors",
        y="score",
        ax=axs[0],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[0].set_title("n_neighbors")
    axs[0].set_xlabel("")
    axs[0].legend(loc="upper right")

    # Score vs weights
    sns.lineplot(
        data=df_simple_expanded,
        x="weights",
        y="score",
        ax=axs[1],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="weights",
        y="score",
        ax=axs[1],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[1].set_title("weights")
    axs[1].set_xlabel("")
    axs[1].set_ylabel("Mean Score")
    axs[1].legend(loc="upper right")

    # Score vs metric
    sns.lineplot(
        data=df_simple_expanded,
        x="metric",
        y="score",
        ax=axs[2],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="metric",
        y="score",
        ax=axs[2],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[2].tick_params(axis="x", rotation=30)
    axs[2].set_title("metric")
    axs[2].set_xlabel("")
    axs[2].set_ylabel("Mean Score")
    axs[2].legend(loc="upper right")

    # Save figure
    plt.tight_layout()
    plt.savefig(img_dir / "knn_hyperparameters_evolution.png")
    plt.show()
