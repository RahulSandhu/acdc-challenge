from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, cast

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from utils.CoefficientThresholdLasso import CoefficientThresholdLasso


def svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_temp: pd.DataFrame,
    y_temp: pd.Series,
    cv: StratifiedKFold,
    svm_kernel: Sequence,
    svm_gamma: Sequence,
    svm_c: Sequence,
    SEED: int = 42,
    pen: float = 0.5,
) -> Tuple[Dict[str, Optional[Pipeline]], pd.DataFrame, pd.DataFrame, str]:
    """
    Train and evaluate SVM models using two selection strategies: simple
    train/validation split and K-fold cross-validation, both incorporating CTL
    (Coefficient Threshold Lasso) and LDA (Linear Discriminant Analysis) as
    preprocessing steps.

    The function performs:
    1. A manual grid search on a train/validation split using CTL → LDA → SVM,
    selecting the model based on a penalized validation accuracy that accounts
    for the overfitting gap.
    2. A GridSearchCV over a combined training+validation set using the same
    CTL → LDA → SVM pipeline and penalized scoring.

    Inputs:
        - X_train (pd.DataFrame): Training features (already scaled).
        - y_train (pd.Series): Labels for the training set.
        - X_val (pd.DataFrame): Validation features (already scaled).
        - y_val (pd.Series): Labels for the validation set.
        - X_temp (pd.DataFrame): Combined training+validation features.
        - y_temp (pd.Series): Labels corresponding to X_temp.
        - cv (StratifiedKFold): Stratified K-Fold cross-validation strategy.
        - svm_kernel (Sequence): List of kernel types to try (e.g., 'linear',
          'rbf').
        - svm_gamma (Sequence): List of gamma values to try (e.g., 'scale',
          'auto').
        - svm_c (Sequence): List of C values to try for the SVM classifier.
        - SEED (int, optional): Random seed for reproducibility. Default is 42.
        - pen (float, optional): Penalty weight applied to overfitting
          (train-val gap). Default is 0.5.

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

    # Grid search pipeline: CTL → LDA → SVM
    grid_pipeline_simple = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
        ]
    )
    grid_parameters_simple = [
        {'kernel': k, 'gamma': g, 'C': c}
        for k in svm_kernel
        for g in svm_gamma
        for c in svm_c
    ]

    # Fit on training data and transform both train and val
    X_train_lda = grid_pipeline_simple.fit_transform(X_train.values, y_train)
    X_val_lda = grid_pipeline_simple.transform(X_val.values)

    # Manual grid search with scoring collected in a dataframe
    df_simple = []

    # Manual grid search (simple train-val split)
    for params in grid_parameters_simple:
        # Fit SVM with current parameters
        svm_mdl = SVC(**params, probability=True, random_state=SEED)
        svm_mdl.fit(X_train_lda, y_train)

        # Evaluate and penalize overfitting
        train_score = accuracy_score(y_train, svm_mdl.predict(X_train_lda))
        val_score = accuracy_score(y_val, svm_mdl.predict(X_val_lda))
        gap = abs(train_score - val_score)
        penalized = val_score - pen * gap

        # Append metrics
        df_simple.append(
            {
                'params': params,
                'train_score': train_score,
                'val_score': val_score,
                'gap': gap,
                'score': penalized,
            }
        )
    df_simple = pd.DataFrame(df_simple)

    # Pick best model from penalized score
    best_idx_simple = df_simple['score'].idxmax()
    best_parameters_simple = df_simple.loc[best_idx_simple, 'params']
    performance_simple = {
        'train_acc': df_simple.loc[best_idx_simple, 'train_score'],
        'val_acc': df_simple.loc[best_idx_simple, 'val_score'],
        'gap': df_simple.loc[best_idx_simple, 'gap'],
        'score': df_simple.loc[best_idx_simple, 'score'],
    }

    # Refit best model on the full training set
    model_simple = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
            (
                'svm',
                SVC(
                    **best_parameters_simple,
                    probability=True,
                    random_state=SEED,
                ),
            ),
        ]
    )
    model_simple.fit(X_train, y_train)

    # Update models
    models['simple'] = model_simple

    # Report results
    summary = "=" * 50 + "\n"
    summary += "  Best Parameters (SIMPLE + CTL/LDA + SVM)\n"
    summary += "=" * 50 + "\n"
    summary += f"  C       : {best_parameters_simple['C']}\n"
    summary += f"  kernel  : {best_parameters_simple['kernel']}\n"
    summary += f"  gamma   : {best_parameters_simple['gamma']}\n"
    summary += "-" * 50 + "\n"
    summary += (
        f"Train Acc: {performance_simple['train_acc']:.4f}, "
        f"Val Acc: {performance_simple['val_acc']:.4f}, "
        f"Score: {performance_simple['score']:.4f}"
    )

    # Grid search pipeline: CTL → LDA → SVM
    grid_pipeline_kfold = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
            ('svm', SVC(probability=True, random_state=SEED)),
        ]
    )
    grid_parameters_kfold = {
        'svm__kernel': svm_kernel,
        'svm__gamma': svm_gamma,
        'svm__C': svm_c,
    }

    # Run stratified K-fold grid search
    grid_search_kfold = GridSearchCV(
        estimator=grid_pipeline_kfold,
        param_grid=grid_parameters_kfold,
        cv=cv,
        scoring='accuracy',
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
        refit=False,
    )
    grid_search_kfold.fit(X_temp, y_temp)

    # Penalize overfit models
    df_kfold = pd.DataFrame(grid_search_kfold.cv_results_)
    df_kfold['gap'] = abs(
        df_kfold['mean_train_score'] - df_kfold['mean_test_score']
    )
    df_kfold['score'] = df_kfold['mean_test_score'] - pen * df_kfold['gap']

    # Pick best model from penalized score
    best_idx_kfold = df_kfold['score'].idxmax()
    best_parameters_kfold = df_kfold.loc[best_idx_kfold, 'params']
    performance_kfold = {
        'train_acc': df_kfold.loc[best_idx_kfold, 'mean_train_score'],
        'val_acc': df_kfold.loc[best_idx_kfold, 'mean_test_score'],
        'gap': df_kfold.loc[best_idx_kfold, 'gap'],
        'score': df_kfold.loc[best_idx_kfold, 'score'],
    }

    # Refit best model on the full training+val set
    best_parameters_kfold = {
        k.replace('svm__', ''): v for k, v in best_parameters_kfold.items()
    }
    model_kfold = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
            (
                'svm',
                SVC(
                    **best_parameters_kfold,
                    probability=True,
                    random_state=SEED,
                ),
            ),
        ]
    )
    model_kfold.fit(X_temp, y_temp)

    # Update models
    models['kfold'] = model_kfold

    # Report results
    summary += "\n\n" + "=" * 50 + "\n"
    summary += "  Best Parameters (KFOLD + CTL/LDA + SVM)\n"
    summary += "=" * 50 + "\n"
    summary += f"  C       : {best_parameters_kfold['C']}\n"
    summary += f"  kernel  : {best_parameters_kfold['kernel']}\n"
    summary += f"  gamma   : {best_parameters_kfold['gamma']}\n"
    summary += "-" * 50 + "\n"
    summary += (
        f"Train Acc: {performance_kfold['train_acc']:.4f}, "
        f"Val Acc: {performance_kfold['val_acc']:.4f}, "
        f"Score: {performance_kfold['score']:.4f}\n"
    )

    return models, df_simple, df_kfold, summary


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/datasets/raw_acdc_radiomics.csv")

    # Encode labels
    le = LabelEncoder()
    df["class"] = le.fit_transform(df["class"])

    # Save the label encoder
    Path("../../results/models/").mkdir(parents=True, exist_ok=True)
    joblib.dump(le, "../../results/models/label_encoder.pkl")

    # Separate features and classes
    X = df.drop(columns=["class"])
    y = df["class"]

    # Apply StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(scaled_features, columns=X.columns)

    # Normalized dataframe
    norm_df = pd.concat([scaled_df, y.reset_index(drop=True)], axis=1)

    # Separate features and classes
    X = norm_df.drop(columns=["class"])
    y = norm_df["class"]

    # 80% Train+Val, 20% Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 75% Train, 25% Val from the 80% (→ 60% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    # Fix for Pyright
    X_train = cast(pd.DataFrame, X_train)
    X_val = cast(pd.DataFrame, X_val)
    X_test = cast(pd.DataFrame, X_test)
    X_temp = cast(pd.DataFrame, X_temp)
    y_train = cast(pd.Series, y_train)
    y_val = cast(pd.Series, y_val)
    y_test = cast(pd.Series, y_test)
    y_temp = cast(pd.Series, y_temp)

    # Save test set
    Path("../../data/testing/").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_test).to_csv("../../data/testing/X_test.csv", index=False)
    y_test.to_frame().to_csv("../../data/testing/y_test.csv", index=False)

    # Define stratified K-Fold cross-validator
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Define SVM hyperparameters
    svm_kernel = ['linear', 'rbf', 'sigmoid', 'poly']
    svm_gamma = ['scale', 'auto', 0.1, 1, 10, 100]
    svm_c = [0.1, 1, 10, 100]

    # Run SVM
    models, df_simple, df_kfold, summary = svm(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_temp=X_temp,
        y_temp=y_temp,
        cv=cv,
        svm_kernel=svm_kernel,
        svm_gamma=svm_gamma,
        svm_c=svm_c,
    )

    # Ensure output directory exists
    Path("../../results/models/svm").mkdir(parents=True, exist_ok=True)
    img_dir = Path("../../images/models/")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    Path("../../results/models/svm/svm_summary.txt").write_text(summary)

    # Expand parameters for simple strategy
    df_simple_expanded = df_simple.copy()
    df_simple_expanded['kernel'] = df_simple_expanded['params'].apply(
        lambda d: d['kernel']
    )
    df_simple_expanded['gamma'] = df_simple_expanded['params'].apply(
        lambda d: d['gamma']
    )
    df_simple_expanded['C'] = df_simple_expanded['params'].apply(
        lambda d: d['C']
    )

    # Expand parameters for kfold strategy
    df_kfold_expanded = df_kfold.copy()
    df_kfold_expanded['kernel'] = df_kfold_expanded['param_svm__kernel']
    df_kfold_expanded['gamma'] = df_kfold_expanded['param_svm__gamma']
    df_kfold_expanded['C'] = df_kfold_expanded['param_svm__C']

    # Make sure gamma is a string
    df_simple_expanded['gamma'] = df_simple_expanded['gamma'].astype(str)
    df_kfold_expanded['gamma'] = df_kfold_expanded['gamma'].astype(str)

    # Custom style
    plt.style.use("../../misc/custom_style.mplstyle")

    # Hyperparameters evolution plot
    fig, axs = plt.subplots(3, 1)

    # Score vs kernel
    sns.lineplot(
        data=df_simple_expanded,
        x='kernel',
        y='score',
        ax=axs[0],
        marker='o',
        label='Simple',
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x='kernel',
        y='score',
        ax=axs[0],
        marker='X',
        linestyle='--',
        label='K-Fold',
    )
    axs[0].set_title('kernel')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Mean Score')
    axs[0].legend(loc='upper right')

    # Score vs gamma
    sns.lineplot(
        data=df_simple_expanded,
        x='gamma',
        y='score',
        ax=axs[1],
        marker='o',
        label='Simple',
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x='gamma',
        y='score',
        ax=axs[1],
        marker='X',
        linestyle='--',
        label='K-Fold',
    )
    axs[1].set_title('gamma')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Mean Score')
    axs[1].legend(loc='upper right')

    # Score vs C
    sns.lineplot(
        data=df_simple_expanded,
        x='C',
        y='score',
        ax=axs[2],
        marker='o',
        label='Simple',
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x='C',
        y='score',
        ax=axs[2],
        marker='X',
        linestyle='--',
        label='K-Fold',
    )
    axs[2].set_title('C')
    axs[2].set_xlabel('')
    axs[2].set_ylabel('Mean Score')
    axs[2].legend(loc='upper right')

    # Save figure
    plt.tight_layout()
    plt.savefig(img_dir / "svm_hyperparameters_evolution.png")
    plt.show()
