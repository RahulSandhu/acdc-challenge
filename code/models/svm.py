from typing import Any, Dict, Optional, Sequence, cast

import pandas as pd
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
from tools.CoefficientThresholdLasso import CoefficientThresholdLasso


def svm(
    X_train_lda: pd.DataFrame,
    y_train: pd.Series,
    X_val_lda: pd.DataFrame,
    y_val: pd.Series,
    X_temp_lda: pd.DataFrame,
    y_temp: pd.Series,
    cv: StratifiedKFold,
    svm_c: Sequence[int],
    SEED: int = 42,
    pen: float = 0.75,
) -> Dict[str, Optional[Pipeline]]:
    """
    Perform a two-step selection process for SVM classifiers using
    LDA-transformed input.

    First, it performs a manual grid search on the training and validation
    sets, selecting the best model based on a penalized score that accounts for
    the train-validation gap. Second, it performs a GridSearchCV using
    cross-validation and selects the model with the best penalized score.

    Inputs:
        - X_train_lda (pd.DataFrame): LDA-transformed training feature matrix
          (60%).
        - y_train (pd.Series): Training labels.
        - X_val_lda (pd.DataFrame): LDA-transformed validation feature matrix
          (20%).
        - y_val (pd.Series): Validation labels.
        - X_temp_lda (pd.DataFrame): LDA-transformed combined training +
          validation feature matrix (80%) used for cross-validation.
        - y_temp (pd.Series): Corresponding labels for X_temp_lda.
        - cv (StratifiedKFold): StratifiedKFold object for cross-validation.
        - svm_c (Sequence[int]): List of C values (inverse of regularization
          strength) for SVM.
        - SEED (int, optional): Random seed for reproducibility. Default is 42.
        - pen (float, optional): Penalty weight for the train-validation score
          gap. Default is 0.75.

    Outputs:
        - Dict[str, Optional[Pipeline]]: Dictionary with keys 'simple' and
          'kfold' mapping to the best SVM pipeline from manual and CV-based
          selection, respectively.
    """
    # Initialize result container
    models: Dict[str, Optional[Pipeline]] = {}

    # Initialize tracking variables for best model
    best_metric = -float('inf')
    best_model: Optional[Pipeline] = None
    best_parameters: Dict[str, Any] = {}
    performance: Dict[str, float] = {}

    # Build parameter combinations
    parameters_grid = [
        {'C': c, 'kernel': k, 'gamma': g}
        for c in svm_c
        for k in ['linear', 'rbf', 'poly', 'sigmoid']
        for g in ['scale', 'auto']
    ]

    # Manual grid search (simple train-val split)
    for params in parameters_grid:
        # Fit SVM with current parameters
        svm_mdl = SVC(**params, probability=True, random_state=SEED)
        svm_mdl.fit(X_train_lda, y_train)

        # Evaluate and penalize overfitting
        train_score = accuracy_score(y_train, svm_mdl.predict(X_train_lda))
        val_score = accuracy_score(y_val, svm_mdl.predict(X_val_lda))
        gap = abs(train_score - val_score)
        penalized = val_score - pen * gap

        # Track best performing model
        if penalized > best_metric:
            best_metric = penalized
            best_model = Pipeline([('svm', svm_mdl)])
            best_parameters = params
            performance = {
                'train_acc': train_score,
                'val_acc': val_score,
                'gap': gap,
                'score': penalized,
            }

    # Fix for Pyright
    best_parameters = cast(Dict[str, Any], best_parameters)
    best_model = cast(Pipeline, best_model)

    # Update models
    models['simple'] = best_model

    print("\n" + "=" * 50)
    print("  Best Parameters (K-FOLD + CTL/LDA + SVM)")
    print("=" * 50)
    print(f"  C       : {best_parameters['C']}")
    print(f"  kernel  : {best_parameters['kernel']}")
    print(f"  gamma   : {best_parameters['gamma']}")
    print("-" * 50)
    print(
        f"Train Acc: {performance['train_acc']:.4f}, "
        f"Val Acc: {performance['val_acc']:.4f}, "
        f"Score: {performance['score']:.4f}"
    )
    print("=" * 50 + "\n")

    # Grid search with transformed input
    grid_pipeline = Pipeline(
        [('svm', SVC(probability=True, random_state=SEED))]
    )
    grid_parameters = {
        'svm__C': svm_c,
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svm__gamma': ['scale', 'auto'],
    }

    # Run grid search CV
    grid_search = GridSearchCV(
        estimator=grid_pipeline,
        param_grid=grid_parameters,
        cv=cv,
        scoring='accuracy',
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
        refit=False,
    )
    grid_search.fit(X_temp_lda, y_temp)

    # Penalize overfit models
    df = pd.DataFrame(grid_search.cv_results_)
    df['gap'] = abs(df['mean_train_score'] - df['mean_test_score'])
    df['score'] = df['mean_test_score'] - pen * df['gap']

    # Get best based on penalized score
    best_idx = df['score'].idxmax()
    best_parameters = df.loc[best_idx, 'params']
    performance = {
        'train_acc': df.loc[best_idx, 'mean_train_score'],
        'val_acc': df.loc[best_idx, 'mean_test_score'],
        'gap': df.loc[best_idx, 'gap'],
        'score': df.loc[best_idx, 'score'],
    }

    # Reconstruct and fit best parameter dictionary
    best_parameters = {
        k.replace('svm__', ''): v for k, v in best_parameters.items()
    }
    best_svm = SVC(**best_parameters, probability=True, random_state=SEED)
    best_svm.fit(X_temp_lda, y_temp)
    final_model = Pipeline([('svm', best_svm)])

    # Update models
    models['kfold'] = final_model

    print("\n" + "=" * 50)
    print("  Best Parameters (K-FOLD + CTL/LDA + SVM)")
    print("=" * 50)
    print(f"  C       : {best_parameters['C']}")
    print(f"  kernel  : {best_parameters['kernel']}")
    print(f"  gamma   : {best_parameters['gamma']}")
    print("-" * 50)
    print(
        f"Train Acc: {performance['train_acc']:.4f}, "
        f"Val Acc: {performance['val_acc']:.4f}, "
        f"Score: {performance['score']:.4f}"
    )
    print("=" * 50 + "\n")

    return models


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/raw/acdc_radiomics.csv")

    # Encode labels
    le = LabelEncoder()
    df["class"] = le.fit_transform(df["class"])

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

    # Get total number of training samples
    total_samples = len(X_temp)

    # Strategy for selecting the number of folds based on dataset size:
    #   - For very small datasets (< 100 samples), use leave-one-out CV
    #   - For moderate datasets (< 1000 samples), use 10 folds
    #   - Otherwise, use 5 folds
    if total_samples < 100:
        heuristic_K = total_samples
    elif total_samples < 1000:
        heuristic_K = 10
    else:
        heuristic_K = 5

    # Limit K to the smallest class count to ensure stratification works
    min_class_count = y_train.value_counts().min()
    K = min(heuristic_K, min_class_count)

    # Define stratified K-Fold cross-validator
    cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    # Apply CTL
    ctl = CoefficientThresholdLasso()
    ctl.fit(X_train.values, y_train.values)

    X_train_lasso = ctl.transform(X_train.values)
    X_val_lasso = ctl.transform(X_val.values)
    X_test_lasso = ctl.transform(X_test.values)
    X_temp_lasso = ctl.transform(X_temp.values)

    # Apply LDA
    lda = LinearDiscriminantAnalysis()
    X_train_lda = lda.fit_transform(X_train_lasso, y_train)

    X_val_lda = lda.transform(X_val_lasso)
    X_test_lda = lda.transform(X_test_lasso)
    X_temp_lda = lda.transform(X_temp_lasso)

    # Define SVM hyperparameters
    svm_c = [0.1, 1, 10, 100]

    # Run SVM
    models = svm(
        X_train_lda=X_train_lda,
        y_train=y_train,
        X_val_lda=X_val_lda,
        y_val=y_val,
        X_temp_lda=X_temp_lda,
        y_temp=y_temp,
        cv=cv,
        svm_c=svm_c,
        SEED=42,
        pen=0.75,
    )
