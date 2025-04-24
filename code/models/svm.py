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
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_temp: pd.DataFrame,
    y_temp: pd.Series,
    cv: StratifiedKFold,
    svm_c: Sequence[int],
    SEED: int = 42,
    pen: float = 0.5,
) -> Dict[str, Optional[Pipeline]]:
    """
    Perform a two-step SVM selection pipeline using CTL and LDA.

    This function first performs a manual grid search using a simple train/val
    split. Then, it uses cross-validation on the full train+val set to find the
    best SVM parameters using CTL and LDA inside the CV pipeline.

    Inputs:
        - X_train (pd.DataFrame): Training features (raw, already scaled).
        - y_train (pd.Series): Labels for the training set.
        - X_val (pd.DataFrame): Validation features (raw, already scaled).
        - y_val (pd.Series): Labels for the validation set.
        - X_temp (pd.DataFrame): Combined training+validation features (80% of
          total).
        - y_temp (pd.Series): Corresponding labels for X_temp.
        - cv (StratifiedKFold): K-Fold strategy to apply during GridSearchCV.
        - svm_c (Sequence[int]): List of C values for the SVM grid.
        - SEED (int, optional): Random seed for reproducibility. Default is 42.
        - pen (float, optional): Penalty weight for the train-validation score
          gap. Default is 0.5.

    Returns:
        - Dict[str, Optional[Pipeline]]: Dictionary with keys 'simple' and 'kfold',
          each containing the best pipeline for their respective strategy.
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

    # Preprocessing pipeline: CTL → LDA
    preprocess = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
        ]
    )

    # Fit on training data and transform both train and val
    X_train_lda = preprocess.fit_transform(X_train.values, y_train)
    X_val_lda = preprocess.transform(X_val.values)

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

    # Rebuild full model
    best_model = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
            (
                'svm',
                SVC(**best_parameters, probability=True, random_state=SEED),
            ),
        ]
    )
    best_model.fit(X_train, y_train)

    # Fix for Pyright
    best_parameters = cast(Dict[str, Any], best_parameters)
    best_model = cast(Pipeline, best_model)

    # Update models
    models['simple'] = best_model

    print("\n" + "=" * 50)
    print("  Best Parameters (SIMPLE + CTL/LDA + SVM)")
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

    # Grid search pipeline: CTL → LDA → SVM
    grid_pipeline = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
            ('svm', SVC(probability=True, random_state=SEED)),
        ]
    )
    grid_parameters = {
        'svm__C': svm_c,
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svm__gamma': ['scale', 'auto'],
    }

    # Run stratified K-fold grid search
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
    grid_search.fit(X_temp, y_temp)

    # Penalize overfit models
    df = pd.DataFrame(grid_search.cv_results_)
    df['gap'] = abs(df['mean_train_score'] - df['mean_test_score'])
    df['score'] = df['mean_test_score'] - pen * df['gap']

    # Pick best model from penalized score
    best_idx = df['score'].idxmax()
    best_parameters = df.loc[best_idx, 'params']
    performance = {
        'train_acc': df.loc[best_idx, 'mean_train_score'],
        'val_acc': df.loc[best_idx, 'mean_test_score'],
        'gap': df.loc[best_idx, 'gap'],
        'score': df.loc[best_idx, 'score'],
    }

    # Refit best model on the full training+val set
    best_parameters = {
        k.replace('svm__', ''): v for k, v in best_parameters.items()
    }
    final_model = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
            (
                'svm',
                SVC(**best_parameters, probability=True, random_state=SEED),
            ),
        ]
    )
    final_model.fit(X_temp, y_temp)

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

    # Get the minimum number of samples across all classes
    min_class_count = y_temp.value_counts().min()

    # Strategy for selecting the number of folds based on dataset size:
    #   - For very small datasets (< 100 samples), use 4 folds (or less if class count is lower)
    #   - For moderate datasets (< 1000 samples), use 5 folds
    #   - For large datasets (>= 1000 samples), use 10 folds
    if total_samples < 100:
        heuristic_K = min(4, min_class_count)
    elif total_samples < 1000:
        heuristic_K = min(5, min_class_count)
    else:
        heuristic_K = min(10, min_class_count)

    # Define stratified K-Fold cross-validator
    cv = StratifiedKFold(n_splits=heuristic_K, shuffle=True, random_state=42)

    # Define SVM hyperparameters
    svm_c = [0.1, 1, 10, 100]

    # Run SVM
    models = svm(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_temp=X_temp,
        y_temp=y_temp,
        cv=cv,
        svm_c=svm_c,
        SEED=42,
        pen=0.5,
    )
