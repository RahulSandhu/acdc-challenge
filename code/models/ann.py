from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skorch import NeuralNetClassifier
from utils.CoefficientThresholdLasso import CoefficientThresholdLasso


class SimpleANN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        hidden_layers,
        output_dim,
        activation_fn,
    ):
        super(SimpleANN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(activation_fn())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fn())
        layers.append(nn.Linear(hidden_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def ann(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_temp: pd.DataFrame,
    y_temp: pd.Series,
    cv: StratifiedKFold,
    hidden_layers=Sequence[int],
    hidden_size=Sequence[int],
    learning_rate=Sequence[float],
    activation_fn=Sequence[Any],
    pen: float = 0.5,
) -> Tuple[Dict[str, Optional[Pipeline]], pd.DataFrame, pd.DataFrame, str]:
    """
    Train and evaluate ANN models using two selection strategies: simple
    train/validation split and K-fold cross-validation, both incorporating CTL
    (Coefficient Threshold Lasso) and LDA (Linear Discriminant Analysis) as
    preprocessing steps.

    The function performs:
    1. A manual grid search on a train/validation split using CTL → LDA → ANN,
    selecting the model based on a penalized validation accuracy that accounts
    for overfitting (train-validation gap).
    2. A GridSearchCV over a combined training+validation set using the same
    CTL → LDA → ANN pipeline and penalized scoring.

    Inputs:
        - X_train (pd.DataFrame): Training features (already scaled).
        - y_train (pd.Series): Labels for the training set.
        - X_val (pd.DataFrame): Validation features (already scaled).
        - y_val (pd.Series): Labels for the validation set.
        - X_temp (pd.DataFrame): Combined training+validation features.
        - y_temp (pd.Series): Labels corresponding to X_temp.
        - cv (StratifiedKFold): Stratified K-Fold cross-validation strategy.
        - hidden_layers (Sequence[int]): Number of hidden layers to try.
        - hidden_size (Sequence[int]): Number of neurons in each hidden layer
          to try.
        - learning_rate (Sequence[float]): Learning rates to try.
        - activation_fn (Sequence[Any]): Activation functions to try (e.g.,
          nn.ReLU, nn.Tanh).
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

    # Grid search pipeline: CTL → LDA → ANN
    grid_pipeline_simple = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
        ]
    )
    grid_parameters_simple = [
        {
            'hidden_layers': hl,
            'hidden_size': hs,
            'learning_rate': lr,
            'activation_fn': afn,
        }
        for hl in hidden_layers
        for hs in hidden_size
        for lr in learning_rate
        for afn in activation_fn
    ]

    # Fit on training data and transform both train and val
    X_train_lda = grid_pipeline_simple.fit_transform(X_train.values, y_train)
    X_val_lda = grid_pipeline_simple.transform(X_val.values)

    # Manual grid search with scoring collected in a dataframe
    df_simple = []

    # Manual grid search (simple train-val split)
    for params in grid_parameters_simple:
        # Fit ANN with current parameters
        ann_mdl = NeuralNetClassifier(
            module=SimpleANN,
            module__input_dim=X_train_lda.shape[1],
            module__hidden_size=params['hidden_size'],
            module__hidden_layers=params['hidden_layers'],
            module__output_dim=len(set(y_train)),
            module__activation_fn=params['activation_fn'],
            max_epochs=30,
            lr=params['learning_rate'],
            optimizer=torch.optim.Adam,
            criterion=nn.CrossEntropyLoss,
            verbose=0,
        )
        ann_mdl.fit(
            X_train_lda.astype('float32'), y_train.values.astype('long')
        )

        # Evaluate and penalize overfitting
        train_score = accuracy_score(
            y_train, ann_mdl.predict(X_train_lda.astype('float32'))
        )
        val_score = accuracy_score(
            y_val, ann_mdl.predict(X_val_lda.astype('float32'))
        )
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
                'ann',
                NeuralNetClassifier(
                    module=SimpleANN,
                    module__input_dim=X_train_lda.shape[1],
                    module__hidden_size=best_parameters_simple['hidden_size'],
                    module__hidden_layers=best_parameters_simple[
                        'hidden_layers'
                    ],
                    module__output_dim=len(set(y_train)),
                    module__activation_fn=best_parameters_simple[
                        'activation_fn'
                    ],
                    max_epochs=30,
                    lr=best_parameters_simple['learning_rate'],
                    optimizer=torch.optim.Adam,
                    criterion=nn.CrossEntropyLoss,
                    verbose=0,
                ),
            ),
        ]
    )
    model_simple.fit(
        X_train.values.astype('float32'), y_train.values.astype('long')
    )

    # Update models
    models['simple'] = model_simple

    summary = "=" * 50 + "\n"
    summary += "  Best Parameters (SIMPLE + CTL/LDA + ANN)\n"
    summary += "=" * 50 + "\n"
    summary += f"  hidden_layers : {best_parameters_simple['hidden_layers']}\n"
    summary += f"  hidden_size   : {best_parameters_simple['hidden_size']}\n"
    summary += f"  learning_rate : {best_parameters_simple['learning_rate']}\n"
    summary += f"  activation_fn : {best_parameters_simple['activation_fn'].__name__}\n"
    summary += "-" * 50 + "\n"
    summary += (
        f"Train Acc: {performance_simple['train_acc']:.4f}, "
        f"Val Acc: {performance_simple['val_acc']:.4f}, "
        f"Score: {performance_simple['score']:.4f}"
    )

    # Grid search pipeline: CTL → LDA → ANN
    grid_pipeline_kfold = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
            (
                'ann',
                NeuralNetClassifier(
                    module=SimpleANN,
                    module__input_dim=len(np.unique(y_temp)) - 1,
                    module__output_dim=len(set(y_temp)),
                    max_epochs=30,
                    optimizer=torch.optim.Adam,
                    criterion=nn.CrossEntropyLoss,
                    verbose=0,
                ),
            ),
        ]
    )
    grid_parameters_kfold = {
        'ann__module__hidden_layers': hidden_layers,
        'ann__module__hidden_size': hidden_size,
        'ann__module__activation_fn': activation_fn,
        'ann__lr': learning_rate,
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
    grid_search_kfold.fit(
        X_temp.values.astype('float32'), y_temp.values.astype('long')
    )

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
        k.replace('ann__', ''): v for k, v in best_parameters_kfold.items()
    }
    model_kfold = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
            (
                'ann',
                NeuralNetClassifier(
                    module=SimpleANN,
                    module__input_dim=len(np.unique(y_temp)) - 1,
                    module__hidden_size=best_parameters_kfold[
                        'module__hidden_size'
                    ],
                    module__hidden_layers=best_parameters_kfold[
                        'module__hidden_layers'
                    ],
                    module__output_dim=len(set(y_temp)),
                    module__activation_fn=best_parameters_kfold[
                        'module__activation_fn'
                    ],
                    max_epochs=30,
                    lr=best_parameters_kfold['lr'],
                    optimizer=torch.optim.Adam,
                    criterion=nn.CrossEntropyLoss,
                    verbose=0,
                ),
            ),
        ]
    )
    model_kfold.fit(
        X_temp.values.astype('float32'), y_temp.values.astype('long')
    )

    # Update models
    models['kfold'] = model_kfold

    # Report results
    summary += "\n\n" + "=" * 50 + "\n"
    summary += "  Best Parameters (KFOLD + CTL/LDA + ANN)\n"
    summary += "=" * 50 + "\n"
    summary += (
        f"  hidden_layers : {best_parameters_kfold['module__hidden_layers']}\n"
    )
    summary += (
        f"  hidden_size   : {best_parameters_kfold['module__hidden_size']}\n"
    )
    summary += f"  learning_rate : {best_parameters_kfold['lr']}\n"
    summary += f"  activation_fn : {best_parameters_kfold['module__activation_fn'].__name__}\n"
    summary += "-" * 50 + "\n"
    summary += (
        f"Train Acc: {performance_kfold['train_acc']:.4f}, "
        f"Val Acc: {performance_kfold['val_acc']:.4f}, "
        f"Score: {performance_kfold['score']:.4f}\n"
    )

    return models, df_simple, df_kfold, summary


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/raw/acdc_radiomics.csv")

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
    Path("../../data/processed/").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_test).to_csv("../../data/processed/X_test.csv", index=False)
    y_test.to_frame().to_csv("../../data/processed/y_test.csv", index=False)

    # Define stratified K-Fold cross-validator
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Define ANN hyperparameters
    hidden_layers_list = [1, 2, 3]
    hidden_size_list = [32, 100, 128]
    learning_rate_list = [0.01, 0.001, 0.0001]
    activation_fn_list = [nn.ReLU, nn.Tanh, nn.Sigmoid]

    # Run ANN
    models, df_simple, df_kfold, summary = ann(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_temp=X_temp,
        y_temp=y_temp,
        cv=cv,
        hidden_layers=hidden_layers_list,
        hidden_size=hidden_size_list,
        learning_rate=learning_rate_list,
        activation_fn=activation_fn_list,
    )

    # Ensure output directory exists
    Path("../../results/models/ann").mkdir(parents=True, exist_ok=True)
    img_dir = Path("../../images/models/")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    Path("../../results/models/ann/ann_summary.txt").write_text(summary)

    # Expand parameters for simple strategy
    df_simple_expanded = df_simple.copy()
    df_simple_expanded['hidden_layers'] = df_simple_expanded['params'].apply(
        lambda d: d['hidden_layers']
    )
    df_simple_expanded['hidden_size'] = df_simple_expanded['params'].apply(
        lambda d: d['hidden_size']
    )
    df_simple_expanded['learning_rate'] = df_simple_expanded['params'].apply(
        lambda d: d['learning_rate']
    )
    df_simple_expanded['activation_fn'] = df_simple_expanded['params'].apply(
        lambda d: d['activation_fn'].__name__
    )

    # Expand parameters for kfold strategy
    df_kfold_expanded = df_kfold.copy()
    df_kfold_expanded['hidden_layers'] = df_kfold_expanded[
        'param_ann__module__hidden_layers'
    ]
    df_kfold_expanded['hidden_size'] = df_kfold_expanded[
        'param_ann__module__hidden_size'
    ]
    df_kfold_expanded['learning_rate'] = df_kfold_expanded['param_ann__lr']
    df_kfold_expanded['activation_fn'] = df_kfold_expanded[
        'param_ann__module__activation_fn'
    ].apply(lambda fn: fn.__name__)

    # Custom style
    plt.style.use("../../misc/custom_style.mplstyle")

    # Hyperparameters evolution plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Score vs hidden_layers
    sns.lineplot(
        data=df_simple_expanded,
        x='hidden_layers',
        y='score',
        ax=axs[0],
        marker='o',
        label='Simple',
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x='hidden_layers',
        y='score',
        ax=axs[0],
        marker='X',
        linestyle='--',
        label='KFold',
    )

    # Score vs hidden_size
    sns.lineplot(
        data=df_simple_expanded,
        x='hidden_size',
        y='score',
        ax=axs[1],
        marker='o',
        label='Simple',
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x='hidden_size',
        y='score',
        ax=axs[1],
        marker='X',
        linestyle='--',
        label='KFold',
    )

    # Score vs learning_rate
    sns.lineplot(
        data=df_simple_expanded,
        x='learning_rate',
        y='score',
        ax=axs[2],
        marker='o',
        label='Simple',
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x='learning_rate',
        y='score',
        ax=axs[2],
        marker='X',
        linestyle='--',
        label='KFold',
    )

    # Score vs activation_fn
    sns.lineplot(
        data=df_simple_expanded,
        x='activation_fn',
        y='score',
        ax=axs[3],
        marker='o',
        label='Simple',
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x='activation_fn',
        y='score',
        ax=axs[3],
        marker='X',
        linestyle='--',
        label='KFold',
    )
    axs[3].tick_params(axis='x', rotation=30)

    # Save figure
    fig.suptitle('ANN Hyperparameters Evolution', fontweight='bold')
    plt.tight_layout()
    plt.savefig(img_dir / "ann_hyperparameters_evolution.png")
    plt.show()

    # Prepare X_test
    X_test_tensor = X_test.values.astype('float32')

    # Test simple model
    simple_test_preds = models['simple'].predict(X_test_tensor)
    simple_test_acc = accuracy_score(y_test, simple_test_preds)

    # Test kfold model
    kfold_test_preds = models['kfold'].predict(X_test_tensor)
    kfold_test_acc = accuracy_score(y_test, kfold_test_preds)

    # Print results
    print("=" * 50)
    print("  Test Results")
    print("=" * 50)
    print(f"Simple Model Test Accuracy : {simple_test_acc:.4f}")
    print(f"KFold Model Test Accuracy  : {kfold_test_acc:.4f}")
