import random
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import torch
from metrics.metrics import evaluate_model
from models.ann import SimpleANN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from skorch import NeuralNetClassifier
from torch import nn
from utils.CoefficientThresholdLasso import CoefficientThresholdLasso
from utils.parse_best_params import parse_best_params

# Load dataset
df = pd.read_csv("../data/datasets/raw_acdc_radiomics.csv")

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

# Parse best hyperparameters
best_params_knn_simple = parse_best_params(
    "../results/models/knn/knn_summary.txt", line_range=(3, 6)
)
best_params_knn_kfold = parse_best_params(
    "../results/models/knn/knn_summary.txt", line_range=(12, 15)
)
best_params_rf_simple = parse_best_params(
    "../results/models/rf/rf_summary.txt", line_range=(3, 6)
)
best_params_rf_kfold = parse_best_params(
    "../results/models/rf/rf_summary.txt", line_range=(12, 15)
)
best_params_svm_simple = parse_best_params(
    "../results/models/svm/svm_summary.txt", line_range=(3, 6)
)
best_params_svm_kfold = parse_best_params(
    "../results/models/svm/svm_summary.txt", line_range=(12, 15)
)
best_params_ann_simple = parse_best_params(
    "../results/models/ann/ann_summary.txt", line_range=(3, 9)
)
best_params_ann_kfold = parse_best_params(
    "../results/models/ann/ann_summary.txt", line_range=(12, 20)
)

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Model configurations
model_configs = {
    'knn_simple': (
        KNeighborsClassifier(**best_params_knn_simple),
        X_train,
        y_train,
    ),
    'knn_kfold': (
        KNeighborsClassifier(**best_params_knn_kfold),
        X_temp,
        y_temp,
    ),
    'rf_simple': (
        RandomForestClassifier(**best_params_rf_simple, random_state=42),
        X_train,
        y_train,
    ),
    'rf_kfold': (
        RandomForestClassifier(**best_params_rf_kfold, random_state=42),
        X_temp,
        y_temp,
    ),
    'svm_simple': (
        SVC(**best_params_svm_simple, probability=True, random_state=42),
        X_train,
        y_train,
    ),
    'svm_kfold': (
        SVC(**best_params_svm_kfold, probability=True, random_state=42),
        X_temp,
        y_temp,
    ),
    'ann_simple': (
        NeuralNetClassifier(
            module=SimpleANN,
            module__input_dim=len(np.unique(y)) - 1,
            module__output_dim=len(np.unique(y)),
            module__hidden_layers=best_params_ann_simple['hidden_layers'],
            module__hidden_size=best_params_ann_simple['hidden_size'],
            module__activation_fn=best_params_ann_simple['activation_fn'],
            max_epochs=best_params_ann_simple['max_epochs'],
            lr=best_params_ann_simple['learning_rate'],
            optimizer=best_params_ann_simple['optimizer'],
            criterion=cast(Any, nn.CrossEntropyLoss),
            verbose=0,
        ),
        X_train,
        y_train,
    ),
    'ann_kfold': (
        NeuralNetClassifier(
            module=SimpleANN,
            module__input_dim=len(np.unique(y)) - 1,
            module__output_dim=len(np.unique(y)),
            module__hidden_layers=best_params_ann_kfold['hidden_layers'],
            module__hidden_size=best_params_ann_kfold['hidden_size'],
            module__activation_fn=best_params_ann_kfold['activation_fn'],
            max_epochs=best_params_ann_kfold['max_epochs'],
            lr=best_params_ann_kfold['learning_rate'],
            optimizer=best_params_ann_kfold['optimizer'],
            criterion=cast(Any, nn.CrossEntropyLoss),
            verbose=0,
        ),
        X_temp,
        y_temp,
    ),
}

# Training/Testing loop
for name, (model, X_fit, y_fit) in model_configs.items():
    # Define pipeline
    pipeline = Pipeline(
        [
            ('ctl', CoefficientThresholdLasso()),
            ('lda', LinearDiscriminantAnalysis()),
            (name.split('_')[0], model),
        ]
    )

    # Fit model
    pipeline.fit(
        X_fit.values.astype('float32') if 'ann' in name else X_fit,
        y_fit.values.astype('long') if 'ann' in name else y_fit,
    )

    # For ANN models, cast X_test to float32 before evaluating
    if 'ann' in name:
        X_input = X_test.values.astype('float32')
    else:
        X_input = X_test

    # Evaluate model
    metrics = evaluate_model(
        pipeline,
        X_input,
        y_test,
        le,
    )

    # Save model
    save_path = Path(f"../results/models/{name.split('_')[0]}")
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path / f"{name}.pkl")
