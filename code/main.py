import random
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import torch
from models.ann import SimpleANN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skorch import NeuralNetClassifier
from torch import nn
from utils.CoefficientThresholdLasso import CoefficientThresholdLasso
from utils.parse_best_params import parse_best_params

# Load dataset
raw_df = pd.read_csv("../data/datasets/raw_acdc_radiomics.csv")
norm_df = pd.read_csv("../data/datasets/norm_acdc_radiomics.csv")

X_temp_norm = pd.read_csv("../data/kfold/X_temp_norm.csv").squeeze()
X_test_raw = pd.read_csv("../data/simple/X_test_norm.csv").squeeze()
X_test_norm = pd.read_csv("../data/simple/X_test_norm.csv").squeeze()
X_train_norm = pd.read_csv("../data/simple/X_train_norm.csv").squeeze()
X_train_raw = pd.read_csv("../data/simple/X_train_raw.csv").squeeze()
y_temp_norm = pd.read_csv("../data/kfold/y_temp_norm.csv").squeeze()
y_test_raw = pd.read_csv("../data/simple/y_test_raw.csv").squeeze()
y_test_norm = pd.read_csv("../data/simple/y_test_norm.csv").squeeze()
y_train_raw = pd.read_csv("../data/simple/y_train_raw.csv").squeeze()
y_train_norm = pd.read_csv("../data/simple/y_train_norm.csv").squeeze()

# Separate features and classes
X_raw = raw_df.drop(columns=["class"])
y_raw = raw_df["class"]
X_norm = norm_df.drop(columns=["class"])
y_norm = norm_df["class"]

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
    "knn_baseline": (
        KNeighborsClassifier(),
        X_train_raw,
        y_train_raw,
    ),
    "knn_simple": (
        KNeighborsClassifier(**best_params_knn_simple),
        X_train_norm,
        y_train_norm,
    ),
    "knn_kfold": (
        KNeighborsClassifier(**best_params_knn_kfold),
        X_temp_norm,
        y_temp_norm,
    ),
    "rf_baseline": (
        RandomForestClassifier(random_state=42),
        X_train_raw,
        y_train_raw,
    ),
    "rf_simple": (
        RandomForestClassifier(**best_params_rf_simple, random_state=42),
        X_train_norm,
        y_train_norm,
    ),
    "rf_kfold": (
        RandomForestClassifier(**best_params_rf_kfold, random_state=42),
        X_temp_norm,
        y_temp_norm,
    ),
    "svm_baseline": (
        SVC(probability=True, random_state=42),
        X_train_raw,
        y_train_raw,
    ),
    "svm_simple": (
        SVC(**best_params_svm_simple, probability=True, random_state=42),
        X_train_norm,
        y_train_norm,
    ),
    "svm_kfold": (
        SVC(**best_params_svm_kfold, probability=True, random_state=42),
        X_temp_norm,
        y_temp_norm,
    ),
    "ann_baseline": (
        NeuralNetClassifier(
            module=SimpleANN,
            module__input_dim=X_train_raw.shape[1],
            module__output_dim=len(np.unique(y_train_raw)),
            module__hidden_layers=1,
            module__hidden_size=1,
            module__activation_fn=nn.ReLU,
            criterion=cast(Any, nn.CrossEntropyLoss),
            verbose=0,
        ),
        X_train_raw,
        y_train_raw,
    ),
    "ann_simple": (
        NeuralNetClassifier(
            module=SimpleANN,
            module__input_dim=len(np.unique(y_norm)) - 1,
            module__output_dim=len(np.unique(y_norm)),
            module__hidden_layers=best_params_ann_simple["hidden_layers"],
            module__hidden_size=best_params_ann_simple["hidden_size"],
            module__activation_fn=best_params_ann_simple["activation_fn"],
            max_epochs=best_params_ann_simple["max_epochs"],
            lr=best_params_ann_simple["learning_rate"],
            optimizer=best_params_ann_simple["optimizer"],
            criterion=cast(Any, nn.CrossEntropyLoss),
            verbose=0,
        ),
        X_train_norm,
        y_train_norm,
    ),
    "ann_kfold": (
        NeuralNetClassifier(
            module=SimpleANN,
            module__input_dim=len(np.unique(y_norm)) - 1,
            module__output_dim=len(np.unique(y_norm)),
            module__hidden_layers=best_params_ann_kfold["hidden_layers"],
            module__hidden_size=best_params_ann_kfold["hidden_size"],
            module__activation_fn=best_params_ann_kfold["activation_fn"],
            max_epochs=best_params_ann_kfold["max_epochs"],
            lr=best_params_ann_kfold["learning_rate"],
            optimizer=best_params_ann_kfold["optimizer"],
            criterion=cast(Any, nn.CrossEntropyLoss),
            verbose=0,
        ),
        X_temp_norm,
        y_temp_norm,
    ),
}

# Model generation loop
for name, (model, X_fit, y_fit) in model_configs.items():
    if "baseline" in name:
        pipeline = Pipeline(
            [
                (name.split("_")[0], model),
            ]
        )
        X_input = X_test_raw.values.astype("float32") if "ann" in name else X_test_raw
        y_eval = y_test_raw
    else:
        pipeline = Pipeline(
            [
                ("ctl", CoefficientThresholdLasso()),
                ("lda", LinearDiscriminantAnalysis()),
                (name.split("_")[0], model),
            ]
        )
        X_input = X_test_norm.values.astype("float32") if "ann" in name else X_test_norm
        y_eval = y_test_norm

    # Fit pipeline
    pipeline.fit(
        X_fit.values.astype("float32") if "ann" in name else X_fit,
        y_fit.values.astype("long") if "ann" in name else y_fit,
    )

    # Save model
    save_path = Path(f"../results/models/{name.split('_')[0]}")
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path / f"{name}.pkl")
