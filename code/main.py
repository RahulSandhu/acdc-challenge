from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
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
from torch import nn, optim
from utils.CoefficientThresholdLasso import CoefficientThresholdLasso
from utils.parse_best_params import parse_best_params

# Load dataset
df = pd.read_csv("../data/raw/acdc_radiomics.csv")

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

# Parse best hyperparameters
best_params_knn_simple = parse_best_params(
    "../results/models/knn/knn_summary.txt", simple=True
)
best_params_knn_kfold = parse_best_params(
    "../results/models/knn/knn_summary.txt", simple=False
)
best_params_rf_simple = parse_best_params(
    "../results/models/rf/rf_summary.txt", simple=True
)
best_params_rf_kfold = parse_best_params(
    "../results/models/rf/rf_summary.txt", simple=False
)
best_params_svm_simple = parse_best_params(
    "../results/models/svm/svm_summary.txt", simple=True
)
best_params_svm_kfold = parse_best_params(
    "../results/models/svm/svm_summary.txt", simple=False
)
best_params_ann_simple = parse_best_params(
    "../results/models/ann/ann_summary.txt",
    simple=True,
    simple_lines=(3, 7),
    kfold_lines=(13, 17),
)
best_params_ann_kfold = parse_best_params(
    "../results/models/ann/ann_summary.txt",
    simple=False,
    simple_lines=(3, 7),
    kfold_lines=(13, 17),
)

# Setup dictionaries
models = {}
metrics = {}

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
            module__input_dim=4,
            module__hidden_layers=best_params_ann_simple['hidden_layers'],
            module__hidden_size=best_params_ann_simple['hidden_size'],
            module__output_dim=len(np.unique(y)),
            module__activation_fn=best_params_ann_simple['activation_fn'],
            max_epochs=30,
            lr=best_params_ann_simple['learning_rate'],
            optimizer=optim.Adam,
            criterion=nn.CrossEntropyLoss,
            verbose=0,
        ),
        X_train,
        y_train,
    ),
    'ann_kfold': (
        NeuralNetClassifier(
            module=SimpleANN,
            module__input_dim=4,
            module__hidden_layers=best_params_ann_kfold['hidden_layers'],
            module__hidden_size=best_params_ann_kfold['hidden_size'],
            module__output_dim=len(np.unique(y)),
            module__activation_fn=best_params_ann_kfold['activation_fn'],
            max_epochs=30,
            lr=best_params_ann_kfold['learning_rate'],
            optimizer=optim.Adam,
            criterion=nn.CrossEntropyLoss,
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
        X_fit.values.astype('float32'),
        y_fit.values.astype('long') if 'ann' in name else y_fit,
    )

    # Update models
    models[name] = pipeline

    # Evaluate model
    metrics[name] = evaluate_model(
        pipeline,
        X_test,
        y_test,
        le,
    )

    # Save model
    strategy = name.split('_')[0]
    Path(f"../results/models/{strategy}").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, f"../results/models/{strategy}/{name}.pkl")

# Organize metrics into a DataFrame
metrics_df = pd.DataFrame(metrics).T

# Metrics to plot
metrics_to_plot = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc_ovr_macro",
]

# Simplified model names
simplified_names = {
    "knn_simple": "KNN Simple",
    "knn_kfold": "KNN Kfold",
    "rf_simple": "RF Simple",
    "rf_kfold": "RF Kfold",
    "svm_simple": "SVM Simple",
    "svm_kfold": "SVM Kfold",
    "ann_simple": "ANN Simple",
    "ann_kfold": "ANN Kfold",
}

# Ensure output directory exists
img_dir = Path("../images/metrics/")
img_dir.mkdir(parents=True, exist_ok=True)

# Custom style
plt.style.use("../misc/custom_style.mplstyle")

# Setup for grouped bars
num_metrics = len(metrics_to_plot)
model_names = metrics_df.index.tolist()
num_models = len(model_names)
bar_width = 0.12
x = np.arange(num_metrics) * 2

# Color palette
colors = colormaps['Set2'](np.linspace(0, 1, num_models))

# Create figure
plt.figure(figsize=(14, 8))

# Plot each model's bars
for idx, (model_name, color) in enumerate(zip(model_names, colors)):
    # Calculate horizontal offset for each model so bars are grouped side by side
    offset = (idx - num_models / 2) * bar_width + bar_width / 2

    # Plot the bars for this model at the correct offset
    plt.bar(
        x + offset,
        metrics_df.loc[model_name, metrics_to_plot],
        width=bar_width,
        label=simplified_names.get(model_name, model_name),
        color=color,
        edgecolor="black",
        linewidth=0.7,
    )

# Add red stars for best models
for i, metric in enumerate(metrics_to_plot):
    # Find the model name with the highest score for this metric
    best_model_idx = str(metrics_df[metric].idxmax())
    best_model_position = model_names.index(best_model_idx)
    best_offset = (
        best_model_position - num_models / 2
    ) * bar_width + bar_width / 2
    best_score = metrics_df.loc[best_model_idx, metric]

    # Plot a red star above the best model's bar
    plt.text(
        x[i] + best_offset,
        best_score + 0.01,
        "*",
        color="red",
        fontsize=20,
        ha='center',
    )

# Formatting and save figure
plt.xticks(
    x,
    [metric.replace('_', ' ').title() for metric in metrics_to_plot],
    fontsize=12,
)
plt.ylabel("Score")
plt.title("Model Comparison Across Metrics")
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
)
plt.ylim(0, 1.15)
plt.tight_layout()
plt.savefig(img_dir / "best_model.png")
plt.show()
