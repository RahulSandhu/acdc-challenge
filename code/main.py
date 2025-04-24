from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from models.knn import knn
from models.rf import rf
from models.svm import svm
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("../data/raw/acdc_radiomics.csv")

# Encode labels
le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])

# Save the label encoder
Path("../results/models/").mkdir(parents=True, exist_ok=True)
joblib.dump(le, "../results/models/label_encoder.pkl")

# Separate features and classes
X = df.drop(columns=["class"])
y = df["class"]

# Apply StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_features, columns=X.columns)

# Normalized dataframe
norm_df = pd.concat([scaled_df, y.reset_index(drop=True)], axis=1)

# Save normalized dataframe
Path("../data/processed").mkdir(parents=True, exist_ok=True)
norm_df.to_csv("../data/processed/norm_acdc_radiomics.csv", index=False)

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
pd.DataFrame(X_test).to_csv("../data/processed/X_test.csv", index=False)
y_test.to_frame().to_csv("../data/processed/y_test.csv", index=False)

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

# Define hyperparameters for the different models
knn_n = list(range(1, 57, 2))

n_estimators = [10, 50, 100, 200]
max_depth = [None, 3, 5, 10, 20, 30]
min_samples_leaf = [1, 2, 4, 8, 10, 12, 14, 16, 18, 20]

svm_c = [0.1, 1, 10, 100]

# Run and save KNN
knn_models = knn(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_temp=X_temp,
    y_temp=y_temp,
    cv=cv,
    knn_n=knn_n,
    pen=0.5,
)

Path("../results/models/knn").mkdir(parents=True, exist_ok=True)
joblib.dump(knn_models['simple'], "../results/models/knn/knn_simple.pkl")
joblib.dump(knn_models['kfold'], "../results/models/knn/knn_kfold.pkl")

# Run and save RF
rf_models = rf(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_temp=X_temp,
    y_temp=y_temp,
    cv=cv,
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    SEED=42,
    pen=0.5,
)

Path("../results/models/rf").mkdir(parents=True, exist_ok=True)
joblib.dump(rf_models['simple'], "../results/models/rf/rf_simple.pkl")
joblib.dump(rf_models['kfold'], "../results/models/rf/rf_kfold.pkl")

# Run and save SVM
svm_models = svm(
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

Path("../results/models/svm").mkdir(parents=True, exist_ok=True)
joblib.dump(svm_models['simple'], "../results/models/svm/svm_simple.pkl")
joblib.dump(svm_models['kfold'], "../results/models/svm/svm_kfold.pkl")
