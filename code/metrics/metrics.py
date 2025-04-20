from pathlib import Path
from typing import Any, Dict, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample


def evaluate_model(
    model,
    X_test: Union[pd.Series, np.ndarray, pd.DataFrame],
    y_test: Union[pd.Series, np.ndarray, pd.DataFrame],
    label_encoder,
    model_name: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Evaluate a trained classification model on a test set and save its report.

    Inputs:
        - model: Trained sklearn estimator with predict and predict_proba
          methods.
        - X_test (pd.DataFrame or np.ndarray): Test feature matrix.
        - y_test (pd.Series or np.ndarray): True test labels.
        - label_encoder: Fitted label encoder for decoding class
          names.
        - model_name (str): Identifier for the model used in printing and
          filenames.
        - out_dir (Path): Directory path to save the classification
          report CSV.

    Outputs:
        - dict: A dictionary of evaluation metrics including.
    """
    # Convert DataFrame to NumPy array to avoid warnings
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    # Predict labels for the test set
    y_pred = model.predict(X_test)

    # Compute performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division='warn'
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division='warn'
    )
    f1 = f1_score(y_test, y_pred, average="macro", zero_division='warn')

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate 95% confidence interval for accuracy
    n = len(y_test)
    z = 1.96
    margin = z * np.sqrt((accuracy * (1 - accuracy)) / n)
    acc_ci = (accuracy - margin, accuracy + margin)

    # Predict probabilities and prepare binarized true labels for ROC-AUC
    y_proba = model.predict_proba(X_test)
    y_test_bin = label_binarize(
        y_test, classes=np.arange(len(label_encoder.classes_))
    )
    roc_auc = roc_auc_score(
        y_test_bin, y_proba, average="macro", multi_class="ovr"
    )

    # Aggregate all metrics into a dictionary
    metrics = {
        "accuracy": accuracy,
        "accuracy_ci_95": acc_ci,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "roc_auc_ovr_macro": roc_auc,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
        ),
        "y_proba": y_proba,
        "y_test_bin": y_test_bin,
    }

    # Save classification report to CSV
    df_report = pd.DataFrame(metrics["classification_report"]).T
    df_report.to_csv(out_dir / f"{model_name}_classification_report.csv")

    print("\n" + "=" * 48)
    print(f"  Evaluation Summary: {model_name}")
    print("=" * 48)
    print(
        f"  Accuracy (95% CI) : {accuracy:.4f} ({acc_ci[0]:.4f}, {acc_ci[1]:.4f})"
    )
    print(f"  Precision (macro) : {precision:.4f}")
    print(f"  Recall (macro)    : {recall:.4f}")
    print(f"  F1 Score (macro)  : {f1:.4f}")
    print(f"  ROC AUC (macro)   : {roc_auc:.4f}")
    print("=" * 48 + "\n")

    return metrics


if __name__ == "__main__":
    # Custom style
    plt.style.use("../../misc/custom_style.mplstyle")

    # Load test datasets
    X_test = pd.read_csv("../../data/processed/X_test_lda.csv")
    y_test = pd.read_csv("../../data/processed/y_test.csv").squeeze()

    # Locate result directories
    base_dir = Path("../../results/models/")
    le = joblib.load(base_dir / "label_encoder.pkl")
    model_types = [d.name for d in base_dir.iterdir() if d.is_dir()]
    variants = ["simple", "kfold"]

    # Loop through each model type
    for model_type in model_types:
        # Set paths for metrics and results
        metrics_dir = Path("../../results/metrics/") / model_type
        metrics_dir.mkdir(parents=True, exist_ok=True)

        results_dir = base_dir / model_type
        results_dir.mkdir(parents=True, exist_ok=True)

        # Prepare directory for saving metric plots
        images_dir = Path(f"../../images/metrics/{model_type}/")
        images_dir.mkdir(parents=True, exist_ok=True)

        # Initialize figures for confusion matrices and ROC curves
        fig_cm, axs_cm = plt.subplots(1, 2, figsize=(10, 8))
        fig_auc, axs_auc = plt.subplots(1, 2, figsize=(10, 8))
        axs_cm = axs_cm.ravel()
        axs_auc = axs_auc.ravel()

        # Iterate over each variant of the model
        for i, variant in enumerate(variants):
            # Define full model name and path
            model_name = f"{model_type}_{variant}"
            model_path = results_dir / f"{model_name}.pkl"

            # Load model and evaluate performance
            model = joblib.load(model_path)
            metrics = evaluate_model(
                model,
                X_test,
                (
                    pd.Series(y_test)
                    if not isinstance(y_test, (pd.Series, np.ndarray))
                    else y_test
                ),
                le,
                model_name,
                metrics_dir,
            )

            # Plot confusion matrix for this model
            cm = metrics["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                ax=axs_cm[i],
            )
            axs_cm[i].set_title(model_name)
            axs_cm[i].set_xlabel("Predicted")
            axs_cm[i].set_ylabel("True")

            # Prepare data for ROC curve computation
            y_test_bin = metrics["y_test_bin"]
            y_proba = metrics["y_proba"]
            n_classes = y_test_bin.shape[1]

            # Set bootstrap parameters for ROC variability
            n_bootstrap = 100
            mean_fpr = np.linspace(0, 1, 2000)

            # Compute ROC curve with bootstrap for each class
            for j in range(n_classes):
                # Collect true positive rates across bootstraps
                tprs = []

                # Perform bootstrap sampling
                for _ in range(n_bootstrap):
                    # Resample test indices with stratification
                    indices = resample(
                        np.arange(len(y_test_bin)),
                        replace=True,
                        stratify=y_test_bin[:, j],
                        random_state=42,
                    )

                    # Compute ROC for resampled data
                    fpr, tpr, _ = roc_curve(
                        y_test_bin[indices, j], y_proba[indices, j]
                    )

                    # Interpolate TPR at common FPR grid
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)

                # Calculate mean ROC and AUC
                tprs = np.array(tprs)
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)

                # Plot mean ROC curve
                axs_auc[i].plot(
                    mean_fpr,
                    mean_tpr,
                    label=f"{le.classes_[j]} (AUC = {mean_auc:.2f})",
                )

            # Add diagonal and labels to ROC plot
            axs_auc[i].plot(
                [0, 1], [0, 1], color="navy", lw=1.5, linestyle="--"
            )
            axs_auc[i].set_title(model_name)
            axs_auc[i].set_xlabel("FPR")
            axs_auc[i].set_ylabel("TPR")
            axs_auc[i].legend(loc="lower right")

        # Save figures
        fig_cm.tight_layout()
        fig_auc.tight_layout()
        fig_cm.savefig(images_dir / f"{model_type}_confusion_matrix.png")
        fig_auc.savefig(images_dir / f"{model_type}_roc_auc.png")
        plt.show()
