from pathlib import Path
from typing import Any, Dict

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


def evaluate_model(
    model,
    X_test,
    y_test,
    label_encoder,
) -> Dict[str, Any]:
    """
    Evaluate a trained classification model on a test set.

    Inputs:
        - model: Trained sklearn estimator.
        - X_test (pd.Series): Test feature matrix.
        - y_test (pd.Series): True test labels.
        - label_encoder: Fitted label encoder for decoding class names.

    Outputs:
        - dict: A dictionary of evaluation metrics including.
    """
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

    return metrics


if __name__ == "__main__":
    # Load test datasets
    X_test = pd.read_csv("../../data/testing/X_test.csv").squeeze()
    y_test = pd.read_csv("../../data/testing/y_test.csv").squeeze()

    # Locate result directories
    base_dir = Path("../../results/models/")
    le = joblib.load(base_dir / "label_encoder.pkl")
    model_types = [d.name for d in base_dir.iterdir() if d.is_dir()]
    variants = ["simple", "kfold"]

    # Custom style
    plt.style.use("../../misc/custom_style.mplstyle")

    # Loop through each model type
    for model_type in model_types:
        # Set paths for metrics, results, and images
        results_dir = base_dir / model_type
        results_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = Path("../../results/metrics/") / model_type
        metrics_dir.mkdir(parents=True, exist_ok=True)
        images_dir = Path(f"../../images/metrics/{model_type}/")
        images_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over each variant of the model
        for variant in variants:
            # Define full model name and path
            model_name = f"{model_type}_{variant}"
            model_path = results_dir / f"{model_name}.pkl"

            # Load model and evaluate performance
            model = joblib.load(model_path)

            # For ANN models, cast X_test to float32 before evaluating
            if model_type == 'ann':
                X_input = X_test.values.astype('float32')
            else:
                X_input = X_test

            # Evaluate model
            metrics = evaluate_model(model, X_input, y_test, le)

            # Save classification report to CSV
            df_report = pd.DataFrame(metrics["classification_report"]).T
            df_report.to_csv(
                metrics_dir / f"{model_name}_classification_report.csv"
            )

            # Create evaluation summary
            summary = "=" * 48 + "\n"
            summary += f"  Evaluation Summary: {model_name}\n"
            summary += "=" * 48 + "\n"
            summary += (
                f"  Accuracy (95% CI) : {metrics['accuracy']:.4f} "
                f"({metrics['accuracy_ci_95'][0]:.4f}, {metrics['accuracy_ci_95'][1]:.4f})\n"
            )
            summary += (
                f"  Precision (macro) : {metrics['precision_macro']:.4f}\n"
            )
            summary += f"  Recall (macro)    : {metrics['recall_macro']:.4f}\n"
            summary += f"  F1 Score (macro)  : {metrics['f1_macro']:.4f}\n"
            summary += (
                f"  ROC AUC (macro)   : {metrics['roc_auc_ovr_macro']:.4f}\n"
            )
            summary += "=" * 48 + "\n\n"

            # Create a figure with 2 subplots: Confusion matrix and ROC-AUC
            fig, axs = plt.subplots(1, 2)

            # Plot confusion matrix
            cm = metrics["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                ax=axs[0],
            )
            axs[0].set_title(f"Confusion Matrix - {model_name}")
            axs[0].set_xlabel("Predicted")
            axs[0].set_ylabel("True")

            # Plot ROC AUC curves
            y_test_bin = metrics["y_test_bin"]
            y_proba = metrics["y_proba"]
            n_classes = y_test_bin.shape[1]

            # Store the best FPR-TPR pair and AUC for each class
            best_fpr_tpr = {}
            auc_values = {}

            # Compute ROC-AUC curve for each clas
            for j in range(n_classes):
                # Collect true positive rates, false positive rates and thresholds
                fpr, tpr, thresholds = roc_curve(
                    y_test_bin[:, j], y_proba[:, j]
                )
                auc_value = auc(fpr, tpr)
                auc_values[j] = auc_value

                # Find best threshold using Youden's J
                youden_j = tpr - fpr
                best_idx = np.argmax(youden_j)
                best_fpr_tpr[j] = (
                    fpr[best_idx],
                    tpr[best_idx],
                    thresholds[best_idx],
                )
                summary += (
                    f"Class {le.classes_[j]}: Best FPR={fpr[best_idx]:.2f}, "
                    f"TPR={tpr[best_idx]:.2f}, Threshold={thresholds[best_idx]:.2f}, "
                    f"AUC={auc_value:.2f}\n"
                )

                # Plot ROC curve
                axs[1].plot(
                    fpr,
                    tpr,
                    label=f"{le.classes_[j]} (AUC = {auc_value:.2f})",
                    lw=2,
                )

            # Add diagonal and labels to ROC plot
            axs[1].plot([0, 1], [0, 1], color="navy", linestyle="--")
            axs[1].set_title(f"ROC-AUC - {model_name}")
            axs[1].set_xlabel("False Positive Rate")
            axs[1].set_ylabel("True Positive Rate")
            axs[1].legend(loc="lower right")

            # Save evaluation summary
            summary_path = (
                metrics_dir / f"{model_name}_classification_summary.txt"
            )
            open(summary_path, "w", encoding="utf-8").write(summary)

            # Save figure
            fig.tight_layout()
            fig.savefig(images_dir / f"{model_name}_metrics.png")
            plt.show()
            plt.close(fig)
