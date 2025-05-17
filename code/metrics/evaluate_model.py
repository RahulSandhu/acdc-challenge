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
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate 95% confidence interval for accuracy
    n = len(y_test)
    z = 1.96
    margin = z * np.sqrt((accuracy * (1 - accuracy)) / n)
    acc_ci = (accuracy - margin, accuracy + margin)

    # Predict probabilities and prepare binarized true labels for ROC-AUC
    y_proba = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
    roc_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

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
            zero_division=0,
        ),
        "y_proba": y_proba,
        "y_test_bin": y_test_bin,
    }

    return metrics


if __name__ == "__main__":
    # Load normalized test dataset
    X_test_norm = pd.read_csv("../../data/simple/X_test_norm.csv").squeeze()
    y_test_norm = pd.read_csv("../../data/simple/y_test_norm.csv").squeeze()
    X_test_raw = pd.read_csv("../../data/simple/X_test_raw.csv").squeeze()
    y_test_raw = pd.read_csv("../../data/simple/y_test_raw.csv").squeeze()

    # Custom style
    plt.style.use("../../config/custom_style.mplstyle")

    # Locate base directory where models are saved
    base_dir = Path("../../results/models/")

    # Load the label encoder used during training
    le = joblib.load(base_dir / "label_encoder.pkl")

    # Identify model type folders
    model_types = [d.name for d in base_dir.iterdir() if d.is_dir()]

    # Define model variants to evaluate
    variants = ["baseline", "simple", "kfold"]

    # Loop through each model type directory
    for model_type in model_types:
        # Define result, metrics, and image output directories
        results_dir = base_dir / model_type
        metrics_dir = Path("../../results/metrics/") / model_type
        images_dir = Path(f"../../images/metrics/{model_type}/")
        results_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # Iterate through each model variant
        for variant in variants:
            # Construct model name and load the corresponding file
            model_name = f"{model_type}_{variant}"
            model_path = results_dir / f"{model_name}.pkl"
            model = joblib.load(model_path)

            # Use raw data if model has "baseline" in its name, otherwise use normalized
            if "baseline" in model_name:
                X_test = X_test_raw
                y_test = y_test_raw
            else:
                X_test = X_test_norm
                y_test = y_test_norm

            # If the model is ANN, convert input to float32
            if model_type == "ann":
                X_input = X_test.values.astype("float32")
            else:
                X_input = X_test

            # Evaluate the model and collect metrics
            metrics = evaluate_model(model, X_input, y_test, le)

            # Save classification report to CSV
            df_report = pd.DataFrame(metrics["classification_report"]).T
            df_report.to_csv(metrics_dir / f"{model_name}_classification_report.csv")

            summary = "=" * 48 + "\n"
            summary += f"  Evaluation Summary: {model_name}\n"
            summary += "=" * 48 + "\n"
            summary += (
                f"  Accuracy (95% CI) : {metrics['accuracy']:.4f} "
                f"({metrics['accuracy_ci_95'][0]:.4f}, {metrics['accuracy_ci_95'][1]:.4f})\n"
            )
            summary += f"  Precision (macro) : {metrics['precision_macro']:.4f}\n"
            summary += f"  Recall (macro)    : {metrics['recall_macro']:.4f}\n"
            summary += f"  F1 Score (macro)  : {metrics['f1_macro']:.4f}\n"
            summary += f"  ROC AUC (macro)   : {metrics['roc_auc_ovr_macro']:.4f}\n"
            summary += "=" * 48 + "\n\n"

            # Define figures
            figcm, axscm = plt.subplots(1, 2)

            # Plot confusion matrix
            cm = metrics["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                ax=axscm[0],
            )
            axscm[0].set_title(f"Confusion Matrix - {model_name}")
            axscm[0].set_xlabel("Predicted")
            axscm[0].set_ylabel("True")

            # Prepare data for ROC-AUC curves
            y_test_bin = metrics["y_test_bin"]
            y_proba = metrics["y_proba"]
            n_classes = y_test_bin.shape[1]

            # Initialize storage for best thresholds and AUCs
            best_fpr_tpr = {}
            auc_values = {}

            # Plot ROC curves per class and store AUC info
            for j in range(n_classes):
                fpr, tpr, thresholds = roc_curve(y_test_bin[:, j], y_proba[:, j])
                auc_value = auc(fpr, tpr)
                auc_values[j] = auc_value
                youden_j = tpr - fpr
                best_idx = np.argmax(youden_j)
                best_fpr_tpr[j] = (fpr[best_idx], tpr[best_idx], thresholds[best_idx])

                summary += (
                    f"Class {le.classes_[j]}: Best FPR={fpr[best_idx]:.2f}, "
                    f"TPR={tpr[best_idx]:.2f}, Threshold={thresholds[best_idx]:.2f}, "
                    f"AUC={auc_value:.2f}\n"
                )

                axscm[1].plot(
                    fpr,
                    tpr,
                    label=f"{le.classes_[j]} (AUC = {auc_value:.2f})",
                    lw=2,
                )

            # Finalize ROC plot formatting
            axscm[1].plot([0, 1], [0, 1], color="navy", linestyle="--")
            axscm[1].set_title(f"ROC-AUC - {model_name}")
            axscm[1].set_xlabel("False Positive Rate")
            axscm[1].set_ylabel("True Positive Rate")
            axscm[1].legend(loc="lower right")

            # Write summary text to file
            summary_path = metrics_dir / f"{model_name}_classification_summary.txt"
            open(summary_path, "w", encoding="utf-8").write(summary)

            # Save and display figure
            figcm.tight_layout()
            figcm.savefig(images_dir / f"{model_name}_metrics.png")
            plt.show()

            # Extract and rename per-class metrics for plotting
            df_class_metrics = df_report.loc[
                le.classes_, ["precision", "recall", "f1-score"]
            ].copy()
            df_class_metrics.columns = ["Precision", "Recall", "F1-Score"]

            # Setup for bar chart of class-wise metrics
            x = np.arange(len(df_class_metrics))
            width = 0.25
            figbar, axbar = plt.subplots(figsize=(14, 10))

            # Define colors for each metric
            colors = {
                "Precision": "#7FB3D5",
                "Recall": "#F7DC6F",
                "F1-Score": "#82E0AA",
            }

            # Plot grouped bars
            bars1 = axbar.bar(
                x - width,
                df_class_metrics["Precision"],
                width,
                label="Precision",
                color=colors["Precision"],
                edgecolor="black",
                linewidth=0.8,
            )
            bars2 = axbar.bar(
                x,
                df_class_metrics["Recall"],
                width,
                label="Recall",
                color=colors["Recall"],
                edgecolor="black",
                linewidth=0.8,
            )
            bars3 = axbar.bar(
                x + width,
                df_class_metrics["F1-Score"],
                width,
                label="F1-Score",
                color=colors["F1-Score"],
                edgecolor="black",
                linewidth=0.8,
            )

            # Configure axes and legend
            axbar.set_ylabel("Score", fontsize=12)
            axbar.set_title(
                f"Class-wise Precision, Recall and F1-Score ({model_name})",
                fontsize=14,
                weight="bold",
            )
            axbar.set_xticks(x)
            axbar.set_xticklabels(df_class_metrics.index, fontsize=11)
            axbar.set_ylim(0, 1.1)
            axbar.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

            # Annotate bars with values
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    axbar.annotate(
                        f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

            # Save and display figure
            figbar.tight_layout()
            figbar.savefig(images_dir / f"{model_name}_class_wise_metrics.png", dpi=300)
            plt.show()
